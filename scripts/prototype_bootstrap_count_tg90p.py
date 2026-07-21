"""Prototype an exact TG90p bootstrap count engine outside icclim runtime."""
# ruff: noqa: ANN001, ANN202

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from xclim.core.calendar import percentile_doy, resample_doy
from xclim.core.units import convert_units_to
from xclim.core.utils import nan_calc_percentiles

if TYPE_CHECKING:
    from collections.abc import Mapping


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prototype exact TG90p bootstrap count without dask graph expansion.",
    )
    parser.add_argument("--file-glob", required=True)
    parser.add_argument("--lat-min", type=float, default=35.0)
    parser.add_argument("--lat-max", type=float, default=45.0)
    parser.add_argument("--lon-min", type=float, default=0.0)
    parser.add_argument("--lon-max", type=float, default=10.0)
    parser.add_argument("--time-range-start", default="1970-01-01")
    parser.add_argument("--time-range-end", default="1974-12-31")
    parser.add_argument("--base-period-start", default="1970-01-01")
    parser.add_argument("--base-period-end", default="1972-12-31")
    parser.add_argument(
        "--target-unit",
        default="degC",
        help="Unit used before computing TG90P, matching icclim standard indices.",
    )
    parser.add_argument("--reference-result", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--engine",
        default="xarray-loop",
        choices=[
            "xarray-loop",
            "numpy-index",
            "numpy-numba",
            "numpy-numba-presort",
        ],
        help="Prototype implementation to run.",
    )
    return parser.parse_args()


def _year_groups(da: xr.DataArray) -> Mapping[np.datetime64, slice]:
    return da.resample(time="YS").groups


def _contains_year(da: xr.DataArray, label: np.datetime64) -> bool:
    return int(np.datetime_as_string(label, unit="Y")) in da.get_index("time").year


def _replace_year(
    ref: xr.DataArray,
    groups: Mapping[np.datetime64, slice],
    target_label: np.datetime64,
    donor_slice: slice,
) -> xr.DataArray:
    target_time = ref.time[groups[target_label]]
    donor = ref.isel(time=donor_slice)
    out = ref.copy(deep=True)
    if donor.sizes["time"] == target_time.size:
        replacement = donor.data
    elif target_time.size == 365:
        replacement = donor.convert_calendar("noleap").data
    elif target_time.size == 366:
        replacement = donor.convert_calendar("366_day", missing=np.nan).data
    else:
        replacement = donor.data[: target_time.size]
    out.loc[{"time": target_time}] = replacement
    return out


def _tg90p_bootstrap_count(
    da: xr.DataArray,
    *,
    base_period: tuple[str, str],
) -> xr.DataArray:
    ref = da.sel(time=slice(*base_period)).load()
    study = da.load()
    base_per = percentile_doy(
        ref,
        window=5,
        per=90,
        alpha=1.0 / 3.0,
        beta=1.0 / 3.0,
        copy=False,
    )
    ref_groups = _year_groups(ref)
    study_groups = _year_groups(study)
    pieces = []
    for year_label, year_slice in study_groups.items():
        year_da = study.isel(time=year_slice)
        if _contains_year(ref, year_label):
            donor_counts = []
            for donor_label, donor_slice in ref_groups.items():
                if donor_label == year_label:
                    continue
                boot_ref = _replace_year(ref, ref_groups, year_label, donor_slice)
                boot_per = percentile_doy(
                    boot_ref,
                    window=5,
                    per=90,
                    alpha=1.0 / 3.0,
                    beta=1.0 / 3.0,
                    copy=False,
                )
                threshold = resample_doy(boot_per.squeeze("percentiles"), year_da)
                donor_counts.append((year_da > threshold).sum(dim="time"))
            value = xr.concat(donor_counts, dim="_bootstrap").mean(
                dim="_bootstrap",
                keep_attrs=True,
            )
        else:
            threshold = resample_doy(base_per.squeeze("percentiles"), year_da)
            value = (year_da > threshold).sum(dim="time")
        value = value.expand_dims(time=[year_label])
        pieces.append(value)
    result = xr.concat(pieces, dim="time")
    result.name = "TG90p"
    result.attrs["units"] = "d"
    return result


def _tg90p_bootstrap_count_numpy_index(
    da: xr.DataArray,
    *,
    base_period: tuple[str, str],
) -> xr.DataArray:
    study = da.load()
    ref = study.sel(time=slice(*base_period))
    values = np.asarray(study.transpose("time", ...).data)
    ref_values = np.asarray(ref.transpose("time", ...).data)
    flat_ref = ref_values.reshape(ref.sizes["time"], -1)
    flat_study = values.reshape(study.sizes["time"], -1)

    ref_time = pd.DatetimeIndex(ref.time.values)
    study_time = pd.DatetimeIndex(study.time.values)
    ref_year_indices = _indices_by_year(ref_time)
    study_year_indices = _indices_by_year(study_time)
    sample_indices = _rolling_sample_index_matrix(ref_time, window=5)
    source_max_doy = int(ref_time.dayofyear.max())
    base_per = _percentiles_from_sample_indices(flat_ref, sample_indices)

    pieces = []
    for year, study_indices in study_year_indices.items():
        year_da = study.isel(time=study_indices)
        year_values = flat_study[study_indices]
        target_max_doy = (
            source_max_doy
            if source_max_doy == 366
            else year_da.time.dt.dayofyear.max().item()
        )
        if year in ref_year_indices:
            donor_counts = []
            for donor_year, donor_indices in ref_year_indices.items():
                if donor_year == year:
                    continue
                remapped = _remap_target_year_indices(
                    sample_indices,
                    ref_time,
                    ref_year_indices[year],
                    donor_indices,
                )
                per = _percentiles_from_sample_indices(flat_ref, remapped)
                donor_counts.append(
                    _count_year_exceedances(
                        year_values,
                        per,
                        year_da.time.dt.dayofyear.to_numpy(),
                        target_max_doy,
                    ),
                )
            flat_count = np.mean(donor_counts, axis=0)
        else:
            flat_count = _count_year_exceedances(
                year_values,
                base_per,
                year_da.time.dt.dayofyear.to_numpy(),
                target_max_doy,
            )
        pieces.append(flat_count.reshape(study.shape[1:]))

    result = xr.DataArray(
        np.stack(pieces, axis=0),
        dims=study.dims,
        coords={
            "time": [np.datetime64(f"{year}-01-01") for year in study_year_indices],
            **{coord: study.coords[coord] for coord in study.dims if coord != "time"},
        },
        name="TG90p",
        attrs={"units": "d"},
    )
    for coord in study.coords:
        if coord not in result.coords and "time" not in study[coord].dims:
            result = result.assign_coords({coord: study[coord]})
    return result.assign_coords(percentiles=90)


def _tg90p_bootstrap_count_numpy_numba(
    da: xr.DataArray,
    *,
    base_period: tuple[str, str],
) -> xr.DataArray:
    study = da.load()
    ref = study.sel(time=slice(*base_period))
    flat_ref = np.asarray(ref.transpose("time", ...).data).reshape(
        ref.sizes["time"],
        -1,
    )
    flat_study = np.asarray(study.transpose("time", ...).data).reshape(
        study.sizes["time"],
        -1,
    )
    ref_time = pd.DatetimeIndex(ref.time.values)
    study_time = pd.DatetimeIndex(study.time.values)
    ref_year_indices = _indices_by_year(ref_time)
    study_year_indices = _indices_by_year(study_time)
    ref_years = np.asarray(list(ref_year_indices), dtype=np.int64)
    study_years = np.asarray(list(study_year_indices), dtype=np.int64)
    study_starts = np.asarray(
        [indices[0] for indices in study_year_indices.values()],
        dtype=np.int64,
    )
    study_lengths = np.asarray(
        [len(indices) for indices in study_year_indices.values()],
        dtype=np.int64,
    )
    source_max_doy = int(ref_time.dayofyear.max())
    study_threshold_max_doys = np.asarray(
        [
            source_max_doy
            if source_max_doy == 366
            else int(study_time[indices].dayofyear.max())
            for indices in study_year_indices.values()
        ],
        dtype=np.int64,
    )
    study_to_ref = np.asarray(
        [
            int(np.where(ref_years == year)[0][0]) if year in ref_year_indices else -1
            for year in study_years
        ],
        dtype=np.int64,
    )
    sample_indices = _rolling_sample_index_matrix(ref_time, window=5)
    index_year, index_pos = _ref_index_year_and_position(
        ref_year_indices, len(ref_time)
    )
    donor_aligned = _donor_alignment_matrix(ref_time, ref_year_indices)
    study_doys = study_time.dayofyear.to_numpy(dtype=np.int64)

    result = _bootstrap_counts_numba_kernel(
        flat_ref.astype(np.float64),
        flat_study.astype(np.float64),
        sample_indices,
        index_year,
        index_pos,
        donor_aligned,
        study_starts,
        study_lengths,
        study_threshold_max_doys,
        study_to_ref,
        study_doys,
    )

    data = result.reshape((len(study_years), *study.shape[1:]))
    out = xr.DataArray(
        data,
        dims=study.dims,
        coords={
            "time": [np.datetime64(f"{year}-01-01") for year in study_years],
            **{coord: study.coords[coord] for coord in study.dims if coord != "time"},
        },
        name="TG90p",
        attrs={"units": "d"},
    )
    for coord in study.coords:
        if coord not in out.coords and "time" not in study[coord].dims:
            out = out.assign_coords({coord: study[coord]})
    return out.assign_coords(percentiles=90)


def _tg90p_bootstrap_count_numpy_numba_presort(
    da: xr.DataArray,
    *,
    base_period: tuple[str, str],
) -> xr.DataArray:
    study = da.load()
    ref = study.sel(time=slice(*base_period))
    flat_ref = np.asarray(ref.transpose("time", ...).data).reshape(
        ref.sizes["time"],
        -1,
    )
    flat_study = np.asarray(study.transpose("time", ...).data).reshape(
        study.sizes["time"],
        -1,
    )
    ref_time = pd.DatetimeIndex(ref.time.values)
    study_time = pd.DatetimeIndex(study.time.values)
    ref_year_indices = _indices_by_year(ref_time)
    study_year_indices = _indices_by_year(study_time)
    ref_years = np.asarray(list(ref_year_indices), dtype=np.int64)
    study_years = np.asarray(list(study_year_indices), dtype=np.int64)
    study_starts = np.asarray(
        [indices[0] for indices in study_year_indices.values()],
        dtype=np.int64,
    )
    study_lengths = np.asarray(
        [len(indices) for indices in study_year_indices.values()],
        dtype=np.int64,
    )
    source_max_doy = int(ref_time.dayofyear.max())
    study_threshold_max_doys = np.asarray(
        [
            source_max_doy
            if source_max_doy == 366
            else int(study_time[indices].dayofyear.max())
            for indices in study_year_indices.values()
        ],
        dtype=np.int64,
    )
    study_to_ref = np.asarray(
        [
            int(np.where(ref_years == year)[0][0]) if year in ref_year_indices else -1
            for year in study_years
        ],
        dtype=np.int64,
    )
    sample_indices = _rolling_sample_index_matrix(ref_time, window=5)
    index_year, index_pos = _ref_index_year_and_position(
        ref_year_indices, len(ref_time)
    )
    donor_aligned = _donor_alignment_matrix(ref_time, ref_year_indices)
    study_doys = study_time.dayofyear.to_numpy(dtype=np.int64)
    sorted_samples, sample_counts = _sorted_samples_from_sample_indices(
        flat_ref.astype(np.float64),
        sample_indices,
    )

    result = _bootstrap_counts_numba_presort_kernel(
        flat_ref.astype(np.float64),
        flat_study.astype(np.float64),
        sample_indices,
        sorted_samples,
        sample_counts,
        index_year,
        index_pos,
        donor_aligned,
        study_starts,
        study_lengths,
        study_threshold_max_doys,
        study_to_ref,
        study_doys,
    )

    data = result.reshape((len(study_years), *study.shape[1:]))
    out = xr.DataArray(
        data,
        dims=study.dims,
        coords={
            "time": [np.datetime64(f"{year}-01-01") for year in study_years],
            **{coord: study.coords[coord] for coord in study.dims if coord != "time"},
        },
        name="TG90p",
        attrs={"units": "d"},
    )
    for coord in study.coords:
        if coord not in out.coords and "time" not in study[coord].dims:
            out = out.assign_coords({coord: study[coord]})
    return out.assign_coords(percentiles=90)


def _ref_index_year_and_position(
    ref_year_indices: dict[int, np.ndarray],
    n_ref_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    index_year = np.full(n_ref_time, -1, dtype=np.int64)
    index_pos = np.full(n_ref_time, -1, dtype=np.int64)
    for year_index, indices in enumerate(ref_year_indices.values()):
        index_year[indices] = year_index
        index_pos[indices] = np.arange(len(indices), dtype=np.int64)
    return index_year, index_pos


def _donor_alignment_matrix(
    ref_time: pd.DatetimeIndex,
    ref_year_indices: dict[int, np.ndarray],
) -> np.ndarray:
    max_year_len = max(len(indices) for indices in ref_year_indices.values())
    n_years = len(ref_year_indices)
    aligned = np.full((n_years, n_years, max_year_len), -1, dtype=np.int64)
    years = list(ref_year_indices)
    for target_i, target_year in enumerate(years):
        target_indices = ref_year_indices[target_year]
        target_time = ref_time[target_indices]
        for donor_i, donor_year in enumerate(years):
            donor_indices = ref_year_indices[donor_year]
            aligned[target_i, donor_i, : len(target_indices)] = (
                _donor_indices_aligned_to_target(
                    target_time,
                    ref_time[donor_indices],
                    donor_indices,
                )
            )
    return aligned


try:
    from numba import njit, prange
except Exception:  # noqa: BLE001
    njit = None
    prange = range


if njit is not None:

    @njit(parallel=True, cache=True)
    def _bootstrap_counts_numba_kernel(  # noqa: C901
        flat_ref,
        flat_study,
        sample_indices,
        index_year,
        index_pos,
        donor_aligned,
        study_starts,
        study_lengths,
        study_max_doys,
        study_to_ref,
        study_doys,
    ):
        n_years = len(study_starts)
        n_cells = flat_study.shape[1]
        out = np.empty((n_years, n_cells), dtype=np.float64)
        n_ref_years = donor_aligned.shape[1]
        max_samples = sample_indices.shape[1]
        for flat_i in prange(n_years * n_cells):
            year_i = flat_i // n_cells
            cell = flat_i % n_cells
            target_ref_i = study_to_ref[year_i]
            max_target_doy = study_max_doys[year_i]
            start = study_starts[year_i]
            length = study_lengths[year_i]
            if target_ref_i < 0:
                q = np.empty(365, dtype=np.float64)
                buf = np.empty(max_samples, dtype=np.float64)
                for doy_i in range(365):
                    q[doy_i] = _quantile_for_doy_cell(
                        flat_ref,
                        sample_indices,
                        index_year,
                        index_pos,
                        donor_aligned,
                        -1,
                        -1,
                        doy_i,
                        cell,
                        buf,
                    )
                count = 0.0
                for offset in range(length):
                    doy = study_doys[start + offset]
                    threshold = _adjusted_threshold(q, doy, max_target_doy)
                    if flat_study[start + offset, cell] > threshold:
                        count += 1.0
                out[year_i, cell] = count
            else:
                donor_total = 0.0
                donor_count = 0
                for donor_i in range(n_ref_years):
                    if donor_i == target_ref_i:
                        continue
                    q = np.empty(365, dtype=np.float64)
                    buf = np.empty(max_samples, dtype=np.float64)
                    for doy_i in range(365):
                        q[doy_i] = _quantile_for_doy_cell(
                            flat_ref,
                            sample_indices,
                            index_year,
                            index_pos,
                            donor_aligned,
                            target_ref_i,
                            donor_i,
                            doy_i,
                            cell,
                            buf,
                        )
                    count = 0.0
                    for offset in range(length):
                        doy = study_doys[start + offset]
                        threshold = _adjusted_threshold(q, doy, max_target_doy)
                        if flat_study[start + offset, cell] > threshold:
                            count += 1.0
                    donor_total += count
                    donor_count += 1
                out[year_i, cell] = donor_total / donor_count
        return out

    @njit(cache=True)
    def _quantile_for_doy_cell(
        flat_ref,
        sample_indices,
        index_year,
        index_pos,
        donor_aligned,
        target_ref_i,
        donor_i,
        doy_i,
        cell,
        buf,
    ):
        n = 0
        for sample_i in range(sample_indices.shape[1]):
            ref_i = sample_indices[doy_i, sample_i]
            if ref_i < 0:
                continue
            mapped_i = ref_i
            if target_ref_i >= 0 and index_year[ref_i] == target_ref_i:
                mapped_i = donor_aligned[target_ref_i, donor_i, index_pos[ref_i]]
            if mapped_i < 0:
                continue
            value = flat_ref[mapped_i, cell]
            if not np.isnan(value):
                buf[n] = value
                n += 1
        return np.float32(_method8_quantile_90(buf, n))

    @njit(cache=True)
    def _method8_quantile_90(buf, n):
        if n == 0:
            return np.nan
        if n == 1:
            return buf[0]
        for i in range(1, n):
            value = buf[i]
            j = i - 1
            while j >= 0 and buf[j] > value:
                buf[j + 1] = buf[j]
                j -= 1
            buf[j + 1] = value
        q = 0.9
        alpha = 1.0 / 3.0
        beta = 1.0 / 3.0
        virtual = n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0
        if virtual >= n - 1:
            return buf[n - 1]
        if virtual < 0:
            return buf[0]
        previous = int(np.floor(virtual))
        gamma = virtual - previous
        left = buf[previous]
        right = buf[previous + 1]
        diff = right - left
        if gamma >= 0.5:
            return right - diff * (1.0 - gamma)
        return left + diff * gamma

    @njit(cache=True)
    def _adjusted_threshold(q, doy, max_target_doy):
        if max_target_doy == 365:
            return np.float32(q[doy - 1])
        position = (doy - 1.0) * 364.0 / 365.0
        lower = int(np.floor(position))
        if lower >= 364:
            return np.float32(q[364])
        gamma = position - lower
        diff = q[lower + 1] - q[lower]
        if gamma >= 0.5:
            return np.float32(q[lower + 1] - diff * (1.0 - gamma))
        return np.float32(q[lower] + diff * gamma)

    @njit(parallel=True, cache=True)
    def _bootstrap_counts_numba_presort_kernel(  # noqa: C901
        flat_ref,
        flat_study,
        sample_indices,
        sorted_samples,
        sample_counts,
        index_year,
        index_pos,
        donor_aligned,
        study_starts,
        study_lengths,
        study_max_doys,
        study_to_ref,
        study_doys,
    ):
        n_years = len(study_starts)
        n_cells = flat_study.shape[1]
        out = np.empty((n_years, n_cells), dtype=np.float64)
        n_ref_years = donor_aligned.shape[1]
        max_samples = sample_indices.shape[1]
        for flat_i in prange(n_years * n_cells):
            year_i = flat_i // n_cells
            cell = flat_i % n_cells
            target_ref_i = study_to_ref[year_i]
            max_target_doy = study_max_doys[year_i]
            start = study_starts[year_i]
            length = study_lengths[year_i]
            if target_ref_i < 0:
                q = np.empty(365, dtype=np.float64)
                for doy_i in range(365):
                    q[doy_i] = _method8_quantile_90_from_sorted(
                        sorted_samples,
                        sample_counts,
                        doy_i,
                        cell,
                    )
                count = 0.0
                for offset in range(length):
                    doy = study_doys[start + offset]
                    threshold = _adjusted_threshold(q, doy, max_target_doy)
                    if flat_study[start + offset, cell] > threshold:
                        count += 1.0
                out[year_i, cell] = count
            else:
                donor_total = 0.0
                donor_count = 0
                remove_buf = np.empty(max_samples, dtype=np.float64)
                add_buf = np.empty(max_samples, dtype=np.float64)
                used_remove = np.empty(max_samples, dtype=np.uint8)
                q = np.empty(365, dtype=np.float64)
                for donor_i in range(n_ref_years):
                    if donor_i == target_ref_i:
                        continue
                    for doy_i in range(365):
                        q[doy_i] = _quantile_for_doy_cell_from_presorted(
                            flat_ref,
                            sample_indices,
                            sorted_samples,
                            sample_counts,
                            index_year,
                            index_pos,
                            donor_aligned,
                            target_ref_i,
                            donor_i,
                            doy_i,
                            cell,
                            remove_buf,
                            add_buf,
                            used_remove,
                        )
                    count = 0.0
                    for offset in range(length):
                        doy = study_doys[start + offset]
                        threshold = _adjusted_threshold(q, doy, max_target_doy)
                        if flat_study[start + offset, cell] > threshold:
                            count += 1.0
                    donor_total += count
                    donor_count += 1
                out[year_i, cell] = donor_total / donor_count
        return out

    @njit(cache=True)
    def _method8_quantile_90_from_sorted(
        sorted_samples,
        sample_counts,
        doy_i,
        cell,
    ):
        n = sample_counts[doy_i, cell]
        if n == 0:
            return np.nan
        if n == 1:
            return np.float32(sorted_samples[doy_i, 0, cell])
        q = 0.9
        alpha = 1.0 / 3.0
        beta = 1.0 / 3.0
        virtual = n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0
        if virtual >= n - 1:
            return np.float32(sorted_samples[doy_i, n - 1, cell])
        if virtual < 0:
            return np.float32(sorted_samples[doy_i, 0, cell])
        previous = int(np.floor(virtual))
        gamma = virtual - previous
        left = sorted_samples[doy_i, previous, cell]
        right = sorted_samples[doy_i, previous + 1, cell]
        diff = right - left
        if gamma >= 0.5:
            return np.float32(right - diff * (1.0 - gamma))
        return np.float32(left + diff * gamma)

    @njit(cache=True)
    def _quantile_for_doy_cell_from_presorted(
        flat_ref,
        sample_indices,
        sorted_samples,
        sample_counts,
        index_year,
        index_pos,
        donor_aligned,
        target_ref_i,
        donor_i,
        doy_i,
        cell,
        remove_buf,
        add_buf,
        used_remove,
    ):
        n_remove = 0
        n_add = 0
        for sample_i in range(sample_indices.shape[1]):
            ref_i = sample_indices[doy_i, sample_i]
            if ref_i < 0:
                continue
            if index_year[ref_i] != target_ref_i:
                continue
            old_value = flat_ref[ref_i, cell]
            if not np.isnan(old_value):
                remove_buf[n_remove] = old_value
                n_remove += 1
            mapped_i = donor_aligned[target_ref_i, donor_i, index_pos[ref_i]]
            if mapped_i >= 0:
                new_value = flat_ref[mapped_i, cell]
                if not np.isnan(new_value):
                    add_buf[n_add] = new_value
                    n_add += 1
        _sort_prefix(add_buf, n_add)
        n = sample_counts[doy_i, cell] - n_remove + n_add
        return np.float32(_method8_quantile_90_adjusted_sorted(
            sorted_samples,
            doy_i,
            cell,
            n,
            remove_buf,
            n_remove,
            add_buf,
            n_add,
            used_remove,
        ))

    @njit(cache=True)
    def _method8_quantile_90_adjusted_sorted(
        sorted_samples,
        doy_i,
        cell,
        n,
        remove_buf,
        n_remove,
        add_buf,
        n_add,
        used_remove,
    ):
        if n == 0:
            return np.nan
        if n == 1:
            return _adjusted_sorted_value_at(
                sorted_samples,
                doy_i,
                cell,
                0,
                remove_buf,
                n_remove,
                add_buf,
                n_add,
                used_remove,
            )
        q = 0.9
        alpha = 1.0 / 3.0
        beta = 1.0 / 3.0
        virtual = n * q + (alpha + q * (1.0 - alpha - beta)) - 1.0
        if virtual >= n - 1:
            return _adjusted_sorted_value_at(
                sorted_samples,
                doy_i,
                cell,
                n - 1,
                remove_buf,
                n_remove,
                add_buf,
                n_add,
                used_remove,
            )
        if virtual < 0:
            return _adjusted_sorted_value_at(
                sorted_samples,
                doy_i,
                cell,
                0,
                remove_buf,
                n_remove,
                add_buf,
                n_add,
                used_remove,
            )
        previous = int(np.floor(virtual))
        gamma = virtual - previous
        left = _adjusted_sorted_value_at(
            sorted_samples,
            doy_i,
            cell,
            previous,
            remove_buf,
            n_remove,
            add_buf,
            n_add,
            used_remove,
        )
        right = _adjusted_sorted_value_at(
            sorted_samples,
            doy_i,
            cell,
            previous + 1,
            remove_buf,
            n_remove,
            add_buf,
            n_add,
            used_remove,
        )
        diff = right - left
        if gamma >= 0.5:
            return right - diff * (1.0 - gamma)
        return left + diff * gamma

    @njit(cache=True)
    def _adjusted_sorted_value_at(
        sorted_samples,
        doy_i,
        cell,
        rank,
        remove_buf,
        n_remove,
        add_buf,
        n_add,
        used_remove,
    ):
        for i in range(n_remove):
            used_remove[i] = 0
        base_i = 0
        add_i = 0
        out_i = -1
        base_n = sorted_samples.shape[1]
        while base_i < base_n or add_i < n_add:
            have_base = base_i < base_n and not np.isnan(
                sorted_samples[doy_i, base_i, cell],
            )
            have_add = add_i < n_add
            if not have_base and not have_add:
                break
            if have_add and (
                not have_base or add_buf[add_i] <= sorted_samples[doy_i, base_i, cell]
            ):
                value = add_buf[add_i]
                add_i += 1
            else:
                value = sorted_samples[doy_i, base_i, cell]
                base_i += 1
                if _consume_removed_value(value, remove_buf, used_remove, n_remove):
                    continue
            out_i += 1
            if out_i == rank:
                return value
        return np.nan

    @njit(cache=True)
    def _consume_removed_value(value, remove_buf, used_remove, n_remove):
        for remove_i in range(n_remove):
            if used_remove[remove_i] == 0 and remove_buf[remove_i] == value:
                used_remove[remove_i] = 1
                return True
        return False

    @njit(cache=True)
    def _sort_prefix(buf, n):
        for i in range(1, n):
            value = buf[i]
            j = i - 1
            while j >= 0 and buf[j] > value:
                buf[j + 1] = buf[j]
                j -= 1
            buf[j + 1] = value

else:

    def _bootstrap_counts_numba_kernel(*args, **kwargs):  # noqa: ARG001
        msg = "numba is required for --engine numpy-numba"
        raise RuntimeError(msg)


def _indices_by_year(time: pd.DatetimeIndex) -> dict[int, np.ndarray]:
    return {int(year): np.where(time.year == year)[0] for year in np.unique(time.year)}


def _rolling_sample_index_matrix(
    time: pd.DatetimeIndex,
    *,
    window: int,
) -> np.ndarray:
    half_window = window // 2
    sample_indices: dict[int, list[int]] = {doy: [] for doy in range(1, 366)}
    doys = time.dayofyear.to_numpy()
    for center, doy in enumerate(doys):
        if doy == 366:
            continue
        start = max(0, center - half_window)
        stop = min(len(time), center + half_window + 1)
        sample_indices[int(doy)].extend(range(start, stop))
    max_samples = max(len(indices) for indices in sample_indices.values())
    matrix = np.full((365, max_samples), -1, dtype=np.int64)
    for doy, indices in sample_indices.items():
        matrix[doy - 1, : len(indices)] = indices
    return matrix


def _remap_target_year_indices(
    sample_indices: np.ndarray,
    ref_time: pd.DatetimeIndex,
    target_indices: np.ndarray,
    donor_indices: np.ndarray,
) -> np.ndarray:
    index_map = np.arange(len(ref_time), dtype=np.int64)
    donor_map = _donor_indices_aligned_to_target(
        ref_time[target_indices],
        ref_time[donor_indices],
        donor_indices,
    )
    index_map[target_indices] = donor_map
    valid = sample_indices >= 0
    remapped = np.full_like(sample_indices, -1)
    remapped[valid] = index_map[sample_indices[valid]]
    return remapped


def _donor_indices_aligned_to_target(
    target_time: pd.DatetimeIndex,
    donor_time: pd.DatetimeIndex,
    donor_indices: np.ndarray,
) -> np.ndarray:
    if len(target_time) == len(donor_time):
        return donor_indices
    donor_by_month_day = {
        (int(month), int(day)): int(index)
        for month, day, index in zip(
            donor_time.month,
            donor_time.day,
            donor_indices,
            strict=True,
        )
    }
    # Missing dates, normally Feb 29 when injecting no-leap into leap, map to -1.
    return np.asarray(
        [
            donor_by_month_day.get((int(month), int(day)), -1)
            for month, day in zip(target_time.month, target_time.day, strict=True)
        ],
        dtype=np.int64,
    )


def _percentiles_from_sample_indices(
    flat_ref: np.ndarray,
    sample_indices: np.ndarray,
) -> np.ndarray:
    valid = sample_indices >= 0
    safe_indices = np.where(valid, sample_indices, 0)
    samples = flat_ref[safe_indices].astype(float, copy=True)
    samples[~valid] = np.nan
    percentiles = nan_calc_percentiles(
        samples,
        percentiles=[90],
        axis=1,
        alpha=1.0 / 3.0,
        beta=1.0 / 3.0,
        copy=False,
    )
    return percentiles[..., 0].T.astype(flat_ref.dtype, copy=False)


def _sorted_samples_from_sample_indices(
    flat_ref: np.ndarray,
    sample_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = sample_indices >= 0
    safe_indices = np.where(valid, sample_indices, 0)
    samples = flat_ref[safe_indices].astype(float, copy=True)
    samples[~valid] = np.nan
    samples.sort(axis=1)
    counts = np.sum(~np.isnan(samples), axis=1, dtype=np.int64)
    return samples, counts


def _count_year_exceedances(
    year_values: np.ndarray,
    percentile_by_doy: np.ndarray,
    year_doys: np.ndarray,
    max_target_doy: int,
) -> np.ndarray:
    adjusted = _adjust_365_percentiles_to_target(percentile_by_doy, max_target_doy)
    threshold = adjusted[year_doys - 1]
    return np.sum(year_values > threshold, axis=0)


def _adjust_365_percentiles_to_target(
    percentile_by_doy: np.ndarray,
    max_target_doy: int,
) -> np.ndarray:
    if max_target_doy == 365:
        return percentile_by_doy
    source_x = np.linspace(1, max_target_doy, num=percentile_by_doy.shape[0])
    target_x = np.arange(1, max_target_doy + 1)
    adjusted = np.empty(
        (max_target_doy, percentile_by_doy.shape[1]),
        dtype=percentile_by_doy.dtype,
    )
    for cell in range(percentile_by_doy.shape[1]):
        adjusted[:, cell] = np.interp(target_x, source_x, percentile_by_doy[:, cell])
    return adjusted


def main() -> None:
    """Run the prototype and print a JSON summary."""
    args = _parse_args()
    files = sorted(glob.glob(args.file_glob))  # noqa: PTH207
    if not files:
        msg = f"No files matched {args.file_glob!r}."
        raise FileNotFoundError(msg)

    open_start = time.perf_counter()
    ds = xr.open_mfdataset(files, combine="by_coords")
    da = ds["tas"].sel(
        lat=slice(args.lat_min, args.lat_max),
        lon=slice(args.lon_min, args.lon_max),
        time=slice(args.time_range_start, args.time_range_end),
    )
    if args.target_unit:
        da = convert_units_to(da, args.target_unit)
    open_end = time.perf_counter()
    compute_start = time.perf_counter()
    if args.engine == "xarray-loop":
        result = _tg90p_bootstrap_count(
            da,
            base_period=(args.base_period_start, args.base_period_end),
        )
    elif args.engine == "numpy-index":
        result = _tg90p_bootstrap_count_numpy_index(
            da,
            base_period=(args.base_period_start, args.base_period_end),
        )
    elif args.engine == "numpy-numba":
        result = _tg90p_bootstrap_count_numpy_numba(
            da,
            base_period=(args.base_period_start, args.base_period_end),
        )
    else:
        result = _tg90p_bootstrap_count_numpy_numba_presort(
            da,
            base_period=(args.base_period_start, args.base_period_end),
        )
    result.load()
    compute_end = time.perf_counter()

    summary: dict[str, object] = {
        "open_seconds": open_end - open_start,
        "engine": args.engine,
        "compute_seconds": compute_end - compute_start,
        "total_seconds": compute_end - open_start,
        "result_shape": tuple(int(x) for x in result.shape),
        "result_mean": float(result.mean().item()),
    }
    if args.reference_result is not None:
        reference = xr.open_dataarray(args.reference_result).load()
        comparable = result
        if result.shape == reference.shape:
            comparable = result.copy()
            comparable.coords.update(reference.coords)
        diff = comparable - reference
        summary.update(
            {
                "reference_mean": float(reference.mean().item()),
                "max_abs_diff": float(abs(diff).max().item()),
                "changed_cells": int((abs(diff) > 0).sum().item()),
                "changed_cells_gt_1e-9": int((abs(diff) > 1e-9).sum().item()),
            },
        )
    if args.output is not None:
        result.to_netcdf(args.output)
        summary["result_path"] = str(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
