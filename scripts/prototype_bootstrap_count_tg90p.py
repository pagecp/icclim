"""Prototype an exact TG90p bootstrap count engine outside icclim runtime."""

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
    parser.add_argument("--reference-result", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--engine",
        default="xarray-loop",
        choices=["xarray-loop", "numpy-index"],
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
    base_per = _percentiles_from_sample_indices(flat_ref, sample_indices)

    pieces = []
    for year, study_indices in study_year_indices.items():
        year_da = study.isel(time=study_indices)
        year_values = flat_study[study_indices]
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
                        study.time.dt.dayofyear.max().item(),
                    ),
                )
            flat_count = np.mean(donor_counts, axis=0)
        else:
            flat_count = _count_year_exceedances(
                year_values,
                base_per,
                year_da.time.dt.dayofyear.to_numpy(),
                study.time.dt.dayofyear.max().item(),
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
    return percentiles[..., 0].T


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
    adjusted = np.empty((max_target_doy, percentile_by_doy.shape[1]))
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
    open_end = time.perf_counter()
    compute_start = time.perf_counter()
    if args.engine == "xarray-loop":
        result = _tg90p_bootstrap_count(
            da,
            base_period=(args.base_period_start, args.base_period_end),
        )
    else:
        result = _tg90p_bootstrap_count_numpy_index(
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
            },
        )
    if args.output is not None:
        result.to_netcdf(args.output)
        summary["result_path"] = str(args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
