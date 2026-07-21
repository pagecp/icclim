"""Prototype an exact TG90p bootstrap count engine outside icclim runtime."""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from xclim.core.calendar import percentile_doy, resample_doy

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
    result = _tg90p_bootstrap_count(
        da,
        base_period=(args.base_period_start, args.base_period_end),
    )
    result.load()
    compute_end = time.perf_counter()

    summary: dict[str, object] = {
        "open_seconds": open_end - open_start,
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
