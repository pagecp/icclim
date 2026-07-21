"""Diagnose percentile-threshold exactness before bootstrap optimization."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
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
        description=(
            "Compare icclim-prepared day-of-year percentiles against candidate "
            "threshold construction paths before running bootstrap counts."
        ),
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the icclim repository under test.",
    )
    parser.add_argument("--file-glob", required=True)
    parser.add_argument("--var-name", default="tas")
    parser.add_argument("--lat-min", type=float, default=35.0)
    parser.add_argument("--lat-max", type=float, default=70.0)
    parser.add_argument("--lon-min", type=float, default=0.0)
    parser.add_argument("--lon-max", type=float, default=40.0)
    parser.add_argument("--time-range-start", default="1950-01-01")
    parser.add_argument("--time-range-end", default="2014-12-31")
    parser.add_argument("--base-period-start", default="1961-01-01")
    parser.add_argument("--base-period-end", default="1990-12-31")
    parser.add_argument("--time-chunk", type=int, default=365)
    parser.add_argument("--lat-chunk", type=int, default=24)
    parser.add_argument("--lon-chunk", type=int, default=32)
    parser.add_argument("--percentile", type=float, default=90.0)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument(
        "--operator",
        choices=[">", ">="],
        default=">",
        help="Comparison operator used when reporting count flips.",
    )
    parser.add_argument(
        "--count-year",
        action="append",
        type=int,
        default=[],
        help="Calendar year to compare exceedance counts for. Repeatable.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _git_rev_parse(repo: Path, ref: str) -> str | None:
    try:
        return subprocess.check_output(  # noqa: S603
            ["git", "-C", str(repo), "rev-parse", ref],  # noqa: S607
            text=True,
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def _load_input(args: argparse.Namespace) -> xr.DataArray:
    files = sorted(glob.glob(args.file_glob))  # noqa: PTH207
    if not files:
        msg = f"No files matched {args.file_glob!r}."
        raise FileNotFoundError(msg)
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={
            "time": args.time_chunk,
            "lat": args.lat_chunk,
            "lon": args.lon_chunk,
        },
    )
    return ds[args.var_name].sel(
        lat=slice(args.lat_min, args.lat_max),
        lon=slice(args.lon_min, args.lon_max),
        time=slice(args.time_range_start, args.time_range_end),
    )


def _icclim_prepared_threshold(
    da: xr.DataArray,
    args: argparse.Namespace,
) -> xr.DataArray:
    from icclim._core.generic.threshold.percentile import (  # noqa: PLC0415
        PercentileThreshold,
    )
    from icclim._core.model.operator import OperatorRegistry  # noqa: PLC0415
    from icclim._core.model.quantile_interpolation import (  # noqa: PLC0415
        QuantileInterpolationRegistry,
    )

    operator = (
        OperatorRegistry.GREATER
        if args.operator == ">"
        else OperatorRegistry.GREATER_OR_EQUAL
    )
    threshold = PercentileThreshold(
        operator=operator,
        value=args.percentile,
        unit="doy_per",
        initial_query=f"{args.operator} {args.percentile:g} doy_per",
        threshold_min_value=None,
        reference_period=(args.base_period_start, args.base_period_end),
        doy_window_width=args.window,
        only_leap_years=False,
        interpolation=QuantileInterpolationRegistry.MEDIAN_UNBIASED,
    )
    threshold.prepare(da)
    return threshold.value.sel(percentiles=args.percentile)


def _direct_xclim_threshold(
    da: xr.DataArray,
    args: argparse.Namespace,
    *,
    load_reference: bool,
) -> xr.DataArray:
    ref = da.sel(time=slice(args.base_period_start, args.base_period_end))
    if load_reference:
        ref = ref.load()
    return percentile_doy(
        arr=ref,
        window=args.window,
        per=[args.percentile],
        alpha=1.0 / 3.0,
        beta=1.0 / 3.0,
    ).sel(percentiles=args.percentile)


def _native_numpy_threshold(
    da: xr.DataArray,
    args: argparse.Namespace,
    reference: xr.DataArray,
) -> xr.DataArray:
    from prototype_bootstrap_count_tg90p import (  # noqa: PLC0415
        _adjust_365_percentiles_to_target,
        _percentiles_from_sample_indices,
        _rolling_sample_index_matrix,
    )

    ref = da.sel(time=slice(args.base_period_start, args.base_period_end)).load()
    ref_time = ref.get_index("time")
    sample_indices = _rolling_sample_index_matrix(ref_time, window=args.window)
    flat_ref = np.asarray(ref.transpose("time", ...).data).reshape(
        ref.sizes["time"],
        -1,
    )
    per = _percentiles_from_sample_indices(flat_ref, sample_indices)
    max_source_doy = int(reference.dayofyear.max().item())
    if max_source_doy != per.shape[0]:
        per = _adjust_365_percentiles_to_target(per, max_source_doy)
    data = per.reshape((max_source_doy, *ref.shape[1:]))
    out = xr.DataArray(
        data,
        dims=("dayofyear", *ref.dims[1:]),
        coords={
            "dayofyear": np.arange(1, max_source_doy + 1),
            **{dim: ref.coords[dim] for dim in ref.dims if dim != "time"},
        },
        name="native_numpy_threshold",
    )
    return out.transpose(*reference.dims)


def _summarize_threshold_diff(
    reference: xr.DataArray,
    candidate: xr.DataArray,
) -> dict[str, object]:
    candidate = _align_like_reference(candidate, reference)
    diff = candidate - reference
    abs_diff = abs(diff)
    return {
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_abs_diff": float(abs_diff.mean().item()),
        "changed_cells": int((abs_diff > 0).sum().item()),
        "changed_cells_gt_1e-9": int((abs_diff > 1e-9).sum().item()),
        "changed_cells_gt_1e-6": int((abs_diff > 1e-6).sum().item()),
    }


def _align_like_reference(
    candidate: xr.DataArray,
    reference: xr.DataArray,
) -> xr.DataArray:
    if candidate.shape == reference.shape:
        candidate = candidate.copy()
        candidate.coords.update(reference.coords)
        return candidate
    return candidate.interp(dayofyear=reference.dayofyear)


def _compare_counts_for_years(
    da: xr.DataArray,
    reference_threshold: xr.DataArray,
    candidates: Mapping[str, xr.DataArray],
    years: list[int],
    operator: str,
) -> dict[str, object]:
    out: dict[str, object] = {}
    for year in years:
        year_da = da.sel(time=slice(f"{year}-01-01", f"{year}-12-31")).load()
        if year_da.sizes.get("time", 0) == 0:
            out[str(year)] = {"error": "year not present"}
            continue
        reference_resampled = resample_doy(reference_threshold, year_da)
        reference_count = _count_exceedances(year_da, reference_resampled, operator)
        year_summary: dict[str, object] = {
            "reference_mean_count": float(reference_count.mean().item()),
        }
        for name, threshold in candidates.items():
            candidate_resampled = resample_doy(
                _align_like_reference(threshold, reference_threshold),
                year_da,
            )
            candidate_count = _count_exceedances(
                year_da,
                candidate_resampled,
                operator,
            )
            count_diff = candidate_count - reference_count
            year_summary[name] = {
                "max_abs_count_diff": float(abs(count_diff).max().item()),
                "changed_cells": int((abs(count_diff) > 0).sum().item()),
                "changed_cells_gt_1e-9": int((abs(count_diff) > 1e-9).sum().item()),
                "mean_count": float(candidate_count.mean().item()),
            }
        out[str(year)] = year_summary
    return out


def _count_exceedances(
    da: xr.DataArray,
    threshold: xr.DataArray,
    operator: str,
) -> xr.DataArray:
    exceedances = da > threshold if operator == ">" else da >= threshold
    return exceedances.sum(dim="time")


def main() -> None:
    """Run the threshold diagnostic and print a JSON summary."""
    args = _parse_args()
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "src"))

    import icclim  # noqa: PLC0415

    start = time.perf_counter()
    da = _load_input(args)
    open_end = time.perf_counter()
    icclim_threshold = _icclim_prepared_threshold(da, args).load()
    icclim_end = time.perf_counter()
    candidates = {
        "direct_xclim_dask": _direct_xclim_threshold(
            da,
            args,
            load_reference=False,
        ).load(),
        "direct_xclim_loaded": _direct_xclim_threshold(
            da,
            args,
            load_reference=True,
        ).load(),
    }
    candidates["native_numpy_loaded"] = _native_numpy_threshold(
        da,
        args,
        icclim_threshold,
    )
    candidates_end = time.perf_counter()

    summary: dict[str, object] = {
        "repo": str(repo),
        "head_commit": _git_rev_parse(repo, "HEAD"),
        "icclim_version": icclim.__version__,
        "subset_sizes": {k: int(v) for k, v in da.sizes.items()},
        "source_dtype": str(da.dtype),
        "threshold_shape": tuple(int(x) for x in icclim_threshold.shape),
        "threshold_dtype": str(icclim_threshold.dtype),
        "threshold_dayofyear_max": int(icclim_threshold.dayofyear.max().item()),
        "open_seconds": open_end - start,
        "icclim_threshold_seconds": icclim_end - open_end,
        "candidate_threshold_seconds": candidates_end - icclim_end,
        "threshold_diffs": {
            name: _summarize_threshold_diff(icclim_threshold, threshold)
            for name, threshold in candidates.items()
        },
        "count_diffs": _compare_counts_for_years(
            da,
            icclim_threshold,
            candidates,
            args.count_year,
            args.operator,
        ),
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
