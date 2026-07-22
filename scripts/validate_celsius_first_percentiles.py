"""Validate Celsius-first temperature percentile behavior on NetCDF inputs."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import xarray as xr
from xclim.core.units import convert_units_to

import icclim


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Kelvin and Celsius inputs for temperature percentiles.",
    )
    parser.add_argument("--file-glob", required=True)
    parser.add_argument("--lat-min", type=float, default=45.0)
    parser.add_argument("--lat-max", type=float, default=55.0)
    parser.add_argument("--lon-min", type=float, default=20.0)
    parser.add_argument("--lon-max", type=float, default=35.0)
    parser.add_argument("--time-range-start", default="1950-01-01")
    parser.add_argument("--time-range-end", default="2014-12-31")
    parser.add_argument("--base-period-start", default="1961-01-01")
    parser.add_argument("--base-period-end", default="1990-12-31")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the Kelvin-versus-Celsius validation and print JSON results."""
    args = _parse_args()
    files = sorted(glob.glob(args.file_glob))  # noqa: PTH207
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        chunks={"time": 365, "lat": 4, "lon": 4},
    )
    tas_k = ds["tas"].sel(
        lat=slice(args.lat_min, args.lat_max),
        lon=slice(args.lon_min, args.lon_max),
        time=slice(args.time_range_start, args.time_range_end),
    )
    tas_c = convert_units_to(tas_k, "degree_Celsius", context="hydro")

    results = []
    common_kwargs = {
        "index_name": "tg90p",
        "var_name": "tas",
        "slice_mode": "year",
        "time_range": (args.time_range_start, args.time_range_end),
        "base_period_time_range": (args.base_period_start, args.base_period_end),
        "bootstrap": False,
    }
    from_kelvin = icclim.index(in_files=tas_k, **common_kwargs).load()
    from_celsius = icclim.index(in_files=tas_c, **common_kwargs).load()
    var_name = next(v for v in from_kelvin.data_vars if not v.endswith("_thresholds"))
    diff = from_kelvin[var_name] - from_celsius[var_name]
    results.append(
        {
            "index": common_kwargs["index_name"],
            "shape": list(from_kelvin[var_name].shape),
            "max_abs_diff_days": float(abs(diff).max().item()),
            "changed_values_gt_1e-9": int((abs(diff) > 1e-9).sum().item()),
            "kelvin_mean": float(from_kelvin[var_name].mean().item()),
            "celsius_mean": float(from_celsius[var_name].mean().item()),
        }
    )
    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
