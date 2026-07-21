# Bootstrap Performance Plan (2026-07-21)

## Goal

Improve runtime for percentile count-index bootstrap without weakening the
reliability safeguards merged in PR #420.

The baseline is now:

- automatic safe tiling for dask-backed percentile count indices,
- memory-budgeted tile sizing,
- retry with smaller tiles on memory-like failures,
- no huge dask graph returned to users.

Any optimized path must keep those safeguards or fall back to them.

## Non-Negotiable Correctness Rules

1. `bootstrap=False` remains an explicit user shortcut for exploratory runs only.
2. One-year, no-overlap, and all-overlap cases must not run bootstrap.
3. Results must match the current exact bootstrap path within strict tolerance.
4. Mean-only agreement is not enough: compare max absolute difference, changed
   cells, and year-by-year differences.
5. Do not change WSDI/CSDI or other spell-index behavior in the first
   performance phase.

## Benchmark Matrix

Always compare the same dataset and chunking across:

- `auto`: merged safe tiled behavior.
- `legacy`: `ICCLIM_BOOTSTRAP_MODE=default`.
- `false`: `bootstrap=False`, only as a speed/reference lower bound.
- `candidate`: any optimized engine.

Track:

- wall time,
- build time,
- compute time,
- graph task count,
- safe tile count,
- peak memory on Kraken,
- result mean,
- max absolute difference versus exact bootstrap.

## Current Local Baseline

Local real-NetCDF TG90P subset:

- file: `tas_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_19700101-19741231.nc`
- subset: `lat 35:45`, `lon 0:10`
- chunks: `time=365`, `lat=4`, `lon=4`
- period: `1970-01-01` to `1974-12-31`
- reference: `1970-01-01` to `1972-12-31`

Measured on 2026-07-21:

- safe auto, `512MB`: `12.03s`, `graph_tasks=0`, mean `73.60082644628099`
- legacy graph: `10.75s`, `graph_tasks=215188`, mean `73.60082644628099`
- `bootstrap=False`: `10.33s`, `graph_tasks=47823`, mean `43.70578512396694`

This confirms PR #420 is a reliability tradeoff, not a speedup.

## Highest-Probability Optimization Path

Target only percentile count indices first (`TG90P`, `TX90P`, `TN90P`).

The likely expensive operation is repeated percentile construction for each
overlap year and donor-year replacement. A useful speedup probably requires an
exact specialized count-index engine that avoids recomputing the full percentile
sort from scratch for every donor/year.

Promising direction:

1. Work tile-by-tile exactly like safe mode.
2. For each day-of-year window and spatial cell, gather reference samples.
3. Pre-sort reference samples once when possible.
4. For each target year/donor replacement, compute the exact percentile threshold
   from the adjusted sample set.
5. Count exceedances and aggregate by output period.
6. If unsupported calendar/dimension/chunking is encountered, fall back to safe
   xclim bootstrap.

## Rejected Or Low-Value Paths

- Reintroducing the previous native-bootstrap branch as-is. It was useful for
  learning, but it did not show a clear speed win against the merged baseline.
- Optimizing graph construction alone. Earlier splice-builder work reduced task
  count but did not improve end-to-end runtime.
- Accepting numerical drift. Previous experiments showed that small-looking mean
  differences can hide real local differences.

## First Implementation Gate

Before touching production runtime code, an optimized candidate must pass:

- local TG90P benchmark faster than safe auto,
- exact or near-exact comparison against legacy bootstrap,
- leap-year and 2-year/3-year overlap tests,
- no returned huge dask graph,
- fallback path preserved.

## Rejected Candidate: Xarray Donor-Year Loop

Implemented and then removed a conservative exact fast path for annual
percentile count indices:

- only single day-of-year percentile thresholds,
- only annual output for now,
- only dask-backed count indices already routed through the safe bootstrap gate,
- keeps the safe xclim tiled path available with `ICCLIM_BOOTSTRAP_MODE=safe`,
- keeps the legacy xclim graph diagnostic path available with
  `ICCLIM_BOOTSTRAP_MODE=default`,
- falls back to safe xclim tiling for unsupported cases.

The candidate computes each bounded spatial tile in memory, reuses xclim's exact
`percentile_doy` and `resample_doy` kernels, and avoids building a large dask
graph. It also caches the raw studied array only when that raw array fits within
`ICCLIM_BOOTSTRAP_SAFE_TILE_MEMORY`; bootstrap temporaries remain tiled.

Local real-NetCDF TG90P subset, same data as above:

- hostile chunks (`time=365`, `lat=4`, `lon=4`), one tile, `512MB`:
  fast `11.14s`, safe `12.10s`, exact (`max_abs_diff=0`).
- good spatial chunks (`time=365`, `lat=99`, `lon=99`), one tile, `512MB`:
  fast `2.36s`, safe `2.47s`, exact.
- hostile chunks, forced 3 tiles, `10MB`:
  fast with raw cache `11.43s`, safe `32.81s`, exact.

Kraken 65-year TG90P benchmark:

- file glob: `/scratch/globc/page/models/tas_day_ACCESS-CM2_historical_*.nc`
- subset: `lat=28`, `lon=21`, `time=23741`
- chunks: `time=365`, `lat=24`, `lon=32`
- `bootstrap=False`: `61.63s`, `graph_tasks=23089`, mean
  `43.74709576138147`
- legacy graph: `122.64s`, `graph_tasks=4707056`, mean
  `45.13441238564391`
- xarray donor-year fast candidate: cancelled after ~5m50s without completing
  icclim execution
- forced safe tiled path with default `2GB`: cancelled after ~5m49s without
  completing icclim execution

Interpretation:

- The arithmetic part is now very cheap on this subset; load/chunk topology is
  the dominant cost.
- Repeated tile reads were the major local multi-tile bottleneck. The raw-cache
  guard removes that when the raw studied data fits the memory budget.
- The xarray donor-year loop does not scale to realistic 30-year reference
  periods because it calls `percentile_doy` for every target-year/donor-year
  pair. This effectively reintroduces the expensive repeated percentile work we
  are trying to remove.
- Do not merge this production approach.

## Next Candidate

The next candidate must operate below the xarray donor-year loop:

- load one bounded tile,
- build reference rolling-window samples as compact NumPy arrays,
- sort or partially sort samples once per day-of-year/cell,
- compute each donor replacement threshold by removing target-year window
  values and injecting donor-year window values without reconstructing a full
  xarray object,
- count exceedances directly on NumPy arrays,
- wrap the result back into xarray only after computation,
- compare exactly against legacy/safe outputs on local and Kraken data.

This is closer to the previous developer's original idea and has a better chance
of real speedup because it attacks the repeated sort/rebuild cost directly.
