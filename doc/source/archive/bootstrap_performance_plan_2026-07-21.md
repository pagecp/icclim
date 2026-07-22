# Bootstrap Reliability And Performance Notes (2026-07-21)

This document records the bootstrap work done after PR #420 so future
developers can continue from evidence instead of repeating the same
experiments.

For the broader scientific and numerical review, read
`bootstrap_state_of_art_2026-07-21.md` before attempting new prototypes.

## Context

Percentile-based count indices such as `TG90P`, `TX90P`, and `TN90P` require
bootstrap when the reference period overlaps the calculation period. The goal is
not optional decoration: without bootstrap, the years inside the reference
period can have biased percentile exceedance counts.

The immediate user problem was reliability:

- dask-backed percentile bootstrap could build enormous graphs,
- users had to guess chunking manually,
- bad chunking could either exhaust memory or spend days building/executing a
  graph and return nothing,
- icclim v4's C backend was slow but usually finished, so icclim v7 needs the
  same "eventually returns a result" property.

PR #420 fixed reliability first. Performance work must not remove those
safeguards.

## Current Baseline

The merged reliability path now does the following for dask-backed percentile
count indices:

- uses automatic safe spatial tiling,
- derives tile size from `ICCLIM_BOOTSTRAP_SAFE_TILE_MEMORY`,
- allows an expert override with `ICCLIM_BOOTSTRAP_SAFE_TILE_CELLS`,
- retries with smaller tiles on memory-like failures,
- returns a computed xarray object instead of handing users a huge dask graph,
- keeps `ICCLIM_BOOTSTRAP_MODE=default` as a diagnostic way to force the old
  xclim graph path,
- keeps `bootstrap=False` as an explicit user shortcut only.

`bootstrap=False` is not scientifically equivalent. It is useful for fast
exploration and lower-bound timing only.

## Correctness Rules

Any optimized bootstrap path must satisfy these rules before production use:

1. `bootstrap=False` remains user-selected only.
2. One-year reference periods do not bootstrap.
3. No-overlap periods do not bootstrap.
4. All-overlap periods do not bootstrap because bootstrap is unnecessary.
5. Mean agreement is not enough.
6. Compare `max_abs_diff`, all changed cells, changed cells above `1e-9`, and
   year/cell locations for meaningful changes.
7. One-day flips must be investigated, not waved away.
8. First optimize percentile count indices only; do not change `WSDI`, `CSDI`,
   or spell-index behavior in the first performance phase.

## Benchmark Data

Local small TG90P benchmark:

- file: `tas_day_MPI-ESM1-2-HR_historical_r1i1p1f1_gn_19700101-19741231.nc`
- subset: `lat 35:45`, `lon 0:10`
- chunks: `time=365`, `lat=4`, `lon=4`
- period: `1970-01-01` to `1974-12-31`
- reference: `1970-01-01` to `1972-12-31`

Kraken ACCESS-CM2 benchmark:

- file glob: `/scratch/globc/page/models/tas_day_ACCESS-CM2_historical_*.nc`
- subset: `lat 35:70`, `lon 0:40`
- resulting subset: `time=23741`, `lat=28`, `lon=21`
- period: `1950-01-01` to `2014-12-31`
- reference: `1961-01-01` to `1990-12-31`
- chunks: `time=365`, `lat=24`, `lon=32`

Kraken environment:

- clone: `/scratch/globc/page/src/icclim`
- Python: `/scratch/globc/page/.conda/envs/icclimv7/bin/python`
- benchmark cache: `/scratch/globc/page/icclim-bench`

## Benchmark Results

Local small TG90P results:

| Candidate | Time | Exactness | Notes |
| --- | ---: | --- | --- |
| safe auto, `512MB` | `12.03s` | exact | Reliable, not faster. |
| legacy graph | `10.75s` | exact | `215188` graph tasks. |
| `bootstrap=False` | `10.33s` | not equivalent | Mean `43.70578512396694`. |
| xarray donor-year loop | `11.14s` | exact | Only small local win. |
| xarray donor-year loop, forced 3 tiles | `11.43s` | exact | Safe tiled path was `32.81s`. |
| Numba full-sort prototype | `2.26s` warm compute | exact locally | First promising low-level path. |
| Numba presorted replacement | `1.40s` warm compute | exact locally | Local-only win; rejected at scale. |

Kraken ACCESS-CM2 full-subset results:

| Candidate | Time | Memory/Graph | Result | Status |
| --- | ---: | --- | --- | --- |
| `bootstrap=False` | `61.63s` | `23089` graph tasks | mean `43.74709576138147` | Lower-bound only. |
| legacy xclim graph | `122.64s` total, `78.26s` compute | `4707056` graph tasks | mean `45.13441238564391` | Exact reference, risky graph. |
| safe tiled path | cancelled after `~5m49s` | no huge returned graph | no result | Reliable but slow on this subset. |
| xarray donor-year loop | cancelled after `~5m50s` | no huge returned graph | no result | Rejected. |
| Numba full-sort prototype | `101.91s` total | about `1.1GB` observed earlier | mean `45.13414893808983` | Promising but not exact yet. |
| Numba presorted replacement | `201.10s` total, `198.70s` compute | low-memory compiled path | mean `45.13414893808983` | Rejected; slower than full-sort. |
| Numba rank-select prototype, Celsius-first medium subset | `39.59s` total | low-memory compiled path | exact vs safe within `1e-9` | Promising; `~7.4x` faster than safe on `8x5` ACCESS-CM2 subset. |

## Rejected Candidate: Xarray Donor-Year Loop

This candidate computed each bounded spatial tile in memory, reused xclim's
`percentile_doy` and `resample_doy`, and avoided returning a huge dask graph.

Why it looked attractive:

- exact on local small tests,
- conceptually simple,
- preserved fallback to safe tiling,
- worked well when forced to avoid repeated tile reads locally.

Why it was rejected:

- it still calls `percentile_doy` for every target-year/donor-year pair,
- realistic 30-year reference periods reintroduce the repeated percentile work,
- Kraken did not finish the full benchmark within the useful window,
- it is reliability-compatible but not a real performance path.

## Rejected Candidate: Numba Presorted Replacement

This candidate followed the original presort idea:

- pre-sort base samples for each day-of-year and cell,
- for each bootstrap replacement, remove target-year window values,
- merge donor-year values,
- compute the method-8 percentile from the adjusted sorted stream.

Why it looked attractive:

- it directly attacks repeated sorting,
- it was exact and faster on the small local case,
- it kept memory low and avoided xarray object rebuilds.

Why it was rejected:

- Kraken full benchmark was slower than both the Numba full-sort prototype and
  the legacy graph path,
- scalar remove/merge scans dominated more than expected,
- sorting about 150 samples with simple insertion sort was cheaper than the
  presorted adjustment machinery at realistic geometry.

Keep the implementation in `scripts/prototype_bootstrap_count_tg90p.py` as a
negative benchmark. Do not promote it to production as written.

## Correctness Trap: One-Day Flips

The Numba prototypes match local small safe results, but Kraken comparisons show
rare meaningful one-day flips against cached safe/legacy outputs.

Observed fresh small-control differences before deeper investigation:

- two meaningful cells above `1e-9`,
- both outside the reference period,
- safe and `bootstrap=False` agreed with each other,
- the Numba prototype gave one fewer day.

This means the issue is likely in reproducing icclim/xclim's prepared percentile
threshold path, not in donor-year replacement itself.

Important details:

- icclim prepares percentile thresholds through `PercentileThreshold.prepare`.
- xclim's `percentile_doy` may produce a 366-day day-of-year climatology when the
  reference contains leap years.
- `resample_doy` adjusts/reindexes based on the source day-of-year coordinate and
  target time axis.
- dask/output dtype behavior can matter: tiny threshold differences around
  `1e-5` can flip a strict `>` comparison by one day.

Future optimized production code should first reproduce the prepared threshold
exactly for non-overlap years, because this isolates percentile construction
from bootstrap replacement.

Follow-up result:

- icclim's current threshold path is hybrid. Non-overlap years use thresholds
  prepared from the original input units and then converted to the studied-data
  unit. Bootstrap replacement years are recomputed by xclim's decorator from the
  already-normalized comparison data.
- The full-sort Numba prototype now mirrors this ordering. On the two Kraken
  one-cell controls that previously failed, the hybrid prototype matches safe
  output within floating tolerance (`changed_cells_gt_1e-9=0`).
- A medium Kraken validation on an ACCESS-CM2 `8x8` grid also matched the safe
  tiled output within floating tolerance: `changed_cells_gt_1e-9=0`,
  `max_abs_diff=4.263256414560601e-14`, equal means
  `43.815865384615385`.
- On that same `8x8` subset, safe tiled runtime was about `199.91s` with two
  tiles under `ICCLIM_BOOTSTRAP_SAFE_TILE_MEMORY=512MB`; the hybrid Numba
  prototype runtime was about `60.73s`.
- A full `28x21` hybrid prototype comparison against cached legacy output ran in
  about `124.60s`, compared with the cached legacy runtime of about `122.64s`.
  It reduced meaningful differences from the previous prototype but still had
  three values above `1e-9` (`max_abs_diff=1.0`). Fresh one-cell safe controls
  were submitted for those coordinates before drawing a final conclusion.
- A fresh one-cell safe control for the first full-subset mismatch was
  OOM-killed at `32GB` before producing output. The `64GB` retry also ran for
  more than half an hour and was OOM-killed before producing output. This
  reinforces the original reliability problem: the safe path avoids huge dask
  graphs by tiling, but xclim bootstrap work can still be memory-expensive and
  slow even at very small spatial sizes.
- Moving unit conversion after leap-year day-of-year interpolation and keeping
  adjusted thresholds in float64 fixed the tested non-overlap boundary flips.
  The latest full `28x21` hybrid prototype run took about `114.60s` and had only
  one meaningful difference against cached legacy output, on a reference-period
  bootstrap year. We decided not to chase exact legacy reproduction further
  because the hybrid current behavior is scientifically inconsistent; future
  work should move temperature percentile indices to a coherent Celsius-first
  path.
- Production direction changed accordingly: temperature standard-index data are
  now normalized using the known standard variable, not only metadata guessing,
  and temperature percentile thresholds are prepared from the full normalized
  series so reference periods outside `time_range` remain valid. This makes base
  and bootstrap percentile construction use the same Celsius-first scientific
  convention.

## Follow-up After Celsius-first Merge

After PR #422, old hybrid cached comparisons are no longer the right production
reference. Fresh ACCESS-CM2 medium validation was run with Celsius-first
threshold semantics:

- subset: `lat 35:45`, `lon 0:10`, resulting shape `65x8x5`,
- safe tiled reference: `293.80s`, mean `48.84263925729442`,
- `bootstrap=False` lower bound: `48.81s`, mean `47.39730769230769`,
- Numba full-sort: `45.34s`, `max_abs_diff=4.26e-14`,
  `changed_cells_gt_1e-9=0`,
- Numba rank-select: `39.59s`, `max_abs_diff=4.26e-14`,
  `changed_cells_gt_1e-9=0`.

The rank-select prototype replaces full insertion-sort percentile computation
with selection of the two ranks needed by method-8 p90. It was exact on local
smoke tests and the medium Kraken validation, and is the best current
performance candidate. A larger safe-vs-rank-select validation was submitted
next before considering production integration.

## Useful Scripts

Use these scripts rather than ad-hoc notebooks:

- `scripts/benchmark_bootstrap_tg90p.py`
- `scripts/prototype_bootstrap_count_tg90p.py`
- `scripts/compare_bootstrap_cached_outputs.py`
- `scripts/slurm_benchmark_bootstrap_tg90p.sh`

Recommended comparison fields:

- wall time,
- build time,
- compute time,
- graph task count,
- safe tile count,
- peak memory,
- result mean,
- `max_abs_diff`,
- changed cells,
- changed cells above `1e-9`,
- coordinates of meaningful changed cells.

## Next Ideas

Priority 1: larger-scale exactness validation.

- Compare the hybrid full-sort Numba prototype against cached safe outputs on
  the ACCESS-CM2 subset.
- Report `max_abs_diff`, `changed_cells_gt_1e-9`, wall time, and peak memory.
- Inspect any remaining mismatches as one-cell controls before optimizing.

Priority 2: improve the Numba full-sort prototype, not the presort prototype.

- The full-sort prototype is the best speed signal so far on Kraken.
- It avoids huge dask graphs and stayed low-memory.
- Its simple insertion sort over small samples appears cache-friendly.
- The next optimization should reduce duplicated threshold computation across
  cells/years without adding scalar merge overhead.

Priority 3: tile production integration only after exactness.

- Production code must keep the safe tiled fallback.
- The optimized path should be gated narrowly: daily day-of-year percentile,
  annual count output, simple strict comparison, supported calendars only.
- Unsupported cases should silently fall back to the safe xclim path.

Priority 4: only then consider an xclim PR.

- icclim should prove the implementation first.
- Once exactness and reliability are demonstrated, the lower-level bootstrap
  helper could be proposed back to xclim.

## Do Not Repeat

- Do not optimize only graph construction; it does not solve the user failure
  mode by itself.
- Do not accept mean-only agreement.
- Do not use `bootstrap=False` as a correctness reference.
- Do not merge a faster path that can silently change rare cells.
- Do not assume the presorted idea is automatically faster; the Kraken result
  showed the opposite for the tested implementation.
