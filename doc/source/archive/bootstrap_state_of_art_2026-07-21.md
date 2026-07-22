# Bootstrap State Of The Art For Percentile Climate Indices

Date: 2026-07-21

This note summarizes the scientific method, comparable implementations, and
numerical optimization options for icclim's percentile bootstrap work. It should
be read before attempting another performance prototype.

## Why Bootstrap Exists Here

Percentile-based temperature indices such as `TX90P`, `TN90P`, `TG90P`,
`TX10P`, and `TN10P` compare daily values against thresholds computed from a
reference/base period. Zhang et al. (2005) showed that this creates artificial
inhomogeneity at the start and end of the base period: the threshold is affected
by sampling error, and exceedance rates outside the base period are biased
relative to rates inside it.

The Zhang bootstrap correction is not the generic bootstrap-confidence-interval
problem. In this context, it is a bias correction for percentile exceedance
counts in the base/reference period.

The operational rule is:

- For years outside the reference period, compute the index with the original
  day-of-year percentile threshold.
- For each target year inside the reference period, replace that year's data in
  the reference sample with each other reference year in turn.
- Recompute the day-of-year percentile threshold for each replacement.
- Recompute the index for the target year against each replacement threshold.
- Average the resulting `N - 1` index values.

For a 30-year reference period, each overlapping year therefore requires 29
threshold/index recomputations.

Source: Zhang et al. 2005, Journal of Climate, DOI `10.1175/JCLI3366.1`.

## Existing Implementations

### xclim

xclim implements this through a `percentile_bootstrap` decorator. Its documented
algorithm is year-group based:

- detect whether a target group overlaps the reference period,
- build altered reference periods by replacing the target group with every other
  reference group,
- recompute `percentile_doy`,
- compute the index,
- average over the bootstrap dimension.

Important implementation traits:

- dask inputs chunked on time are rechunked to full-time chunks,
- `xarray.map_blocks` is used for percentile recomputation,
- the approach is exact and generic,
- it can create very large graphs or force expensive memory layouts,
- it is computationally expensive by design.

This is the implementation icclim currently relies on under the hood, with
icclim's safe tiling wrapped around it for reliability.

### climdex.pcic

`climdex.pcic` is a well-tested R/C implementation of CLIMDEX routines. Its
source exposes a `zhang.running.qtile` path. When bootstrap data are requested,
the C routine returns an array shaped conceptually like:

```text
day_of_year x reference_year x donor_year x quantile
```

The package documentation also distinguishes in-base and out-base quantiles:

- out-base quantiles are one value per day of year,
- in-base quantiles are day-of-year by target year by replacement year.

This is a very useful design clue for icclim: precomputing all in-base
replacement thresholds as a compact tile-local array may be more robust than
trying to drive xarray/dask through the same replacement loop.

### CDO ETCCDI Operators

CDO has dedicated `etccdi_*` operators for percentile-based indices. They differ
from older `eca_*` operators by applying bootstrapping over a reference period
and using R Type 8 percentile calculation.

Important CDO lessons:

- exact ETCCDI percentile calculation can require high working memory,
- when not enough bins/memory are configured, histogram approximations may be
  used and results can differ,
- CDO explicitly discusses circular day-of-year windows for correct boundary
  handling,
- exactness and memory are treated as a tradeoff, not a trivial implementation
  detail.

For icclim, this supports our current stance: it is acceptable to be slower if
the calculation finishes and is exact, but approximate histogram shortcuts should
not be silently substituted for default scientific output.

## Percentile Definition

ETCCDI-style temperature percentile indices use R Type 8 / Hyndman-Fan method 8.
In icclim/xclim terms this corresponds to:

```text
alpha = 1 / 3
beta = 1 / 3
```

This matters. Different quantile definitions can change thresholds enough to
flip strict exceedance counts on individual days.

For optimized code, "close" is not enough:

- preserve method 8,
- preserve xclim's interpolation branch,
- preserve day-of-year calendar adjustment,
- preserve dtype behavior where strict `>` comparisons are close to threshold.

## Calendar And Day-Of-Year Details

These details are correctness traps:

- Feb 29 handling differs between 365-day, 366-day, no-leap, and 360-day
  calendars.
- xclim can produce a 366-day day-of-year percentile climatology when the
  reference contains leap years.
- `resample_doy` does not merely index a 365-value vector; it may adjust the
  source day-of-year coordinate to the target time axis.
- CDO notes that ETCCDI calculations use circular day-of-year windows at time
  boundaries.
- climdex.pcic excludes Feb 29 from the bootstrap set for ordinary 365-day
  calendars.

Any native icclim implementation must first reproduce icclim/xclim's prepared
thresholds for non-bootstrap years before optimizing bootstrap years.

## Numerical Optimization Options

### Option 1: Safe Tiled xclim Fallback

This is the current reliability baseline.

Strengths:

- exact by construction,
- already merged,
- prevents huge returned dask graphs,
- can retry smaller spatial tiles after memory-like errors.

Weaknesses:

- slow,
- still repeatedly invokes xclim's generic bootstrap machinery,
- not a performance solution.

Keep as the fallback no matter what.

### Option 2: Numba Full-Sort Count Kernel

The current best speed signal is the Numba full-sort prototype:

- load one spatial tile,
- build rolling day-of-year sample indices,
- for each year/cell/donor/day-of-year, gather the sample,
- insertion-sort the small sample,
- compute method-8 quantile,
- count exceedances directly.

Why this is promising:

- it avoids huge dask graphs,
- it uses low memory,
- it was faster than legacy total runtime on Kraken,
- small samples are only about `window * reference_years`, e.g. `5 * 30 = 150`
  values, so simple sorting is not absurd.

Current blocker:

- exactness is not fully proven on Kraken due rare one-day threshold flips.

Best next step:

- reproduce icclim/xclim prepared threshold arrays exactly, then re-run the
  kernel comparison.

### Option 3: Presorted Replacement

This was tested and rejected as implemented.

The idea:

- sort the base sample once,
- remove target-year values,
- merge donor-year values,
- read the needed order statistics.

What happened:

- it was exact and faster on the small local case,
- it was slower on the realistic Kraken benchmark,
- scalar remove/merge scans cost more than re-sorting small samples.

Do not continue this exact implementation unless the data structure changes
substantially.

### Option 4: Selection Instead Of Sorting

For method-8 percentile at a fixed percentile, only two adjacent order
statistics are needed. A selection algorithm could find those ranks without
fully sorting the sample.

Possible algorithms:

- quickselect / introselect,
- `np.partition`-style partial sorting,
- Numba-compatible custom selection for two ranks.

Potential benefit:

- lower asymptotic cost than full sort.

Risk:

- for `n ~= 150`, constant factors can dominate,
- exact NaN and tie behavior must match xclim,
- implementing robust selection may not beat insertion sort in practice.

This is worth benchmarking only after threshold exactness is solved.

### Option 5: Rank/Count Delta From Base Sorted Samples

For each day-of-year/cell, keep the sorted base sample and derive the adjusted
bootstrap quantile rank by accounting for removed and added values.

Potentially better than the rejected presort prototype if:

- only the two needed ranks are queried,
- removal/addition are represented by compact sorted small arrays,
- duplicate values and NaNs are handled carefully,
- the merge stops once the needed ranks are reached.

This is a refined version of the presort idea, not the same implementation.

Risk:

- exact duplicate-value removal is tricky,
- branch-heavy scalar code may again lose to simple sorting.

### Option 6: Precompute In-Base Threshold Cube Per Tile

This follows the climdex.pcic data model:

```text
day_of_year x target_year x donor_year x cell
```

Then counting becomes a simple threshold comparison phase.

Strengths:

- separates threshold construction from counting,
- makes exactness diagnostics easier,
- may support caching thresholds for multiple indices using the same
  percentile/reference period.

Weaknesses:

- memory can grow quickly,
- for 366 days, 30 years, 29 donors, and many cells, tile sizing is essential,
- still needs a fast exact threshold-construction kernel.

This is a strong architectural candidate after the threshold kernel is exact.

### Option 7: Approximate Quantile Algorithms

Streaming and sliding-window quantile algorithms such as Greenwald-Khanna and
related summaries can estimate quantiles with bounded rank error and low memory.

They are not appropriate as the default icclim bootstrap method because:

- icclim's scientific default should be exact,
- ETCCDI/CDO emphasize Type-8 exactness,
- small threshold perturbations can flip day counts,
- users expect reproducibility across platforms and versions.

Approximate methods could be offered only as an explicit expert approximation
mode, with clear metadata and warnings. That is out of scope for the current
reliability/performance fix.

### Option 8: Parallelism Strategy

The bootstrap is naturally parallel across:

- spatial tiles,
- cells within a tile,
- target years,
- donor years,
- variables/indices when thresholds are independent.

Best practical strategy:

- keep dask out of the inner bootstrap replacement loop,
- use Numba threads or process-level tile parallelism,
- keep each worker's memory bounded,
- avoid nested parallelism that oversubscribes cores,
- make `NUMBA_NUM_THREADS` or an icclim expert setting explicit if needed.

## Recommended Research Path

Do this before any new production code:

1. Build a threshold-only comparison harness.
2. For non-overlap years, compare icclim-prepared threshold values and native
   threshold values before counting.
3. Match dtype, calendar, NaN, and Type-8 interpolation behavior exactly.
4. Re-run the Numba full-sort prototype against fresh safe references.
5. Only if exact, benchmark selection-based and refined rank-delta variants.
6. If still promising, integrate as a narrow production fast path with safe
   fallback.

## Threshold Diagnostic Result

Implemented `scripts/diagnose_bootstrap_thresholds.py` to compare percentile
threshold construction paths before running bootstrap counts.

The first results confirm a critical exactness detail:

- `icclim` prepared thresholds match direct xclim thresholds when xclim is run
  on the same dask-backed input.
- Eager-loaded xclim thresholds differ from the dask-prepared threshold by about
  `1e-5`.
- The native NumPy threshold path used by the prototypes matches the eager-loaded
  xclim path, not the dask-prepared icclim path.

Local MPI-ESM subset:

- `direct_xclim_dask`: `max_abs_diff=0`.
- `direct_xclim_loaded`: `max_abs_diff=1.52587890625e-05`.
- `native_numpy_loaded`: `max_abs_diff=1.52587890625e-05`.
- No annual count flips for tested non-reference years `1970` and `2000`.

Kraken ACCESS-CM2 small subset:

- `direct_xclim_dask`: `max_abs_diff=0`.
- `direct_xclim_loaded`: `max_abs_diff=1.4241536462122895e-05`.
- `native_numpy_loaded`: `max_abs_diff=1.4241536462122895e-05`.
- No annual count flips for tested non-reference years `1951` and `1952`.

Interpretation:

- Threshold construction path and dtype/chunk behavior are real, measurable
  sources of numerical differences.
- However, the remaining one-day Numba mismatches were not explained by the
  aggregate threshold diagnostic alone.

Follow-up exact-cell diagnostic:

- Fresh one-cell controls from the current branch reproduced the difference.
- Cell A `(lat=49.375, lon=25.3125)`: safe and `bootstrap=False` both give `40`
  for 1951; Numba gives `39`.
- Cell B `(lat=51.875, lon=30.9375)`: safe and `bootstrap=False` both give `68`
  for 1952; Numba gives `67`.
- Saving thresholds from full `icclim.index(..., save_thresholds=True)` showed
  that full-pipeline `TG90P` uses Celsius-standardized data and float32
  day-of-year thresholds.
- Updating the prototype to convert `tas` to `degC` and cast native thresholds
  to float32 is necessary but not sufficient.
- The first remaining flip was caused by threshold unit-conversion ordering. For
  non-overlap years, icclim prepares percentile thresholds on the original data
  units, then converts the prepared threshold to the studied-data unit. This is
  not bitwise equivalent to converting the full input array to Celsius before
  computing the percentile. For cell A, the flip is on `1951-06-15`: the data
  value is `20.037994384765625`, the saved full-pipeline threshold is
  `20.037988662719727`, and the converted-first native threshold is
  `20.037994384765625`.
- The second remaining flip was in a reference-period year (`1984`) and came
  from the opposite path: xclim's `percentile_bootstrap` recomputes replacement
  thresholds from the comparison data passed to the decorated function, which in
  icclim is already Celsius-normalized. Therefore current icclim behavior is
  hybrid:
  non-overlap years use original-units percentile then threshold conversion;
  bootstrap replacement years use normalized comparison data before percentile
  computation.
- Updating the prototype to reproduce this hybrid path made the two known
  one-cell Kraken controls exact within floating tolerance:
  `changed_cells_gt_1e-9=0` for both cell A and cell B.
- A medium Kraken validation on an ACCESS-CM2 `8x8` subset also matched the
  safe tiled output within floating tolerance: `changed_cells_gt_1e-9=0`,
  `max_abs_diff=4.263256414560601e-14`.
- A full `28x21` comparison against cached legacy output still had three values
  above `1e-9`: `1961-07-02` at `(lat=64.375, lon=27.1875)`,
  `1988-07-01` at `(lat=50.625, lon=6.5625)`, and `2012-07-01` at
  `(lat=43.125, lon=15.9375)`. Fresh one-cell safe controls are required to
  determine whether these are optimized-path differences or cached legacy
  reference differences.
- Therefore a production fast path must reproduce icclim's threshold preparation
  and xclim bootstrap recomputation order, not only the Zhang replacement rule.
- After scientific review, we chose not to preserve this hybrid behavior as the
  desired production target. For temperature percentile indices, icclim should
  normalize both the study data and the full reference series to Celsius before
  percentile construction. Exact legacy reproduction remains useful for
  diagnostics, but the release-candidate behavior should be internally coherent
  and scientifically explicit.

Next diagnostic:

- Run the hybrid prototype against larger multi-cell safe outputs.
- If larger outputs remain exact, treat the full-sort Numba prototype as the
  first viable fast-path candidate.
- If larger outputs reveal new differences, add targeted one-cell diagnostics
  before changing performance code.

## Source Links

- Zhang et al. (2005), "Avoiding inhomogeneity in percentile-based indices of
  temperature extremes", DOI `10.1175/JCLI3366.1`:
  https://www.research.ed.ac.uk/en/publications/avoiding-inhomogeneity-in-percentile-based-indices-of-temperature/
- xclim bootstrap source documentation:
  https://xclim.readthedocs.io/en/v0.44.0/_modules/xclim/core/bootstrapping.html
- xclim indices documentation:
  https://xclim.readthedocs.io/en/stable/indices.html
- climdex.pcic source:
  https://github.com/cran/climdex.pcic/blob/master/R/climdex.r
- PCIC software library:
  https://www.uvic.ca/pcic/research-resources/software-library/index.php
- CDO ETCCDI operator discussion:
  https://code.mpimet.mpg.de/boards/2/topics/6173
- CDO ETCCDI tutorial:
  https://tutorials.dkrz.de/use-case_climate-extremes-indices_cdo.html
- CDO percentile/memory discussion:
  https://code.mpimet.mpg.de/boards/2/topics/12035
- Hyndman and Fan (1996), "Sample quantiles in statistical packages":
  https://robjhyndman.com/papers/sample_quantiles.pdf
- Greenwald-Khanna quantile estimator overview:
  https://aakinshin.net/posts/greenwald-khanna-quantile-estimator/
