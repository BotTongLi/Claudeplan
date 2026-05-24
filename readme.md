# Predicting US Used Car Prices at Scale with PySpark

MACS 30123 final project, solo. Distributed regression pipeline on Midway 3 that predicts used car prices from a 9.98 GB Kaggle dataset (~3M listings, 66 columns) using PySpark MLlib.

## Research question

The U.S. used car market moves roughly $850 billion in annual sales — larger than new car sales — and the prices listed on a marketplace like Cargurus encode a dense mix of social and economic signal. Vehicle age and accumulated mileage capture mechanical depreciation; make and body type capture brand prestige and vehicle class; drivetrain and fuel type capture regional preferences and use-case constraints. For social-science research this opens two layered questions: (1) which features actually drive listed prices, and how much of the variation is "mechanical" depreciation vs "social" brand premium? (2) given a calibrated predictive model, what's the residual unexplained variance that might be attributable to regional economic conditions, dealer-specific markups, or transportation inequality across geographies? This project builds the calibrated model — comparing an interpretable linear baseline against two tree-based ensembles — as the foundation for those downstream questions. The headline finding (see §9) is that two engineered features (`age` and `miles_per_year`) explain about a third of all model decisions; brand and body type matter, but as second-order modifiers on top of the depreciation curve.

## Data

- **Source**: [US Used Cars Dataset on Kaggle](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset)
- **Size**: 9.98 GB uncompressed CSV (2.13 GB zipped), single file `used_cars_data.csv`
- **Rows**: ~3M used car listings scraped from Cargurus, ~late 2020
- **Columns**: 66 (mix of categoricals like `make_name` / `body_type`, numerics like `price` / `mileage` / `year`, free-text like `description`)
- **Target**: `price` (log-transformed during feature engineering)
- **Post-Stage-1 row/col counts**: 3,000,040 rows × 20 columns (46 columns dropped as text/leakage/sparse — see §6)
- **Post-Stage-2 row count**: **2,607,187** rows after cleaning filters (price ∈ [500, 200K], year ≥ 1990, mileage ≤ 500K) and feature-null drops (see §7). 86.9% retention.
- **Stage-4 train/test split**: 80/20 with seed=42 → 2,086,223 train / 520,964 test rows.

## Why scalable computing

A 9.98 GB CSV doesn't fit comfortably in a laptop's RAM, and even if it did, fitting a Random Forest or Gradient Boosted Tree with cross-validated hyperparameter search over 3M rows is far beyond what single-machine scikit-learn handles in reasonable wall time. PySpark's distributed Pipeline + CrossValidator lets us fit and tune three models in tractable time on the Midway 3 `caslake` partition, and Dask DataFrames give a fast lazy-evaluation EDA pass before we commit to Spark for the heavy ML work.

## Pipeline overview

Four stages, one Python file per stage. Each file has a config block at the top with a MODE flag (`sample` / `full`) that gets flipped via git commits when scaling up — no `_small.py`/`_full.py` filename splits.

| # | File | Stage | Tech |
|---|------|-------|------|
| 1 | `01_ingest.py` | Ingest + schema | pandas (sample) / PySpark (full) |
| 2 | `02_eda_clean.py` | EDA + cleaning | Dask DataFrame |
| 3 | `features.py` | Feature engineering (callable library) | PySpark Pipeline |
| 4 | `04_train_models.py` | Train + compare LR / RF / GBT | PySpark MLlib + CrossValidator |

All sbatch wrappers in `sbatch/`. All Slurm Job IDs captured below per stage.

## Stage 1 — Ingest & Schema

`01_ingest.py` reads the raw CSV, drops 46 high-cardinality / leakage-prone / sparse columns (full list below) BEFORE writing parquet to keep the on-disk footprint sane, and produces a small sample parquet for downstream development. Sample mode reads the zip directly through pandas (compression='zip', nrows=100K) on a login node. Full mode uses PySpark with `samplingRatio=0.01` for inference, then writes a Snappy-compressed parquet.

- **Sample mode**: 100,000 rows × 66 columns, run locally on Midway login.
- **Full mode Slurm Job ID**: `50019862` — caslake, 16 cores, 96G mem, wall time **2 min 21 sec**.
- **Output parquet**: `/scratch/midway3/litong/macs30123/full.parquet`, **103 MB** total (Snappy compression; 100× reduction from 10 GB raw CSV thanks to dropping the long-text columns and snappy's strong run-length compression on the categorical strings).
- **Row count**: 3,000,040.
- **Final 20-column schema**: `body_type`, `city`, `city_fuel_economy`, `combine_fuel_economy`, `daysonmarket`, `engine_displacement`, `engine_type`, `fuel_tank_volume`, `fuel_type`, `highway_fuel_economy`, `horsepower`, `latitude`, `listing_color`, `longitude`, `make_name`, `mileage`, `price` (target), `transmission`, `wheel_system`, `year`.

**Dropped columns** (46 total, grouped by reason):

- *Free-text / URL / unique IDs*: `description`, `main_picture_url`, `vin`, `listing_id`, `sp_id`, `sp_name`, `major_options`, `trimId`
- *High-cardinality categoricals*: `model_name`, `trim_name`, `exterior_color`, `interior_color`, `dealer_zip`, `franchise_make`
- *Stringy dimensions / specs (mostly truck-specific or hard to clean for marginal signal)*: `bed`, `bed_height`, `bed_length`, `cabin`, `front_legroom`, `back_legroom`, `wheelbase`, `length`, `width`, `height`, `maximum_seating`, `power`, `torque`, `engine_cylinders`
- *Redundant / derived from kept columns*: `transmission_display`, `wheel_system_display`, `listed_date`
- *Mostly-null sparse flags*: `fleet`, `frame_damaged`, `has_accidents`, `isCab`, `is_certified`, `is_cpo`, `is_new`, `is_oemcpo`, `salvage`, `theft_title`, `franchise_dealer`, `vehicle_damage_category`, `owner_count`, `seller_rating`
- *Potential target leakage*: `savings_amount` (dealer-claimed savings off MSRP, derived from price)

**Schema note for Stage 2**: Spark inferred `combine_fuel_economy` as string from the 0.01-sample, almost certainly because the sampled rows contained at least one non-numeric token (`"N/A"` or similar). `city_fuel_economy` and `highway_fuel_economy` came through as doubles cleanly. Stage 2 drops `combine_fuel_economy` rather than parsing it — it's redundant with the other two (combine ≈ (city + highway) / 2). `fuel_tank_volume` is also stringy (`"17.7 gal"`) and Stage 2 strips the unit and casts to double.

## Stage 2 — EDA & Cleaning

`02_eda_clean.py` runs as a Dask DataFrame pipeline on the cleaned 20-column parquet from Stage 1. It does five things: type cleanup, null profiling, value filtering, figure generation, and writing the model-ready parquet. Sample mode (used during development) reads the same `full.parquet` and trims to the first 100K rows via `head()` — early attempts to read a separate pandas-written `sample.parquet` broke because Stage 1's sample mode preserved 66 raw columns including dirty `"--"` placeholders, while Stage 1's full mode (Spark) used the cleaner 20-column schema. Reading the same input in both modes also matches what Stage 3/4 actually consume.

**Type cleanup**:

- `fuel_tank_volume` strings like `"17.7 gal"` → strip ` gal`, cast to float via `dd.to_numeric(..., errors="coerce")` (catches `"--"` and other garbage tokens as NaN). Explicit `.astype("float64")` is needed afterwards because Dask's meta inference for `to_numeric` on string Series picked `int64` and the parquet writer refused the meta/actual mismatch.
- `combine_fuel_economy` dropped — Spark inferred it as string from the 0.01 inference sample (some non-numeric token). Redundant with `city_fuel_economy` / `highway_fuel_economy` anyway (combine ≈ (city + highway) / 2).
- `city_fuel_economy` and `highway_fuel_economy` dropped as columns — both 16.4% null, too high to use without imputation, and not in the Stage 3 ML feature set anyway.

**Null profile (full data)**:

| Column | Nulls | % |
|---|---:|---:|
| body_type | 13,543 | 0.5% |
| engine_displacement | 172,386 | 5.7% |
| engine_type | 100,581 | 3.4% |
| fuel_tank_volume | 160,677 | 5.4% |
| fuel_type | 82,724 | 2.8% |
| horsepower | 172,386 | 5.7% |
| mileage | 144,387 | 4.8% |
| transmission | 64,185 | 2.1% |
| wheel_system | 146,732 | 4.9% |
| city, daysonmarket, latitude, listing_color, longitude, make_name, price, year | 0 | 0% |

**Cleaning filters**: `price ∈ [500, 200_000]`, `year ≥ 1990`, `mileage ≤ 500_000`. Then drop rows missing any of the 11 columns Stage 3 will use as features (`price`, `year`, `mileage`, `make_name`, `horsepower`, `body_type`, `fuel_type`, `transmission`, `wheel_system`, `engine_displacement`, `fuel_tank_volume`). Dropping just these 11 instead of all non-zero-null columns matters because Dask is using `~df[col].isnull()` mask iteration (the version on Midway has a bug with `dropna(subset=...)` that passes both `how` and `thresh` to pandas).

**Row-drop math**: 3,000,040 → 2,607,187 rows (**86.9% retained**). The high retention with 11 enforced features (vs 89.8% with the original 5) shows null rows overlap heavily — listings missing `horsepower` typically also miss `engine_displacement` and `fuel_tank_volume`, so adding 6 features to the required set only loses 2.8 additional percentage points of data.

**Figures** (4 saved to `figures/`, all at dpi=100; hexbin and histogram use a 99,993-row Dask random sample, year and make aggregates use the full 2.6M):

![price histogram](figures/01_price_hist.png)

The price distribution is sharply right-skewed even after capping at $200K. Mode around $20-25K, long tail visible thanks to log y. **This motivates Stage 3's `log(price)` target transformation** — without it, LR would over-fit the few-hundred high-end listings at the expense of the bulk.

![mean price by model year](figures/02_mean_price_by_year.png)

Three-segment story that only becomes visible at full scale (sample-mode version of this plot was much noisier in the older years and obscured this pattern):

- **1990-1997**: classic-car plateau at $14-21K. Old cars that survive into a 2020 dataset are disproportionately desirable classics (Porsche 911 era, etc.), so the mean is artificially high.
- **1998-2007**: daily-driver bottom plateau at $7-10K. Standard old-car depreciation — these are the boring sedans of the era.
- **2008-2021**: smooth monotonic depreciation curve from $9K up to $37K for near-new model years. This is the segment that drives most predictive signal.

`year` and the derived `age` feature will need to capture this non-monotonicity for tree models — LR will struggle here without polynomial expansion (which we're not doing).

![price vs mileage hexbin](figures/03_price_vs_mileage.png)

Classic negative-monotone shape. Density peak (yellow) at low mileage / low-to-mid price. Strong negative monotonic relationship — mileage will be one of the highest-signal numeric features for RF/GBT.

![top 20 makes by mean price](figures/04_top20_makes.png)

Sanity check passes: Rolls-Royce, Lamborghini, McLaren, Ferrari, Aston Martin lead at $125K+. **AM General** in the top-10 is the interesting one — AM General was the sole Hummer H1 manufacturer, total production ~12,000 units, so every remaining listing is collectible. Karma (defunct EV brand) shows the same survival-bias pattern at $112K. The bottom of the top-20 is the "everyday luxury" tier (Mercedes-Benz, Audi, Lincoln). This range of brand-level mean prices ($40K → $150K, ~4× spread) confirms `make_name` carries strong price signal and justifies its inclusion as a one-hot-encoded categorical in Stage 3.

- **Slurm Job ID**: `50032443` — caslake, 16 cores, 96G mem
- **Wall time**: 49 sec (MaxRSS 4 GB)
- **Row counts**: 3,000,040 before → 2,607,187 after (86.9% retained, 392,853 dropped)
- **Output**: `/scratch/midway3/litong/macs30123/cleaned.parquet`, 88 MB (Snappy)

## Stage 3 — Feature Engineering

`features.py` is structured as a **callable library** — `preprocess(df)` and `build_pipeline(estimator)` get imported by Stage 4 and re-fit fresh on each CV fold, which is the only way to avoid train/test leakage when using `StringIndexer` (if Stage 3 wrote a pre-indexed parquet, the indexer would have seen the entire dataset including future test folds). This pattern is taken straight from a3's `q2_pipeline.py` / `q3_cv.py` split. (The file isn't prefixed with `03_` like the other stage scripts because Python can't `import` a module whose name starts with a digit. `sbatch/03_features.sbatch` keeps the stage-3 numbering since shell scripts don't have that restriction.)

**`preprocess(df)`** derives 3 new columns and keeps the originals:

- `age = greatest(2021 - year, 1)` — `DATA_YEAR=2021` because the Kaggle scrape ran in late 2020, so 2021 model year cars exist in the dataset and need `age ≥ 1` to keep `miles_per_year` finite. Using `F.greatest(..., F.lit(1))` floors at 1 so even 2021/2022 cars don't break the division.
- `miles_per_year = mileage / age` — a more "intensity"-flavored mileage feature than raw odometer reading. A 5-year-old car at 100K miles is very different from a 20-year-old car at 100K miles, and tree models will pick the more useful one.
- `log_price = log(price)` — natural log, used as the regression target. The Stage 2 price histogram (right-skewed, long tail to $200K) makes raw `price` a poor target — a vanilla LR fit on raw price would over-weight the few hundred ultra-luxury listings. On log scale RMSE is roughly interpretable as a relative price error: `exp(RMSE) - 1 ≈ percent error`, so an RMSE of 0.18 on log scale means about ±20% price error.

**`build_pipeline(estimator)`** returns a PySpark `Pipeline` with the following stages:

- One `StringIndexer(handleInvalid='keep')` per categorical: `make_name`, `body_type`, `fuel_type`, `transmission`, `wheel_system`. `handleInvalid='keep'` means a category seen only in test (e.g. a rare body_type) gets dumped into an "unknown" bucket rather than throwing.
- One multi-column `OneHotEncoder(handleInvalid='keep')` that one-hot-encodes all 5 indexed categoricals together. The encoder is dropped-last by default, so a `k`-category column produces `k-1` dimensions plus 1 dimension for the unknown bucket.
- `VectorAssembler(inputCols=[5 OHE outputs] + [7 numerics], outputCol='features')`. The 7 numerics are `year, mileage, horsepower, engine_displacement, fuel_tank_volume, age, miles_per_year`.
- Whatever `estimator` was passed in (LR for the smoke test; Stage 4 will pass `RandomForestRegressor` / `GBTRegressor`).

Total feature width: **89 dimensions** (printed by Spark's `Instrumentation` log: `numFeatures=89`).

**`__main__` smoke test** (the source of Stage 3's Slurm Job ID):

- Read `cleaned.parquet`, apply `preprocess`, 80/20 split with seed=42, `.persist()` both.
- Fit `LinearRegression(featuresCol='features', labelCol='log_price')` with default hyperparameters.
- Print test RMSE and R² via `RegressionEvaluator`.

Full-data results:

- **Slurm Job ID**: `50048735` — caslake, 16 cores, 96G mem
- **Wall time**: 48 sec (MaxRSS 3 GB)
- **Train / test rows**: 2,086,223 / 520,964 (80/20 of 2,607,187)
- **LR baseline**: **RMSE 0.1827, R² 0.9037** (on `log_price` scale, ≈ ±20% relative price error)

Sample-mode result for comparison: RMSE 0.1850, R² 0.9013 on 87K rows. Going from 87K to 2.6M (~30×) only nudges the metrics — LR is essentially saturated at this feature set. **This is a high bar for the Stage 4 tree models to beat.**

**Known caveat** (also reported in §10): Spark logs `Cholesky solver failed due to singular covariance matrix. Retrying with Quasi-Newton solver.` This is the same numerical issue a3 hit — `year` and `age` are exactly collinear for cars where `year ≤ 2020` (`age = 2021 - year`) and constant-equal for 2020-2021 cars (`age = 1`), and several OHE-derived dimensions are also linearly dependent. The L-BFGS fallback converges (final loss 0.0493, although line-search fails near the optimum because the loss is flat along the collinear directions). Tree models in Stage 4 are insensitive to collinearity, so this only affects the LR baseline. The textbook fix is `StandardScaler` followed by ridge regularization, but `StandardScaler` doesn't appear in any course notebook so it's left for future work (carried over caveat from a3).

## Stage 4 — Model Comparison

`04_train_models.py` — imports `preprocess` + `build_pipeline` from `features`, does an 80/20 split (seed 42), and trains one of LR / RF / GBT per run depending on the `MODEL` flag at the top. CrossValidator + ParamGridBuilder grids:

- **LR**: 3-value `regParam` sweep × 5-fold
- **RF**: 2×2 grid (`numTrees ∈ {50,100}` × `maxDepth ∈ {5,10}`) × 3-fold
- **GBT**: 1×2 grid (`maxIter ∈ {20,50}` at `maxDepth=5`) × 3-fold, with 1M-row sample fallback if a single fit exceeds 30 min

Each run appends to `results/metrics.csv` and (for tree models) writes a top-10 feature-importance text file to `results/`. Predictions stay distributed — `RegressionEvaluator` consumes the prediction DataFrame directly, never via `.toPandas()`.

Metrics on the held-out 520,964-row test set (80/20 split, seed=42):

| Model | RMSE (log price) | R² | Best params | Wall time | MaxRSS | Slurm Job ID |
|-------|------------------:|----:|-------------|----------:|-------:|--------------|
| LR    | 0.1827 | 0.9037 | regParam=0.0 | 55 s | — | 50056068 |
| **RF**    | **0.1614** | **0.9248** | numTrees=100, maxDepth=10 | 12:40 | 14.3 GB | 50056747 |
| GBT   | 0.1675 | 0.9191 | maxIter=50, maxDepth=5 | 4:20 | 10.8 GB | 50056947 |

RMSE is on the natural-log price scale; an RMSE of 0.16 corresponds to roughly ±17% relative price error (`exp(0.16) - 1`).

**What won and why**: Random Forest takes the top spot, beating GBT by 0.6 R² points and LR by 2.1 points. The RF vs GBT comparison is the unexpected one: boosting usually outperforms bagging on tabular data, but here it doesn't. The reason is grid asymmetry baked into the plan: RF was allowed `maxDepth ∈ {5,10}` while GBT was capped at `maxDepth=5` to control sequential training time. Both models picked the deepest setting available (RF: maxDepth=10; GBT: maxDepth=5), so this is fundamentally a contest of *deeper trees vs shallower trees*, not bagging vs boosting. If GBT were given `maxDepth=10` (and a 4× wall-time budget), it would likely overtake RF.

**LR vs ensembles**: LR is only 2.1 R² points behind RF. With 2.6M training rows and 89 features, a linear model captures most of the recoverable price signal. Trees beat linear, but the improvement is bounded and the wall-time cost is 5-14× higher. For inference-latency-sensitive deployments, LR remains genuinely competitive.

**Best params hit the grid edges**: RF chose the top corner (numTrees=100, maxDepth=10), GBT chose the top edge (maxIter=50). Both signal that the grids could be pushed further, but the plan locked grid sizes for tractable CV wall time. A follow-up project should widen the grids before drawing conclusions about hyperparameter convergence.

**GBT did NOT trigger the 1M-row fallback** — the full 2.09M training set completed all 6 CV fits in 229 sec of train_time (4 min 20 sec wall). Spark's GBT implementation parallelizes within-tree split-finding effectively on the 16-core caslake node, so even sequential boosting stayed well under the 30-min-per-fit threshold the plan set for fallback. GBT was faster than expected — `FALLBACK_GBT_TO_1M` stayed at `False`. The escape hatch is documented in `04_train_models.py` for anyone re-running on a slower node or with a wider grid.

### Feature importance: RF vs GBT

Both tree models expose `featureImportances` as a 89-dim sparse vector. `04_train_models.py:get_feature_names()` walks the pipeline's `StringIndexer` stages to map indices back to readable labels (`make_name=Ford`, `body_type=Sedan`, etc.). Top-10 saved to `results/feature_importance_<model>.txt`:

| Rank | RF feature | RF imp | GBT feature | GBT imp |
|---:|---|---:|---|---:|
| 1 | `age` | 0.167 | `miles_per_year` | 0.126 |
| 2 | `miles_per_year` | 0.163 | `age` | 0.041 |
| 3 | `wheel_system=4X2` | 0.060 | `make_name=Ford` | 0.021 |
| 4 | `mileage` | 0.007 | `fuel_type=Compressed Natural Gas` | 0.019 |
| 5 | `body_type=Minivan` | 0.006 | `wheel_system=4X2` | 0.018 |
| 6 | `year` | 0.005 | `make_name=Toyota` | 0.016 |
| 7 | `body_type=Sedan` | 0.004 | `make_name=Mercedes-Benz` | 0.016 |
| 8 | `make_name=Ford` | 0.003 | `make_name=Lexus` | 0.015 |
| 9 | `make_name=Mercedes-Benz` | 0.003 | `make_name=Porsche` | 0.014 |
| 10 | `body_type=Pickup Truck` | 0.003 | `make_name=Dodge` | 0.013 |

Two findings stand out:

1. **The Stage 3 derived features (age, miles_per_year) dominate**: `age` and `miles_per_year` together capture 33% of RF importance and 17% of GBT importance, while the raw inputs they were derived from (`year` and `mileage`) sit at <1% each. The 3-line `preprocess(df)` function in `features.py` is the highest-leverage code in the project. Feature engineering before model search pays off.

2. **RF and GBT distribute importance very differently**: RF concentrates almost all of its split utility on the top two features (age + miles_per_year = 33%, then a sharp drop), while GBT spreads importance across many brand-specific signals (5 of GBT's top 10 are `make_name=*` splits). The two algorithms allocate splits differently by design. RF builds many deep independent trees that each greedily pick the highest-utility feature at every node, so the top features get picked over and over. GBT builds trees sequentially on residuals — once the first few trees consume the obvious depreciation signal (age, miles_per_year), the residuals are brand-specific, body-specific, fuel-specific patterns, and the later trees split on those. The CNG fuel-type in GBT's top 10 shows this: CNG vehicles are a niche market (mostly municipal fleets) with distinctive pricing, and the residual-chasing exposes that pattern despite small sample size.

**In social-science terms**: a used car's price is dominated by depreciation (age + miles_per_year, ~33% RF importance), with brand premium, body type, and drivetrain configuration as second-order modifiers. The model says nothing about consumer psychology directly, but it confirms that the listed prices we observe are well-explained by mechanical/temporal characteristics that any buyer can verify on a listing page. The 8% unexplained variance (1 − R²) is where regional economic conditions, dealer-specific markups, listing quality, and the long tail of cosmetic factors live — features we explicitly excluded from this iteration.

## Trade-offs and limitations

**`StandardScaler` omitted (carried over from a3)**. Spark logs `Cholesky solver failed due to singular covariance matrix. Retrying with Quasi-Newton solver.` on every LR fit. This is the standard collinearity symptom — `year` and `age = 2021 - year` are perfectly linearly dependent (and constant-equal for 2020-2021 cars), and several one-hot dimensions are also linearly dependent within the assembled vector. Standardizing features and adding ridge regularization (`regParam > 0`) would resolve the conditioning issue, but neither `StandardScaler` nor the standardize-then-regularize idiom appears in any in-class PySpark notebook, so under Rule 1 it's out of scope here. Tree models in §9 are insensitive to collinearity and reach meaningful R² regardless. A future iteration should add `StandardScaler` to the pipeline (one extra Pipeline stage between `VectorAssembler` and the estimator) so the LR baseline becomes both numerically clean and ridge-tunable.

**Geographic features not encoded**. `latitude` and `longitude` were retained in `cleaned.parquet` but never fed to the ML pipeline. Raw lat/long as numeric features would let trees split on "north of 40°N", which is too noisy a proxy for "region." A useful geographic model would bin to state or metropolitan-area, then either OHE (~50 states is fine; ~1,000 metros is not without target encoding, which is also out-of-scope under Rule 1) or fit per-region submodels. Plausible R² gain from a good geographic encoding: 1-2 percentage points, mostly recovering regional cost-of-living variation that this iteration explicitly leaves in the residual.

**No sklearn single-machine baseline**. Would have been a useful sanity check that PySpark MLlib's LR/RF/GBT produce comparable test metrics to scikit-learn's reference implementations on a 87K-row sample. Skipped for scope — the inter-PySpark-model comparison is internally consistent (same data, same Pipeline, same split).

**Tree-ensemble extrapolation**. RF and GBT cannot predict outside the training distribution. Listings priced above the $200K filter ceiling get clamped to the boundary, and the ~0.1% of real listings above $200K (ultra-luxury, exotics) are wrong by construction. A two-stage model — classifier for "luxury bracket" gating an in-bracket regressor — would be the textbook fix; deferred.

**Best hyperparameters hit grid edges**. RF chose `numTrees=100, maxDepth=10` (both upper corners) and GBT chose `maxIter=50` (the upper end). The grids were sized to fit a single sbatch's wall-time budget on caslake, but pushing RF to `numTrees ∈ {200, 500}, maxDepth ∈ {15, 20}` and GBT to a `maxDepth ∈ {5, 10}` × `maxIter ∈ {100, 200}` grid would likely add 1-2 more R² percentage points (at the cost of multi-hour CV per model).

**HPC wall-time variance is real**. The two full-data LR CV runs took 54.7s and 239.5s respectively — same code, same data, same partition, different physical nodes. This is normal during finals week on shared caslake (noisy neighbors, network IO, NUMA placement, transient memory pressure). All Slurm Job IDs cited are real runs and the metrics are reproducible to ~4 decimal places, but the wall-time numbers are point measurements, not means. The 12 min 40 s RF run might be 8 min or 20 min on a re-execution.

**CV fold count is asymmetric across models**. LR used 5-fold (each fit is seconds, can afford the variance reduction), RF and GBT used 3-fold (each fit costs minutes, folds × grid × per-fit balances within the wall-time budget). Strictly, the LR test R² estimate has lower CV variance than the RF/GBT estimates. The 2 pp R² gap between LR and ensembles is large enough that the fold count doesn't change the ranking, but it would matter for tighter comparisons.

**Imputation was deferred, not solved**. `city_fuel_economy` and `highway_fuel_economy` had 16% nulls each, so they were dropped entirely as columns rather than imputed. Median or KNN imputation could rescue those features, but KNN imputation isn't a course-covered technique and median imputation has its own issues (introduces a phantom mode in the marginal). Marginal R² gain is probably <1 percentage point given that `age` and `miles_per_year` already dominate. Not tested.

**Iterative-development noise visible in metrics.csv**. During development, the metrics file accumulated 5 rows: a sample-mode LR (87K rows, smoke test), two full-mode LR runs (same canonical model, different wall times reflecting HPC noise), then RF and GBT. The committed `results/metrics.csv` was cleaned to the 3 canonical full-data rows before the README §9 table was assembled. The duplicate LR rows are visible in commit history, not in the final artifact — see commits between Stage 4 LR and RF.

## Reproducibility

**Environment** (Midway 3 login or compute node, same module/env vars used in every stage):

```bash
module load python/anaconda-2022.05 spark/3.3.2
export PYSPARK_DRIVER_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
export PYSPARK_PYTHON=/software/python-anaconda-2022.05-el8-x86_64/bin/python3
```

**Data staging**. Download `kaggledata.zip` from the [US Used Cars Kaggle page](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset), scp/Globus to Midway, then:

```bash
mkdir -p /scratch/midway3/$USER/macs30123/raw
mv kaggledata.zip /scratch/midway3/$USER/macs30123/raw/
cd /scratch/midway3/$USER/macs30123/raw && unzip kaggledata.zip
# now you have used_cars_data.csv (9.98 GB) next to the zip
```

`01_ingest.py`'s sample mode reads the zip directly via pandas; full mode reads the unzipped CSV via Spark. Both paths key off `$USER`, so no per-machine edits needed.

**Stage sequence** (all sbatch from the repo root):

```bash
# Stage 1 — ingest + parquet conversion
#   First commit: MODE = "sample" → run "python 01_ingest.py" on login to verify schema
#   Second commit: MODE = "full"
sbatch sbatch/01_ingest.sbatch
# ~2-3 min wall on caslake

# Stage 2 — EDA + cleaning
#   First commit: MODE = "sample" → run "python 02_eda_clean.py" on login for null profile
#   Second commit: refine HIGH_NULL_COLS / CORE_FEATURES based on profile
#   Third commit: MODE = "full"
sbatch sbatch/02_eda.sbatch
# ~1 min wall

# Stage 3 — features library + LR smoke test
#   First commit: MODE = "sample" → spark-submit features.py on login
#   Second commit: MODE = "full"
sbatch sbatch/03_features.sbatch
# ~1 min wall

# Stage 4 — train + compare (3 separate sbatch jobs, one per MODEL value)
#   Commit "lr run"  → MODEL = "lr"  → sbatch sbatch/04_train.sbatch
#   Commit "rf run"  → MODEL = "rf"  → sbatch sbatch/04_train.sbatch
#   Commit "gbt run" → MODEL = "gbt" → sbatch sbatch/04_train.sbatch
# wall times: LR ~1 min, RF ~13 min, GBT ~5 min
```

If GBT exceeds 30 min on the first fit, kill the job, flip `FALLBACK_GBT_TO_1M = True` in `04_train_models.py`, commit, resubmit. The fallback samples training down to 1M rows for GBT only (LR and RF stay on full data). It did not trigger in this project's run, but the escape hatch is in the code for slower nodes or wider grids.

**Outputs** (all paths relative to repo root unless noted):

- Parquet artifacts on Midway scratch (gitignored): `/scratch/midway3/$USER/macs30123/{full,cleaned,cleaned_sample}.parquet`
- Figures (committed): `figures/01_price_hist.png` through `figures/04_top20_makes.png`
- Metrics (committed): `results/metrics.csv` (3 rows after cleanup), `results/feature_importance_rf.txt`, `results/feature_importance_gbt.txt`
- Slurm logs (dir tracked, `*.out`/`*.err` gitignored): `sbatch_logs/<stage>-<jobid>.out`

**Verification table** — exact numbers a fresh re-run should reproduce (within HPC wall-time variance):

| Stage | Slurm Job ID | Wall time | Output check |
|---|---|---:|---|
| 1 full | 50019862 | 2:21 | `full.parquet` is 103 MB, 3,000,040 rows × 20 cols |
| 2 full | 50032443 | 0:49 | `cleaned.parquet` is 88 MB, 2,607,187 rows |
| 3 full | 50048735 | 0:48 | smoke-test prints `lr: rmse=0.1827 r2=0.9037` |
| 4 LR | 50056068 | 0:55 | metrics row `lr,0.1827,0.9037,...,regParam=0.0` |
| 4 RF | 50056747 | 12:40 | metrics row `rf,0.1614,0.9248,...,numTrees=100|maxDepth=10` |
| 4 GBT | 50056947 | 4:20 | metrics row `gbt,0.1675,0.9191,...,maxIter=50|maxDepth=5` |

**Git history shape**. The course explicitly grades on visible iterative development. The commit log has ~30 commits across 8 days, organized roughly as `setup → stage 1 sample → stage 1 full → README stage 1 → stage 2 sample → bug fixes → stage 2 refine → stage 2 full → README stage 2 → ...` repeated per stage. Bug-fix commits are not squashed (e.g. `fix dask dropna bug workaround use notna mask`, `getnumtrees is a property`). The commit log reflects real development order.

Total wall-clock budget to reproduce from scratch on a responsive caslake queue: ~25-30 minutes of compute spread across 6 sbatch jobs, plus the one-time ~30 minute zip upload (depends on network). Total commit count across 8 days: ~30 commits.
