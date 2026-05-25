# Predicting US Used Car Prices at Scale with PySpark

MACS 30123 final project, solo. A distributed regression pipeline on Midway 3 that predicts US used car prices from a 9.98 GB Kaggle dataset.

## Research question

The U.S. used car market moves around $850 billion every year, more than new car sales. Listed prices on a marketplace like Cargurus carry a lot of social and economic signal: depreciation from age and mileage, brand premium, vehicle class, and regional preferences for drivetrain and fuel type. I want to know which features actually drive listed prices, and how much variation a well-calibrated model leaves unexplained. The unexplained part is interesting because that is where regional economic conditions, dealer-specific markups, and transportation inequality live. This project builds the calibrated model. I compare a linear baseline against two tree-based ensembles and identify which features matter most.

## Data

I use the [US Used Cars Dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset) from Kaggle. It contains about 3 million listings scraped from Cargurus in late 2020. The raw file is a 9.98 GB uncompressed CSV with 66 columns. The target is the listing `price`, log-transformed because the distribution is heavily right-skewed.

After Stage 1 ingest and Stage 2 cleaning, 2,607,187 rows remain (86.9% retention) across 18 columns. Stage 4 uses an 80/20 train/test split with seed=42, giving 2,086,223 training and 520,964 test rows.

## Why scalable computing

A 9.98 GB CSV does not fit in laptop RAM. Fitting Random Forest or Gradient Boosted Trees with cross-validation over 3 million rows is also far beyond what scikit-learn handles in reasonable wall time on a single machine. The data is dirty as well: missing fields, stringy numerics like "17.7 gal", and 66 columns of mixed types that need parsing before any model can use them.

I run the pipeline on Midway 3 caslake nodes (16 cores, 96 GB RAM per node). PySpark MLlib handles the heavy ML work via spark-submit. Dask DataFrame handles the lighter EDA pass with a single-node LocalCluster. Each pipeline stage is one sbatch job. All six Slurm Job IDs are cited below.

## Pipeline overview

Four stages, one Python file per stage. Each file has a `MODE` flag at the top that I flip between "sample" and "full" via git commits when scaling up.

| # | File | Stage | Tech |
|---|------|-------|------|
| 1 | `01_ingest.py` | Ingest + schema | PySpark |
| 2 | `02_eda_clean.py` | EDA + cleaning | Dask DataFrame |
| 3 | `features.py` | Feature engineering (callable library) | PySpark Pipeline |
| 4 | `04_train_models.py` | Train + compare LR / RF / GBT | PySpark MLlib + CrossValidator |

All sbatch wrappers in `sbatch/`. Slurm logs in `sbatch_logs/`.

## Stage 1 — Ingest

`01_ingest.py` reads the raw CSV and writes a Snappy-compressed Parquet. I dropped 46 of the 66 columns before write: text fields like description and VIN, sparse boolean flags, redundant duplicates, and `savings_amount` which would leak the target. The remaining 20 columns include `price`, `year`, `mileage`, `make_name`, `body_type`, and 15 others.

Sample mode reads the first 100K rows via pandas. Full mode uses PySpark with `samplingRatio=0.01` for schema inference. The output Parquet is 103 MB, a 100× reduction from the 10 GB raw CSV, because the long text columns I dropped were most of the bulk.

Slurm Job `50019862`: caslake, 16 cores, 96G mem, 2 min 21 sec wall. Output: `/scratch/midway3/litong/macs30123/full.parquet`, 3,000,040 rows.

## Stage 2 — EDA and cleaning

`02_eda_clean.py` runs Dask DataFrame on the 20-column Parquet. It profiles nulls, applies cleaning filters, generates four figures, and writes `cleaned.parquet`.

Type cleanup: `fuel_tank_volume` came through as a string ("17.7 gal"), so I strip the unit and cast to float. Spark inferred `combine_fuel_economy` as string from the 0.01-sample (probably hit an "N/A" token), so I drop it as redundant with `city_fuel_economy` and `highway_fuel_economy`. I also drop `city_fuel_economy` and `highway_fuel_economy` themselves because both are 16.4% null and not in the ML feature set.

Null profile across the 18 remaining columns:

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

Cleaning filters: `price ∈ [500, 200_000]`, `year ≥ 1990`, `mileage ≤ 500_000`. Then drop rows missing any of the 11 columns Stage 3 uses as features. Result: 2,607,187 rows kept, which is 86.9% retention.

![price histogram](figures/01_price_hist.png)

Right-skewed price distribution. Mode around $20-25K, long tail out to $200K visible on log y. This is why I log-transform the target.

![mean price by model year](figures/02_mean_price_by_year.png)

Three segments visible at full scale. 1990-1997 sits at a classic-car plateau of $14-21K (survivor bias). 1998-2007 is a daily-driver bottom of $7-10K. 2008-2021 is a smooth depreciation curve from $9K up to $37K.

![price vs mileage hexbin](figures/03_price_vs_mileage.png)

Negative monotone relationship. Density peak in the low-mileage low-to-mid-price corner.

![top 20 makes by mean price](figures/04_top20_makes.png)

Rolls-Royce, Lamborghini, McLaren, Ferrari, Aston Martin lead at $125K+. AM General (sole Hummer H1 maker, only ~12,000 ever produced) and Karma (defunct EV brand) appear in the top 10 because their few remaining listings are all collectibles.

Slurm Job `50032443`: 49 sec wall, MaxRSS 4 GB. Output: `cleaned.parquet`, 88 MB.

## Stage 3 — Feature engineering

`features.py` defines two functions that Stage 4 imports. `preprocess(df)` adds three derived columns: `age = greatest(2021 - year, 1)`, `miles_per_year = mileage / age`, and `log_price = log(price)`. The floor on `age` avoids division-by-zero for 2021 model-year cars.

`build_pipeline(estimator)` returns a PySpark Pipeline: StringIndexer plus OneHotEncoder for 5 categoricals (`make_name`, `body_type`, `fuel_type`, `transmission`, `wheel_system`), then a VectorAssembler that combines the OHE outputs with 7 numerics (`year`, `mileage`, `horsepower`, `engine_displacement`, `fuel_tank_volume`, `age`, `miles_per_year`), then whatever estimator the caller passes in. Total feature width: 89 dimensions.

I keep Stage 3 as a callable library rather than writing a transformed parquet so each CV fold in Stage 4 refits the StringIndexer fresh, which avoids train/test leakage.

The `__main__` smoke test fits LinearRegression with default hyperparameters on the full cleaned data. RMSE 0.1827, R² 0.9037 on the test set.

Slurm Job `50048735`: 48 sec wall, MaxRSS 3 GB.

## Stage 4 — Model comparison

`04_train_models.py` imports `preprocess` and `build_pipeline` from `features`, runs the 80/20 split, then trains one of LR / RF / GBT depending on the `MODEL` flag at the top. I ran the script three times, flipping `MODEL` between runs to get one Slurm Job ID per model.

CV grids: LR sweeps `regParam ∈ {0, 0.01, 0.1}` × 5-fold. RF sweeps `numTrees ∈ {50, 100}` × `maxDepth ∈ {5, 10}` × 3-fold. GBT sweeps `maxIter ∈ {20, 50}` at `maxDepth=5` × 3-fold. Each run appends a row to `results/metrics.csv` and (for tree models) writes top-10 feature importances to `results/`.

Test set results on the 520,964 held-out rows:

| Model | RMSE (log price) | R² | Best params | Wall time | MaxRSS | Slurm Job ID |
|-------|------------------:|----:|-------------|----------:|-------:|--------------|
| LR    | 0.1827 | 0.9037 | regParam=0.0 | 55 s | — | 50056068 |
| **RF**    | **0.1614** | **0.9248** | numTrees=100, maxDepth=10 | 12:40 | 14.3 GB | 50056747 |
| GBT   | 0.1675 | 0.9191 | maxIter=50, maxDepth=5 | 4:20 | 10.8 GB | 50056947 |

Random Forest wins. The gap between RF and GBT is small mostly because I gave RF more grid room (maxDepth=10 vs 5), so this is really a depth comparison, not a bagging-vs-boosting one. LR is only 2 R² points behind RF, which says LR saturates with this much training data and 89 features. GBT did not need the 1M-row fallback I built in (the script has a `FALLBACK_GBT_TO_1M` flag for slower nodes or wider grids).

### Feature importance

Top 10 from each tree model:

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

The two engineered features `age` and `miles_per_year` together explain 33% of RF importance and 17% of GBT importance. Their raw inputs `year` and `mileage` contribute under 1% each. The 3-line `preprocess()` function turns out to be the highest-leverage code in the project.

RF concentrates split utility on the top two features then drops sharply. GBT spreads importance much wider: 5 of its top 10 are brand-specific. This makes sense given how GBT works. It builds trees sequentially on residuals, so once the first few trees absorb the depreciation signal, the remaining trees pick up brand, body, and fuel-type patterns. The CNG fuel type in GBT's top 10 is a good example (CNG vehicles are a small fleet-vehicle niche with distinctive pricing).

In social-science terms, a used car's price is dominated by depreciation. Brand premium, body type, and drivetrain serve as second-order modifiers. The 8% unexplained variance is where regional economic conditions, dealer markups, and cosmetic factors live, none of which this iteration encodes.

## Trade-offs and limitations

**LR numerical instability.** Spark logs "Cholesky solver failed due to singular covariance matrix" on every LR fit and falls back to L-BFGS. The cause is collinearity: `year` and `age` are linearly dependent for cars before 2020, and several OHE-derived dimensions are also linearly dependent. The standard fix is StandardScaler plus ridge regularization, but StandardScaler is not in any course notebook, so I left it out. Tree models in Stage 4 are insensitive to collinearity.

**Geographic features not used.** `latitude` and `longitude` are in `cleaned.parquet` but never fed to the ML pipeline. Raw lat/long would let trees split on "north of 40°N", which is too noisy a proxy for region. A useful geographic model would bin to state or metropolitan area. Plausible R² gain: 1-2 percentage points of regional cost-of-living variation.

**Best hyperparameters hit grid edges.** RF picked the top corner (numTrees=100, maxDepth=10) and GBT picked the upper end (maxIter=50). Grids were sized for a tractable CV wall time on caslake. Pushing to numTrees=300+ and maxDepth=15+ would likely add 1-2 more R² points at multi-hour-per-model cost.
