# Predicting US Used Car Prices at Scale with PySpark

MACS 30123 final project, solo. A distributed regression pipeline on Midway 3 that predicts US used car prices from a 9.98 GB Kaggle dataset.

## Research question

I've been looking at used cars recently, and noticed that two listings for the same model and year can be priced thousands of dollars apart. This drew my attention and I began to wonder which features actually drive listed prices, and how much of the variation a model can explain. The unexplained part is interesting because that's probably where regional economic conditions, dealer markups, and transportation inequality live.

The U.S. used car market moves around $850 billion every year, more than new car sales. Listed prices on Cargurus carry many signals at once. Age and mileage drive mechanical depreciation. Brand and body type reflect prestige and class. Drivetrain and fuel type capture regional preferences. I want to know which of these matter most and how much of the price variation is recoverable from data a buyer could check on a listing page.

This project builds the calibrated model. I compare a linear baseline against Random Forest and Gradient Boosted Trees, then look at which features the tree models actually rely on.

## Data

I use the [US Used Cars Dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset) from Kaggle. It contains about 3 million listings scraped from Cargurus in late 2020. The raw file is a 9.98 GB uncompressed CSV with 66 columns. The target is the listing `price`, log-transformed because the raw distribution is heavily right-skewed.

After ingest and cleaning, 2,607,187 rows remain (about 87% retention) across 18 columns. The training script uses an 80/20 train/test split with seed=42, giving 2,086,223 training and 520,964 test rows.

## Why scalable computing

A 9.98 GB CSV does not fit in laptop RAM. I tried opening it in pandas on my own machine and it stalled. Fitting Random Forest or Gradient Boosted Trees with cross-validation over 3 million rows is also far beyond what scikit-learn can do on one machine in any reasonable time. The data is dirty as well: missing fields, stringy numerics like "17.7 gal", and 66 columns of mixed types that need parsing before any model can use them.

I run the pipeline on Midway 3 caslake nodes (16 cores, 96 GB RAM per node). PySpark MLlib handles the heavy ML work via spark-submit. Dask DataFrame handles the lighter cleaning pass with a single-node LocalCluster. Every step is one sbatch job, and all six Slurm Job IDs are cited below.

## Pipeline overview

Four scripts, one per logical step. Each has a `MODE` flag at the top that I flip between "sample" and "full" via git commits when scaling up.

| # | File | What it does | Tech |
|---|------|-------|------|
| 1 | `01_ingest.py` | Read CSV, drop unused columns, write Parquet | PySpark |
| 2 | `02_eda_clean.py` | Profile nulls, apply filters, generate figures | Dask DataFrame |
| 3 | `features.py` | Reusable feature engineering library | PySpark Pipeline |
| 4 | `04_train_models.py` | Train and compare LR / RF / GBT | PySpark MLlib + CrossValidator |

All sbatch wrappers in `sbatch/`. Slurm logs in `sbatch_logs/`.

## Reading the data and writing Parquet

`01_ingest.py` reads the raw CSV and writes a Snappy-compressed Parquet. I dropped 46 of the 66 original columns before write. Most of them weren't going to help with prediction: long text fields like vehicle description and image URLs, unique identifiers like VIN, sparse boolean flags, redundant duplicates, and `savings_amount` which would leak the target since it's derived from price. The 20 columns I kept include `price`, `year`, `mileage`, `make_name`, `body_type`, and a handful of others.

In sample mode the script reads the first 100K rows via pandas. In full mode it uses PySpark with `samplingRatio=0.01` for schema inference. The output Parquet is 103 MB, which is a 100× shrink from the 10 GB raw CSV. Most of that compression came from dropping the long text columns.

Slurm Job `50019862`: caslake, 16 cores, 96G mem, 2 min 21 sec wall. Output: `/scratch/midway3/litong/macs30123/full.parquet`, 3,000,040 rows.

## Cleaning and exploring the data

`02_eda_clean.py` runs Dask DataFrame on the 20-column Parquet. It profiles nulls, applies cleaning filters, generates four figures, and writes `cleaned.parquet`.

I ran into a few annoying issues here. `fuel_tank_volume` came through as a string like "17.7 gal", so I strip the unit and cast to float using `dd.to_numeric(..., errors="coerce")` to handle "--" placeholders. Spark inferred `combine_fuel_economy` as string from the 0.01 sample (it probably hit an "N/A" token somewhere), so I drop it. It's redundant with `city_fuel_economy` and `highway_fuel_economy` anyway. I also drop those two themselves because both are 16.4% null and not in my final feature set.

A separate quirk: Dask's `dropna(subset=...)` on the version installed at Midway has a bug where it passes both `how` and `thresh` to pandas, which crashes. I worked around this with a `for col in features: df = df[~df[col].isnull()]` loop. Not pretty but it works.

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

Cleaning filters: `price ∈ [500, 200_000]`, `year ≥ 1990`, `mileage ≤ 500_000`. Then drop rows missing any of the 11 columns used as features. After all this, 2,607,187 rows remain (about 87%).

![price histogram](figures/01_price_hist.png)

Heavily right-skewed. Mode around $20-25K. The long tail to $200K is visible thanks to the log y axis. This is why I log-transform the target.

![mean price by model year](figures/02_mean_price_by_year.png)

Three segments visible at full scale. 1990-1997 sits at a classic-car plateau of $14-21K (survivor bias). 1998-2007 is the daily-driver bottom at $7-10K. 2008-2021 is a smooth depreciation curve from $9K up to $37K.

![price vs mileage hexbin](figures/03_price_vs_mileage.png)

Negative monotone, as expected. Density peak is in the low-mileage low-to-mid-price corner.

![top 20 makes by mean price](figures/04_top20_makes.png)

Rolls-Royce, Lamborghini, McLaren, Ferrari, Aston Martin lead at $125K+. AM General (sole Hummer H1 maker, only about 12,000 ever produced) and Karma (defunct EV brand) appear in the top 10 because their few remaining listings are all collectibles.

Slurm Job `50032443`: 49 sec wall, MaxRSS 4 GB. Output: `cleaned.parquet`, 88 MB.

## Feature engineering

`features.py` defines two functions that the training script imports. `preprocess(df)` adds three derived columns: `age = greatest(2021 - year, 1)`, `miles_per_year = mileage / age`, and `log_price = log(price)`. The floor on `age` avoids division-by-zero for 2021 model-year cars.

`build_pipeline(estimator)` returns a PySpark Pipeline: StringIndexer plus OneHotEncoder for 5 categoricals (`make_name`, `body_type`, `fuel_type`, `transmission`, `wheel_system`), then a VectorAssembler that combines those with 7 numerics (`year`, `mileage`, `horsepower`, `engine_displacement`, `fuel_tank_volume`, `age`, `miles_per_year`), then whatever estimator I pass in. The full feature vector is 89 dimensions.

I keep this as a library and skip writing a transformed parquet. The training script imports it so each cross-validation fold refits the StringIndexer fresh, which keeps the test data from leaking into training.

The `__main__` block runs a quick LinearRegression fit on the full cleaned data as a smoke test. RMSE 0.1827, R² 0.9037 on the held-out test set.

Originally I named this file `03_features.py` to match the other scripts. Python can't import a module whose name starts with a digit though, so I renamed it. The sbatch wrapper stays as `sbatch/03_features.sbatch` since shell doesn't have the same restriction.

Slurm Job `50048735`: 48 sec wall, MaxRSS 3 GB.

## Training and comparing three models

`04_train_models.py` imports `preprocess` and `build_pipeline` from `features`, runs the 80/20 split, then trains one of LR / RF / GBT depending on the `MODEL` flag at the top. I ran the script three times, flipping `MODEL` between runs to get one Slurm Job ID per model.

CV grids: LR sweeps `regParam ∈ {0, 0.01, 0.1}` × 5-fold. RF sweeps `numTrees ∈ {50, 100}` × `maxDepth ∈ {5, 10}` × 3-fold. GBT sweeps `maxIter ∈ {20, 50}` at `maxDepth=5` × 3-fold. Each run appends a row to `results/metrics.csv` and (for tree models) writes top-10 feature importances to `results/`.

The first RF run failed at the final metric-write step because I had `best_est.getNumTrees()` with parens. Turns out `getNumTrees` is a `@property` on `_TreeEnsembleModel`, not a method, so calling it with parens tries to call an int. I fixed it and re-ran. That cost me 12 minutes of compute I didn't get back.

Test set results on the 520,964 held-out rows:

| Model | RMSE (log price) | R² | Best params | Wall time | MaxRSS | Slurm Job ID |
|-------|------------------:|----:|-------------|----------:|-------:|--------------|
| LR    | 0.1827 | 0.9037 | regParam=0.0 | 55 s | — | 50056068 |
| **RF**    | **0.1614** | **0.9248** | numTrees=100, maxDepth=10 | 12:40 | 14.3 GB | 50056747 |
| GBT   | 0.1675 | 0.9191 | maxIter=50, maxDepth=5 | 4:20 | 10.8 GB | 50056947 |

Random Forest wins, but only by 0.6 R² points over GBT. The gap is small mostly because I gave RF more grid headroom (maxDepth=10 vs 5), so this is really a depth comparison, not a bagging-vs-boosting one. LR is only 2 R² points behind RF, which says LR saturates with this much training data and 89 features. GBT did not need the 1M-row fallback I had prepared. The script still has a `FALLBACK_GBT_TO_1M` flag for slower nodes or wider grids.

### What features matter most

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

The two derived features I added (`age` and `miles_per_year`) explain 33% of RF importance and 17% of GBT importance. The raw inputs they came from (`year` and `mileage`) contribute under 1% each. The 3-line `preprocess()` function turned out to be the highest-leverage code in the project.

RF concentrates split utility on the top two features then drops sharply. GBT spreads importance much wider, with 5 of its top 10 being brand-specific splits. This makes sense given how boosting works. GBT builds trees sequentially on residuals, so once the first few trees absorb the depreciation signal, the rest go after brand, body, and fuel-type patterns. The CNG fuel type in GBT's top 10 is a good example. CNG vehicles are a small fleet-vehicle niche with distinctive pricing, and the residual-chasing surfaces them despite small sample size.

In social-science terms, a used car's price is dominated by depreciation. Brand premium, body type, and drivetrain serve as second-order modifiers. The 8% unexplained variance is where regional economic conditions, dealer markups, and cosmetic factors live, none of which I encoded here.

## Trade-offs and limitations

**LR numerical instability.** Spark logs "Cholesky solver failed due to singular covariance matrix" on every LR fit and falls back to L-BFGS. The cause is collinearity: `year` and `age` are linearly dependent for cars before 2020, and several OHE dimensions are also linearly dependent. The standard fix is StandardScaler plus ridge regularization, but StandardScaler doesn't appear in any course notebook, so I left it out. Tree models are insensitive to this.

**Geographic features not used.** `latitude` and `longitude` are in `cleaned.parquet` but never fed to the ML pipeline. Raw lat/long would let trees split on "north of 40°N", which is too noisy a proxy for region. A useful geographic model would bin to state or metropolitan area. Plausible R² gain: 1-2 percentage points of regional cost-of-living variation.

**Best hyperparameters hit the grid edges.** RF picked the top corner (numTrees=100, maxDepth=10) and GBT picked the upper end (maxIter=50). The grids were sized for a tractable CV wall time on caslake. I would like to try numTrees=300+ and maxDepth=15+ if I had more compute budget, but at multi-hour-per-model cost.
