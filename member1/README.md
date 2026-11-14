# Member1 — EDA and Ridge contribution

Start with **[the executed notebook](../member1_eda_ridge.ipynb)** for the report,
code, figures and results.

## Reproduce

From the repository root, using the team's Python 3.11.1:

```bash
uv sync --frozen
```

Open `member1_eda_ridge.ipynb`, choose the repository `.venv` interpreter and use
**Restart Kernel and Run All**. Run it from the repository root so it can read the
existing `data/` CSV and import the analysis code in `member1/`.

For an automated clean-kernel check, without registering a user kernel:

```bash
uv run python member1/verify_notebook.py
```

For the analysis and exported figures without the notebook UI:

```bash
MPLBACKEND=Agg uv run python -m member1.analysis
```

`verify_notebook.py` uses the `nbclient` and `nbformat` development dependencies.
`build_notebook.py` regenerates the notebook source. Run `verify_notebook.py` after
rebuilding to populate its outputs.

## Performance evaluation

- Question: contemporaneous demand estimation given recorded weather and calendar inputs.
- Source: 52,416 ten-minute records, 364 complete days, 2017-01-01 through 2017-12-30.
- Development: 2017-01-01–2017-10-18; final holdout: 2017-10-19–2017-12-30.
- Five expanding-window folds with 28 days of validation per fold.
- Standardization fitted on each fold's training subset.
- Candidate models: training-mean B0; calendar R1; calendar/weather R2; expanded-basis R3.
- Each zone/feature-set combination selects alpha from 11 values via CV.
- The common primary feature set minimizes mean three-zone CV RMSE: **R3**.

| R3 | Alpha | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|
| Zone 1 | 10,000 | 3,671.03 | 3,231.86 | 0.645 |
| Zone 2 | 0.0001 | 4,545.98 | 3,846.11 | 0.313 |
| Zone 3 | 0.0001 | 3,511.80 | 2,884.23 | −0.136 |

RMSE/MAE use source data units, which the UCI variable table does not specify.
Macro CV RMSE is 4,640.59 / 4,534.74 / 4,278.47 for R1/R2/R3.
R3's macro holdout RMSE is 3,909.60, versus 7,067.47 for B0.

On the test set, the lowest RMSE comes from R2 in Zone 1, R1 in Zone 2 and R3 in
Zone 3. In Zone 3, R3 improves on the training-mean benchmark. The test-period
mean still has lower squared error, which gives R3 a negative R².

## Files

| File | Purpose |
|---|---|
| `analysis.py` | Data checks, feature engineering, chronological CV, evaluation, plots and exports |
| `verify_notebook.py` | Clean-kernel sequential execution and error checking |
| `build_notebook.py` | Authoring source for the English notebook |
| `figures/01_...`–`10_...` | Five EDA figures and five performance evaluation figures |
| `results/data_audit.json` | Data quality checks and column map |
| `results/protocol.json`, `cv_folds.csv` | Reusable experiment settings and exact fold boundaries |
| `results/cv_search.csv` | All 495 candidate-zone-fold metric records |
| `results/cv_selected_fold_metrics.csv`, `cv_summary.csv` | CV results at the selected alpha for each candidate/zone |
| `results/final_metrics.csv` | Test and development-fit metrics for each model |
| `results/predictions_oof.csv.gz` | 241,920 rows: four models × three zones × 20,160 validation timestamps |
| `results/predictions_holdout.csv.gz` | 126,144 rows: four models × three zones × 10,512 holdout timestamps |
| `results/split_assignments.csv` | All timestamps with development/holdout and optional CV validation fold |
| `results/daily_load_72d.csv` | 364 raw daily curves, three zones × 24 hourly means |
| `results/daily_metadata.csv` | Daily mean weather, weekday and development/holdout membership |
| `results/features_R*.csv`, `primary_coefficients.csv` | Feature/scaling definitions and conditional coefficient values |
| `results/holdout_*_errors.csv`, `weather_range_diagnostics.csv` | Test errors by hour and temperature band, plus weather ranges |
| `results/notebook_validation.json` | Record of the latest clean-kernel run |

Compressed CSV files can be read directly with `pd.read_csv(path)`.

## Interfaces for other members

### Member2: comparable nonlinear regression

```python
from pathlib import Path
from member1.analysis import audit_data, split_data, make_features, cv_splits

data, _ = audit_data(Path.cwd())
development, holdout = split_data(data)
X = make_features(development, 'R3')
splits, fold_table = cv_splits(development)
```

Use these `splits` and calendar/weather information for RF/GB selection. Fit any learned
preprocessing in the training fold. For common-input comparison use the same feature
set. Select settings on the development dates and report test scores on the common
year-end dates.

### Members3/4: daily representations

`daily_load_72d.csv` uses zone-major column order: `zone_1_h00`…`zone_3_h23`.
Each hourly value is the mean of six ten-minute demand readings.
Join `daily_metadata.csv` on `date`. For unseen-date claims, select/fix the scaler,
PCA/KPCA and clustering models using development dates only. Raw hourly means retain
absolute load level; shape normalization is a distinct modeling choice.

### Member5: error-by-regime analysis

Prediction key: `(timestamp, zone, model)`. Day-level joins use `date` (YYYY-MM-DD).
Residual is **observed minus predicted**. Filter `model == 'R3'` for the primary model.
OOF predictions cover the validation dates from 2017-06-01 to 2017-10-18 and use
the hyperparameters selected on those folds. Test predictions cover 2017-10-19
to 2017-12-30.

Use clusters of completed days to compare prediction errors by load pattern.

## Report integration

Integrate data preprocessing and EDA into Section 2, feature engineering and
Ridge into Section 3.1, and performance evaluation into Section 3.3.

Dataset credit: Salam & El Hibaoui (2018), *Power Consumption of Tetouan City*, UCI,
[DOI 10.24432/C5B034](https://doi.org/10.24432/C5B034), CC BY 4.0.
