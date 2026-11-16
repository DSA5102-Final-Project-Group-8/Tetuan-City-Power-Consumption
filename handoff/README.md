# Handoff data for teammates

All files are produced by the Task 1 / Task 2 cells of `DSA5102_Final_Project_Group8.ipynb`
(re-running the notebook rewrites them). The final notebook itself only needs `data/`.

| File | Content | Used by |
|---|---|---|
| `daily_load_72d.csv` | 364 days × 72 columns: hourly mean demand `zone_1_h00` … `zone_3_h23` (mean of six 10-minute readings) | Task 2 (PCA/KPCA), Task 3 (K-means, GMM) |
| `daily_metadata.csv` | Daily mean weather, `weekday` (0 = Monday), `split` (development / holdout); join on `date` | Tasks 2–3 |
| `task1_predictions_holdout.csv.gz` | Test-period (2017-10-19 – 2017-12-30) predictions of every Task 1 model: B0, R1, R2, R3 (Ridge), RF, GB, HYB | Section 6 integrated analysis |
| `task1_predictions_oof.csv.gz` | Validation-fold predictions (2017-06-01 – 2017-10-18) of the same models with their selected settings | Section 6 |
| `task1_metrics.csv` | Mean CV and test RMSE / MAE / R² for every model and zone | Report tables |
| `task2_pca_scores.csv` | Date-indexed scores from the PCA scale and dimension selected by development-period validation | Task 3 and Section 6 |
| `task2_kpca_scores.csv` | Date-indexed scores from the selected RBF Kernel PCA configuration | Task 3 sensitivity analysis |
| `task2_reconstruction_metrics.csv` | Validation and holdout reconstruction RMSE for raw PCA, standardized PCA and RBF Kernel PCA | Task 2 results and Section 6 |

Prediction files: key `(timestamp, zone, model)`, a `date` column (YYYY-MM-DD) for joining to
daily clusters, `residual = y_true - y_pred` (observed minus predicted). Primary models:
`R3` (linear) and `RF` (nonlinear). `pd.read_csv` reads the `.gz` files directly.

Protocol: development period 2017-01-01 – 2017-10-18 for all model choices, final test period
2017-10-19 – 2017-12-30. For claims about unseen days, fit scalers / PCA / clustering on
development dates only.
