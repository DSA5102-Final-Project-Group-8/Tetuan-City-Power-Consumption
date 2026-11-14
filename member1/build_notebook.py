"""Build the readable report notebook; execute it with verify_notebook.py."""
from pathlib import Path
import textwrap

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(textwrap.dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(textwrap.dedent(text).strip()))


md('''
    # Task 1 — Exploratory Data Analysis and Ridge Regression
    **DSA5102 Final Project · Group 8 · WangKe (Member 1)**

    How well can calendar information and weather recorded at the same time
    estimate electricity demand in the three Tetouan distribution zones?
    We use Ridge as the linear benchmark for the group's model comparison.

    From the repository root, run `uv sync --frozen`, open this notebook and choose
    the repository `.venv` interpreter. Then **Restart Kernel and Run All**.
    The model, plotting and export functions are in [member1/analysis.py](member1/analysis.py).
''')
code('''
    from pathlib import Path
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd
    from IPython.display import display

    PROJECT_ROOT = Path.cwd()
    if not (PROJECT_ROOT / "data/Tetuan City power consumption.csv").exists():
        raise FileNotFoundError("Open and run this notebook from the repository root.")
    from member1 import analysis as m1

    RESULTS = PROJECT_ROOT / "member1/results"
    FIGURES = PROJECT_ROOT / "member1/figures"
    RESULTS.mkdir(parents=True, exist_ok=True)
    m1.setup_style()
    print("Python:", sys.version.split()[0])

    def display_metrics(frame):
        """Format tiny alphas without an optional table-rendering dependency."""
        pretty = frame.copy()
        for column in pretty.select_dtypes(include="number"):
            if column != "alpha":
                pretty[column] = pretty[column].round(3)
        if "alpha" in pretty:
            pretty["alpha"] = pretty["alpha"].map(lambda a: "—" if pd.isna(a) else f"{a:g}")
        display(pretty)
''')
md('''
    ## 1. Data Preprocessing
    The [UCI Power Consumption of Tetouan City dataset](https://archive.ics.uci.edu/dataset/849/power%2Bconsumption%2Bof%2Btetouan%2Bcity)
    describes three distribution networks in northern Morocco. This CSV has one
    timestamp, five weather/radiation-related inputs and three demand targets.
    The CSV contains **52,416 rows** from 1 January to 30 December 2017.
    We standardize column names, parse the timestamps and check for missing
    values, duplicates and gaps. Demand is reported in the source data units.
''')
code('''
    data, audit = m1.audit_data(PROJECT_ROOT)
    m1.write_json(RESULTS / "data_audit.json", audit)
    display(pd.Series({key: audit[key] for key in
                       ["rows", "columns", "time_start", "time_end", "complete_days", "cleaning"]}))
    display(pd.Series(audit["checks"], name="Passed"))
    display(data.head())
''')
md('''
    All 364 observed dates have 144 unique ten-minute records, with no missing
    values, duplicate timestamps or gaps. The CSV ends on **30 December**.
''')
code('''
    dictionary = pd.DataFrame({
        "field": ["timestamp", *m1.WEATHER, *m1.TARGETS],
        "role": ["calendar input", *(["observed weather input"] * 5), *(["target only"] * 3)],
        "use": ["recorded calendar", "temperature", "humidity", "wind speed",
                "general diffuse flows", "diffuse flows", "network 1 demand",
                "network 2 demand", "network 3 demand"],
    })
    display(dictionary)
''')
md('''
    ## 2. Exploratory Data Analysis
    We use the data through 18 October to examine daily trends, hours of peak
    demand, weekday patterns and associations with weather. The daily trend shows
    mean demand. Shading on the daily profiles spans the 10th to 90th percentiles
    across dates.
''')
code('''
    development, holdout = m1.split_data(data)
    eda_figures = m1.plot_eda(development, FIGURES)
    for name, figure in eda_figures.items():
        display(figure)
        plt.close(figure)
''')
code('''
    fractional_hour = development.index.hour + development.index.minute / 60
    mean_profile = development[m1.TARGETS].groupby(fractional_hour).mean()
    display(pd.DataFrame({"mean_profile_peak_hour": mean_profile.idxmax(),
                          "mean_profile_trough_hour": mean_profile.idxmin(),
                          "temperature_correlation": development[m1.WEATHER + m1.TARGETS].corr()
                              .loc["temperature", m1.TARGETS]}).rename(index=m1.LABELS).round(3))
    display(development[m1.TARGETS].corr().round(3))
''')
md('''
    **Observations and modeling implications.** All three mean daily profiles peak
    around **20:10–20:20**, motivating cyclic daily features. Early-morning troughs
    and weekday patterns support the inclusion of calendar information. Zones 1
    and 2 have a demand correlation of **0.921**.
    Zone 3 shows a pronounced mid-year rise and subsequent decline; a smooth annual
    basis may not explain that change adequately.

    Temperature has positive marginal correlations of about **0.418 / 0.515 / 0.443**
    with the three targets. The hour-coloured scatter shows how temperature and
    time of day vary together, so we compare time-only and time-plus-weather models.
''')
md('''
    ## 3. Feature Engineering
    We compare three sets of inputs to see what weather and interactions add
    once the model already has calendar information.

    | Feature set | Inputs | Features |
    |---|---|---:|
    | R1 | Three daily sine/cosine pairs, weekday indicators and annual sine/cosine | 14 |
    | R2 | R1 + the five weather fields | 19 |
    | R3 | R2 + T², T×humidity, T×daily sin/cos k=1 | 23 |

    Fractional hours preserve ten-minute resolution. The daily sine and cosine
    terms put 23:50 and 00:00 close together. Monday is the weekday reference.
    Annual sine and cosine terms allow a smooth seasonal pattern within 2017.

''')
code('''
    example_x = m1.make_features(development, "R3")
    display(example_x.head(3))
''')
md(r'''
    ## 4. Ridge Regression
    The baseline (B0) predicts each zone's mean demand from the training data.
    We train a separate Ridge model for each zone and feature set. Ridge minimizes

    $$\sum_{i\in\mathrm{train}} (y_i-b-\boldsymbol\phi(x_i)^T w)^2
       +\alpha\lVert w\rVert_2^2.$$

    Here, `alpha` controls the penalty on the feature coefficients. The
    intercept is unpenalized. Ridge is linear in the fitted coefficients,
    including those on temperature squared and the interaction terms.
    The pipeline standardizes continuous features and keeps weekday indicators as 0 or 1.
''')
code('''
    display(m1.ridge_pipeline(example_x.columns))
''')
md('''
    ## 5. Performance Evaluation

    ### Cross-validation and model selection
    We use **1 January–18 October** to select the model and
    **19 October–30 December** for the final test. Each of five expanding
    training windows is followed by 28 days of validation. The scaler is fitted
    on the training data within each fold.

    For each zone and feature set, `GridSearchCV` tests **10⁻⁴, 10⁻³, …, 10⁶**
    and picks the alpha with the lowest mean validation RMSE. We then average
    the three zone RMSEs to choose one feature set, and report zone scores
    individually because the absolute error scales differ.
''')
code('''
    _, fold_table = m1.cv_splits(development)
    display(fold_table)
''')
code('''
    development_result = m1.run_development(development, RESULTS)
    display_metrics(development_result["summary"])
    display(development_result["cv_macro_rmse"].rename("Mean zone CV RMSE").round(3))
    primary = development_result["primary_model"]
    print("CV-selected feature set:", primary)
''')
md('''
    Mean validation RMSE falls from **4,640.59 (R1)** to **4,534.74 (R2)** and
    **4,278.47 (R3)**. R3 gains most in Zone 3, while R2 has the lowest validation
    RMSE in Zone 1. The selected R3 alphas are **10,000 / 0.0001 / 0.0001** for
    Zones 1/2/3. Zones 2 and 3 selected the smallest value in the grid.
''')
md('''
    ### Test results and error analysis
    We fit each configuration on the full development period and evaluate it
    on the year-end test set. RMSE gives more weight to large errors, and MAE
    reports the average absolute error. Both use the original demand scale.
''')
code('''
    final_metrics, holdout_predictions = m1.evaluate_holdout(
        development, holdout, development_result, RESULTS)
    test_metrics = final_metrics.query("split == 'holdout'")
    display_metrics(test_metrics)
    display(test_metrics.groupby("model")[["rmse", "mae", "r2"]].mean().round(3))
''')
code('''
    result_figures = m1.plot_results(development_result, final_metrics, holdout_predictions, FIGURES)
    for name, figure in result_figures.items():
        display(figure)
        plt.close(figure)
''')
code('''
    primary_scores = test_metrics.query("model == @primary").set_index("zone")
    baseline_scores = test_metrics.query("model == 'B0'").set_index("zone")
    improvement = 100 * (1 - primary_scores.rmse / baseline_scores.rmse)
    display(primary_scores[["rmse", "mae", "r2"]]
            .assign(rmse_reduction_vs_B0_percent=improvement).rename(index=m1.LABELS).round(3))
    m1.diagnostic_tables(development, holdout, holdout_predictions, primary, RESULTS)
''')
md('''
    R3 lowers the average test RMSE from **4,124.70** for calendar-only Ridge
    to **3,909.60**, a reduction of **5.2%**. Relative to B0, the RMSE reductions
    are **48.6%, 26.2% and 55.5%** for Zones 1, 2 and 3.
    Zone 3 still has an **R² of −0.136**. B0 uses the development-period mean.
    R² uses the test-period mean as its reference, and that reference has lower
    squared error than R3 in Zone 3.

    On the test set, R2 has the lowest RMSE in Zone 1, R1 in Zone 2 and R3 in
    Zone 3. The hourly residuals show that R3 underestimates Zone 2 demand
    around the evening peak. These differences give the team's nonlinear
    models concrete cases to compare.
''')
md('''
    ### Conclusion
    Calendar, weather and interaction features give the lowest average validation
    RMSE among the Ridge configurations. The year-end results show where the
    linear benchmark struggles, particularly with variation in Zone 3 and the
    evening peak in Zone 2.

    The data cover one city and 364 days. The year-end test measures performance
    on later dates in 2017. A second year would allow a separate test of seasonal
    generalization.
''')
md('''
    ## Appendix

    ### Files for the other tasks
    The prediction files include **residual = observed − predicted** and a `date`
    field for joining to daily clusters. Validation-fold predictions cover
    1 June–18 October and use the hyperparameters selected on those folds.

    Daily load curves use the hourly mean of six ten-minute readings per zone:
    **364 dates × 72 features**, ordered by zone and hour. These curves describe
    completed days for PCA and clustering. The shared dates and feature functions
    are documented in [member1/README.md](member1/README.md).
''')
code('''
    m1.export_handoff(data, development, development_result, RESULTS)
    display(pd.read_csv(RESULTS / "daily_load_72d.csv", nrows=3))
''')
md('''
    ### Contribution and references
    **WangKe (Member 1):** data auditing and preprocessing,
    exploratory visualizations, calendar/weather feature engineering, Ridge baselines,
    chronological cross-validation and performance/error analysis, reproducible handoff
    outputs for the other team tasks.

    **Dataset:** Salam, A. & El Hibaoui, A. (2018). *Power Consumption of Tetouan City*.
    UCI Machine Learning Repository. [DOI: 10.24432/C5B034](https://doi.org/10.24432/C5B034).
    CC BY 4.0. The CSV is included in the group repository.

    **Course:** DSA5102 Lectures 1–3 (linear basis models, kernels, CV/ensembles),
    Lectures 6–8 (generalization, PCA and clustering), and the Final Group Project instructions.

    **Implementation references:**
    [Ridge objective](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
    [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
    [preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html).
    See [member1/README.md](member1/README.md) for run instructions and file descriptions.
''')

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata.kernelspec = {'display_name': 'Python (DSA5102 Group 8)',
                               'language': 'python', 'name': 'python3'}
nbf.validate(notebook)
nbf.write(notebook, ROOT / 'member1_eda_ridge.ipynb')
print(f'Built {len(cells)} cells.')
