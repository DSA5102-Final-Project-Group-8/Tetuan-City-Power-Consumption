"""Readable helpers for member1_eda_ridge.ipynb.

Select Ridge features and regularization with chronological cross-validation.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

SEED = 42
WEATHER = ['temperature', 'humidity', 'wind_speed', 'general_diffuse_flows', 'diffuse_flows']
TARGETS = ['zone_1', 'zone_2', 'zone_3']
LABELS = {'zone_1': 'Zone 1', 'zone_2': 'Zone 2', 'zone_3': 'Zone 3'}
MODELS = ['B0', 'R1', 'R2', 'R3']
COLORS = ['#25617a', '#c06d38', '#718b4b']
ALPHAS = np.logspace(-4, 6, 11)
PROTOCOL = {
    'task': 'Contemporaneous demand estimation given recorded weather and calendar information',
    'development_start': '2017-01-01', 'development_end': '2017-10-18',
    'holdout_start': '2017-10-19', 'holdout_end': '2017-12-30',
    'n_splits': 5, 'validation_days': 28, 'rows_per_day': 144, 'gap_rows': 0,
    'alpha_grid': ALPHAS.tolist(), 'seed': SEED,
    'alpha_selection': 'Minimum mean 5-fold validation RMSE separately for each zone and feature set',
    'feature_set_selection': 'Minimum equally weighted mean of three zone CV RMSEs',
    'residual_definition': 'y_true - y_pred',
    'units': 'Source data units: UCI variable table does not specify measurement units',
}


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def setup_style():
    sns.set_theme(style='whitegrid', context='notebook', palette=COLORS)
    plt.rcParams.update({'figure.dpi': 110, 'savefig.dpi': 160, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})


def audit_data(root):
    """Read the CSV and check the data needed for time-based analysis."""
    path = Path(root) / 'data' / 'Tetuan City power consumption.csv'
    raw = pd.read_csv(path)
    timestamp = pd.to_datetime(raw['DateTime'], format='%m/%d/%Y %H:%M', errors='raise')
    rename = {c: '_'.join(c.lower().split()) for c in raw.columns}
    target_names = {f'{zone}_power_consumption': zone for zone in TARGETS}
    rename = {original: target_names.get(name, name) for original, name in rename.items()}
    data = raw.rename(columns=rename).drop(columns='datetime')
    data.index = pd.DatetimeIndex(timestamp, name='timestamp')
    numeric = data[WEATHER + TARGETS]
    per_day = data.groupby(data.index.normalize()).size()
    expected = pd.date_range(data.index.min(), data.index.max(), freq='10min')
    full_days = pd.date_range('2017-01-01', '2017-12-30', freq='D')
    checks = {
        'no_missing_values': not raw.isna().any().any(),
        'all_numeric_values_finite': bool(np.isfinite(numeric.to_numpy()).all()),
        'timestamps_sorted_unique': bool(data.index.is_monotonic_increasing and data.index.is_unique),
        'regular_10min_grid': data.index.equals(expected),
        '364_complete_days': bool(per_day.index.equals(full_days) and per_day.eq(144).all()),
        'no_duplicate_rows': not raw.duplicated().any(),
    }
    failed = [check for check, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f'Data checks failed: {", ".join(failed)}')
    audit = {'rows': len(raw), 'columns': len(raw.columns),
             'time_start': str(data.index.min()), 'time_end': str(data.index.max()),
             'complete_days': len(per_day), 'checks': {k: bool(v) for k, v in checks.items()},
             'missing_by_column': {c: int(n) for c, n in raw.isna().sum().items()},
             'column_mapping': rename, 'cleaning': 'Rename columns only; no imputation, deletion or clipping',
             'coverage_note': 'Complete within observed Jan 1-Dec 30 range; Dec 31 is not present',
             'units': PROTOCOL['units'],
             'uci_metadata_note': 'Web metadata reports 52417 instances; actual source CSV contains 52416 rows',
             'timezone': 'Not provided; calendar is preserved as recorded'}
    return data, audit


def split_data(data):
    development = data.loc[PROTOCOL['development_start']:PROTOCOL['development_end']].copy()
    holdout = data.loc[PROTOCOL['holdout_start']:PROTOCOL['holdout_end']].copy()
    return development, holdout


def make_features(data, feature_set='R3'):
    """Build calendar and weather features for one Ridge configuration."""
    if feature_set not in ['R1', 'R2', 'R3']:
        raise ValueError(feature_set)
    t = data.index
    hour = t.hour.to_numpy() + t.minute.to_numpy() / 60
    x = pd.DataFrame(index=t)
    for k in (1, 2, 3):
        x[f'daily_sin_{k}'] = np.sin(2 * np.pi * k * hour / 24)
        x[f'daily_cos_{k}'] = np.cos(2 * np.pi * k * hour / 24)
    for weekday in range(1, 7):
        x[f'weekday_{weekday}'] = (t.dayofweek == weekday).astype(float)
    year_position = t.dayofyear.to_numpy() - 1 + hour / 24
    x['annual_sin'] = np.sin(2 * np.pi * year_position / 365)
    x['annual_cos'] = np.cos(2 * np.pi * year_position / 365)
    if feature_set in ['R2', 'R3']:
        x[WEATHER] = data[WEATHER]
    if feature_set == 'R3':
        x['temperature_squared'] = x['temperature'] ** 2
        x['temperature_x_humidity'] = x['temperature'] * x['humidity']
        x['temperature_x_daily_sin'] = x['temperature'] * x['daily_sin_1']
        x['temperature_x_daily_cos'] = x['temperature'] * x['daily_cos_1']
    return x


def ridge_pipeline(columns, alpha=1.):
    weekdays = [c for c in columns if c.startswith('weekday_')]
    continuous = [c for c in columns if c not in weekdays]
    preprocess = ColumnTransformer([
        ('continuous', StandardScaler(), continuous),
        ('weekday', 'passthrough', weekdays),
    ], verbose_feature_names_out=False)
    return Pipeline([('preprocess', preprocess),
                     ('ridge', Ridge(alpha=alpha, fit_intercept=True, solver='cholesky'))])


def cv_splits(development):
    rows_per_day = PROTOCOL['rows_per_day']
    validation_days = PROTOCOL['validation_days']
    splitter = TimeSeriesSplit(n_splits=PROTOCOL['n_splits'],
                               test_size=validation_days * rows_per_day, gap=PROTOCOL['gap_rows'])
    splits, rows = list(splitter.split(development)), []
    for fold, (tr, va) in enumerate(splits, 1):
        a, b = development.index[tr], development.index[va]
        rows.append(dict(fold=fold, train_start=str(a[0]), train_end=str(a[-1]), train_rows=len(a),
                         train_days=len(a) // rows_per_day, validation_start=str(b[0]),
                         validation_end=str(b[-1]), validation_rows=len(b), validation_days=validation_days))
    return splits, pd.DataFrame(rows)


def metric_values(y, prediction):
    return {'rmse': float(np.sqrt(mean_squared_error(y, prediction))),
            'mae': float(mean_absolute_error(y, prediction)), 'r2': float(r2_score(y, prediction))}


def prediction_frame(index, y, prediction, zone, model, alpha, split, fold=None):
    frame = pd.DataFrame({'timestamp': index, 'date': index.strftime('%Y-%m-%d'),
                          'zone': zone, 'model': model, 'alpha': alpha, 'split': split,
                          'fold': fold, 'y_true': np.asarray(y), 'y_pred': prediction})
    frame['residual'] = frame['y_true'] - frame['y_pred']
    return frame


def run_development(development, result_dir):
    """Select Ridge settings from the development period."""
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    splits, fold_table = cv_splits(development)
    fold_table.to_csv(result_dir / 'cv_folds.csv', index=False)
    write_json(result_dir / 'protocol.json', PROTOCOL)
    search_rows, best, score_rows, predictions, fitted = [], {}, [], [], {}
    with threadpool_limits(limits=1):
        for feature_set in ['R1', 'R2', 'R3']:
            x = make_features(development, feature_set)
            best[feature_set] = {}
            for zone in TARGETS:
                grid = GridSearchCV(
                    ridge_pipeline(x.columns), {'ridge__alpha': ALPHAS}, cv=splits,
                    scoring={'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error',
                             'r2': 'r2'}, refit='rmse', n_jobs=1, return_train_score=False,
                    error_score='raise')
                grid.fit(x, development[zone])
                alpha = float(grid.best_params_['ridge__alpha'])
                best[feature_set][zone] = alpha
                fitted[(feature_set, zone)] = grid.best_estimator_
                for candidate, params in enumerate(grid.cv_results_['params']):
                    for fold in range(len(splits)):
                        search_rows.append(dict(
                            model=feature_set, zone=zone, alpha=float(params['ridge__alpha']), fold=fold+1,
                            rmse=float(-grid.cv_results_[f'split{fold}_test_rmse'][candidate]),
                            mae=float(-grid.cv_results_[f'split{fold}_test_mae'][candidate]),
                            r2=float(grid.cv_results_[f'split{fold}_test_r2'][candidate])))
                for fold, (tr, va) in enumerate(splits, 1):
                    pipe = ridge_pipeline(x.columns, alpha).fit(x.iloc[tr], development[zone].iloc[tr])
                    pred = pipe.predict(x.iloc[va])
                    score_rows.append(dict(model=feature_set, zone=zone, alpha=alpha, fold=fold,
                                           **metric_values(development[zone].iloc[va], pred)))
                    predictions.append(prediction_frame(x.index[va], development[zone].iloc[va], pred,
                                                        zone, feature_set, alpha, 'oof', fold))
                print(f'{feature_set} / {zone}: alpha={alpha:g}, CV RMSE={-grid.best_score_:.2f}')
        for zone in TARGETS:
            fitted[('B0', zone)] = DummyRegressor(strategy='mean').fit(
                np.zeros((len(development), 1)), development[zone])
            for fold, (tr, va) in enumerate(splits, 1):
                pred = np.full(len(va), development[zone].iloc[tr].mean())
                score_rows.append(dict(model='B0', zone=zone, alpha=np.nan, fold=fold,
                                       **metric_values(development[zone].iloc[va], pred)))
                predictions.append(prediction_frame(development.index[va], development[zone].iloc[va],
                                                     pred, zone, 'B0', np.nan, 'oof', fold))
    scores = pd.DataFrame(score_rows)
    summary = scores.groupby(['model', 'zone'], sort=True).agg(
        alpha=('alpha', 'first'), mean_rmse=('rmse', 'mean'), std_rmse=('rmse', 'std'),
        mean_mae=('mae', 'mean'), mean_r2=('r2', 'mean')).reset_index()
    macro = summary.groupby('model')['mean_rmse'].mean()
    selected = macro.loc[['R1', 'R2', 'R3']].idxmin()
    search = pd.DataFrame(search_rows)
    search.to_csv(result_dir / 'cv_search.csv', index=False)
    scores.to_csv(result_dir / 'cv_selected_fold_metrics.csv', index=False)
    summary.to_csv(result_dir / 'cv_summary.csv', index=False)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(result_dir / 'predictions_oof.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    return {'fitted': fitted, 'primary_model': selected, 'best_alphas': best,
            'cv_macro_rmse': macro, 'summary': summary, 'search': search, 'folds': fold_table}


def evaluate_holdout(development, holdout, dev_result, result_dir):
    """Evaluate the models fitted on the development period."""
    result_dir = Path(result_dir)
    metrics, predictions, coefficients = [], [], []
    for model in MODELS:
        x_test = np.zeros((len(holdout), 1)) if model == 'B0' else make_features(holdout, model)
        x_dev = np.zeros((len(development), 1)) if model == 'B0' else make_features(development, model)
        for zone in TARGETS:
            fitted = dev_result['fitted'][(model, zone)]
            alpha = np.nan if model == 'B0' else dev_result['best_alphas'][model][zone]
            prediction = fitted.predict(x_test)
            metrics.append(dict(model=model, zone=zone, alpha=alpha, split='holdout',
                                **metric_values(holdout[zone], prediction)))
            metrics.append(dict(model=model, zone=zone, alpha=alpha, split='development_fit',
                                **metric_values(development[zone], fitted.predict(x_dev))))
            predictions.append(prediction_frame(holdout.index, holdout[zone], prediction,
                                                 zone, model, alpha, 'holdout'))
            if model == dev_result['primary_model']:
                names = fitted.named_steps['preprocess'].get_feature_names_out()
                for name, coef in zip(names, fitted.named_steps['ridge'].coef_):
                    coefficients.append(dict(zone=zone, feature=name, coefficient=float(coef),
                        interpretation='Scaled continuous inputs and 0/1 weekday indicators'))
    all_metrics = pd.DataFrame(metrics)
    holdout_predictions = pd.concat(predictions, ignore_index=True)
    all_metrics.to_csv(result_dir / 'final_metrics.csv', index=False)
    holdout_predictions.to_csv(result_dir / 'predictions_holdout.csv.gz', index=False,
                               compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(coefficients).to_csv(result_dir / 'primary_coefficients.csv', index=False)
    return all_metrics, holdout_predictions


def export_handoff(data, development, dev_result, result_dir):
    result_dir = Path(result_dir)
    folds, assigned = dev_result['folds'], pd.Series(pd.NA, index=data.index, dtype='Int64')
    for row in folds.itertuples():
        assigned.loc[row.validation_start:row.validation_end] = row.fold
    assignment = pd.DataFrame({'timestamp': data.index, 'date': data.index.strftime('%Y-%m-%d'),
                               'split': np.where(data.index <= development.index[-1], 'development', 'holdout'),
                               'cv_validation_fold': assigned.to_numpy()})
    assignment.to_csv(result_dir / 'split_assignments.csv', index=False)
    hourly = data[TARGETS].resample('h').mean()
    # The audit guarantees six complete 10-minute records per hour.
    daily = pd.DataFrame(index=pd.date_range(data.index[0].normalize(), data.index[-1].normalize(), freq='D'))
    daily.index.name = 'date'
    for zone in TARGETS:
        for hour in range(24):
            series = hourly.loc[hourly.index.hour == hour, zone].copy()
            series.index = series.index.normalize()
            daily[f'{zone}_h{hour:02d}'] = series
    if daily.shape != (364, 72) or daily.isna().any().any():
        raise ValueError('Daily curve matrix is incomplete.')
    daily.to_csv(result_dir / 'daily_load_72d.csv')
    metadata = data[WEATHER].resample('D').mean()
    metadata['weekday'] = metadata.index.dayofweek
    metadata['split'] = np.where(metadata.index <= development.index[-1].normalize(), 'development', 'holdout')
    metadata.index.name = 'date'
    metadata.to_csv(result_dir / 'daily_metadata.csv')
    for feature_set in ['R1', 'R2', 'R3']:
        features = make_features(development.iloc[:1], feature_set)
        pd.DataFrame({'feature': features.columns,
                      'scaling': ['passthrough 0/1' if c.startswith('weekday_') else 'train-fold StandardScaler'
                                  for c in features.columns]}).to_csv(result_dir / f'features_{feature_set}.csv', index=False)


def save_figure(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(directory / f'{name}.png', bbox_inches='tight')
    return fig


def plot_eda(development, figure_dir):
    """All charts receive development data only."""
    setup_style()
    figures = {}
    daily = development[TARGETS].resample('D').mean()
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, zone, color in zip(axes, TARGETS, COLORS):
        ax.plot(daily.index, daily[zone], color=color, lw=1.1)
        ax.set_ylabel(LABELS[zone])
    axes[0].set_title('Development period: daily mean demand (source data units)')
    figures['01_daily_trend'] = save_figure(fig, figure_dir, '01_daily_trend')

    profiles = development[TARGETS].copy()
    profiles['hour'] = development.index.hour + development.index.minute / 60
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for ax, zone, color in zip(axes, TARGETS, COLORS):
        grouped = profiles.groupby('hour')[zone]
        mean, low, high = grouped.mean(), grouped.quantile(.1), grouped.quantile(.9)
        ax.fill_between(mean.index, low, high, color=color, alpha=.16, label='10th-90th percentile')
        ax.plot(mean.index, mean, color=color, label='Mean')
        ax.set(title=LABELS[zone], xlabel='Hour of day', ylabel='Demand (source units)', xticks=[0, 6, 12, 18, 24])
    axes[0].legend(fontsize=8)
    fig.suptitle('Development period: within-day demand profiles', y=1.03)
    figures['02_daily_profile'] = save_figure(fig, figure_dir, '02_daily_profile')

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    work = development[TARGETS].assign(weekday=development.index.dayofweek, hour=development.index.hour)
    for ax, zone in zip(axes, TARGETS):
        table = work.groupby(['weekday', 'hour'])[zone].mean().unstack()
        sns.heatmap(table, ax=ax, cmap='YlGnBu', yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    xticklabels=4, cbar_kws={'label': 'Source units'})
        ax.tick_params(axis='y', labelrotation=0)
        ax.set(title=LABELS[zone], xlabel='Hour', ylabel='Weekday')
    fig.suptitle('Development period: weekday-by-hour mean demand', y=1.03)
    figures['03_weekday_hour'] = save_figure(fig, figure_dir, '03_weekday_hour')

    sample = development.sample(n=min(5000, len(development)), random_state=SEED)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    scatter = None
    for ax, zone in zip(axes, TARGETS):
        scatter = ax.scatter(sample.temperature, sample[zone], c=sample.index.hour,
                             cmap='viridis', vmin=0, vmax=23, alpha=.3, s=6, rasterized=True)
        ax.set(title=LABELS[zone], xlabel='Temperature (source units)', ylabel='Demand (source units)')
    fig.suptitle('Development sample: temperature association, coloured by recorded hour', y=1.03)
    # An explicit colourbar axis avoids squeezing/overlapping the three panels.
    fig.subplots_adjust(right=.9, wspace=.35)
    cax = fig.add_axes([.92, .2, .015, .6])
    fig.colorbar(scatter, cax=cax, label='Hour of day')
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(Path(figure_dir) / '04_temperature_relationship.png', bbox_inches='tight')
    figures['04_temperature_relationship'] = fig

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(development[WEATHER + TARGETS].corr(), vmin=-1, vmax=1, center=0,
                cmap='vlag', annot=True, fmt='.2f', ax=ax, annot_kws={'size': 8})
    ax.set_title('Development period: Pearson correlations (associations only)')
    figures['05_correlations'] = save_figure(fig, figure_dir, '05_correlations')
    return figures


def plot_results(dev_result, final_metrics, holdout_predictions, figure_dir):
    setup_style()
    figures, primary = {}, dev_result['primary_model']
    search = dev_result['search']
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for ax, zone in zip(axes, TARGETS):
        for model, color in zip(['R1', 'R2', 'R3'], COLORS):
            means = search[(search.zone == zone) & (search.model == model)].groupby('alpha').rmse.mean()
            ax.semilogx(means.index, means, '-o', ms=3, label=model, color=color)
        ax.set(title=LABELS[zone], xlabel='Ridge alpha', ylabel='Mean CV RMSE (source units)')
    axes[0].legend()
    figures['06_alpha_search'] = save_figure(fig, figure_dir, '06_alpha_search')

    cv = dev_result['summary'].pivot(index='zone', columns='model', values='mean_rmse').reindex(TARGETS)[MODELS]
    final = final_metrics.query("split == 'holdout'").pivot(index='zone', columns='model', values='rmse').reindex(TARGETS)[MODELS]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    cv.rename(index=LABELS).plot.bar(ax=axes[0], rot=0, color=['#adb5bd', *COLORS])
    final.rename(index=LABELS).plot.bar(ax=axes[1], rot=0, color=['#adb5bd', *COLORS])
    for ax, title in zip(axes, ['Development cross-validation', 'Year-end test']):
        ax.set(title=title, xlabel='', ylabel='RMSE (source units)')
        ax.legend(fontsize=8)
    figures['07_model_comparison'] = save_figure(fig, figure_dir, '07_model_comparison')

    chosen = holdout_predictions.query('model == @primary').copy()
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, zone, color in zip(axes, TARGETS, COLORS):
        z = chosen[chosen.zone == zone]
        ax.hexbin(z.y_true, z.y_pred, gridsize=45, mincnt=1, cmap='Blues')
        low, high = min(z.y_true.min(), z.y_pred.min()), max(z.y_true.max(), z.y_pred.max())
        ax.plot([low, high], [low, high], '--', c=color, lw=1)
        ax.set(title=LABELS[zone], xlabel='Observed demand', ylabel='Predicted demand')
        ax.set_aspect('equal', adjustable='box')
    fig.suptitle(f'Final holdout: CV-selected {primary} (source units)', y=1.02)
    figures['08_observed_predicted'] = save_figure(fig, figure_dir, '08_observed_predicted')

    chosen['timestamp'] = pd.to_datetime(chosen.timestamp)
    chosen['hour'] = chosen.timestamp.dt.hour
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for zone, color in zip(TARGETS, COLORS):
        z = chosen[chosen.zone == zone]
        hourly = z.groupby('hour').agg(mae=('residual', lambda a: a.abs().mean()), bias=('residual', 'mean'))
        axes[0].plot(hourly.index, hourly.mae, label=LABELS[zone], color=color)
        axes[1].plot(hourly.index, hourly.bias, label=LABELS[zone], color=color)
    axes[0].set(title='Absolute error by hour', xlabel='Recorded hour', ylabel='MAE (source units)')
    axes[1].set(title='Signed error by hour', xlabel='Recorded hour', ylabel='Mean observed - predicted')
    axes[1].axhline(0, color='#666666', ls='--', lw=1)
    axes[0].legend()
    figures['09_hourly_errors'] = save_figure(fig, figure_dir, '09_hourly_errors')

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    start = chosen.timestamp.min()
    week = chosen[(chosen.timestamp >= start) & (chosen.timestamp < start + pd.Timedelta(days=7))]
    for ax, zone, color in zip(axes, TARGETS, COLORS):
        z = week[week.zone == zone]
        ax.plot(z.timestamp, z.y_true, color='#374151', lw=1, label='Observed')
        ax.plot(z.timestamp, z.y_pred, color=color, lw=1, alpha=.85, label=primary)
        ax.set_ylabel(LABELS[zone])
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].set_title('First seven test days (source units)')
    figures['10_first_holdout_week'] = save_figure(fig, figure_dir, '10_first_holdout_week')
    return figures


def diagnostic_tables(development, holdout, holdout_predictions, primary, result_dir):
    """Summarize test errors by hour and weather conditions."""
    selected = holdout_predictions[holdout_predictions.model == primary].copy()
    selected['timestamp'] = pd.to_datetime(selected.timestamp)
    selected['hour'] = selected.timestamp.dt.hour
    hourly = selected.groupby(['zone', 'hour']).agg(
        n=('residual', 'size'), mae=('residual', lambda r: r.abs().mean()), bias=('residual', 'mean')).reset_index()
    hourly.to_csv(Path(result_dir) / 'holdout_hourly_errors.csv', index=False)
    ranges = development[WEATHER].agg(['min', 'max']).T
    ranges.index.name = 'feature'
    ranges.to_csv(Path(result_dir) / 'development_weather_ranges.csv')
    range_rows = []
    for feature in WEATHER:
        lower, upper = ranges.loc[feature, ['min', 'max']]
        outside = ((holdout[feature] < lower) | (holdout[feature] > upper)).mean()
        range_rows.append(dict(feature=feature, development_min=lower, development_max=upper,
                               holdout_outside_fraction=float(outside)))
    pd.DataFrame(range_rows).to_csv(Path(result_dir) / 'weather_range_diagnostics.csv', index=False)
    # Define temperature bands from the development period.
    lower, upper = development.temperature.quantile([1/3, 2/3])
    selected = selected.merge(holdout[['temperature']], left_on='timestamp', right_index=True, validate='many_to_one')
    selected['temperature_band'] = pd.cut(selected.temperature, [-np.inf, lower, upper, np.inf],
                                          labels=['lower', 'middle', 'upper'])
    bands = selected.groupby(['zone', 'temperature_band'], observed=True).agg(
        n=('residual', 'size'), mae=('residual', lambda r: r.abs().mean()),
        bias=('residual', 'mean')).reset_index()
    bands.to_csv(Path(result_dir) / 'holdout_temperature_band_errors.csv', index=False)
    return hourly


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    results = root / 'member1' / 'results'
    data, audit = audit_data(root)
    results.mkdir(parents=True, exist_ok=True)
    write_json(results / 'data_audit.json', audit)
    development, holdout = split_data(data)
    dev_result = run_development(development, results)
    metrics, predictions = evaluate_holdout(development, holdout, dev_result, results)
    export_handoff(data, development, dev_result, results)
    diagnostic_tables(development, holdout, predictions, dev_result['primary_model'], results)
    plot_eda(development, root / 'member1' / 'figures')
    plot_results(dev_result, metrics, predictions, root / 'member1' / 'figures')
    print('Primary feature set:', dev_result['primary_model'])
    print(metrics.query("split == 'holdout'").to_string(index=False))
