import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import FeatureAgglomeration
from sklearn.base import clone
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import xgboost as xgb

# ===============================================================
# 1. Load and preprocess data
# ===============================================================
df = pd.read_csv('ev_sales_by_make_year_comprehensive.csv')

# Filter for Model Years 2023-2026
df = df[(df['Model Year'] >= 2023) & (df['Model Year'] <= 2026)].reset_index(drop=True)

print(f"Dataset shape after filtering for years 2023-2026: {df.shape}")
print(f"Target (sales) distribution:\n{df['sales'].describe()}")

# Encode categorical variables (one encoder per column)
categorical_cols = ['Make', 'Model_most_common', 'Electric Vehicle Type_most_common',
                    'Clean Alternative Fuel Vehicle (CAFV) Eligibility_most_common']

for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# Separate features, target, and grouping variable
X = df.drop('sales', axis=1)
y = df['sales']
groups = df['Model Year'].values
years = df['Model Year'].values

print(X.shape, y.shape)


# ===============================================================
# 2. VIF analysis for multicollinearity (predictors and target)
# ===============================================================
def compute_vif(X, y=None, label="Feature set"):
    """
    Variance Inflation Factor for each predictor.
    If y is provided, the target is included in the design matrix so that
    predictor-target collinearity is reflected in the VIF values.
    """
    X_num = X.select_dtypes(include=[np.number]).copy()

    constant_cols = X_num.columns[X_num.nunique() <= 1].tolist()
    if constant_cols:
        print(f"[{label}] Dropping constant columns for VIF: {constant_cols}")
        X_num = X_num.drop(columns=constant_cols)

    design = X_num.copy()
    if y is not None:
        design['sales_target'] = np.asarray(y, dtype=float)

    design = design.replace([np.inf, -np.inf], np.nan).dropna()
    design_const = sm.add_constant(design, has_constant='add')

    vif_rows = []
    for i, col in enumerate(design_const.columns):
        if col == 'const':
            continue
        vif_rows.append({'Feature': col,
                         'VIF': variance_inflation_factor(design_const.values, i)})

    vif_df = pd.DataFrame(vif_rows).sort_values('VIF', ascending=False).reset_index(drop=True)
    vif_df.insert(0, 'Scope', label)

    print(f"\nVIF table [{label}]:")
    print(vif_df.to_string(index=False))

    high = vif_df[vif_df['VIF'] > 10]
    if not high.empty:
        print(f"Features with VIF > 10 ({label}): {high['Feature'].tolist()}")
    else:
        print(f"No features with VIF > 10 ({label}).")

    return vif_df


vif_results = [compute_vif(X, y, label="Full feature set (with target)")]


# ===============================================================
# 3. Feature selection functions
# ===============================================================
def rf_feature_importance(X, y, n_features=5):
    """Random Forest feature importance"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    indices = np.argsort(model.feature_importances_)[::-1]
    return X.columns[indices[:n_features]].tolist(), X.columns[indices[0]]


def xgb_feature_importance(X, y, n_features=5):
    """XGBoost feature importance"""
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    indices = np.argsort(model.feature_importances_)[::-1]
    return X.columns[indices[:n_features]].tolist(), X.columns[indices[0]]


def feature_agglomeration_selection(X, y, n_features=5):
    """Feature Agglomeration: select top features by variance across clusters"""
    n_clusters = min(5, X.shape[1])
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X)

    feature_variances = [(X.columns[i], X[X.columns[i]].var(), fa.labels_[i])
                         for i in range(len(fa.labels_))]
    feature_variances.sort(key=lambda x: x[1], reverse=True)
    selected_features = [f for f, _, _ in feature_variances[:n_features]]
    return selected_features, selected_features[0] if selected_features else None


def hvgs_selection(X, y, n_features=5):
    """Highly Variable Gene Selection (variance-based)"""
    variance = X.var().sort_values(ascending=False)
    selected_features = variance.index[:n_features].tolist()
    return selected_features, selected_features[0] if selected_features else None


def spearman_selection(X, y, n_features=5):
    """Spearman's correlation with the target"""
    correlation = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlation.append((col, abs(corr)))
    correlation.sort(key=lambda x: x[1], reverse=True)
    selected_features = [item[0] for item in correlation[:n_features]]
    return selected_features, selected_features[0] if selected_features else None


methods = {
    'RF': rf_feature_importance,
    'XGB': xgb_feature_importance,
    'FA': feature_agglomeration_selection,
    'HVGS': hvgs_selection,
    'Spearman': spearman_selection
}


# ===============================================================
# 4. Confidence interval and validation utilities
# ===============================================================
def bootstrap_ci(scores, n_boot=10000, confidence=0.95, seed=42):
    """Percentile bootstrap CI for the mean of fold-level R² scores"""
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    boot_means = np.array([
        rng.choice(scores, size=len(scores), replace=True).mean()
        for _ in range(n_boot)
    ])
    alpha = (1 - confidence) / 2
    lower, upper = np.percentile(boot_means, [100 * alpha, 100 * (1 - alpha)])
    return scores.mean(), lower, upper


def grouped_cv_scores(model, X_sub, y, groups):
    """Year-grouped CV: each fold holds out whole Model Year(s)"""
    n_groups = len(np.unique(groups))
    gkf = GroupKFold(n_splits=min(5, n_groups))
    fold_scores, fold_years = [], []
    for train_idx, test_idx in gkf.split(X_sub, y, groups):
        m = clone(model)
        m.fit(X_sub.iloc[train_idx], y.iloc[train_idx])
        pred = m.predict(X_sub.iloc[test_idx])
        fold_scores.append(r2_score(y.iloc[test_idx], pred))
        fold_years.append(sorted(np.unique(groups[test_idx]).tolist()))
    return np.array(fold_scores), fold_years


def timeseries_cv_scores(model, X_sub, y, years):
    """Forward-chaining validation: train on earlier years, test on the next year"""
    unique_years = sorted(np.unique(years))
    scores, test_years = [], []
    for i in range(1, len(unique_years)):
        train_mask = years <= unique_years[i - 1]
        test_mask = years == unique_years[i]
        if train_mask.sum() < 2 or test_mask.sum() < 2:
            continue
        m = clone(model)
        m.fit(X_sub[train_mask], y[train_mask])
        pred = m.predict(X_sub[test_mask])
        scores.append(r2_score(y[test_mask], pred))
        test_years.append(unique_years[i])
    return np.array(scores), test_years


# ===============================================================
# 5. Main loop: feature selection, VIF, validation, reporting
# ===============================================================
results = []
fold_detail = []

for method_name, method_func in methods.items():
    print(f"\n{'=' * 60}\nProcessing {method_name}...\n{'=' * 60}")

    try:
        # Top-5 features from the full dataset
        top5_features, top_feature = method_func(X, y, 5)
        print(f"Top 5 features: {top5_features}")

        # VIF on the selected top-5 predictors (with target)
        vif_top5 = compute_vif(X[top5_features], y, label=f"{method_name} Top-5 (with target)")
        vif_results.append(vif_top5)

        X_top5 = X[top5_features]
        model = (RandomForestRegressor(n_estimators=100, random_state=42)
                 if method_name == 'RF'
                 else xgb.XGBRegressor(n_estimators=100, random_state=42))

        # (a) Year-grouped CV
        g_scores, g_years = grouped_cv_scores(model, X_top5, y, groups)
        g_mean, g_lo, g_hi = bootstrap_ci(g_scores)
        g_sd = g_scores.std(ddof=1) if len(g_scores) > 1 else np.nan
        print(f"Grouped CV R²: {g_mean:.4f} (bootstrap 95% CI: [{g_lo:.4f}, {g_hi:.4f}])")
        print(f"Fold-level R²: {np.round(g_scores, 4).tolist()} | SD = {g_sd:.4f}")

        # (b) Forward-chaining time-series CV
        t_scores, t_years = timeseries_cv_scores(model, X_top5, y, years)
        if len(t_scores) > 0:
            t_mean, t_lo, t_hi = bootstrap_ci(t_scores)
            t_sd = t_scores.std(ddof=1) if len(t_scores) > 1 else np.nan
            print(f"Time-series CV R²: {t_mean:.4f} (bootstrap 95% CI: [{t_lo:.4f}, {t_hi:.4f}])")
            print(f"Per-year R² (test years {t_years}): {np.round(t_scores, 4).tolist()} | SD = {t_sd:.4f}")
        else:
            t_mean = t_lo = t_hi = t_sd = np.nan
            print("Time-series CV: insufficient years for forward-chaining.")

        # Record fold-level details
        for k, (s, yrs) in enumerate(zip(g_scores, g_years), start=1):
            fold_detail.append({'Method': method_name, 'Scheme': 'GroupKFold',
                                'Fold': k, 'Held_out_years': str(yrs), 'R2': round(s, 4)})
        for yr, s in zip(t_years, t_scores):
            fold_detail.append({'Method': method_name, 'Scheme': 'TimeSeries',
                                'Fold': yr, 'Held_out_years': str(yr), 'R2': round(s, 4)})

        # Top-4 from the reduced dataset (top feature removed)
        if top_feature is None:
            top_feature = top5_features[0]
        X_reduced = X.drop(top_feature, axis=1)
        top4_features, _ = method_func(X_reduced, y, 4)
        print(f"Top 4 features after removing {top_feature}: {top4_features}")

        results.append({
            'Method': method_name,
            'GroupCV_R2': f"{g_mean:.4f}",
            'GroupCV_R2_bootCI95': f"[{g_lo:.4f}, {g_hi:.4f}]",
            'GroupCV_FoldSD': f"{g_sd:.4f}",
            'TS_R2': f"{t_mean:.4f}" if not np.isnan(t_mean) else "NA",
            'TS_R2_bootCI95': f"[{t_lo:.4f}, {t_hi:.4f}]" if not np.isnan(t_mean) else "NA",
            'TS_FoldSD': f"{t_sd:.4f}" if not np.isnan(t_sd) else "NA",
            'Top5_Features': ', '.join(top5_features),
            'Top4_Features_Reduced': ', '.join(top4_features)
        })

    except Exception as e:
        print(f"Error with {method_name}: {e}")
        results.append({'Method': method_name, 'GroupCV_R2': "ERROR"})


# ===============================================================
# 6. Save and display outputs
# ===============================================================
results_df = pd.DataFrame(results)
print("\nSummary Table:")
print(results_df.to_string(index=False))

results_df.to_csv('result_grouped_cv.csv', index=False)
pd.DataFrame(fold_detail).to_csv('fold_level_r2.csv', index=False)

vif_all = pd.concat(vif_results, ignore_index=True)
vif_all.to_csv('vif_result.csv', index=False)

print("\nSaved: result_grouped_cv.csv, fold_level_r2.csv, vif_result.csv")