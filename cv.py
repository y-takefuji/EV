import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import FeatureAgglomeration
from scipy.stats import spearmanr
import xgboost as xgb

# Load the dataset
df = pd.read_csv('ev_sales_by_make_year_comprehensive.csv')

# Filter for Model Years 2023-2026
df = df[(df['Model Year'] >= 2023) & (df['Model Year'] <= 2026)]

# Display dataset shape
print(f"Dataset shape after filtering for years 2023-2026: {df.shape}")

# Display target distribution
print(f"Target (sales) distribution:\n{df['sales'].describe()}")

# Preprocess data - encode categorical variables
le = LabelEncoder()
categorical_cols = ['Make', 'Model_most_common', 'Electric Vehicle Type_most_common', 
                    'Clean Alternative Fuel Vehicle (CAFV) Eligibility_most_common']

for col in categorical_cols:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Separate features and target
X = df.drop('sales', axis=1)
y = df['sales']

# Feature selection functions
def rf_feature_importance(X, y, n_features=5):
    """Random Forest feature importance"""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return X.columns[indices[:n_features]].tolist(), X.columns[indices[0]]

def xgb_feature_importance(X, y, n_features=5):
    """XGBoost feature importance"""
    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return X.columns[indices[:n_features]].tolist(), X.columns[indices[0]]

# 5. Feature Agglomeration - Pure version selecting top features across all clusters
def feature_agglomeration_selection(X, y, n_features=5):
    # Always set 5 clusters for top 5 features, as requested
    # If we have fewer features than 5, we need to adjust
    n_clusters = min(5, X.shape[1])
    
    # Create and fit the feature agglomeration
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X)
    
    # Get all features and their cluster assignments
    feature_clusters = [(X.columns[i], fa.labels_[i]) for i in range(len(fa.labels_))]
    
    # Calculate variance for each feature
    feature_variances = []
    for feature, cluster in feature_clusters:
        variance = X[feature].var()
        feature_variances.append((feature, variance, cluster))
    
    # Sort all features by variance (descending)
    feature_variances.sort(key=lambda x: x[1], reverse=True)
    
    # Simply select top n_features across all clusters by variance
    selected_features = [f for f, _, _ in feature_variances[:n_features]]
    
    return selected_features, selected_features[0] if selected_features else None

# 6. Highly Variable Gene Selection (based on variance)
def hvgs_selection(X, y, n_features=5):
    variance = X.var().sort_values(ascending=False)
    selected_features = variance.index[:n_features].tolist()
    return selected_features, selected_features[0] if selected_features else None

# 7. Spearman's Correlation
def spearman_selection(X, y, n_features=5):
    correlation = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlation.append((col, abs(corr)))
    
    correlation.sort(key=lambda x: x[1], reverse=True)
    selected_features = [item[0] for item in correlation[:n_features]]
    return selected_features, selected_features[0] if selected_features else None

# Main feature selection and cross-validation process
results = []
methods = {
    'RF': rf_feature_importance,
    'XGB': xgb_feature_importance,
    'FA': feature_agglomeration_selection,
    'HVGS': hvgs_selection,
    'Spearman': spearman_selection
}

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for method_name, method_func in methods.items():
    print(f"\nProcessing {method_name}...")
    
    try:
        # Get top 5 features from full dataset
        top5_features, top_feature = method_func(X, y, 5)
        print(f"Top 5 features: {top5_features}")
        
        # Cross-validation on top 5 features
        X_top5 = X[top5_features]
        
        if method_name in ['RF', 'XGB']:
            if method_name == 'RF':
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        cv_scores = cross_val_score(model, X_top5, y, cv=kf, scoring='r2')
        cv5_score = np.mean(cv_scores)
        print(f"CV5 R² score: {cv5_score:.4f}")
        
        # Remove top feature and get top 4 from reduced dataset
        if top_feature is not None:
            X_reduced = X.drop(top_feature, axis=1)
        else:
            # If top feature is None, just use the first feature
            top_feature = top5_features[0]
            X_reduced = X.drop(top_feature, axis=1)
            
        top4_features, _ = method_func(X_reduced, y, 4)
        print(f"Top 4 features after removing {top_feature}: {top4_features}")
        
        results.append({
            'Method': method_name,
            'CV5_R2': f"{cv5_score:.4f}",
            'Top5_Features': ', '.join(top5_features),
            'Top4_Features_Reduced': ', '.join(top4_features)
        })
    except Exception as e:
        print(f"Error with {method_name}: {e}")
        results.append({
            'Method': method_name,
            'CV5_R2': "ERROR",
            'Top5_Features': "ERROR",
            'Top4_Features_Reduced': "ERROR"
        })

# Create summary table
results_df = pd.DataFrame(results)
print("\nSummary Table:")
print(results_df)

# Format the results table for better readability
formatted_results = results_df.copy()
formatted_results = formatted_results[['Method', 'CV5_R2', 'Top5_Features', 'Top4_Features_Reduced']]
print("\nFormatted Results Table:")
print(formatted_results)

# Save results to CSV
results_df.to_csv('result.csv', index=False)
print("\nResults saved to result.csv")
