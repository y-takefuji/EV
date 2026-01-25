import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
import xgboost as xgb
from scipy.stats import spearmanr

def main():
    # Load the dataset
    df = pd.read_csv('ev_battery_synth.csv')

    # Show shape of dataset
    print(f"Dataset shape: {df.shape}")

    # Display target distribution
    print("\nTarget distribution:")
    print(df['capacity_retained_percent'].describe())

    # Delete instances with NaNs in any column
    df = df.dropna()
    print(f"\nDataset shape after deleting NaNs: {df.shape}")

    # Prepare features and target
    X = df.drop(columns=['capacity_retained_percent'])
    y = df['capacity_retained_percent']
    
    # Convert string columns to categorical integers
    for col in X.columns[:3]:  # First three columns are strings
        X[col] = X[col].astype('category').cat.codes

    # Convert regression target to binary classification using mean as threshold
    threshold = y.mean()
    print(f"\nUsing threshold for binary classification: {threshold:.4f}")
    y_binary = (y >= threshold).astype(int)
    
    # Results dictionary
    results = {
        'method': [],
        'CV6': [],
        'CV5': [],
        'top6_features': [],
        'top5_features': []
    }

    # Apply feature selection methods
    feature_selection_methods = {
        'RandomForest': rf_feature_selection,
        'XGBoost': xgb_feature_selection,
        'FA': fa_feature_selection,
        'HVGS': hvgs_feature_selection,
        'PCA': pca_feature_selection,
        'Spearman': spearman_feature_selection
    }

    for method_name, method_func in feature_selection_methods.items():
        # Get top 6 features from full dataset
        if method_name in ['FA', 'HVGS', 'PCA']:
            top6_features = method_func(X, k=6)
        else:
            top6_features = method_func(X, y_binary, k=6)
        
        # Calculate CV6 score
        cv6_score = evaluate_features(X[top6_features], y_binary)
        
        # Create reduced dataset by removing the highest feature
        reduced_X = X.drop(columns=[top6_features[0]])
        
        # Reselect top 5 features from reduced dataset by re-fitting
        if method_name in ['FA', 'HVGS', 'PCA']:
            # Re-fit on reduced dataset
            top5_features = method_func(reduced_X, k=5)
        else:
            # Re-fit on reduced dataset with target
            top5_features = method_func(reduced_X, y_binary, k=5)
        
        # Calculate CV5 score
        cv5_score = evaluate_features(reduced_X[top5_features], y_binary)
        
        # Add results to dictionary
        results['method'].append(method_name)
        results['CV6'].append(cv6_score)
        results['CV5'].append(cv5_score)
        results['top6_features'].append(', '.join(top6_features))
        results['top5_features'].append(', '.join(top5_features))

    # Create summary table
    results_df = pd.DataFrame(results)

    # Format to 4 decimal places
    results_df['CV6'] = results_df['CV6'].map(lambda x: f"{x:.4f}")
    results_df['CV5'] = results_df['CV5'].map(lambda x: f"{x:.4f}")

    # Save results to CSV
    results_df.to_csv('result.csv', index=False)

    # Display the results
    print("\nResults Summary:")
    print(results_df)

# Function for cross-validation
def evaluate_features(X_selected, y, cv=5):
    model = RandomForestClassifier(random_state=42)
    # Use accuracy instead of r2
    scores = cross_val_score(model, X_selected, y, cv=cv, scoring='accuracy')
    return np.mean(scores)

# 1. Random Forest Feature Selection
def rf_feature_selection(X, y, k=6):
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances.nlargest(k).index.tolist()

# 2. XGBoost Feature Selection
def xgb_feature_selection(X, y, k=6):
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances.nlargest(k).index.tolist()

# 3. Feature Agglomeration (FA)
def fa_feature_selection(X, k=6):
    n_clusters = min(len(X.columns) - k, len(X.columns) - 1)
    if n_clusters <= 0:
        # If we have fewer features than k, just return all features
        return list(X.columns[:k])
    
    # Apply feature agglomeration
    agglo = FeatureAgglomeration(n_clusters=n_clusters)
    agglo.fit(X)
    
    # Get feature importances from agglomeration
    feature_importances = {}
    for i, cluster_label in enumerate(np.unique(agglo.labels_)):
        cluster_features = X.columns[agglo.labels_ == cluster_label]
        
        # Within each cluster, rank features by variance
        for feature in cluster_features:
            feature_importances[feature] = X[feature].var()
    
    # Convert to pandas Series for easier handling
    importances_series = pd.Series(feature_importances)
    
    # Return top k features based on importance across all clusters
    return importances_series.nlargest(k).index.tolist()

# 4. Highly Variable Gene Selection (HVGS)
def hvgs_feature_selection(X, k=6):
    # Pure variance-based selection
    variances = X.var().sort_values(ascending=False)
    return variances.index[:k].tolist()

# 5. PCA-based Feature Selection
def pca_feature_selection(X, k=6):
    pca = PCA(n_components=min(10, X.shape[1]))
    pca.fit(X)
    
    # Get feature contributions to principal components
    components = pd.DataFrame(pca.components_, columns=X.columns)
    
    # Calculate feature importance as sum of absolute loadings
    feature_importance = pd.Series(index=X.columns, data=0)
    for i, var_explained in enumerate(pca.explained_variance_ratio_):
        feature_importance += abs(components.iloc[i]) * var_explained
        
    return feature_importance.nlargest(k).index.tolist()

# 6. Spearman's Correlation
def spearman_feature_selection(X, y, k=6):
    correlations = []
    for col in X.columns:
        corr, _ = spearmanr(X[col], y)
        correlations.append((col, abs(corr)))
    
    sorted_features = sorted(correlations, key=lambda x: x[1], reverse=True)
    return [feature[0] for feature in sorted_features[:k]]

if __name__ == "__main__":
    main()
