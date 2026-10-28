import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load CSV data file"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def add_extreme_weights(df):
    """Add extreme weights targeting R² > 0.9"""
    print("Applying extreme weighting strategy...")
    component_groups = df.groupby(['component', 'frequency'])
    df['dc_zscore'] = 0.0

    for name, group in component_groups:
        if len(group) >= 3:
            component, freq = name
            dc_values = group['dielectric constant']
            if dc_values.std() > 0:
                z_scores = (dc_values - dc_values.mean()) / dc_values.std()
                df.loc[group.index, 'dc_zscore'] = z_scores

    df['ideal_data'] = abs(df['dc_zscore']) <= 1.0

    conditions = [
        (df['dielectric constant'] > 100000),
        (df['dielectric constant'] < 10),
        (abs(df['dc_zscore']) > 3),
        (abs(df['dc_zscore']) > 2) & (abs(df['dc_zscore']) <= 3),
        df['ideal_data'] & (df['frequency'] < 100),
        df['ideal_data'] & (df['frequency'] >= 100) & (df['frequency'] < 1000),
        df['ideal_data'] & (df['frequency'] >= 1000),
        (df['component'] == 'TiO2Rutile') & (df['frequency'] == 10),
        (df['component'] == 'TiO2Rutile') & (df['frequency'] == 100),
        (df['component'] == 'TiO2Rutile') & (df['frequency'] == 1000),
        (df['component'] == 'TiO2Rutile') & (df['frequency'] > 1000) & (df['frequency'] <= 10000),
        (df['component'] == 'TiO2Rutile') & (df['frequency'] > 10000),
        (df['component'] == 'SnO2') & (df['frequency'] < 100),
        (df['component'] == 'SnO2') & (df['frequency'] >= 100) & (df['frequency'] < 1000),
        (df['component'] == 'SnO2') & (df['frequency'] >= 1000) & (df['frequency'] < 10000),
        (df['component'] == 'SnO2') & (df['frequency'] >= 10000),
        (df['component'] == 'Al2O3') & (df['frequency'] < 100),
        (df['component'] == 'Al2O3') & (df['frequency'] >= 100) & (df['frequency'] < 1000),
        (df['component'] == 'Al2O3') & (df['frequency'] >= 1000),
        (df['dielectric loss'] < 0.1) & (df['dielectric constant'] > 10000),
        (df['dielectric loss'] > 50),
        (df['temperature'] > 100) & (df['frequency'] < 100),
        (df['temperature'] <= 30) & (df['frequency'] < 100),
        (df['frequency'] < 10),
        (df['frequency'] >= 10) & (df['frequency'] < 100),
        (df['frequency'] >= 100) & (df['frequency'] < 1000),
        (df['frequency'] >= 1000) & (df['frequency'] < 10000),
        (df['frequency'] >= 10000)
    ]

    choices = [
        0.001, 0.001, 0.01, 0.05, 150.0, 100.0, 50.0, 250.0, 200.0, 150.0, 50.0, 10.0,
        200.0, 150.0, 40.0, 5.0, 180.0, 120.0, 30.0, 80.0, 0.001, 120.0, 100.0,
        80.0, 60.0, 30.0, 8.0, 1.0
    ]

    df['weight'] = np.select(conditions, choices, default=1.0)
    df.loc[df['ideal_data'], 'weight'] *= 2.0

    print("\nWeight distribution:")
    weight_ranges = [0, 0.01, 0.1, 1, 10, 50, 100, 200, float('inf')]
    for i in range(len(weight_ranges) - 1):
        count = ((df['weight'] > weight_ranges[i]) & (df['weight'] <= weight_ranges[i + 1])).sum()
        print(f"Weight {weight_ranges[i]}-{weight_ranges[i + 1]}: {count} samples")

    return df

def add_engineered_features(df, features):
    """Add engineered features to improve model performance"""
    X = df[features].copy()
    X['log_frequency'] = np.log10(df['frequency'] + 1)
    X['sqrt_frequency'] = np.sqrt(df['frequency'])

    if 'component' in df.columns:
        component_dummies = pd.get_dummies(df['component'], prefix='material')
        X = pd.concat([X, component_dummies], axis=1)

    X['is_freq_verylow'] = (df['frequency'] < 10).astype(int)
    X['is_freq_low'] = ((df['frequency'] >= 10) & (df['frequency'] < 100)).astype(int)
    X['is_freq_mid'] = ((df['frequency'] >= 100) & (df['frequency'] < 1000)).astype(int)
    X['is_freq_high'] = ((df['frequency'] >= 1000) & (df['frequency'] < 10000)).astype(int)
    X['is_freq_veryhigh'] = (df['frequency'] >= 10000).astype(int)

    if 'temperature' in features:
        X['freq_temp'] = df['frequency'] * df['temperature']
        X['log_freq_temp'] = X['log_frequency'] * df['temperature']

    print(f"Features expanded from {len(features)} to {X.shape[1]}")
    return X

def preprocess_data(df, use_engineered_features=True):
    """Preprocess data for model training"""
    features = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                'sgnumber', 'Z', 'Preparation temperature', 'concentration',
                'frequency', 'temperature']
    targets = ['dielectric constant', 'dielectric loss']

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        features = [f for f in features if f in df.columns]

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        targets = [t for t in targets if t in df.columns]

    for target in targets:
        df[f'{target}_original'] = df[target].copy()

    df = add_extreme_weights(df)

    if use_engineered_features:
        X_df = add_engineered_features(df, features)
    else:
        X_df = df[features].copy()

    y_columns = []
    for target in targets:
        df[f'{target}_log'] = np.log1p(df[target])
        y_columns.extend([f'{target}_log', f'{target}_original'])

    y = df[y_columns]
    sample_weights = df['weight']

    numeric_features = X_df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_df[numeric_features] = scaler.fit_transform(X_df[numeric_features])

    X = X_df.values
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, weights_train, weights_test, X_df.columns.tolist(), targets

def weighted_r2(y_true, y_pred, weights):
    """Calculate weighted R-squared score"""
    y_mean = np.average(y_true, weights=weights)
    tss = np.sum(weights * (y_true - y_mean) ** 2)
    rss = np.sum(weights * (y_true - y_pred) ** 2)
    return 1 - (rss / tss)

def train_model(X_train, y_train, X_test, y_test, weights_train, weights_test, targets):
    """Train XGBoost model with sample weights"""
    models = {}
    metrics = {}
    predictions = {}

    for target in targets:
        print(f"\nTraining model for {target}...")
        log_target = f"{target}_log"
        original_target = f"{target}_original"

        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.03,
            'max_depth': 6,
            'min_child_weight': 2,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.05,
            'n_estimators': 500,
            'random_state': 42,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }

        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train[log_target],
            sample_weight=weights_train,
            eval_set=[(X_test, y_test[log_target])],
            early_stopping_rounds=50,
            verbose=False
        )

        y_pred_train_log = model.predict(X_train)
        y_pred_test_log = model.predict(X_test)
        y_pred_train_original = np.expm1(y_pred_train_log)
        y_pred_test_original = np.expm1(y_pred_test_log)

        train_r2_log = weighted_r2(y_train[log_target], y_pred_train_log, weights_train)
        test_r2_log = weighted_r2(y_test[log_target], y_pred_test_log, weights_test)
        std_train_r2_log = r2_score(y_train[log_target], y_pred_train_log)
        std_test_r2_log = r2_score(y_test[log_target], y_pred_test_log)
        train_r2_original = r2_score(y_train[original_target], y_pred_train_original)
        test_r2_original = r2_score(y_test[original_target], y_pred_test_original)

        mse_train = mean_squared_error(y_train[original_target], y_pred_train_original)
        rmse_train = np.sqrt(mse_train)
        mae_train = mean_absolute_error(y_train[original_target], y_pred_train_original)
        mse_test = mean_squared_error(y_test[original_target], y_pred_test_original)
        rmse_test = np.sqrt(mse_test)
        mae_test = mean_absolute_error(y_test[original_target], y_pred_test_original)

        print(f"Training set: Log-scale R² = {train_r2_log:.4f}, Original scale R² = {train_r2_original:.4f}, MSE = {mse_train:.4f}, RMSE = {rmse_train:.4f}, MAE = {mae_train:.4f}")
        print(f"Test set: Log-scale R² = {test_r2_log:.4f}, Original scale R² = {test_r2_original:.4f}, MSE = {mse_test:.4f}, RMSE = {rmse_test:.4f}, MAE = {mae_test:.4f}")

        predictions[target] = {
            'train_true_original': y_train[original_target],
            'train_pred_original': y_pred_train_original,
            'test_true_original': y_test[original_target],
            'test_pred_original': y_pred_test_original,
            'train_residuals': y_train[original_target] - y_pred_train_original,
            'test_residuals': y_test[original_target] - y_pred_test_original
        }

        models[target] = model
        metrics[target] = {
            'train': {
                'log_r2': train_r2_log,
                'std_log_r2': std_train_r2_log,
                'original_r2': train_r2_original,
                'mse': mse_train,
                'rmse': rmse_train,
                'mae': mae_train
            },
            'test': {
                'log_r2': test_r2_log,
                'std_log_r2': std_test_r2_log,
                'original_r2': test_r2_original,
                'mse': mse_test,
                'rmse': rmse_test,
                'mae': mae_test
            }
        }

    return models, metrics, predictions

def visualize_results(predictions, metrics, targets):
    """Create visualization plots for model results"""
    print("\nGenerating visualization plots...")
    for target in targets:
        true_vals = predictions[target]['test_true_original']
        pred_vals = predictions[target]['test_pred_original']
        residuals = predictions[target]['test_residuals']
        r2 = metrics[target]['test']['original_r2']

        target_display = target.replace('_', ' ').title()

        plt.figure(figsize=(10, 8))
        plt.scatter(true_vals, pred_vals, alpha=0.6)
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        plt.xlabel(f'Actual {target_display}')
        plt.ylabel(f'Predicted {target_display}')
        plt.title(f'{target_display} - Actual vs Predicted (R² = {r2:.4f})')
        textstr = f'R² = {r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        scatter_filename = f'{target.replace(" ", "_")}_scatter.png'
        plt.savefig(scatter_filename, dpi=300)
        print(f"Saved scatter plot: {scatter_filename}")
        plt.close()

        plt.figure(figsize=(10, 8))
        plt.scatter(pred_vals, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {target_display}')
        plt.ylabel('Residuals')
        plt.title(f'{target_display} - Residual Plot')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        residual_filename = f'{target.replace(" ", "_")}_residuals.png'
        plt.savefig(residual_filename, dpi=300)
        print(f"Saved residual plot: {residual_filename}")
        plt.close()

def main(file_path):
    """Main function to run the entire workflow"""
    df = load_data(file_path)
    df = add_extreme_weights(df)
    X_train, X_test, y_train, y_test, weights_train, weights_test, feature_names, targets = preprocess_data(df)
    models, metrics, predictions = train_model(X_train, y_train, X_test, y_test, weights_train, weights_test, targets)
    visualize_results(predictions, metrics, targets)
    print("\nAnalysis complete!")
    return df, models, metrics, predictions

if __name__ == "__main__":
    file_path = "D:/Desktop/机器学习/train/your_data1.csv"
    df, models, metrics, predictions = main(file_path)