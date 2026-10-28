import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
import json

warnings.filterwarnings('ignore')

print(f"XGBoost version: {xgb.__version__}")
print(f"XGBoost path: {xgb.__file__}")


def load_data(file_path):
    print("Loading dataset...")
    df = pd.read_csv(file_path, encoding='gbk')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print("Columns:", df.columns.tolist())
    print("First few rows:\n", df.head().to_string())
    
    # 检查并重命名 frequency 列
    possible_freq_columns = ['freq', 'Frequency', 'FREQUENCY', 'Freq']
    for col in possible_freq_columns:
        if col in df.columns:
            print(f"Renaming '{col}' to 'frequency'")
            df = df.rename(columns={col: 'frequency'})
            break
    else:
        if 'frequency' not in df.columns:
            print("Warning: 'frequency' column not found. Adding with default value 0.")
            df['frequency'] = 0  # 默认值，可以根据需要调整
    
    return df


def add_extreme_weights(df):
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
        0.001, 0.001, 0.01, 0.05, 200.0, 150.0, 75.0, 300.0, 250.0, 200.0, 70.0, 15.0,
        250.0, 200.0, 60.0, 8.0, 230.0, 170.0, 45.0, 100.0, 0.0005, 150.0, 130.0,
        100.0, 80.0, 40.0, 12.0, 1.5
    ]

    df['weight'] = np.select(conditions, choices, default=1.0)
    df.loc[df['ideal_data'], 'weight'] *= 2.5

    print("\nWeight distribution:")
    weight_ranges = [0, 0.01, 0.1, 1, 10, 50, 100, 200, float('inf')]
    for i in range(len(weight_ranges) - 1):
        count = ((df['weight'] > weight_ranges[i]) & (df['weight'] <= weight_ranges[i + 1])).sum()
        print(f"Weight range {weight_ranges[i]}-{weight_ranges[i + 1]}: {count} samples")

    return df


def add_engineered_features(df, features):
    X = df[features].copy()
    X['log_frequency'] = np.log10(df['frequency'] + 1)
    X['sqrt_frequency'] = np.sqrt(df['frequency'])
    X['frequency_reciprocal'] = 1 / (df['frequency'] + 1)

    if 'component' in df.columns:
        try:
            with open("component_categories.json", "r") as f:
                component_categories = json.load(f)
            print("Training component categories:", component_categories)
        except FileNotFoundError:
            print("Warning: 'component_categories.json' not found. Using categories from prediction data.")
            component_categories = df['component'].unique().tolist()
            print("Prediction component categories (used as fallback):", component_categories)

        print("Prediction component categories:", df['component'].unique().tolist())

        component_dummies = pd.get_dummies(df['component'], prefix='material')
        for category in component_categories:
            dummy_col = f"material_{category}"
            if dummy_col not in component_dummies.columns:
                component_dummies[dummy_col] = 0
        expected_dummy_cols = [f"material_{cat}" for cat in component_categories]
        component_dummies = component_dummies[expected_dummy_cols]
        X = pd.concat([X, component_dummies], axis=1)
    else:
        print("Warning: 'component' column missing. Adding default dummy columns.")
        try:
            with open("component_categories.json", "r") as f:
                component_categories = json.load(f)
        except FileNotFoundError:
            print("Error: 'component_categories.json' not found and 'component' column missing. Cannot proceed.")
            raise
        for category in component_categories:
            X[f"material_{category}"] = 0

    X['is_freq_verylow'] = (df['frequency'] < 10).astype(int)
    X['is_freq_low'] = ((df['frequency'] >= 10) & (df['frequency'] < 100)).astype(int)
    X['is_freq_mid'] = ((df['frequency'] >= 100) & (df['frequency'] < 1000)).astype(int)
    X['is_freq_high'] = ((df['frequency'] >= 1000) & (df['frequency'] < 10000)).astype(int)
    X['is_freq_veryhigh'] = (df['frequency'] >= 10000).astype(int)

    if 'temperature' in features:
        X['freq_temp'] = df['frequency'] * df['temperature']
        X['log_freq_temp'] = X['log_frequency'] * df['temperature']
        X['temp_squared'] = df['temperature'] ** 2
        X['log_freq_squared'] = X['log_frequency'] ** 2

    if all(param in features for param in ['a', 'b', 'c']):
        X['a_b_ratio'] = df['a'] / df['b']
        X['b_c_ratio'] = df['b'] / df['c']
        X['a_c_ratio'] = df['a'] / df['c']

    X['dc_zscore_abs'] = np.abs(df['dc_zscore'])
    X['is_ideal_data'] = df['ideal_data'].astype(int)

    print(f"Features expanded from {len(features)} to {X.shape[1]}")
    return X


def preprocess_data_for_prediction(df, use_engineered_features=True):
    features = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                'sgnumber', 'Z', 'Preparation temperature', 'concentration',
                'frequency', 'temperature']
    targets = ['dielectric constant', 'dielectric loss']

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"Missing features: {missing_features}")
        features = [f for f in features if f in df.columns]

    missing_targets = [t for t in targets if t in df.columns]
    if missing_targets:
        print(f"Target columns present in dataset (will be ignored for prediction): {missing_targets}")

    df = add_extreme_weights(df)

    if use_engineered_features:
        X_df = add_engineered_features(df, features)
    else:
        X_df = df[features].copy()

    print("Prediction features:", X_df.columns.tolist())

    with open("training_features.json", "r") as f:
        training_features = json.load(f)
    print("Training features (loaded):", training_features)

    if set(X_df.columns) != set(training_features):
        print("Feature mismatch detected!")
        missing_in_prediction = set(training_features) - set(X_df.columns)
        extra_in_prediction = set(X_df.columns) - set(training_features)
        print(f"Features missing in prediction: {missing_in_prediction}")
        print(f"Extra features in prediction: {extra_in_prediction}")

        for feature in missing_in_prediction:
            X_df[feature] = 0
        X_df = X_df[training_features]

    numeric_features = X_df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler()
    X_df[numeric_features] = scaler.fit_transform(X_df[numeric_features])

    X = X_df.values
    return X, X_df.columns.tolist(), targets


def predict_new_data(new_file_path, model_dir="models"):
    df_new = load_data(new_file_path)
    print("New dataset columns:", df_new.columns.tolist())
    
    # 暂停执行，等待用户确认列名
    input("Press Enter to continue after checking the columns...")

    X_new, feature_names, targets = preprocess_data_for_prediction(df_new)
    print(f"Preprocessed new dataset with shape: {X_new.shape}")

    predictions_new = {}
    for target in targets:
        model_path = f"{model_dir}/xgb_model_{target.replace(' ', '_')}.json"
        print(f"Loading model for {target} from {model_path}...")

        model = xgb.XGBRegressor()
        model.load_model(model_path)

        y_pred_log = model.predict(X_new)
        y_pred_original = np.expm1(y_pred_log)

        predictions_new[target] = y_pred_original
        df_new[f'predicted_{target}'] = y_pred_original

        print(f"Completed predictions for {target}. Sample predictions: {y_pred_original[:5]}")

    output_file = new_file_path.replace('.csv', '_predicted.csv')
    df_new.to_csv(output_file, index=False, encoding='gbk')
    print(f"Saved predictions to {output_file}")

    return df_new, predictions_new


if __name__ == "__main__":
    new_file_path = r"D:\Desktop\机器学习\train\氧化物数据集 - 副本.csv"
    df_new, predictions_new = predict_new_data(new_file_path)