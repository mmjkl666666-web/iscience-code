import pandas as pd
import numpy as np
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap


# 数据加载函数
def load_data(file_path):
    df = pd.read_csv(file_path)
    features = ['vol', 'Bandgap', 'Ionic Radius', 'Preparation temperature',
                'concentration', 'frequency', 'temperature', 'polarizability',
                'electronegativity', 'components_encoded_dielectric constant',
                'components_encoded_dielectric loss']
    targets = ['dielectric constant', 'dielectric loss']

    X = df[features]
    y = df[targets]
    return X, y, features  # 返回特征名称以用于 SHAP 和重要性图


# 绘制 SHAP 和特征重要性图
def plot_shap_and_importance(model, X_scaled, features, target):
    # SHAP 分析
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # SHAP 摘要图（点图）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled, feature_names=features, show=False)
    plt.title(f'SHAP Summary Plot for {target}')
    plt.tight_layout()
    plt.savefig(f'shap_summary_{target.replace(" ", "_")}.png')
    plt.close()

    # SHAP 摘要图（条形图）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_scaled, feature_names=features, plot_type="bar", show=False)
    plt.title(f'SHAP Bar Plot for {target}')
    plt.tight_layout()
    plt.savefig(f'shap_bar_{target.replace(" ", "_")}.png')
    plt.close()

    # 特征重要性图
    plt.figure(figsize=(10, 6))
    plot_importance(model, importance_type='gain', title=f'Feature Importance for {target}')
    plt.tight_layout()
    plt.savefig(f'feature_importance_{target.replace(" ", "_")}.png')
    plt.close()


# XGBoost模型训练及K折交叉验证函数
def train_xgb_model(X, y, features, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    params = {
        'objective': 'reg:squarederror',
        'n_estimators': 1500,
        'max_depth': 10,
        'learning_rate': 0.1,
        'subsample': 0.85,
        'colsample_bytree': 0.8,
        'random_state': 42,
    }

    predictions, actuals, metrics = {target: [] for target in y.columns}, {target: [] for target in y.columns}, {
        target: {'R2': []} for target in y.columns}

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # 将缩放后的数据转换为 DataFrame 以保留特征名称
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=features)
        X_val_scaled_df = pd.DataFrame(X_val_scaled, columns=features)

        for target in y.columns:
            xgb = XGBRegressor(**params)
            xgb.fit(X_train_scaled_df, y_train[target])

            y_pred = xgb.predict(X_val_scaled_df)

            predictions[target].extend(y_pred)
            actuals[target].extend(y_val[target])

            r2 = r2_score(y_val[target], y_pred)
            metrics[target]['R2'].append(r2)

            # 在最后一折生成 SHAP 和特征重要性图
            if train_index[-1] == X.index[-1]:  # 检查是否为最后一折
                plot_shap_and_importance(xgb, X_val_scaled_df, features, target)

    return predictions, actuals, metrics


# 打印评估指标
def print_metrics(metrics, targets):
    for target in targets:
        avg_r2 = np.mean(metrics[target]['R2'])
        print(f"Average R2 for {target}: {avg_r2:.4f}")


# 结果可视化
def plot_scatter(actuals, predictions, targets):
    plt.figure(figsize=(12, 5))

    for i, target in enumerate(targets):
        plt.subplot(1, 2, i + 1)
        plt.scatter(actuals[target], predictions[target], alpha=0.5)
        plt.plot([min(actuals[target]), max(actuals[target])], [min(actuals[target]), max(actuals[target])], 'r--',
                 lw=2)
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{target} Prediction vs Actual')
        plt.tight_layout()

    plt.savefig('xgb_scatter_plot.png')
    plt.show()


# 主程序
def main(file_path):
    X, y, features = load_data(file_path)
    predictions, actuals, metrics = train_xgb_model(X, y, features)
    print_metrics(metrics, y.columns)
    plot_scatter(actuals, predictions, y.columns)


if __name__ == "__main__":
    file_path = r"D:\Desktop\机器学习\train/Supplementary Material 1..csv"
    main(file_path)