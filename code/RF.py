import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 读取 CSV 文件
def load_data(file_path):
    df = pd.read_csv(file_path)
    # 输入特征
    features = ['vol', 'Bandgap', 'Ionic Radius', 'Preparation temperature',
                'concentration', 'frequency', 'temperature', 'polarizability',
                'electronegativity', 'components_encoded_dielectric constant',
                'components_encoded_dielectric loss']
    # 输出变量
    targets = ['dielectric constant', 'dielectric loss']

    X = df[features]
    y = df[targets]
    return X, y

# K 折交叉验证和模型训练
def train_rf_model(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储每个输出的预测和实际值
    predictions = {target: [] for target in y.columns}
    actuals = {target: [] for target in y.columns}

    # 存储评估指标
    metrics = {target: {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': []} for target in y.columns}
    # 针对介电损耗优化的超参数
    rf_params = {
        'n_estimators': 1500,  # 增加树的数量以提高稳定性
        'max_depth': 10,  # 限制树深度以避免过拟合
        'learning_rate': 0.3,
        'min_samples_split': 5,  # 提高分裂所需样本数，减少过拟合
        'min_samples_leaf': 2,  # 控制叶子节点样本数，平滑预测
        'max_features': 'sqrt',  # 使用平方根特征数，适合高维数据
        'random_state': 42
    }


    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        for target in y.columns:
            # 初始化随机森林模型
            rf = RandomForestRegressor(random_state=42)

            # 训练模型
            rf.fit(X_train, y_train[target])

            # 预测
            y_pred = rf.predict(X_val)

            # 存储预测和实际值
            predictions[target].extend(y_pred)
            actuals[target].extend(y_val[target])

            # 计算评估指标
            mse = mean_squared_error(y_val[target], y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val[target], y_pred)
            r2 = r2_score(y_val[target], y_pred)

            metrics[target]['MSE'].append(mse)
            metrics[target]['RMSE'].append(rmse)
            metrics[target]['MAE'].append(mae)
            metrics[target]['R2'].append(r2)

    return predictions, actuals, metrics

# 打印平均评估指标
def print_metrics(metrics, targets):
    for target in targets:
        print(f"\nMetrics for {target}:")
        for metric_name in metrics[target]:
            avg_metric = np.mean(metrics[target][metric_name])
            print(f"Average {metric_name}: {avg_metric:.4f}")

# 绘制预测值与实际值的散点图
def plot_scatter(actuals, predictions, targets):
    plt.figure(figsize=(12, 5))

    for i, target in enumerate(targets):
        plt.subplot(1, 2, i+1)
        plt.scatter(actuals[target], predictions[target], alpha=0.5)
        plt.plot([min(actuals[target]), max(actuals[target])],
                [min(actuals[target]), max(actuals[target])], 'r--', lw=2)
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{target} Prediction vs Actual')
        plt.tight_layout()

    plt.savefig('scatter_plot.png')
    plt.show()

# 主函数
def main(file_path):
    # 加载数据
    X, y = load_data(file_path)

    # 训练模型并进行 K 折交叉验证
    predictions, actuals, metrics = train_rf_model(X, y)

    # 打印评估指标
    print_metrics(metrics, y.columns)

    # 绘制散点图
    plot_scatter(actuals, predictions, y.columns)

if __name__ == "__main__":
    # 替换为你的 CSV 文件路径
    file_path = "D:/Desktop/机器学习/train/data improve3.csv"
    main(file_path)
