import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# 读取 CSV 文件
def load_data(file_path):
    df = pd.read_csv(file_path)
    features = ['vol', 'Bandgap', 'Ionic Radius', 'Preparation temperature',
                'concentration', 'frequency', 'temperature', 'polarizability',
                'electronegativity', 'components_encoded_dielectric constant',
                'components_encoded_dielectric loss']
    targets = ['dielectric constant', 'dielectric loss']

    X = df[features]
    y = df[targets].copy()
    # 对介电损耗进行对数变换
    y['dielectric loss'] = np.log1p(y['dielectric loss'])
    return X, y


# K 折交叉验证和模型训练
def train_gpr_model(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # 存储每个输出的预测和实际值
    predictions = {target: [] for target in y.columns}
    actuals = {target: [] for target in y.columns}

    # 存储评估指标
    metrics = {target: {'MSE': [], 'RMSE': [], 'MAE': [], 'R2': []} for target in y.columns}

    # 使用 RobustScaler 进行特征标准化
    scaler = RobustScaler()

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # 标准化特征
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for target in y.columns:
            # 定义高斯过程回归模型的核函数
            kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + \
                     WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-10, 1e1))
            gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.1, random_state=42, optimizer='fmin_l_bfgs_b')

            # 训练模型
            gpr.fit(X_train_scaled, y_train[target])

            # 预测
            y_pred = gpr.predict(X_val_scaled)

            # 如果是介电损耗，反变换预测和实际值
            if target == 'dielectric loss':
                y_pred = np.expm1(y_pred)
                y_val_target = np.expm1(y_val[target])
            else:
                y_val_target = y_val[target]

            # 存储预测和实际值
            predictions[target].extend(y_pred)
            actuals[target].extend(y_val_target)

            # 计算评估指标
            mse = mean_squared_error(y_val_target, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_target, y_pred)
            r2 = r2_score(y_val_target, y_pred)

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
        plt.subplot(1, 2, i + 1)
        plt.scatter(actuals[target], predictions[target], alpha=0.5)
        plt.plot([min(actuals[target]), max(actuals[target])],
                 [min(actuals[target]), max(actuals[target])], 'r--', lw=2)
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{target} Prediction vs Actual')
        plt.tight_layout()

    plt.savefig('gpr_scatter_plot.png')
    plt.show()


# 主函数
def main(file_path):
    # 加载数据
    X, y = load_data(file_path)

    # 训练模型并进行 K 折交叉验证
    predictions, actuals, metrics = train_gpr_model(X, y)

    # 打印评估指标
    print_metrics(metrics, y.columns)

    # 绘制散点图
    plot_scatter(actuals, predictions, y.columns)


if __name__ == "__main__":
    # 替换为你的 CSV 文件路径
    file_path = "D:/Desktop/机器学习/train/data improve3.csv"
    main(file_path)