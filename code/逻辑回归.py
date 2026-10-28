import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# 配置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path):
    """加载并预处理材料数据"""
    data = pd.read_csv(file_path)

    # 处理缺失值
    print("缺失值统计:\n", data.isnull().sum())
    data = data.dropna()

    return data


def build_model():
    """构建预处理和回归管道"""
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                                    'Z', 'Preparation temperature',
                                   'concentration', 'frequency', 'temperature']),
        ('cat', OneHotEncoder(handle_unknown='ignore'))
    ])

    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])


def evaluate_model(model, X_test, y_test):
    """模型评估与结果可视化"""
    y_pred = model.predict(X_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n模型评估结果:")
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")

    # 可视化预测结果
    plt.figure(figsize=(12, 5))

    # 介电常数预测
    plt.subplot(1, 2, 1)
    plt.scatter(y_test.iloc[:, 0], y_pred[:, 0], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际介电常数')
    plt.ylabel('预测介电常数')
    plt.title('介电常数预测效果')

    # 介电损耗预测
    plt.subplot(1, 2, 2)
    plt.scatter(y_test.iloc[:, 1], y_pred[:, 1], alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('实际介电损耗')
    plt.ylabel('预测介电损耗')
    plt.title('介电损耗预测效果')

    plt.tight_layout()
    plt.savefig('material_prediction.png', dpi=300)
    plt.show()


def cross_validate(model, X, y, k=5):
    """K折交叉验证"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    rmse_scores = []
    r2_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))

    print("\n交叉验证结果:")
    print(f"平均RMSE: {np.mean(rmse_scores):.4f} (±{np.std(rmse_scores):.4f})")
    print(f"平均R²: {np.mean(r2_scores):.4f} (±{np.std(r2_scores):.4f})")


if __name__ == "__main__":
    # 加载数据
    data = load_data("D:/Desktop/机器学习/train/your_data1.csv")

    # 定义输入输出
    features = data[['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                     'sgnumber', 'Z', 'Preparation temperature', 'concentration',
                     'frequency', 'temperature']]
    targets = data[['dielectric constant', 'dielectric loss']]

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        features, targets, test_size=0.2, random_state=42)

    # 构建并训练模型
    model = build_model()
    model.fit(X_train, y_train)

    # 模型评估
    evaluate_model(model, X_test, y_test)

    # 交叉验证
    cross_validate(model, features, targets)