import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------- 数据预处理 -----------------
# 读取CSV文件（处理中文路径）
data = pd.read_csv('D:/Desktop/机器学习/train/your_data1.csv', engine='python')

# 处理缺失值（示例用中位数填充）
print("原始数据缺失统计:\n", data.isnull().sum())
data = data.dropna()
print("清洗后数据量:", len(data))

# 定义特征工程

numeric_features = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol', 'sgnumber',
                    'Z', 'Preparation temperature', 'concentration',
                    'frequency', 'temperature']

# 构建预处理管道
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# ----------------- 模型构建 -----------------
# 创建PLS管道
pls_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('pls', PLSRegression(n_components=3))  # 可调节主成分数
])

# ----------------- 模型评估 -----------------
# 划分数据集
X = data[numeric_features + categorical_features]
y = data[['dielectric constant', 'dielectric loss']]

# 常规评估
pls_pipeline.fit(X, y)
y_pred = pls_pipeline.predict(X)

print("\n【训练集评估】")
for i, target in enumerate(y.columns):
    print(f"\n目标变量: {target}")
    print(f"MSE: {mean_squared_error(y[target], y_pred[:, i]):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y[target], y_pred[:, i])):.4f}")
    print(f"MAE: {mean_absolute_error(y[target], y_pred[:, i]):.4f}")
    print(f"R²: {r2_score(y[target], y_pred[:, i]):.4f}")

# K折交叉验证
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)
cv_metrics = {'RMSE': [], 'R2': []}

for train_idx, val_idx in kf.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    fold_model = pls_pipeline.fit(X_train, y_train)
    y_val_pred = fold_model.predict(X_val)

    cv_metrics['RMSE'].append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    cv_metrics['R2'].append(r2_score(y_val, y_val_pred))

print("\n【交叉验证结果】")
print(f"平均RMSE: {np.mean(cv_metrics['RMSE']):.4f} (±{np.std(cv_metrics['RMSE']):.4f})")
print(f"平均R²: {np.mean(cv_metrics['R2']):.4f} (±{np.std(cv_metrics['R2']):.4f})")
# 设置全局中文字体（Windows推荐）
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 微软雅黑
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 验证字体配置
print("当前字体:", plt.rcParams['font.sans-serif'])


# ----------------- 可视化改进 -----------------
def plot_predictions(y_true, y_pred, targets):
    plt.figure(figsize=(15, 6), dpi=120)
    for i, target in enumerate(targets):
        plt.subplot(1, 2, i + 1)

        # 数据有效性检查
        valid_idx = np.isfinite(y_true[target]) & np.isfinite(y_pred[:, i])
        y_t = y_true[target][valid_idx]
        y_p = y_pred[valid_idx, i]

        # 动态调整坐标范围
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())

        # 绘制散点图
        scatter = plt.scatter(y_t, y_p,
                              c=np.abs(y_t - y_p),
                              cmap='viridis',
                              alpha=0.7,
                              edgecolors='w',
                              linewidth=0.5)

        # 添加辅助元素
        plt.plot([min_val, max_val], [min_val, max_val],
                 'r--', lw=1.5, label='理想预测线')
        plt.colorbar(scatter).set_label('绝对误差', rotation=270, labelpad=15)

        # 添加统计标注
        stats_text = f'R² = {r2_score(y_t, y_p):.2f}\nRMSE = {np.sqrt(mean_squared_error(y_t, y_p)):.2f}'
        plt.annotate(stats_text,
                     xy=(0.05, 0.85),
                     xycoords='axes fraction',
                     bbox=dict(boxstyle="round", fc="w"))

        plt.title(f'{target}预测效果')
        plt.xlabel('实际值')
        plt.ylabel('预测值')
        plt.grid(alpha=0.3, linestyle=':')

    plt.tight_layout()
    plt.savefig('pls_predictions.png', bbox_inches='tight')
    plt.show(block=True)  # 确保窗口保持打开


# 调用绘图函数
plot_predictions(y, y_pred, y.columns)
