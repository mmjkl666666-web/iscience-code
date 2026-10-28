import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# 1. 数据加载
data = pd.read_csv('E:/iscience/end/second/code/代码/daima/data improve4.csv')

# 2. 定义特征和目标变量
X = data[['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol', 'sgnumber',
          'Z', 'Preparation temperature', 'concentration', 'frequency', 'temperature']]
y = data[['dielectric constant', 'dielectric loss']]

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['sgnumber']),
        ('num', StandardScaler(), ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                                  'Z', 'Preparation temperature', 'concentration',
                                  'frequency', 'temperature'])
    ]
)

# 5. 定义高斯过程核函数
kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)

# 6. 创建并训练两个独立模型（分别预测两个目标）
gpr_constant = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-5,
        n_restarts_optimizer=10,
        random_state=42
    ))
])

gpr_loss = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-5,
        n_restarts_optimizer=10,
        random_state=42
    ))
])

# 7. 训练模型
gpr_constant.fit(X_train, y_train['dielectric constant'])
gpr_loss.fit(X_train, y_train['dielectric loss'])

# 8. 预测与评估
y_pred_const = gpr_constant.predict(X_test)
y_pred_loss = gpr_loss.predict(X_test)

r2_const = r2_score(y_test['dielectric constant'], y_pred_const)
r2_loss = r2_score(y_test['dielectric loss'], y_pred_loss)

# 9. 可视化结果
plt.figure(figsize=(14, 6))

# 介电常数预测
plt.subplot(1, 2, 1)
plt.scatter(y_test['dielectric constant'], y_pred_const,
           alpha=0.6, color='#3498db', edgecolor='w', s=80)
plt.plot([y.min()[0], y.max()[0]], [y.min()[0], y.max()[0]],
         'r--', linewidth=2)
plt.xlabel('Actual Dielectric Constant', fontsize=12)
plt.ylabel('Predicted Dielectric Constant', fontsize=12)
plt.title(f'Dielectric Constant (R² = {r2_const:.3f})')
plt.grid(True, alpha=0.3, linestyle=':')

# 介电损耗预测
plt.subplot(1, 2, 2)
plt.scatter(y_test['dielectric loss'], y_pred_loss,
           alpha=0.6, color='#e67e22', edgecolor='w', s=80)
plt.plot([y.min()[1], y.max()[1]], [y.min()[1], y.max()[1]],
         'r--', linewidth=2)
plt.xlabel('Actual Dielectric Loss', fontsize=12)
plt.ylabel('Predicted Dielectric Loss', fontsize=12)
plt.title(f'Dielectric Loss (R² = {r2_loss:.3f})')
plt.grid(True, alpha=0.3, linestyle=':')

plt.tight_layout()
plt.show()