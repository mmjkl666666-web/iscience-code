import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score

# 1. 数据加载与预处理
data = pd.read_csv('D:/Desktop/机器学习/train/your_data1.csv')

# 定义特征和目标变量
X = data[['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol', 'sgnumber',
          'Z', 'Preparation temperature', 'concentration', 'frequency', 'temperature']]
y = data[['dielectric constant', 'dielectric loss']]

# 2. 分类特征处理
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['sgnumber'])
    ],
    remainder='passthrough'
)

X_processed = preprocessor.fit_transform(X)

# 3. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# 4. 创建并训练随机森林模型
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1  # 使用所有CPU核心加速计算
)

rf_model.fit(X_train, y_train)

# 5. 预测与评估
y_pred = rf_model.predict(X_test)

# 计算决定系数
r2_const = r2_score(y_test['dielectric constant'], y_pred[:, 0])
r2_loss = r2_score(y_test['dielectric loss'], y_pred[:, 1])

# 6. 可视化结果
plt.figure(figsize=(14, 6), dpi=100)

# 介电常数预测结果
plt.subplot(1, 2, 1)
plt.scatter(y_test['dielectric constant'], y_pred[:, 0],
           alpha=0.6, color='#2ecc71', edgecolor='w')
plt.plot([y.min()[0], y.max()[0]], [y.min()[0], y.max()[0]],
         'r--', linewidth=2)
plt.xlabel('Actual Dielectric Constant', fontsize=12)
plt.ylabel('Predicted Dielectric Constant', fontsize=12)
plt.title(f'Dielectric Constant (R² = {r2_const:.3f})')
plt.grid(True, alpha=0.3, linestyle='--')

# 介电损耗预测结果
plt.subplot(1, 2, 2)
plt.scatter(y_test['dielectric loss'], y_pred[:, 1],
           alpha=0.6, color='#e74c3c', edgecolor='w')
plt.plot([y.min()[1], y.max()[1]], [y.min()[1], y.max()[1]],
         'r--', linewidth=2)
plt.xlabel('Actual Dielectric Loss', fontsize=12)
plt.ylabel('Predicted Dielectric Loss', fontsize=12)
plt.title(f'Dielectric Loss (R² = {r2_loss:.3f})')
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()