import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# 1. 数据加载与预处理
data = pd.read_csv('D:/Desktop/机器学习/train/your_data1.csv')

# 定义特征和目标变量
X = data[['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol', 'sgnumber',
          'Z', 'Preparation temperature', 'concentration', 'frequency', 'temperature']]
y = data[['dielectric constant', 'dielectric loss']]

# 2. 数据集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. 创建预处理管道
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['sgnumber']),
        ('num', StandardScaler(), ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                                  'Z', 'Preparation temperature', 'concentration',
                                  'frequency', 'temperature'])
    ]
)

# 4. 创建并训练回归模型（分别预测两个目标）
def create_linear_model():
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

model_const = create_linear_model()
model_loss = create_linear_model()

model_const.fit(X_train, y_train['dielectric constant'])
model_loss.fit(X_train, y_train['dielectric loss'])

# 5. 预测与评估
y_pred_const = model_const.predict(X_test)
y_pred_loss = model_loss.predict(X_test)

r2_const = r2_score(y_test['dielectric constant'], y_pred_const)
r2_loss = r2_score(y_test['dielectric loss'], y_pred_loss)

# 6. 可视化结果
plt.figure(figsize=(14, 6))

# 介电常数预测结果
plt.subplot(1, 2, 1)
plt.scatter(y_test['dielectric constant'], y_pred_const,
           alpha=0.6, color='#2980b9', edgecolor='w', s=80)
plt.plot([y.min()[0], y.max()[0]], [y.min()[0], y.max()[0]],
         'r--', linewidth=2)
plt.xlabel('Actual Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.title(f'Dielectric Constant (R² = {r2_const:.3f})', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')

# 介电损耗预测结果
plt.subplot(1, 2, 2)
plt.scatter(y_test['dielectric loss'], y_pred_loss,
           alpha=0.6, color='#c0392b', edgecolor='w', s=80)
plt.plot([y.min()[1], y.max()[1]], [y.min()[1], y.max()[1]],
         'r--', linewidth=2)
plt.xlabel('Actual Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.title(f'Dielectric Loss (R² = {r2_loss:.3f})', fontsize=14)
plt.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()