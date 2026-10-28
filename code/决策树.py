import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline

# 1. 数据加载
data = pd.read_csv('D:/Desktop/机器学习/train/your_data1.csv')

# 2. 数据预处理
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
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['sgnumber'])
    ],
    remainder='passthrough'
)

# 5. 创建决策树模型管道
def create_dtree_pipeline():
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(
            max_depth=5,        # 控制树深度防止过拟合
            min_samples_split=10,
            random_state=42
        ))
    ])

# 6. 初始化并训练两个模型
model_const = create_dtree_pipeline()
model_loss = create_dtree_pipeline()

model_const.fit(X_train, y_train['dielectric constant'])
model_loss.fit(X_train, y_train['dielectric loss'])

# 7. 预测与评估
y_pred_const = model_const.predict(X_test)
y_pred_loss = model_loss.predict(X_test)

r2_const = r2_score(y_test['dielectric constant'], y_pred_const)
r2_loss = r2_score(y_test['dielectric loss'], y_pred_loss)

# 8. 可视化结果
plt.figure(figsize=(14, 6), dpi=100)

# 介电常数预测
plt.subplot(1, 2, 1)
plt.scatter(y_test['dielectric constant'], y_pred_const,
           alpha=0.6, edgecolors='w', color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.title(f'Dielectric Constant (R²={r2_const:.3f})', fontsize=14)
plt.grid(True, alpha=0.3)

# 介电损耗预测
plt.subplot(1, 2, 2)
plt.scatter(y_test['dielectric loss'], y_pred_loss,
           alpha=0.6, edgecolors='w', color='green')
plt.plot([y_test['dielectric loss'].min(), y_test['dielectric loss'].max()],
         [y_test['dielectric loss'].min(), y_test['dielectric loss'].max()],
         'r--', lw=2)
plt.xlabel('Actual Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.title(f'Dielectric Loss (R²={r2_loss:.3f})', fontsize=14)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()