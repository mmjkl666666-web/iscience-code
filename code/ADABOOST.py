import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import AdaBoostRegressor
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
        # 对晶体群编号进行独热编码，处理未知类别
        ('sgnumber', OneHotEncoder(handle_unknown='ignore'), ['sgnumber'])
    ],
    remainder='passthrough'  # 保留其他数值特征
)

# 5. 创建AdaBoost模型管道
def create_adaboost_pipeline():
    return Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor(random_state=42))
    ])

# 6. 分别训练两个模型
model_constant = create_adaboost_pipeline()
model_loss = create_adaboost_pipeline()

model_constant.fit(X_train, y_train['dielectric constant'])
model_loss.fit(X_train, y_train['dielectric loss'])

# 7. 预测与评估
y_pred_const = model_constant.predict(X_test)
y_pred_loss = model_loss.predict(X_test)

r2_const = r2_score(y_test['dielectric constant'], y_pred_const)
r2_loss = r2_score(y_test['dielectric loss'], y_pred_loss)

# 8. 可视化结果
plt.figure(figsize=(14, 6))

# 介电常数子图
plt.subplot(1, 2, 1)
plt.scatter(y_test['dielectric constant'], y_pred_const, alpha=0.5)
plt.plot([min(y_test['dielectric constant']), max(y_test['dielectric constant'])],
         [min(y_test['dielectric constant']), max(y_test['dielectric constant'])],
         'r--')
plt.xlabel('Actual Dielectric Constant')
plt.ylabel('Predicted Dielectric Constant')
plt.title(f'Dielectric Constant Prediction (R² = {r2_const:.3f})')

# 介电损耗子图
plt.subplot(1, 2, 2)
plt.scatter(y_test['dielectric loss'], y_pred_loss, alpha=0.5)
plt.plot([min(y_test['dielectric loss']), max(y_test['dielectric loss'])],
         [min(y_test['dielectric loss']), max(y_test['dielectric loss'])],
         'r--')
plt.xlabel('Actual Dielectric Loss')
plt.ylabel('Predicted Dielectric Loss')
plt.title(f'Dielectric Loss Prediction (R² = {r2_loss:.3f})')

plt.tight_layout()
plt.show()