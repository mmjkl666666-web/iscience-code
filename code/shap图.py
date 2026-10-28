import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import json

# 加载数据集
df = pd.read_csv('D:/Desktop/机器学习/train/your_data1.csv')

# 加载训练好的模型
model = xgb.XGBRegressor()
model.load_model('models/xgb_model_dielectric_constant.json')

# 加载训练时使用的特征名称
with open('training_features.json', 'r') as f:
    training_features = json.load(f)

# 准备输入数据（与训练时相同的特征工程）
features = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol', 'sgnumber',
            'Z', 'Preparation temperature', 'concentration', 'frequency', 'temperature']
X = df[features].copy()

# 添加工程特征（与训练时一致）
X['log_frequency'] = np.log10(df['frequency'] + 1)
X['sqrt_frequency'] = np.sqrt(df['frequency'])
X['frequency_reciprocal'] = 1 / (df['frequency'] + 1)

# 编码 component
component_dummies = pd.get_dummies(df['component'], prefix='material')
X = pd.concat([X, component_dummies], axis=1)

# 添加频率分段特征
X['is_freq_verylow'] = (df['frequency'] < 10).astype(int)
X['is_freq_low'] = ((df['frequency'] >= 10) & (df['frequency'] < 100)).astype(int)
X['is_freq_mid'] = ((df['frequency'] >= 100) & (df['frequency'] < 1000)).astype(int)
X['is_freq_high'] = ((df['frequency'] >= 1000) & (df['frequency'] < 10000)).astype(int)
X['is_freq_veryhigh'] = (df['frequency'] >= 10000).astype(int)

# 添加交互特征
X['freq_temp'] = df['frequency'] * df['temperature']
X['log_freq_temp'] = X['log_frequency'] * df['temperature']
X['temp_squared'] = df['temperature'] ** 2
X['log_freq_squared'] = X['log_frequency'] ** 2

# 添加晶体参数比例特征
X['a_b_ratio'] = df['a'] / df['b']
X['b_c_ratio'] = df['b'] / df['c']
X['a_c_ratio'] = df['a'] / df['c']

# 计算 z-score 和理想数据标志
component_groups = df.groupby(['component', 'frequency'])
df['dc_zscore'] = 0.0
for name, group in component_groups:
    if len(group) >= 3:
        dc_values = group['dielectric constant']
        if dc_values.std() > 0:
            z_scores = (dc_values - dc_values.mean()) / dc_values.std()
            df.loc[group.index, 'dc_zscore'] = z_scores
X['dc_zscore_abs'] = np.abs(df['dc_zscore'])
X['is_ideal_data'] = (abs(df['dc_zscore']) <= 1.0).astype(int)

# 确保特征顺序与训练时一致
X = X[training_features]

# 标准化数值特征（与训练时一致）
from sklearn.preprocessing import StandardScaler
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[numeric_features] = scaler.fit_transform(X[numeric_features])

# 计算 SHAP 值
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 绘制 SHAP 特征重要性图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X, plot_type="dot", show=False)

# 设置标题和标签
plt.title('Feature Importance for Dielectric Constant Prediction')
plt.xlabel('SHAP Value (Impact on Model Output)')

# 保存并显示图形
plt.savefig('shap_feature_importance.png')
plt.show()