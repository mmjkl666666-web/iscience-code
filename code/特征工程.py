import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from category_encoders import TargetEncoder

# 加载数据
df = pd.read_csv("data improve.csv")

# 检查数据类型和缺失值
print(df.info())
print(df.isnull().sum())

# 处理缺失值（若有）
df = df.dropna()  # 或使用填充方法
# 选择数值列（排除分类列）
numeric_cols = ['vol', 'Bandgap', 'Ionic Radius', 'Preparation temperature',
                'concentration', 'frequency(LOG)', 'temperature',
                'dielectric constant', 'dielectric loss']

# 计算 Z 分数
z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
outliers = z_scores > 3  # Z 分数 > 3 视为异常

# 标记并删除异常点
df = df[~outliers.any(axis=1)]
# 方法 1：目标编码（需结合目标变量）
# 假设目标变量为 'dielectric constant'
encoder = TargetEncoder()
df['component_encoded'] = encoder.fit_transform(df['components'], df['dielectric constant'])

# 方法 2：One-Hot Encoding（适用于类别较少的情况）
df = pd.get_dummies(df, columns=['components'], drop_first=True)
# 方法 1：目标编码（需结合目标变量）
# 假设目标变量为 'dielectric constant'
encoder = TargetEncoder()
df['component_encoded'] = encoder.fit_transform(df['components'], df['dielectric constant'])

# 方法 2：One-Hot Encoding（适用于类别较少的情况）
df = pd.get_dummies(df, columns=['components'], drop_first=True)
# 若 RFE 后保留了分类列 'components'
encoder = TargetEncoder()
X_selected['component_encoded'] = encoder.fit_transform(
    X_selected['components'], y
)

# 删除原始分类列
X_selected = X_selected.drop('components', axis=1)
# 合并处理后的特征和目标变量
final_df = pd.concat([X_selected, y], axis=1)

# 保存结果
final_df.to_csv("processed_data.csv", index=False)