import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
import warnings

warnings.filterwarnings('ignore')

# 读取数据
df = pd.read_csv("D:/Desktop/机器学习/train/data improve.csv")


# 1. 稳健的异常点检测（标记但不删除）
def robust_outlier_detection(series, threshold=3.0):
    """更稳健的异常点检测，处理低方差特征"""
    median = np.median(series)
    # 使用IQR作为MAD的替代方案
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # 避免IQR为0的情况（常数特征）
    if IQR < 1e-6:
        return np.zeros_like(series, dtype=bool)

    modified_z = (series - median) / (IQR / 1.349)  # 1.349是标准正态分布IQR/scaling因子
    return (np.abs(modified_z) > threshold)


# 需要检测异常值的数值列
numerical_cols = ['vol', 'Bandgap', 'Ionic Radius', 'Preparation temperature',
                  'concentration', 'frequency(LOG)', 'temperature']

# 创建异常点标记列
df['is_outlier'] = 0

# 对每个数值列检测异常点
for col in numerical_cols:
    outlier_flags = robust_outlier_detection(df[col], threshold=4.0)  # 更高阈值保留更多点
    df.loc[outlier_flags, 'is_outlier'] += 1

# 输出异常点信息
outlier_counts = df['is_outlier'].value_counts().sort_index()
print("异常点标记统计:")
for count, freq in outlier_counts.items():
    print(f"标记为 {count} 个特征异常的行数: {freq}")
print(f"\n总行数保持不变: {len(df)}")

# 2. 增强的目标编码
# 分别对两个目标变量进行编码并取平均
encoder_constant = TargetEncoder(smoothing=2.0)
encoder_loss = TargetEncoder(smoothing=2.0)

components_encoded_constant = encoder_constant.fit_transform(
    df['components'],
    df['dielectric constant']
)

components_encoded_loss = encoder_loss.fit_transform(
    df['components'],
    df['dielectric loss']
)

# 使用两种编码的平均值
df['components_encoded'] = (components_encoded_constant + components_encoded_loss) / 2

# 3. RFE特征选择（增强稳定性）
# 准备特征和目标变量
X = df.drop(columns=['components', 'dielectric constant', 'dielectric loss'])
y_constant = df['dielectric constant']
y_loss = df['dielectric loss']

# 标准化数值特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 初始化估计器（增加树的数量提高稳定性）
estimator = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)


# 对两个目标变量分别进行特征选择（使用更保守的参数）
def stable_rfe(estimator, X, y):
    """更稳定的RFE实现"""
    clone_estimator = clone(estimator)
    selector = RFE(clone_estimator, n_features_to_select=5, step=0.2, verbose=0)
    selector.fit(X, y)
    return selector


selector_constant = stable_rfe(estimator, X_scaled, y_constant)
selector_loss = stable_rfe(estimator, X_scaled, y_loss)

# 获取特征选择结果（并集）
selected_features = []
for i, col in enumerate(X.columns):
    if selector_constant.support_[i] or selector_loss.support_[i]:
        selected_features.append(col)

# 确保Bandgap总是被包含
if 'Bandgap' not in selected_features:
    selected_features.append('Bandgap')
    print("\n注意：Bandgap未被自动选择，已手动添加")

# 4. 生成优化后的数据集
df_optimized = df[['components', 'dielectric constant', 'dielectric loss'] + selected_features].copy()

# 添加目标编码列
df_optimized['components_encoded'] = df['components_encoded']

# 保存优化后的数据集
df_optimized.to_csv('data improve2.csv', index=False)

# 输出最终结果
print("\n=== 优化完成 ===")
print(f"最终数据集形状: {df_optimized.shape}")
print(f"保留的特征 ({len(selected_features)}个): {', '.join(selected_features)}")
print("优化后的数据集已保存为 'data improve2.csv'")
print(f"Bandgap列最小值: {df_optimized['Bandgap'].min()}, 最大值: {df_optimized['Bandgap'].max()}")
