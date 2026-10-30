import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
import warnings
import os
import json

warnings.filterwarnings('ignore')  # 忽略警告信息以保持输出简洁

def load_data(file_path):
    """加载 CSV 数据文件"""
    print("正在加载数据...")
    try:
        df = pd.read_csv(file_path, encoding='gbk')  # 尝试使用 GBK 编码读取
    except UnicodeDecodeError:
        print("GBK 编码加载失败，尝试 UTF-8...")
        df = pd.read_csv(file_path, encoding='utf-8')  # 如果失败则使用 UTF-8
    print(f"数据加载完成：{df.shape[0]} 行，{df.shape[1]} 列")
    print("列名：", df.columns.tolist())
    print("前几行数据：\n", df.head().to_string())

    # 重命名 'frequency(LOG)' 为 'frequency'
    if 'frequency(LOG)' in df.columns:
        print("将 'frequency(LOG)' 重命名为 'frequency'")
        df = df.rename(columns={'frequency(LOG)': 'frequency'})
    elif 'frequency' not in df.columns:
        print("警告：未找到 'frequency' 列，添加默认值 0")
        df['frequency'] = 0

    return df

def detect_outliers(df, targets=['dielectric constant', 'dielectric loss']):
    """使用 Z 分数和领域特定规则检测异常点"""
    print("正在检测异常点...")
    df['is_outlier'] = False  # 初始化异常点标记列

    # 基于 Z 分数的异常点检测
    for target in targets:
        if target in df.columns:
            z_scores = np.abs((df[target] - df[target].mean()) / df[target].std())  # 计算 Z 分数
            df.loc[z_scores > 3, 'is_outlier'] = True  # Z 分数 > 3 标记为异常

    # 领域特定规则检测异常点
    conditions = [
        (df['dielectric constant'] > 100000),  # 介电常数过高
        (df['dielectric constant'] < 10),      # 介电常数过低
        (df['dielectric loss'] > 50),          # 介电损耗过高
        (df['dielectric loss'] < 0.001)       # 介电损耗过低
    ]
    for condition in conditions:
        df.loc[condition, 'is_outlier'] = True  # 满足任一条件标记为异常

    outlier_count = df['is_outlier'].sum()
    print(f"检测到 {outlier_count} 个异常点（占数据的 {outlier_count/len(df)*100:.2f}%）")
    return df

def remove_outliers(df):
    """从数据集中移除异常点"""
    print("正在移除异常点...")
    df_cleaned = df[~df['is_outlier']].drop(columns=['is_outlier'])  # 删除异常点并移除标记列
    print(f"移除异常点后的数据：{df_cleaned.shape[0]} 行，{df_cleaned.shape[1]} 列")
    return df_cleaned

def add_molecular_descriptors(df):
    """基于 components 添加分子描述符（简化示例）"""
    print("正在添加分子描述符...")
    # 简化的分子描述符：为每个组件映射假设的化学性质
    descriptor_map = {
        'TiO2Rutile': {'polarizability': 3.8, 'electronegativity': 2.5},  # 极化率和电负性
        'TiO2Anatase': {'polarizability': 3.9, 'electronegativity': 2.5},
        'Al2O3': {'polarizability': 2.9, 'electronegativity': 3.0},
        'SiO2': {'polarizability': 2.5, 'electronegativity': 3.5},
        'SnO2': {'polarizability': 4.0, 'electronegativity': 2.8},
        'Fe3O4': {'polarizability': 5.0, 'electronegativity': 2.2},
        'CoFe2O3': {'polarizability': 4.8, 'electronegativity': 2.3}
    }
    df['polarizability'] = df['components'].map(lambda x: descriptor_map[x]['polarizability'])  # 添加极化率
    df['electronegativity'] = df['components'].map(lambda x: descriptor_map[x]['electronegativity'])  # 添加电负性
    print("已添加分子描述符：极化率、电负性")
    return df

def apply_target_encoding(df, features, targets):
    """对分类特征（components）应用目标编码"""
    print("正在对 'components' 应用目标编码...")
    encoder = TargetEncoder(cols=['components'])  # 初始化目标编码器
    for target in targets:
        df[f'components_encoded_{target}'] = encoder.fit_transform(df['components'], df[target])  # 针对每个目标变量编码
    print(f"已添加目标编码特征：{', '.join([f'components_encoded_{t}' for t in targets])}")
    return df

def apply_rfe(df, features, targets, n_features_to_select=10):
    """使用递归特征消除（RFE）选择重要特征"""
    print("正在应用 RFE 进行特征选择...")
    X = df[features]
    selected_features = []
    
    for target in targets:
        y = df[target]
        estimator = RandomForestRegressor(random_state=42)  # 使用随机森林作为基础模型
        selector = RFE(estimator, n_features_to_select=n_features_to_select, step=1)  # 初始化 RFE
        selector = selector.fit(X, y)  # 执行 RFE
        selected = X.columns[selector.support_].tolist()  # 获取选中特征
        selected_features.extend(selected)
        print(f"为 {target} 选择的特征：{selected}")
    
    # 去重并保留特征顺序
    selected_features = list(dict.fromkeys(selected_features))
    print(f"RFE 后的最终选中特征：{selected_features}")
    return selected_features

def add_engineered_features(df, features):
    """添加工程特征以提升模型性能"""
    print("正在添加工程特征...")
    X = df[features].copy()
    X['log_frequency'] = np.log10(df['frequency'] + 1)  # 对数频率
    X['sqrt_frequency'] = np.sqrt(df['frequency'])  # 频率平方根
    X['frequency_reciprocal'] = 1 / (df['frequency'] + 1)  # 频率倒数
    
    # 添加频率范围的二值特征
    X['is_freq_verylow'] = (df['frequency'] < 10).astype(int)
    X['is_freq_low'] = ((df['frequency'] >= 10) & (df['frequency'] < 100)).astype(int)
    X['is_freq_mid'] = ((df['frequency'] >= 100) & (df['frequency'] < 1000)).astype(int)
    X['is_freq_high'] = ((df['frequency'] >= 1000) & (df['frequency'] < 10000)).astype(int)
    X['is_freq_veryhigh'] = (df['frequency'] >= 10000).astype(int)

    # 添加温度相关交互特征
    if 'temperature' in features:
        X['freq_temp'] = df['frequency'] * df['temperature']  # 频率与温度交互
        X['log_freq_temp'] = X['log_frequency'] * df['temperature']  # 对数频率与温度交互
        X['temp_squared'] = df['temperature'] ** 2  # 温度平方
        X['log_freq_squared'] = X['log_frequency'] ** 2  # 对数频率平方

    print(f"特征从 {len(features)} 个扩展到 {X.shape[1]} 个")
    return X

def preprocess_and_optimize_data(df):
    """预处理和优化数据集"""
    # 定义初始特征和目标变量
    features = ['vol', 'Bandgap', 'Ionic Radius', 'Preparation temperature', 
                'concentration', 'frequency', 'temperature']
    targets = ['dielectric constant', 'dielectric loss']

    # 检查缺失的特征和目标变量
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"缺失的特征：{missing_features}")
        features = [f for f in features if f in df.columns]

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        print(f"缺失的目标变量：{missing_targets}")
        targets = [t for t in targets if t in df.columns]

    # 检测并移除异常点
    df = detect_outliers(df, targets)
    df = remove_outliers(df)

    # 添加分子描述符
    df = add_molecular_descriptors(df)
    features.extend(['polarizability', 'electronegativity'])

    # 应用目标编码
    df = apply_target_encoding(df, features, targets)
    features.extend([f'components_encoded_{t}' for t in targets])

    # 应用 RFE 进行特征选择
    selected_features = apply_rfe(df, features, targets, n_features_to_select=10)

    # 添加工程特征
    X = add_engineered_features(df, selected_features)
    features = X.columns.tolist()

    # 合并特征和目标变量，生成优化数据集
    df_optimized = pd.concat([df[['components'] + targets], X], axis=1)

    # 保存优化数据集
    output_path = 'data_improve2.csv'
    df_optimized.to_csv(output_path, index=False, encoding='utf-8')
    print(f"优化数据集已保存至 {output_path}")
    
    # 保存特征列表
    with open("optimized_features.json", "w") as f:
        json.dump(features, f)
    print("已保存优化特征列表：", features)

    return df_optimized, features, targets

def main(file_path):
    """主函数，运行数据集优化流程"""
    df = load_data(file_path)  # 加载数据
    df_optimized, features, targets = preprocess_and_optimize_data(df)  # 优化数据
    print("\n数据集优化完成！")
    return df_optimized, features, targets

if __name__ == "__main__":
    file_path = r"D:\机器学习\给闫\data improve.csv"  # 输入文件路径
    df_optimized, features, targets = main(file_path)
