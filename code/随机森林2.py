import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
except:
    pass


def load_data(file_path):
    """加载CSV数据文件"""
    print("正在加载数据...")
    df = pd.read_csv(file_path)
    print(f"数据加载完成，共 {df.shape[0]} 行，{df.shape[1]} 列")

    # 显示每列的数据类型
    print("\n数据类型检查:")
    print(df.dtypes)

    # 检查是否有非数值列应该被转换为数值
    print("\n检查可能的分类列或非数值列:")
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_values = df[col].unique()
            print(f"列 '{col}' 包含以下非数值数据: {unique_values[:5]} {'...' if len(unique_values) > 5 else ''}")

    return df


def preprocess_data(df):
    """数据预处理"""
    print("开始数据预处理...")

    # 首先处理可能的成分列
    if 'component' in df.columns:
        print("发现'component'列，处理为分类特征")
        # 保留原始成分列作为参考
        df_processed = df.copy()
        # 将成分转换为分类编码
        df_processed['component_code'] = pd.Categorical(df['component']).codes
    else:
        df_processed = df.copy()
        # 检查其他可能包含成分的列
        for col in df.columns:
            if df[col].dtype == 'object':
                print(f"将'{col}'列转换为分类编码")
                df_processed[f'{col}_code'] = pd.Categorical(df[col]).codes
                # 可以选择删除原始文本列，或保留用于后续分析
                # df_processed = df_processed.drop(col, axis=1)

    # 检查缺失值
    if df_processed.isnull().sum().sum() > 0:
        print(f"发现缺失值: \n{df_processed.isnull().sum()}")
        df_processed = df_processed.dropna()
        print(f"删除缺失值后，剩余 {df_processed.shape[0]} 行数据")
    else:
        print("数据中没有缺失值")

    # 定义特征和目标变量
    features = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                'sgnumber', 'Z', 'Preparation temperature', 'concentration',
                'frequency', 'temperature']

    # 如果有成分编码，添加到特征列表
    if 'component_code' in df_processed.columns:
        features.append('component_code')

    # 添加其他转换后的分类编码
    for col in df_processed.columns:
        if col.endswith('_code') and col != 'component_code':
            features.append(col)

    targets = ['dielectric constant', 'dielectric loss']

    # 检查所有特征是否都在数据集中
    missing_features = [f for f in features if f not in df_processed.columns]
    if missing_features:
        print(f"警告: 以下特征在数据集中缺失: {missing_features}")
        features = [f for f in features if f in df_processed.columns]

    missing_targets = [t for t in targets if t not in df_processed.columns]
    if missing_targets:
        print(f"警告: 以下目标变量在数据集中缺失: {missing_targets}")
        targets = [t for t in targets if t in df_processed.columns]

    if not features or not targets:
        raise ValueError("没有足够的特征或目标变量来建立模型")

    # 添加特征工程
    df_processed = feature_engineering(df_processed, features)

    # 确保目标变量是浮点数
    for target in targets:
        if target in df_processed.columns:
            try:
                df_processed[target] = df_processed[target].astype(float)
            except ValueError as e:
                print(f"警告: 无法将'{target}'转换为浮点数: {e}")
                print(f"{target}列的样本值: {df_processed[target].head()}")
                # 可能需要进一步处理或排除此目标

    # 准备特征和目标变量
    # 获取所有可用特征（包括新工程的特征）
    all_features = [col for col in df_processed.columns
                    if col not in targets
                    and df_processed[col].dtype in ['int64', 'float64', 'bool']]

    # 检查特征列是否都是数值型
    for col in all_features:
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            print(f"警告: 特征'{col}'不是数值类型，尝试转换")
            try:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                # 填充转换后可能产生的NaN值
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            except Exception as e:
                print(f"无法转换'{col}'为数值: {e}，从特征列表中移除")
                all_features.remove(col)

    X = df_processed[all_features]
    y = df_processed[targets]

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 转换回DataFrame以保留列名
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print("数据预处理完成")
    print(f"最终使用的特征: {all_features}")
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, all_features, targets, scaler


def feature_engineering(df, existing_features):
    """特征工程，添加新特征以提高模型性能"""
    df_new = df.copy()

    # 确保所有基础特征都是数值型
    numeric_features = []
    for feat in existing_features:
        if feat in df.columns:
            try:
                df_new[feat] = pd.to_numeric(df_new[feat], errors='coerce')
                if df_new[feat].isnull().sum() > 0:
                    # 如果有NaN值，用均值填充
                    df_new[feat] = df_new[feat].fillna(df_new[feat].mean())
                numeric_features.append(feat)
            except:
                print(f"警告: 特征'{feat}'无法转换为数值型，跳过相关特征工程")

    # 只使用可以成功转换为数值的特征进行特征工程
    if 'a' in numeric_features and 'b' in numeric_features and 'c' in numeric_features:
        # 1. 添加晶格参数比率
        df_new['a_to_b_ratio'] = df_new['a'] / df_new['b'].replace(0, np.nan).fillna(1)
        df_new['a_to_c_ratio'] = df_new['a'] / df_new['c'].replace(0, np.nan).fillna(1)
        df_new['b_to_c_ratio'] = df_new['b'] / df_new['c'].replace(0, np.nan).fillna(1)

    if 'alpha' in numeric_features and 'beta' in numeric_features and 'gamma' in numeric_features:
        # 2. 添加角度特征
        df_new['angle_sum'] = df_new['alpha'] + df_new['beta'] + df_new['gamma']
        df_new['angle_product'] = df_new['alpha'] * df_new['beta'] * df_new['gamma']

    if 'Z' in numeric_features and 'vol' in numeric_features:
        # 3. 添加密度相关特征
        df_new['density_factor'] = df_new['Z'] / df_new['vol'].replace(0, np.nan).fillna(1)

    if 'temperature' in numeric_features and 'frequency' in numeric_features:
        # 4. 添加温度与频率交互特征
        if df_new['frequency'].min() > 0:
            df_new['temp_freq_interaction'] = df_new['temperature'] * np.log1p(df_new['frequency'])

    if 'sgnumber' in numeric_features:
        # 5. 添加晶体系统分类
        def get_crystal_system(sgnumber):
            try:
                sgnumber = float(sgnumber)
                if 1 <= sgnumber <= 2:
                    return 1  # 三斜晶系
                elif 3 <= sgnumber <= 15:
                    return 2  # 单斜晶系
                elif 16 <= sgnumber <= 74:
                    return 3  # 正交晶系
                elif 75 <= sgnumber <= 142:
                    return 4  # 四方晶系
                elif 143 <= sgnumber <= 167:
                    return 5  # 三方晶系
                elif 168 <= sgnumber <= 194:
                    return 6  # 六方晶系
                elif 195 <= sgnumber <= 230:
                    return 7  # 立方晶系
                else:
                    return 0  # 未知
            except:
                return 0  # 无法转换为数值

        df_new['crystal_system'] = df_new['sgnumber'].apply(get_crystal_system)

    if 'frequency' in numeric_features:
        # 6. 对数转换特征
        # 首先确保数据为正
        if df_new['frequency'].min() > 0:
            df_new['log_frequency'] = np.log(df_new['frequency'])

    if 'Preparation temperature' in numeric_features and 'temperature' in numeric_features:
        # 7. 添加温度相关特征
        df_new['temp_difference'] = df_new['Preparation temperature'] - df_new['temperature']

    if all(feat in numeric_features for feat in ['alpha', 'beta', 'gamma', 'a', 'b', 'c']):
        # 8. 添加晶胞对称性特征
        df_new['is_cubic'] = ((abs(df_new['alpha'] - 90) < 1e-6) &
                              (abs(df_new['beta'] - 90) < 1e-6) &
                              (abs(df_new['gamma'] - 90) < 1e-6) &
                              (abs(df_new['a'] - df_new['b']) < 1e-6) &
                              (abs(df_new['a'] - df_new['c']) < 1e-6)).astype(int)

    if 'concentration' in numeric_features and 'Z' in numeric_features:
        # 9. 添加浓度与Z的交互项
        df_new['conc_Z_interaction'] = df_new['concentration'] * df_new['Z']

    # 10. 添加多项式特征
    for feat in ['a', 'b', 'c', 'vol']:
        if feat in numeric_features:
            df_new[f'{feat}_squared'] = df_new[feat] ** 2

    return df_new


def train_random_forest_models(X_train, y_train, X_test, y_test, targets):
    """训练随机森林模型"""
    models = {}
    predictions = {}
    metrics = {}

    for target in targets:
        print(f"\n训练 {target} 的随机森林模型...")

        # 设置模型参数
        params = {
            'n_estimators': 200,  # 树的数量
            'max_depth': 15,  # 树的最大深度
            'min_samples_split': 5,  # 内部节点需要的最小样本数
            'min_samples_leaf': 2,  # 叶节点需要的最小样本数
            'max_features': 'sqrt',  # 考虑的最大特征数
            'bootstrap': True,  # 使用bootstrap样本
            'random_state': 42
        }

        # 创建并训练模型
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train[target])

        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # 计算评估指标
        train_mse = mean_squared_error(y_train[target], y_pred_train)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train[target], y_pred_train)
        train_r2 = r2_score(y_train[target], y_pred_train)

        test_mse = mean_squared_error(y_test[target], y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test[target], y_pred_test)
        test_r2 = r2_score(y_test[target], y_pred_test)

        print(f"训练集评估指标:")
        print(f"MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, R²: {train_r2:.4f}")
        print(f"测试集评估指标:")
        print(f"MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

        # 存储模型和预测结果
        models[target] = model
        predictions[target] = {
            'train_true': y_train[target],
            'train_pred': y_pred_train,
            'test_true': y_test[target],
            'test_pred': y_pred_test
        }
        metrics[target] = {
            'train': {'mse': train_mse, 'rmse': train_rmse, 'mae': train_mae, 'r2': train_r2},
            'test': {'mse': test_mse, 'rmse': test_rmse, 'mae': test_mae, 'r2': test_r2}
        }

    return models, predictions, metrics


def perform_k_fold_validation(X, y, targets, k=5):
    """执行K折交叉验证"""
    cv_results = {}

    for target in targets:
        print(f"\n对 {target} 进行 {k} 折交叉验证...")

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y[target].iloc[train_idx]
            y_test_fold = y[target].iloc[test_idx]

            # 标准化
            scaler = StandardScaler()
            X_train_fold_scaled = scaler.fit_transform(X_train_fold)
            X_test_fold_scaled = scaler.transform(X_test_fold)

            # 创建DataFrame以保留列名
            X_train_fold_scaled = pd.DataFrame(X_train_fold_scaled, columns=X_train_fold.columns)
            X_test_fold_scaled = pd.DataFrame(X_test_fold_scaled, columns=X_test_fold.columns)

            # 训练模型
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=42
            )

            model.fit(X_train_fold_scaled, y_train_fold)

            # 预测
            y_pred = model.predict(X_test_fold_scaled)

            # 评估
            mse = mean_squared_error(y_test_fold, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_fold, y_pred)

            rmse_scores.append(rmse)
            r2_scores.append(r2)

            print(f"  折 {fold + 1} - RMSE: {rmse:.4f}, R²: {r2:.4f}")

        avg_rmse = np.mean(rmse_scores)
        avg_r2 = np.mean(r2_scores)

        print(f"{k} 折交叉验证平均指标:")
        print(f"平均 RMSE: {avg_rmse:.4f}, 平均 R²: {avg_r2:.4f}")

        cv_results[target] = {
            'rmse_scores': rmse_scores,
            'r2_scores': r2_scores,
            'avg_rmse': avg_rmse,
            'avg_r2': avg_r2
        }

    return cv_results


def plot_feature_importance(models, features, targets):
    """绘制特征重要性图"""
    for target in targets:
        # 获取特征重要性
        importances = models[target].feature_importances_
        indices = np.argsort(importances)[::-1]

        # 只展示前15个最重要的特征
        n_features_to_show = min(15, len(features))
        top_indices = indices[:n_features_to_show]

        plt.figure(figsize=(12, 8))
        plt.title(f'{target} 的特征重要性 (前{n_features_to_show}项)')
        plt.bar(range(n_features_to_show), importances[top_indices])
        plt.xticks(range(n_features_to_show), [features[i] for i in top_indices], rotation=90)
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{target.replace(" ", "_")}.png', dpi=300)
        plt.close()

        # 打印特征重要性
        print(f"\n{target} 的前{n_features_to_show}个重要特征:")
        for i in range(n_features_to_show):
            print(f"  {features[top_indices[i]]}: {importances[top_indices[i]]:.4f}")


def visualize_results(predictions, metrics, targets):
    """可视化预测结果"""
    for target in targets:
        target_name = target.replace(" ", "_")

        # 1. 预测值与实际值的散点图 (测试集)
        plt.figure(figsize=(10, 8))
        plt.scatter(predictions[target]['test_true'], predictions[target]['test_pred'], alpha=0.7)

        # 添加理想的预测线
        min_val = min(predictions[target]['test_true'].min(), predictions[target]['test_pred'].min())
        max_val = max(predictions[target]['test_true'].max(), predictions[target]['test_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')

        plt.xlabel(f'实际 {target}')
        plt.ylabel(f'预测 {target}')
        plt.title(f'{target} 预测值与实际值对比 (R² = {metrics[target]["test"]["r2"]:.4f})')

        # 添加文本框显示指标
        textstr = '\n'.join((
            f'测试集指标:',
            f'RMSE = {metrics[target]["test"]["rmse"]:.4f}',
            f'MAE = {metrics[target]["test"]["mae"]:.4f}',
            f'R² = {metrics[target]["test"]["r2"]:.4f}'
        ))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'scatter_plot_{target_name}.png', dpi=300)
        plt.close()

        # 2. 残差图
        plt.figure(figsize=(10, 8))
        residuals = predictions[target]['test_true'] - predictions[target]['test_pred']
        plt.scatter(predictions[target]['test_pred'], residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'预测 {target}')
        plt.ylabel('残差')
        plt.title(f'{target} 残差图')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'residual_plot_{target_name}.png', dpi=300)
        plt.close()

        # 3. 残差分布图
        plt.figure(figsize=(10, 8))
        sns.histplot(residuals, kde=True)
        plt.xlabel('残差')
        plt.ylabel('频率')
        plt.title(f'{target} 残差分布')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f'residual_dist_{target_name}.png', dpi=300)
        plt.close()

        # 4. 相关性图 (预测值 vs 实际值)
        plt.figure(figsize=(10, 8))
        corr_data = pd.DataFrame({
            'Actual': predictions[target]['test_true'],
            'Predicted': predictions[target]['test_pred']
        })
        sns.heatmap(corr_data.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f'{target} 预测值与实际值的相关性')
        plt.tight_layout()
        plt.savefig(f'correlation_{target_name}.png', dpi=300)
        plt.close()


def main(file_path):
    """主函数"""
    # 1. 加载数据
    df = load_data(file_path)

    # 2. 数据预处理
    X, y, X_train, X_test, y_train, y_test, features, targets, scaler = preprocess_data(df)

    # 3. 训练随机森林模型
    models, predictions, metrics = train_random_forest_models(X_train, y_train, X_test, y_test, targets)

    # 4. K折交叉验证
    cv_results = perform_k_fold_validation(X, y, targets, k=5)

    # 5. 特征重要性可视化
    plot_feature_importance(models, features, targets)

    # 6. 结果可视化
    visualize_results(predictions, metrics, targets)

    print("\n分析完成！所有结果已保存为图表。")
    return models, predictions, metrics, cv_results


if __name__ == "__main__":
    # 替换为你的CSV文件路径
    file_path = "E:/iscience/end/second/code/代码/daima/data improve.csv"  # 请修改为你的文件路径
    models, predictions, metrics, cv_results = main(file_path)