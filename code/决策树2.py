import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
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
    return df


def preprocess_data(df):
    """数据预处理"""
    print("开始数据预处理...")

    # 检查缺失值
    if df.isnull().sum().sum() > 0:
        print(f"发现缺失值: \n{df.isnull().sum()}")
        df = df.dropna()
        print(f"删除缺失值后，剩余 {df.shape[0]} 行数据")
    else:
        print("数据中没有缺失值")

    # 定义特征和目标变量
    features = ['a', 'b', 'c', 'alpha', 'beta', 'gamma', 'vol',
                'sgnumber', 'Z', 'Preparation temperature', 'concentration',
                'frequency', 'temperature']

    targets = ['dielectric constant', 'dielectric loss']

    # 检查所有特征是否都在数据集中
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"警告: 以下特征在数据集中缺失: {missing_features}")
        features = [f for f in features if f in df.columns]

    missing_targets = [t for t in targets if t not in df.columns]
    if missing_targets:
        print(f"警告: 以下目标变量在数据集中缺失: {missing_targets}")
        targets = [t for t in targets if t in df.columns]

    if not features or not targets:
        raise ValueError("没有足够的特征或目标变量来建立模型")

    # 添加特征工程
    df_processed = feature_engineering(df)

    # 准备数据
    X = df_processed[features]
    y = df_processed[targets]

    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("数据预处理完成")
    return X, y, X_train_scaled, X_test_scaled, y_train, y_test, features, targets, scaler


def feature_engineering(df):
    """特征工程，添加新特征以提高模型性能"""
    df_new = df.copy()

    # 1. 添加晶格参数比率
    df_new['a_to_b_ratio'] = df['a'] / df['b']
    df_new['a_to_c_ratio'] = df['a'] / df['c']
    df_new['b_to_c_ratio'] = df['b'] / df['c']

    # 2. 添加角度特征
    df_new['angle_sum'] = df['alpha'] + df['beta'] + df['gamma']
    df_new['angle_product'] = df['alpha'] * df['beta'] * df['gamma']

    # 3. 添加密度相关特征
    df_new['density_factor'] = df['Z'] / df['vol']

    # 4. 添加温度与频率交互特征
    df_new['temp_freq_interaction'] = df['temperature'] * np.log1p(df['frequency'])

    # 5. 添加晶体系统分类
    def get_crystal_system(sgnumber):
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

    df_new['crystal_system'] = df['sgnumber'].apply(get_crystal_system)

    # 6. 对数转换特征
    # 首先确保数据为正
    if df['frequency'].min() > 0:
        df_new['log_frequency'] = np.log(df['frequency'])

    # 7. 对介电常数进行数据类型转换确保是浮点数
    if 'dielectric constant' in df.columns:
        df_new['dielectric constant'] = df['dielectric constant'].astype(float)

    return df_new


def train_decision_tree_models(X_train, y_train, X_test, y_test, targets):
    """训练决策树模型"""
    models = {}
    predictions = {}
    metrics = {}

    for target in targets:
        print(f"\n训练 {target} 的决策树模型...")

        # 设置模型参数
        params = {
            'max_depth': 10,  # 树的最大深度
            'min_samples_split': 5,  # 内部节点需要的最小样本数
            'min_samples_leaf': 2,  # 叶节点需要的最小样本数
            'random_state': 42
        }

        # 创建并训练模型
        model = DecisionTreeRegressor(**params)
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

        # 准备数据
        X_array = X.values
        y_array = y[target].values

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        rmse_scores = []
        r2_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X_array)):
            X_train_fold, X_test_fold = X_array[train_idx], X_array[test_idx]
            y_train_fold, y_test_fold = y_array[train_idx], y_array[test_idx]

            # 训练模型
            model = DecisionTreeRegressor(
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )

            model.fit(X_train_fold, y_train_fold)

            # 预测
            y_pred = model.predict(X_test_fold)

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
        plt.figure(figsize=(12, 8))

        # 获取特征重要性
        importances = models[target].feature_importances_
        indices = np.argsort(importances)[::-1]

        # 绘制条形图
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=90)
        plt.xlabel('特征')
        plt.ylabel('重要性')
        plt.title(f'{target} 的特征重要性')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{target.replace(" ", "_")}.png', dpi=300)
        plt.close()


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

    # 3. 训练决策树模型
    models, predictions, metrics = train_decision_tree_models(X_train, y_train, X_test, y_test, targets)

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
    file_path = "D:/Desktop/机器学习/train/your_data1.csv"  # 请修改为你的文件路径
    models, predictions, metrics, cv_results = main(file_path)