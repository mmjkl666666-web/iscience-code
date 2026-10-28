import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小 - 显著增大所有字体
plt.rcParams.update({
    'font.size': 18,  # 全局字体大小增大
    'axes.titlesize': 24,  # 标题字体大小
    'axes.labelsize': 22,  # 坐标轴标签字体大小
    'xtick.labelsize': 20,  # X轴刻度字体大小
    'ytick.labelsize': 20,  # Y轴刻度字体大小
    'legend.fontsize': 18  # 图例字体大小
})

# 读取CSV文件
file_path = r"D:\Desktop\机器学习\train\Supplementary Material 1..csv"
data = pd.read_csv(file_path)

# 计算相关系数矩阵
corr_matrix = data.corr()

# 创建更大的图形
plt.figure(figsize=(26, 24))  # 进一步增大图形尺寸

# 绘制热图，显著增大字体
heatmap = sns.heatmap(
    corr_matrix,
    annot=True,  # 显示数值
    fmt=".2f",  # 数值格式（保留两位小数）
    annot_kws={"size": 24},  # 注释字体大小增大
    cmap='coolwarm',  # 颜色映射
    vmin=-1,  # 最小值范围
    vmax=1,  # 最大值范围
    linewidths=0.5,  # 单元格间的线宽
    square=True,  # 单元格为正方形
    cbar_kws={
        "shrink": 0.82,  # 调整颜色条大小
        "aspect": 50  # 颜色条纵横比
    }
)

# 设置标题
plt.title('Feature Correlation Heatmap', fontsize=26, pad=28)

# 旋转X轴标签并调整位置 - 显著增大标签字体
plt.xticks(rotation=45, ha='right', fontsize=22, rotation_mode='anchor')
plt.yticks(fontsize=22)

# 添加网格线增强可读性（可选）
heatmap.grid(False)  # 关闭默认网格线

# 调整颜色条标签字体（修复labelsize问题）
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)  # 设置刻度标签字体大小
cbar.set_label('Correlation', size=22)  # 设置颜色条标题字体大小

# 简化可能过长的特征名（可选）
max_name_length = 20
if any(len(col) > max_name_length for col in corr_matrix.columns):
    corr_matrix.columns = [col[:max_name_length - 3] + '..' if len(col) > max_name_length else col
                           for col in corr_matrix.columns]
    corr_matrix.index = [col[:max_name_length - 3] + '..' if len(col) > max_name_length else col
                         for col in corr_matrix.index]

    # 重新绘制热图（如果需要）
    plt.close()
    plt.figure(figsize=(26, 24))
    heatmap = sns.heatmap(
        corr_matrix,
        annot=True, annot_kws={"size": 18},
        fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
        linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.82, "aspect": 50}
    )
    plt.title('Feature Correlation Heatmap', fontsize=26, pad=28)
    plt.xticks(rotation=45, ha='right', fontsize=22, rotation_mode='anchor')
    plt.yticks(fontsize=22)
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Correlation', size=22)

# 紧凑布局
plt.tight_layout()

# 保存高分辨率图像
plt.savefig('correlation_heatmap_large.png', dpi=300, bbox_inches='tight')  # 添加文件名并保存

# 显示图形
plt.show()