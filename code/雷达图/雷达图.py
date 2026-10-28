import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 加载数据集
df = pd.read_csv('D:/Desktop/机器学习/train/氧化物数据集 - 副本_predicted.csv')

# 提取相关列并处理缺失值
df = df[['component', 'predicted_dielectric constant']].dropna()

# 按 predicted_dielectric constant 排序
df_sorted = df.sort_values(by='predicted_dielectric constant', ascending=False)

# 重置索引以便纵坐标使用
df_sorted.reset_index(drop=True, inplace=True)

# 创建瀑布图，调整图形高度以适应所有标签
fig, ax = plt.subplots(figsize=(10, 40))  # 增加高度以容纳所有标签

# 绘制条形图
bars = ax.barh(df_sorted.index, df_sorted['predicted_dielectric constant'], color='skyblue')

# 设置纵坐标标签，显示所有 component
ax.set_yticks(range(len(df_sorted)))
ax.set_yticklabels(df_sorted['component'], fontsize=6)  # 减小字体以减少重叠

# 设置标题和标签
ax.set_title('Predicted Dielectric Constants Across Components')
ax.set_xlabel('Predicted Dielectric Constant')
ax.set_ylabel('Component')

# 添加网格线
ax.grid(True, axis='x', linestyle='--', alpha=0.7)

# 调整布局以防止标签被裁剪
plt.tight_layout()

# 保存图形
plt.savefig('waterfall_plot_all_labels.png')

# 显示图形
plt.show()