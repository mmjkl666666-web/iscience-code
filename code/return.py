import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize

# 强制使用交互式后端（重要！）
matplotlib.use('TkAgg')  # 也可以尝试 'Qt5Agg' 或 'WebAgg'

# 读取CSV文件
try:
    df = pd.read_csv(r'D:\Desktop\机器学习\train\your_data1 - 热图.csv')
except Exception as e:
    print(f"文件读取失败: {str(e)}")
    exit()

# 计算相关性矩阵
corr_matrix = df.corr()
variables = corr_matrix.columns
n_vars = len(variables)

# 生成网格数据
x, y = np.meshgrid(np.arange(n_vars), np.arange(n_vars))
x = x.flatten()
y = y.flatten()
z = np.zeros_like(x)

# 数据转换
corr_values = corr_matrix.values.flatten()
height = (corr_values + 1) / 2

# 可视化参数
bar_dx = bar_dy = 0.8
bar_dz = height

# 颜色映射
cmap = plt.get_cmap('coolwarm')
norm = Normalize(vmin=-1, vmax=1)
colors = cmap(norm(corr_values))

# 创建3D画布
fig = plt.figure(figsize=(14, 12))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D柱状图（启用深度着色）
ax.bar3d(x, y, z, bar_dx, bar_dy, bar_dz,
         color=colors,
         shade=True,         # 启用阴影
         edgecolor='k',      # 黑色边框
         linewidth=0.3,      # 边框粗细
         alpha=0.8)          # 增加透明度

# 颜色条设置
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.12, shrink=0.6)
cbar.set_label('Correlation Coefficient', rotation=270, labelpad=25)
cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

# 坐标轴增强设置
ax.set_xticks(np.arange(n_vars))
ax.set_xticklabels(variables,
                  rotation=45,
                  ha='right',
                  va='top',    # 垂直对齐方式
                  fontsize=10,
                  fontfamily='sans-serif')

ax.set_yticks(np.arange(n_vars))
ax.set_yticklabels(variables,
                  rotation=-30,
                  ha='left',
                  va='bottom',
                  fontsize=10,
                  fontfamily='sans-serif')

ax.set_zlabel('Normalized Correlation',
             rotation=90,
             labelpad=15,
             fontsize=12)

# 视角控制参数（可调节）
ax.view_init(elev=28, azim=-45)  # 初始视角
ax.dist = 10                    # 观察距离

# 启用网格线
ax.xaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "#444444"})
ax.yaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "#444444"})
ax.zaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "#444444"})

# 标题设置
ax.set_title("Interactive 3D Correlation Matrix\n(Rotate with Mouse)",
            fontsize=16,
            pad=30,
            fontweight='bold')

# 布局优化
plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

# 启用交互模式（关键！）
plt.ion()  # 打开交互模式
plt.show(block=True)  # 保持窗口打开

# 运行后可以用以下方式控制：
# - 左键拖动旋转
# - 右键拖动缩放
# - 滚轮缩放
# - ctrl+左键拖动平移