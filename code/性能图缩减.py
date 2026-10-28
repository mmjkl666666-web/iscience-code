import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import to_rgb, to_hex

# 加载CSV文件
data = pd.read_csv('D:/Desktop/机器学习/train/性能图 缩减.csv')

# 提取数据
components = data['component'].tolist()
constants = data['predicted_dielectric constant'].tolist()
losses = data['predicted_dielectric loss'].tolist()

# 按介电损耗排序
sorted_indices = np.argsort(losses)
components = [components[i] for i in sorted_indices]
constants = [constants[i] for i in sorted_indices]
losses = [losses[i] for i in sorted_indices]
N = len(data)

# 计算介电损耗的最小和最大值以创建边界
min_loss = min(losses)
max_loss = max(losses)
num_bins = min(10, N)
boundaries = np.linspace(min_loss, max_loss, num_bins + 1)

# 定义颜色条的颜色
colors = ['#FF0000', '#B2DEF3', '#D6F7F0', '#BC91BE', '#D1E7C1', '#00FFFF', '#FAC8AF', '#F05E5E', '#7B9FFB', '#FFC0CB'][
         :num_bins]
cmap = ListedColormap(colors)


# 生成浅色（用于间隔条纹）
def lighten_color(color, factor=0.5):
    """将颜色变浅，factor控制变浅程度（0到1，越大越浅）"""
    rgb = to_rgb(color)
    # 增加亮度：将RGB值向1（白色）靠拢
    light_rgb = tuple(c + (1 - c) * factor for c in rgb)
    return to_hex(light_rgb)


light_colors = [lighten_color(c, factor=0.5) for c in colors]

# 计算介电常数的最小和最大值
min_constant = min(constants)
max_constant = max(constants)

# 定义半径
R_max = 1.0
R_inner = 0.8 * R_max
R_outer = 1.0 * R_max
R_constant_max = 0.7 * R_max
R_label = (R_inner + R_outer) / 2

# 创建图形
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.set_xlim(-1.3, 1.3)
ax.set_ylim(-1.3, 1.3)
ax.axis('off')

# 绘制外圈扇形区域表示介电损耗，并添加浅色间隔条纹
for i in range(N):
    theta1 = i * 2 * np.pi / N
    theta2 = (i + 1) * 2 * np.pi / N
    loss = losses[i]
    bin_index = np.clip(np.digitize(loss, boundaries) - 1, 0, num_bins - 1)
    color = cmap(bin_index)

    # 绘制主扇形区域
    wedge = Wedge((0, 0), R_outer, theta1 * 180 / np.pi, theta2 * 180 / np.pi,
                  width=R_outer - R_inner, facecolor=color)
    ax.add_patch(wedge)

    # 计算标签位置
    theta_mid = (theta1 + theta2) / 2
    x_label = R_label * np.cos(theta_mid)
    y_label = R_label * np.sin(theta_mid)
    rotation = theta_mid * 180 / np.pi
    if 90 < rotation < 270:
        rotation += 180
    ax.text(x_label, y_label, components[i], fontsize=6, ha='center', va='center',
            rotation=rotation, rotation_mode='anchor')

    # 检查是否需要绘制浅色间隔条纹（与下一个扇形比较）
    if i < N - 1:  # 最后一个扇形不需要与下一个比较
        next_loss = losses[i + 1]
        next_bin_index = np.clip(np.digitize(next_loss, boundaries) - 1, 0, num_bins - 1)
        if bin_index == next_bin_index:  # 如果下一个扇形颜色相同
            # 绘制窄的浅色间隔条纹
            stripe_theta1 = theta2
            stripe_theta2 = theta2 + 0.02 * 2 * np.pi / N  # 条纹宽度为扇形宽度的2%
            light_color = light_colors[bin_index]
            stripe_wedge = Wedge((0, 0), R_outer, stripe_theta1 * 180 / np.pi, stripe_theta2 * 180 / np.pi,
                                 width=R_outer - R_inner, facecolor=light_color)
            ax.add_patch(stripe_wedge)

# 绘制内圈径向线条表示介电常数（由外向内，颜色与外圈一致）
for i in range(N):
    theta = (i + 0.5) * 2 * np.pi / N
    constant = constants[i]
    # 归一化介电常数以确定线条长度
    R_constant_i = (constant - min_constant) / (max_constant - min_constant) * R_constant_max
    # 线条从外圈 (R_outer) 向内到 (R_outer - R_constant_i)
    x_outer = R_outer * np.cos(theta)
    y_outer = R_outer * np.sin(theta)
    x_inner = (R_outer - R_constant_i) * np.cos(theta)
    y_inner = (R_outer - R_constant_i) * np.sin(theta)
    # 使用与外圈相同的颜色
    loss = losses[i]
    bin_index = np.clip(np.digitize(loss, boundaries) - 1, 0, num_bins - 1)
    color = cmap(bin_index)
    # 绘制线条，粗细与外圈一致
    ax.plot([x_outer, x_inner], [y_outer, y_inner], color=color, linewidth=2)

# 添加颜色条
norm = BoundaryNorm(boundaries, num_bins)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
cbar.set_label('预测介电损耗')
cbar.set_ticks(boundaries)

# 绘制各个颜色所代表的区域（图例）
legend_patches = []
for i in range(num_bins):
    label = f'[{boundaries[i]:.2f}, {boundaries[i + 1]:.2f})'
    patch = Patch(color=colors[i], label=label)
    legend_patches.append(patch)

# 添加图例，放置在右上角
ax.legend(handles=legend_patches, title='介电损耗范围', loc='upper right', fontsize=8)

plt.show()