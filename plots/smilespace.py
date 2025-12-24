import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar

# --- 定义曲线函数 ---

# 1. 中心黑色垂直线 (0, 0, z), z 范围 [0, 2]
def curve_black(t):
    # t 范围 [0, 1] 映射到 z 范围 [0, 2]
    return np.array([0, 0, 2 * t])

# 2. 红色抛物线 (t, 0, 2t^2)
def curve_red(t):
    return np.array([t, 0, 2 * (t**2)])

# 3. 绿色正弦线 (-0.7t, 0.7t + 0.1*sin(5t), 2t)
def curve_green(t):
    return np.array([-0.7 * t, 0.7 * t + 0.1 * np.sin(5 * t), 2 * t])

# 4. 蓝色直线 (-0.7t, -0.7t, 2t)
def curve_blue(t):
    return np.array([-0.7 * t, -0.7 * t, 2 * t])

# --- 严谨投影核心逻辑 ---

def get_rigorous_projection(point, curve_func):
    # 定义目标函数：点到曲线上某一点的距离平方
    # 使用平方是为了避免开方运算，加快优化速度且结果一致
    def distance_sq(t):
        curve_pt = curve_func(t)
        return np.sum((curve_pt - point)**2)
    
    # 在给定的参数范围 [0, 1] 内寻找最小值
    # minimize_scalar 专门用于单变量有界优化，非常精确
    res = minimize_scalar(distance_sq, bounds=(0, 1), method='bounded')
    
    # 返回曲线上对应的最优坐标点
    return curve_func(res.x)

# --- 绘图准备 ---

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 设定绘制曲线用的精细采样点（仅用于画线）
t_plot = np.linspace(0, 1, 200)

# 1. 绘制中心黑线
pts_black = np.array([curve_black(t) for t in t_plot])
ax.plot(pts_black[:,0], pts_black[:,1], pts_black[:,2], color='black', linewidth=2)
ax.text(0, 0, -0.1, "v0", color='black', fontweight='bold')
ax.text(0, 0, 1.6, "v1", color='black', fontweight='bold')

# 2. 绘制红色抛物线
pts_red = np.array([curve_red(t) for t in t_plot])
ax.plot(pts_red[:,0], pts_red[:,1], pts_red[:,2], color='red')
ax.text(pts_red[-1,0], pts_red[-1,1], pts_red[-1,2]+0.1, "v1", color='red')

# 3. 绘制绿色正弦线
pts_green = np.array([curve_green(t) for t in t_plot])
ax.plot(pts_green[:,0], pts_green[:,1], pts_green[:,2], color='green')
ax.text(pts_green[-1,0], pts_green[-1,1], pts_green[-1,2]+0.1, "v2", color='green')

# 4. 绘制蓝色直线
pts_blue = np.array([curve_blue(t) for t in t_plot])
ax.plot(pts_blue[:,0], pts_blue[:,1], pts_blue[:,2], color='blue')
ax.text(pts_blue[-1,0], pts_blue[-1,1], pts_blue[-1,2]+0.1, "v3", color='blue')

# --- 随机点生成 ---

# 随机生成一个内部点 (根据此前范围微调)
rand_point = np.array([
    np.random.uniform(-0.2, 0.2),
    np.random.uniform(-0.2, 0.2),
    np.random.uniform(0.6, 1.1)
])

# 绘制随机点
ax.scatter(*rand_point, color='magenta', s=120, zorder=10)
ax.text(rand_point[0], rand_point[1], rand_point[2]+0.15, "a random smile", 
        color='magenta', fontsize=12, ha='center', fontweight='bold')

# --- 执行严谨投影并画虚线 ---
curves = [
    (curve_black, 'black'),
    (curve_red, 'red'),
    (curve_green, 'green'),
    (curve_blue, 'blue')
]

for func, col in curves:
    # 调用优化器获取精确的投影点 Q
    proj_point = get_rigorous_projection(rand_point, func)
    
    # 绘制从 P 到 Q 的虚线
    ax.plot([rand_point[0], proj_point[0]],
            [rand_point[1], proj_point[1]],
            [rand_point[2], proj_point[2]],
            color=col, linestyle='--', linewidth=1.5, alpha=0.8)

# --- 图形修饰 ---
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Smile Space for demonstration', fontsize=16)
ax.view_init(elev=20, azim=35)

# 保持比例尺一致防止视觉扭曲
ax.set_box_aspect([1,1,1]) 

plt.show()