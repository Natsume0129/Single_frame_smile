import matplotlib.pyplot as plt
import numpy as np

# 设置绘图属性以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def draw_clean_projection():
    # 创建画布
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # 定义关键坐标点
    # v0: 坐标原点
    v0 = np.array([0, 0])
    
    # 1. 定义基向量 v1 和 v2
    # v1 (true) 向量，设定为一个锐角方向（约 40 度）
    angle_v1 = np.radians(40)
    v1_direction = np.array([np.cos(angle_v1), np.sin(angle_v1)])
    v1_end = v1_direction * 8
    
    # v2 (other) 向量，设定为水平方向
    v2_end = np.array([9, 0])
    
    # 2. 定义点 P (a specific smile) 的位置
    p = np.array([6, 2.5])
    
    # 3. 计算垂足坐标 (Perpendicular Projection)
    # 投影到 v2 (水平轴) 的垂足坐标为 (p.x, 0)
    p_on_v2 = np.array([p[0], 0])
    
    # 投影到 v1 的垂足公式: proj = (p·v / |v|^2) * v
    # 这里使用单位向量简化计算
    p_on_v1 = np.dot(p, v1_direction) * v1_direction
    
    # 4. 绘制向量射线 (带箭头)
    ax.annotate('', xy=v1_end, xytext=v0, arrowprops=dict(arrowstyle='->', lw=2))
    ax.annotate('', xy=v2_end, xytext=v0, arrowprops=dict(arrowstyle='->', lw=2))
    
    # 5. 绘制从点 P 向两条轴做的垂线 (虚线)
    # 注意：这里虽然没有直角标志，但计算逻辑确保了它们是垂直交汇的
    ax.plot([p[0], p_on_v1[0]], [p[1], p_on_v1[1]], color='black', linestyle='--', linewidth=1.2)
    ax.plot([p[0], p_on_v2[0]], [p[1], p_on_v2[1]], color='black', linestyle='--', linewidth=1.2)
    
    # 6. 添加文字标注
    # 标注原点
    ax.text(-0.4, -0.4, '$v_0$', fontsize=14)
    
    # 标注 v1 及其含义
    ax.text(v1_end[0]-0.2, v1_end[1]+0.4, '$v_1$', fontsize=14)
    ax.text(v1_end[0]+0.3, v1_end[1]+0.3, 'true', fontsize=16)
    
    # 标注 v2 及其含义
    ax.text(v2_end[0]-0.2, v2_end[1]-0.6, '$v_2$', fontsize=14)
    ax.text(v2_end[0]+0.2, v2_end[1], 'other', fontsize=16)
    
    # 标注特征点 "a specific smile"
    ax.text(p[0]+0.3, p[1]+0.1, 'a specific smile', fontsize=16)
    
    # 7. 图表样式配置
    # 设置等比例坐标轴，确保垂直关系在视觉上准确
    ax.set_aspect('equal')
    
    # 设置显示范围
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    
    # 隐藏所有坐标轴线、刻度和边框
    ax.axis('off')
    
    # 自动调整布局
    plt.tight_layout()
    
    # 显示图形
    plt.show()

if __name__ == "__main__":
    draw_clean_projection()