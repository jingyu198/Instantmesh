import numpy as np
import matplotlib.pyplot as plt

# 定义 BCE Loss 函数
def bce_loss(p, gt):
    epsilon = 1e-8  # 防止 log(0)
    p = np.clip(p, epsilon, 1 - epsilon)  # 将 p 限制在 (0, 1) 范围内
    return - (gt * np.log(p) + (1 - gt) * np.log(1 - p))

# Ground truth (gt) 值
gt = 0.49

# 定义预测概率范围 (p)
p_values = np.linspace(0, 1, 500)  # 在 [0, 1] 间取 500 个点

# 计算对应的 BCE Loss
loss_values = bce_loss(p_values, gt)

# 找到最小值及其对应位置
min_loss = np.min(loss_values)
min_p = p_values[np.argmin(loss_values)]

# 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(p_values, loss_values, label=f'BCE Loss (gt = {gt})', color='blue')
plt.title('Binary Cross-Entropy Loss', fontsize=16)
plt.xlabel('Predicted Probability (p)', fontsize=14)
plt.ylabel('BCE Loss', fontsize=14)
plt.ylim(0, 5)  # 设置 y 轴范围
plt.grid(alpha=0.3)

# 标记最小值
plt.scatter(min_p, min_loss, color='red', label=f'Min Loss = {min_loss:.4f} at p = {min_p:.2f}')
plt.annotate(f'Min Loss\n({min_p:.2f}, {min_loss:.4f})',
             xy=(min_p, min_loss), xytext=(min_p + 0.1, min_loss + 0.5),
             arrowprops=dict(facecolor='red', arrowstyle='->'),
             fontsize=12, color='black')

plt.legend(fontsize=12)

# 保存图像
output_path = "/home/gjy/jingyu/InstantMesh/src/bce_loss_with_min.png"  # 保存路径
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 设置分辨率和紧凑布局
plt.close()  # 关闭绘图窗口

print(f"图像已保存到 {output_path}")
