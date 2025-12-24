import matplotlib.pyplot as plt

# 设置绘图风格
plt.style.use('seaborn-v0_8') # 或者使用 'ggplot'

# 1. 准备数据 (根据你提供的日志提取)
epochs = list(range(1, 26))

# 训练集指标
train_loss = [0.3916, 0.1934, 0.1213, 0.1009, 0.0719, 0.0984, 0.0989, 0.0558, 0.0596, 0.0395, 
              0.0489, 0.0691, 0.0343, 0.0279, 0.0264, 0.0285, 0.0244, 0.0338, 0.0216, 0.0277, 
              0.0354, 0.0243, 0.0224, 0.0333, 0.0232]
train_acc = [0.822, 0.912, 0.956, 0.964, 0.969, 0.957, 0.960, 0.980, 0.977, 0.987, 
             0.984, 0.974, 0.987, 0.990, 0.990, 0.987, 0.990, 0.984, 0.993, 0.989, 
             0.989, 0.993, 0.989, 0.989, 0.994]
train_f1 = [0.822, 0.917, 0.954, 0.966, 0.970, 0.953, 0.963, 0.980, 0.979, 0.987, 
            0.984, 0.973, 0.987, 0.990, 0.990, 0.987, 0.990, 0.985, 0.993, 0.989, 
            0.989, 0.993, 0.988, 0.989, 0.994]

# 验证集指标
val_loss = [0.3895, 0.2731, 0.1920, 0.2190, 0.1616, 0.4844, 0.3808, 0.2071, 0.2064, 0.2238, 
            0.2421, 0.2197, 0.1670, 0.2027, 0.2894, 0.1697, 0.1866, 0.2872, 0.1363, 0.1142, 
            0.2196, 0.1764, 0.1955, 0.3436, 0.1849]
val_acc = [0.792, 0.896, 0.935, 0.935, 0.922, 0.740, 0.792, 0.870, 0.922, 0.909, 
           0.896, 0.883, 0.922, 0.948, 0.883, 0.948, 0.922, 0.896, 0.909, 0.974, 
           0.883, 0.935, 0.909, 0.870, 0.909]
val_f1 = [0.789, 0.871, 0.921, 0.918, 0.893, 0.750, 0.667, 0.839, 0.889, 0.868, 
          0.871, 0.866, 0.900, 0.929, 0.824, 0.933, 0.900, 0.875, 0.873, 0.966, 
          0.824, 0.915, 0.885, 0.800, 0.885]

# 2. 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 绘制 Loss 曲线
# 
ax1.plot(epochs, train_loss, label='Train Loss', color='#1f77b4', lw=2)
ax1.plot(epochs, val_loss, label='Val Loss', color='#ff7f0e', linestyle='--', lw=2)
ax1.set_title('Training & Validation Loss', fontsize=14)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# 绘制 Accuracy 和 F1 曲线
# 
ax2.plot(epochs, train_acc, label='Train Acc', color='#2ca02c', lw=1.5)
ax2.plot(epochs, val_acc, label='Val Acc', color='#d62728', linestyle='--', lw=1.5)
ax2.plot(epochs, val_f1, label='Val F1', color='#9467bd', linestyle=':', lw=2)
ax2.set_title('Performance Metrics (Acc & F1)', fontsize=14)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Score')
ax2.legend()
ax2.grid(True)

# 调整布局并展示
plt.tight_layout()
plt.show()