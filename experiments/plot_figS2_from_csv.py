import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取真实数据
csv_path = "results_reproduction/figS3_W_A_per_epoch.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"找不到文件: {csv_path}")

df = pd.read_csv(csv_path)
epochs = df["epoch"].values
wa_means = df["W_A_mean"].values

# 2. 开始绘图（拉高一点画布，给底部文字留空间）
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1]})
fig.suptitle("Fig. S2 | Autonomous parasitic suppression under magnitude dominance", fontsize=14, y=1.02)

# === 左边面板 (a): Full training ===
ax1.plot(epochs, wa_means, color="#1f77b4", linewidth=1.5)
ax1.axhline(y=0.01, color='red', linestyle='--', label=r'\(W_{rest\_floor} \approx 0.01\)')
ax1.set_title("(a) Full training: rapid suppression + floor + CMA decay", fontsize=12, pad=12)
ax1.set_xlabel("Epoch", fontsize=11)
ax1.set_ylabel(r"\(W_{aux}(t)\) (epoch mean)", fontsize=11)
ax1.set_xlim(0, max(epochs) + 10)
ax1.set_ylim(0, max(wa_means)*1.1)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right', frameon=True)

# 标出峰值（稍微拉开一点距离，避免箭头贴线）
peak_idx = np.argmax(wa_means)
ax1.scatter(epochs[peak_idx], wa_means[peak_idx], color='red', s=50, zorder=5)
ax1.annotate(f"peak: {wa_means[peak_idx]:.3f}", xy=(epochs[peak_idx], wa_means[peak_idx]), 
             xytext=(epochs[peak_idx]+25, wa_means[peak_idx]), color='red', fontsize=11,
             arrowprops=dict(arrowstyle="->", color='red', lw=1))

# 加个小插图 (Zoom-in 60~400)，向下向左微调
ax_inset = ax1.inset_axes([0.45, 0.45, 0.45, 0.35])
inset_mask = epochs >= 60
inset_eps = epochs[inset_mask]
inset_wa = wa_means[inset_mask]
ax_inset.plot(inset_eps, inset_wa, color="#1f77b4", linewidth=1)
ax_inset.axhline(y=0.01, color='red', linestyle='--')
ax_inset.set_title("zoom: 60~400 (CMA decay)", fontsize=8, pad=2)
ax_inset.tick_params(labelsize=7)
ax_inset.grid(True, alpha=0.3)

# === 右边面板 (b): Zoom-in 2~10 ===
zoom_mask = (epochs >= 2) & (epochs <= 10)
zoom_eps = epochs[zoom_mask]
zoom_wa = wa_means[zoom_mask]
ax2.plot(zoom_eps, zoom_wa, color="red", marker='o', markersize=8, linewidth=2)

ax2.set_title("(b) Zoom: 84% drop within one epoch (2 → 3)", fontsize=12, pad=12)
ax2.set_xlabel("Epoch", fontsize=11)
ax2.set_ylabel(r"\(W_{aux}(t)\) (epoch mean)", fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1.5, 10.5)
# 稍微扩大Y轴上限，给底部文字留白
ax2.set_ylim(0, max(zoom_wa)*1.15)

# 精准标注各个点（彻底解决重叠）
for e, w in zip(zoom_eps, zoom_wa):
    if e == 2:
        ax2.text(e, w + 0.005, f"{w:.3f}\n(initial)", ha='center', va='bottom', fontsize=10)
    elif e == 3:
        drop_pct = (zoom_wa[0] - w) / zoom_wa[0] * 100
        # 标注放在左侧或右上方，避免压线
        ax2.text(e - 0.15, w + 0.005, f"{w:.3f}\n(-{drop_pct:.0f}%)", ha='right', va='bottom', fontsize=10, fontweight='bold')
    elif e == 5:
        ax2.text(e, w + 0.005, f"{w:.3f}\n(floor)", ha='center', va='bottom', fontsize=10)
    else:
        # Epoch 6~10 的数值统一标在上方
        ax2.text(e, w + 0.002, f"{w:.3f}", ha='center', va='bottom', fontsize=9, color='gray')

# 将底部那行长字彻底下移，放在数据点和X轴之间
ax2.text(7.5, 0.0015, r"floor: \(W_{rest} \times 2 = 0.01 \times 2 = 0.010\)", 
         fontsize=8.5, color='gray', ha='center', va='bottom')

plt.tight_layout()
out_path = "results_reproduction/figures/figS2_parasitic_suppression.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
# 增加 pad_inches 确保边缘元素不被截断
plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"✅ 排版优化后的 Fig. S2 已生成并保存至: {out_path}")
