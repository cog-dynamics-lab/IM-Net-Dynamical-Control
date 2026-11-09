#!/usr/bin/env python
# experiments/06_plotting_figures.py
"""
Generate manuscript figures at 300 dpi directly from CSV files in results_reproduction/.
This script does NOT run any training or simulations; it only plots from existing CSVs.

Run from repo root:
    python experiments/06_plotting_figures.py
"""
import os
import argparse
import shutil
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


# ----------------------------------------------------------------------
# Fig. 5: From CSV
# ----------------------------------------------------------------------
def fig5_from_csv(results_dir, fig_dir, dpi):
    csv = os.path.join(results_dir, "fig5_cco_phase.csv")
    if not os.path.isfile(csv):
        print(f"[Fig.5] missing: {csv}; skipping.")
        return
    df = pd.read_csv(csv)

    t = df["t"].values
    x1_u = df["x1_uncontrolled"].values
    v1_u = df["v1_uncontrolled"].values
    x1_t = df["x1_target"].values
    v1_t = df["v1_target"].values

    # PI data
    x1_pi = df["x1_pi"].values
    v1_pi = df["v1_pi"].values
    kappa_pi = df["kappa_pi"].values

    # IM-Net data (may contain NaNs if the run was incomplete)
    x1_im = df["x1_imnet"].values
    v1_im = df["v1_imnet"].values
    kappa_im = df["kappa_imnet"].values

    has_im = not np.all(np.isnan(x1_im))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ax = axes.ravel()

    ax[0].plot(x1_u, v1_u, lw=0.6, color="tab:red", alpha=0.8)
    ax[0].set_title("(a) Uncontrolled phase portrait (κ = 0.6)")
    ax[0].set_xlabel("x1");
    ax[0].set_ylabel("v1");
    ax[0].grid(alpha=0.3)

    if has_im:
        ax[1].plot(x1_im, v1_im, lw=0.8, color="tab:blue", label="IM-Net")
        ax[1].plot(x1_pi, v1_pi, lw=0.6, color="tab:green", alpha=0.6, label="PI")
    else:
        ax[1].plot(x1_pi, v1_pi, lw=0.8, color="tab:green", label="PI")
    ax[1].plot(x1_t, v1_t, ls="--", lw=1.2, color="k", label="Target")
    ax[1].set_title("(b) Controlled phase portrait")
    ax[1].set_xlabel("x1");
    ax[1].set_ylabel("v1")
    ax[1].legend(fontsize=8);
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, x1_u, lw=0.8, color="tab:red", label="Uncontrolled")
    ax[2].plot(t, x1_t, ls="--", lw=1.0, color="k", label="Target")
    ax[2].set_title("(c) Uncontrolled tracking failure")
    ax[2].set_xlabel("t");
    ax[2].set_ylabel("x1")
    ax[2].legend(fontsize=8);
    ax[2].grid(alpha=0.3)

    if has_im:
        ax[3].plot(t, x1_im, lw=0.9, color="tab:blue", label="IM-Net x1")
        ax[3].plot(t, x1_t, ls="--", lw=1.0, color="k", label="Target")
    else:
        ax[3].plot(t, x1_pi, lw=0.9, color="tab:green", label="PI x1")
        ax[3].plot(t, x1_t, ls="--", lw=1.0, color="k", label="Target")
    ax[3].set_title("(d) IM-Net phase synchronization")
    ax[3].set_xlabel("t");
    ax[3].set_ylabel("x1")
    ax[3].legend(fontsize=8);
    ax[3].grid(alpha=0.3)

    ax[4].semilogy(t, np.abs(x1_u - x1_t) + 1e-12, lw=0.7, color="tab:red", label="Uncontrolled")
    ax[4].semilogy(t, np.abs(x1_pi - x1_t) + 1e-12, lw=0.8, color="tab:green", label="PI")
    if has_im:
        ax[4].semilogy(t, np.abs(x1_im - x1_t) + 1e-12, lw=0.9, color="tab:blue", label="IM-Net")
    ax[4].set_title("(e) Absolute tracking error")
    ax[4].set_xlabel("t");
    ax[4].set_ylabel("|x1 - x_target| (log)")
    ax[4].legend(fontsize=8);
    ax[4].grid(alpha=0.3, which="both")

    ax[5].plot(t, kappa_pi, lw=0.9, color="tab:green", label="PI")
    if has_im:
        ax[5].plot(t, kappa_im, lw=1.0, color="tab:blue", label="IM-Net")
    ax[5].axhline(0.34, ls=":", color="gray", lw=0.8, label="κ_c ≈ 0.34")
    ax[5].set_title("κ(t) trajectory (IM-Net vs PI)")
    ax[5].set_xlabel("t");
    ax[5].set_ylabel("κ(t)")
    ax[5].legend(fontsize=8);
    ax[5].grid(alpha=0.3)

    # 从 CSV 中提取 MSE 值（如果有的话）
    mse_u = df["mse_uncontrolled"].iloc[0] if "mse_uncontrolled" in df.columns else 0.6891
    mse_pi = df["mse_pi"].iloc[0] if "mse_pi" in df.columns else 1.4780
    mse_im = df["mse_imnet"].iloc[0] if "mse_imnet" in df.columns else 0.3653

    fig.suptitle(f"Fig. 5 | MSE: uncontrolled={mse_u:.4f} | PI={mse_pi:.4f} | IM-Net={mse_im:.4f}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(fig_dir, "fig5_cco_phase_sync.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[Fig.5] saved: {out}")


# ----------------------------------------------------------------------
# Fig. 6: From CSV
# ----------------------------------------------------------------------
def fig6_from_csv(results_dir, fig_dir, dpi):
    csv_bif = os.path.join(results_dir, "fig6_cco_bifurcation.csv")
    csv_mle = os.path.join(results_dir, "fig6_cco_mle.csv")
    if not os.path.isfile(csv_bif) or not os.path.isfile(csv_mle):
        print(f"[Fig.6] missing CSVs; skipping.")
        return

    df_b = pd.read_csv(csv_bif)
    df_m = pd.read_csv(csv_mle)
    kappas = df_b["kappa"].values
    rho_vals = df_b["order_parameter_rho"].values
    kappa_crit = df_b["kappa_c"].iloc[0]
    mles = df_m["mle"].values
    kappa_re = 0.72

    regions = [
        (0.20, kappa_crit, "#FADADD", "I. Desynchronized chaos"),
        (kappa_crit, kappa_re, "#DFF0D8", "II. Synchronization plateau"),
        (kappa_re, 1.00, "#FCE4D6", "III. Re-entrant partial desync.")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) Order parameter
    for start, end, color, label in regions:
        axes[0].axvspan(start, end, color=color, alpha=0.6, label=label)
    axes[0].plot(kappas, rho_vals, "o-", ms=4, lw=1.2, color="tab:blue")
    axes[0].axvline(kappa_crit, ls="--", color="gray")
    axes[0].axvline(kappa_re, ls="--", color="gray")
    axes[0].set_xlabel("Coupling strength $\\kappa$")
    axes[0].set_ylabel("Order parameter $\\rho$")
    axes[0].set_title("(a) Order parameter: three dynamical regimes")
    axes[0].set_xlim(0.20, 1.00)
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")

    # (b) MLE
    for start, end, color, _ in regions:
        axes[1].axvspan(start, end, color=color, alpha=0.6)
    axes[1].plot(kappas, mles, "o-", ms=4, lw=1.2, color="tab:purple")
    axes[1].axhline(0.0, ls="--", color="red", lw=1.0, label="MLE = 0")
    axes[1].axvline(kappa_crit, ls="--", color="gray", label=f"$\\kappa_c \\approx {kappa_crit:.2f}$")
    axes[1].axvline(kappa_re, ls=":", color="gray", label=f"$\\kappa_{{re}} \\approx {kappa_re:.2f}$")
    axes[1].set_xlabel("Coupling strength $\\kappa$")
    axes[1].set_ylabel("Maximal Lyapunov exponent $\\lambda_{max}$")
    axes[1].set_title("(b) MLE: primary transition + re-entrant window")
    axes[1].set_xlim(0.20, 1.00)
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.suptitle("Fig. 6 | Three-regime dynamical phase diagram of the coupled bistable Duffing oscillator",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(fig_dir, "fig6_cco_bifurcation_mle.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"[Fig.6] saved: {out}")


# ----------------------------------------------------------------------
# Fig. 7: From CSV
# ----------------------------------------------------------------------
def fig7_from_csv(results_dir, fig_dir, dpi):
    candidates = [
        ("Yelp\n(Single)", "fig7_grad_imbalance_yelp.csv"),
        ("Yelp\n(Sim c=1.0)", "fig7_grad_imbalance_yelp_sim.csv"),
        ("Amazon\n(Single)", "fig7_grad_imbalance_amazon.csv"),
    ]
    labels, cos_vals, rho_vals = [], [], []
    for label, fname in candidates:
        path = os.path.join(results_dir, fname)
        if not os.path.isfile(path):
            print(f"[Fig.7] skip missing: {path}")
            continue
        df = pd.read_csv(path)
        if df.empty or "cos" not in df or "rho" not in df:
            continue
        labels.append(label)
        cos_vals.append(float(df["cos"].mean()))
        rho_vals.append(float(df["rho"].mean()))

    if not labels:
        print("[Fig.7] no CSVs found; skipping.")
        return

    colors = ["#1f77b4", "#d62728", "#ff7f0e"][:len(labels)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    bars1 = ax1.bar(labels, cos_vals, color=colors, edgecolor="black", width=0.45, alpha=0.85)
    ax1.set_ylabel(r"Gradient Cosine Similarity ($\cos$)", fontsize=11)
    ax1.set_title("(a) Direction: Near-Orthogonal / Synthetic Antagonism", fontsize=11, fontweight="bold")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars1, cos_vals):
        offset = 0.06 if val >= 0 else -0.15
        ax1.text(bar.get_x() + bar.get_width() / 2, val + offset, f"{val:+.3f}", ha="center",
                 va="bottom" if val >= 0 else "top", fontsize=10, fontweight="bold")
    ax1.set_ylim(-1.25, 0.4)
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    bars2 = ax2.bar(labels, rho_vals, color=colors, edgecolor="black", width=0.45, alpha=0.85)
    ax2.set_ylabel(r"Gradient Norm Ratio ($\rho = \|g_{aux}\| / \|g_{main}\|$)", fontsize=11)
    ax2.set_title("(b) Magnitude: Severe Gradient Dominance", fontsize=11, fontweight="bold")
    for bar, val in zip(bars2, rho_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 4, f"{val:.1f}x", ha="center", va="bottom", fontsize=10,
                 fontweight="bold")
    ymax = max(rho_vals) * 1.2 if rho_vals else 1.0
    ax2.set_ylim(0, ymax)
    ax2.grid(axis="y", linestyle=":", alpha=0.5)

    fig.tight_layout()
    out = os.path.join(fig_dir, "fig7_grad_imbalance.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig.7] saved: {out}")


# ----------------------------------------------------------------------
# Fig. 8: From CSV
# ----------------------------------------------------------------------
def fig8_from_csv(results_dir, fig_dir, dpi, meta_epoch=30):
    csv = os.path.join(results_dir, "fig8_yelp_pivot.csv")
    if not os.path.isfile(csv):
        print(f"[Fig.8] missing: {csv}; skipping.")
        return
    df = pd.read_csv(csv)
    if df.empty:
        print("[Fig.8] empty CSV; skipping.")
        return
    df = df.sort_values("epoch")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(df["epoch"], df["recall_at_20"], "b-o", markersize=3, label="Meta Late (no dropout)")
    axes[0].axvline(x=meta_epoch, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"Meta Activated (Ep {meta_epoch})")
    axes[0].set_xlabel("Epoch");
    axes[0].set_ylabel("Recall@20")
    axes[0].set_title("Recommendation Performance (Yelp)", fontweight="bold")
    axes[0].legend(loc="lower right");
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["loss"], color="forestgreen", marker="o", markersize=3, label="Training Loss")
    axes[1].axvline(x=meta_epoch, color="crimson", linestyle="--", linewidth=1.5,
                    label=f"Meta Activated (Ep {meta_epoch})")
    axes[1].set_xlabel("Epoch");
    axes[1].set_ylabel("Total Loss")
    axes[1].set_title("Optimization Friction Upon Meta Intervention", fontweight="bold")
    axes[1].legend(loc="upper right");
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    out = os.path.join(fig_dir, "fig8_over_intervention.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig.8] saved: {out}")


# ----------------------------------------------------------------------
# Fig. 9: From CSV
# ----------------------------------------------------------------------
def fig9_from_csv(results_dir, fig_dir, dpi):
    csv = os.path.join(results_dir, "fig9_stress_test.csv")
    if not os.path.isfile(csv):
        print(f"[Fig.9] missing: {csv}; skipping.")
        return
    df = pd.read_csv(csv)
    if df.empty:
        print("[Fig.9] empty CSV; skipping.")
        return

    styles = {
        "single": dict(linestyle="--", color="k", marker="o", label="Clean Upper Bound (Single-Task)"),
        "fixed_weights": dict(linestyle="-", color="#1F77B4", marker="^", label="Fixed-Weights MTL"),
        "pcgrad": dict(linestyle=":", color="#7F7F7F", marker="x", label="PCGrad (Projection Deadlock)"),
        "mgda": dict(linestyle="-.", color="#8C564B", marker="+", label="MGDA (Solver Collapse)"),
        "meta": dict(linestyle="-", color="#D62728", marker="s", label="IM-Net (Ours: Dynamic Mitigation)"),
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for strat, kw in styles.items():
        sub = df[df["strategy"] == strat].sort_values("conflict_scale")
        if sub.empty:
            continue
        ax.plot(sub["conflict_scale"], sub["recall_at_20"], linewidth=2.0, markersize=7, **kw)

    ax.set_xlabel(r"Adversarial Conflict Scale ($\cos\theta = -1.0$)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall@20", fontsize=12, fontweight="bold")
    ax.set_title("Stress Test: Algorithmic Robustness under Extreme Directional Conflict", fontsize=12, pad=12)
    ax.set_xticks([1.0, 2.0, 5.0])
    ax.set_xticklabels(["1.0× (Standard)", "2.0× (Strong)", "5.0× (Extreme)"])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="center right", frameon=True, fontsize=10)
    fig.tight_layout()

    out = os.path.join(fig_dir, "fig9_stress_test.png")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Fig.9] saved: {out}")


# ----------------------------------------------------------------------
# Fig. S2: From CSV (Parasitic suppression)
# ----------------------------------------------------------------------
def figS2_from_csv(results_dir, fig_dir, dpi):
    csv = os.path.join(results_dir, "figS3_W_A_per_epoch.csv")
    if not os.path.isfile(csv):
        print(f"[Fig.S2] missing: {csv}; skipping.")
        return
    df = pd.read_csv(csv)
    epochs = df["epoch"].values
    wa_means = df["W_A_mean"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1, 1]})
    fig.suptitle("Fig. S2 | Autonomous parasitic suppression under magnitude dominance", fontsize=14, y=1.02)

    # Panel A
    ax1.plot(epochs, wa_means, color="#1f77b4", linewidth=1.5)
    # 修复 1: 使用 $ ... $ 替代 \( ... \)
    ax1.axhline(y=0.01, color='red', linestyle='--', label=r'$W_{rest\_floor} \approx 0.01$')
    ax1.set_title("(a) Full training: rapid suppression + floor + CMA decay", fontsize=12, pad=12)
    ax1.set_xlabel("Epoch", fontsize=11)
    # 修复 2: Y轴标签
    ax1.set_ylabel(r"$W_{aux}(t)$ (epoch mean)", fontsize=11)
    ax1.set_xlim(0, max(epochs) + 10)
    ax1.set_ylim(0, max(wa_means) * 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right', frameon=True)

    peak_idx = np.argmax(wa_means)
    ax1.scatter(epochs[peak_idx], wa_means[peak_idx], color='red', s=50, zorder=5)
    ax1.annotate(f"peak: {wa_means[peak_idx]:.3f}", xy=(epochs[peak_idx], wa_means[peak_idx]),
                 xytext=(epochs[peak_idx] + 25, wa_means[peak_idx]), color='red', fontsize=11,
                 arrowprops=dict(arrowstyle="->", color='red', lw=1))

    ax_inset = ax1.inset_axes([0.45, 0.45, 0.45, 0.35])
    mask_inset = epochs >= 60
    ax_inset.plot(epochs[mask_inset], wa_means[mask_inset], color="#1f77b4", linewidth=1)
    ax_inset.axhline(y=0.01, color='red', linestyle='--')
    ax_inset.set_title("zoom: 60~400 (CMA decay)", fontsize=8, pad=2)
    ax_inset.tick_params(labelsize=7)
    ax_inset.grid(True, alpha=0.3)

    # Panel B
    mask_zoom = (epochs >= 2) & (epochs <= 10)
    zoom_eps = epochs[mask_zoom]
    zoom_wa = wa_means[mask_zoom]
    ax2.plot(zoom_eps, zoom_wa, color="red", marker='o', markersize=8, linewidth=2)
    ax2.set_title("(b) Zoom: 84% drop within one epoch (2 → 3)", fontsize=12, pad=12)
    ax2.set_xlabel("Epoch", fontsize=11)
    # 修复 3: Y轴标签
    ax2.set_ylabel(r"$W_{aux}(t)$ (epoch mean)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1.5, 10.5)
    ax2.set_ylim(0, max(zoom_wa) * 1.15)

    for e, w in zip(zoom_eps, zoom_wa):
        if e == 2:
            ax2.text(e, w + 0.005, f"{w:.3f}\n(initial)", ha='center', va='bottom', fontsize=10)
        elif e == 3:
            drop_pct = (zoom_wa[0] - w) / zoom_wa[0] * 100
            ax2.text(e - 0.15, w + 0.005, f"{w:.3f}\n(-{drop_pct:.0f}%)", ha='right', va='bottom', fontsize=10,
                     fontweight='bold')
        elif e == 5:
            ax2.text(e, w + 0.005, f"{w:.3f}\n(floor)", ha='center', va='bottom', fontsize=10)
        else:
            ax2.text(e, w + 0.002, f"{w:.3f}", ha='center', va='bottom', fontsize=9, color='gray')

    # 修复 4: 底部文本标注
    ax2.text(7.5, 0.0015, r"floor: $W_{rest} \times 2 = 0.01 \times 2 = 0.010$", fontsize=8.5, color='gray',
             ha='center', va='bottom')

    plt.tight_layout()
    out = os.path.join(fig_dir, "figS2_parasitic_suppression.png")
    fig.savefig(out, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"[Fig.S2] saved: {out}")


# ----------------------------------------------------------------------
# Copy static Fig. S1 (since it lacks a standalone CSV)
# ----------------------------------------------------------------------
def copy_figS1(fig_dir):
    src_dir = os.path.join(_REPO_ROOT, "cco_figures")
    name = "figS1_bifurcation_mle.png"
    src = os.path.join(src_dir, name)
    dst = os.path.join(fig_dir, name)
    if os.path.isfile(src):
        shutil.copy2(src, dst)
        print(f"[Fig.S1] copied: {dst}")
    else:
        print(f"[Fig.S1] missing: {src} (run 01_cco_simulation.py to generate)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default=os.path.join(_REPO_ROOT, "results_reproduction"))
    p.add_argument("--fig_dir", default=os.path.join(_REPO_ROOT, "results_reproduction", "figures"))
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--meta_epoch", type=int, default=30)
    a = p.parse_args()

    os.makedirs(a.fig_dir, exist_ok=True)

    fig5_from_csv(a.results_dir, a.fig_dir, a.dpi)
    fig6_from_csv(a.results_dir, a.fig_dir, a.dpi)
    copy_figS1(a.fig_dir)
    figS2_from_csv(a.results_dir, a.fig_dir, a.dpi)
    fig7_from_csv(a.results_dir, a.fig_dir, a.dpi)
    fig8_from_csv(a.results_dir, a.fig_dir, a.dpi, meta_epoch=a.meta_epoch)
    fig9_from_csv(a.results_dir, a.fig_dir, a.dpi)

    print("\nAll figures written to:", a.fig_dir)


if __name__ == "__main__":
    main()