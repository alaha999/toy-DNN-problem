import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from .metrics import compute_roc, scan_significance, summarize_best_significance


def _savefig(fig, output_dir, filename, save_png=True):
    if save_png:
        fig.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches="tight")


def _make_meshgrid(X, n_points=250, pad_frac=0.08):
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = np.min(x1), np.max(x1)
    x2_min, x2_max = np.min(x2), np.max(x2)

    dx = x1_max - x1_min
    dy = x2_max - x2_min

    x1_min -= pad_frac * dx
    x1_max += pad_frac * dx
    x2_min -= pad_frac * dy
    x2_max += pad_frac * dy

    xx, yy = np.meshgrid(
        np.linspace(x1_min, x1_max, n_points),
        np.linspace(x2_min, x2_max, n_points),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    return xx, yy, grid


def _predict_on_grid(model, grid):
    return model.predict(grid, verbose=0).ravel()


def make_input_space_figure(X, y, weights=None, bins_1d=40, bins_2d=60,signal_scatter_size=18):
    x1 = X[:, 0]
    x2 = X[:, 1]

    bkg = (y == 0)
    sig = (y == 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    x1_min, x1_max = np.min(x1), np.max(x1)
    x2_min, x2_max = np.min(x2), np.max(x2)

    # 2D unweighted
    h = axes[0, 0].hist2d(
        x1[bkg], x2[bkg],
        bins=bins_2d,
        range=[[x1_min, x1_max], [x2_min, x2_max]],
        cmap="Blues",
        alpha=0.85,
    )
    axes[0, 0].scatter(
        x1[sig],
        x2[sig],
        s=signal_scatter_size,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=1.0,
        label="Signal"
    )
    axes[0, 0].set_title("Input space: unweighted")
    axes[0, 0].set_xlabel("Feature 1")
    axes[0, 0].set_ylabel("Feature 2")
    axes[0, 0].set_xlim(x1_min, x1_max)
    axes[0, 0].set_ylim(x2_min, x2_max)
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=0.25)
    fig.colorbar(h[3], ax=axes[0, 0], label="Background entries")

    # =========================
    # (0,1) 2D weighted
    # =========================
    ax = axes[0, 1]

    h = ax.hist2d(
        x1[bkg],
        x2[bkg],
        bins=bins_2d,
        range=[[x1_min, x1_max], [x2_min, x2_max]],
        weights=None if weights is None else weights[bkg],
        cmap="Blues",
        alpha=0.85
    )

    if weights is None:
        sig_sizes = np.full(np.sum(sig), signal_scatter_size)
    else:
        ws = weights[sig]
        if np.max(ws) > 0:
            sig_sizes = signal_scatter_size * (0.6 + 1.8 * ws / np.max(ws))
        else:
            sig_sizes = np.full(np.sum(sig), signal_scatter_size)

    ax.scatter(
        x1[sig],
        x2[sig],
        s=sig_sizes,
        marker="o",
        facecolors="none",
        edgecolors="red",
        linewidths=1.0,
        label="Signal"
    )

    ax.set_title("Input space: weighted")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label("Weighted background entries")

    # 1D unweighted
    bins1 = np.linspace(x1_min, x1_max, bins_1d + 1)
    bins2 = np.linspace(x2_min, x2_max, bins_1d + 1)

    axes[1, 0].hist(x1[bkg], bins=bins1, density=True, histtype="stepfilled", alpha=0.25, label="Bkg: Feature 1")
    axes[1, 0].hist(x1[sig], bins=bins1, density=True, histtype="step", linewidth=2, label="Sig: Feature 1")
    axes[1, 0].hist(x2[bkg], bins=bins2, density=True, histtype="stepfilled", alpha=0.25, label="Bkg: Feature 2")
    axes[1, 0].hist(x2[sig], bins=bins2, density=True, histtype="step", linewidth=2, label="Sig: Feature 2")
    axes[1, 0].set_title("1D projections: unweighted")
    axes[1, 0].set_xlabel("Feature value")
    axes[1, 0].set_ylabel("A.U.")
    axes[1, 0].legend(frameon=False, ncol=2)
    axes[1, 0].grid(alpha=0.25)

    # 1D weighted
    axes[1, 1].hist(x1[bkg], bins=bins1, weights=weights[bkg], density=True, histtype="stepfilled", alpha=0.25, label="Bkg: Feature 1")
    axes[1, 1].hist(x1[sig], bins=bins1, weights=weights[sig], density=True, histtype="step", linewidth=2, label="Sig: Feature 1")
    axes[1, 1].hist(x2[bkg], bins=bins2, weights=weights[bkg], density=True, histtype="stepfilled", alpha=0.25, label="Bkg: Feature 2")
    axes[1, 1].hist(x2[sig], bins=bins2, weights=weights[sig], density=True, histtype="step", linewidth=2, label="Sig: Feature 2")
    axes[1, 1].set_title("1D projections: weighted")
    axes[1, 1].set_xlabel("Feature value")
    axes[1, 1].set_ylabel("A.U.")
    axes[1, 1].legend(frameon=False, ncol=2)
    axes[1, 1].grid(alpha=0.25)

    plt.tight_layout()
    return fig


def plot_input_space(X, y, weights, output_dir, bins_1d=40, bins_2d=60, save_png=True):
    fig = make_input_space_figure(X, y, weights, bins_1d=bins_1d, bins_2d=bins_2d)
    _savefig(fig, output_dir, "input_space_and_features.png", save_png=save_png)
    plt.close(fig)


def make_score_and_roc_figure(y_test, score_unweighted, score_weighted, w_test, bins=50):

    #Row 1: Unweighted training on unweighted + weighted test events
    #Row 2: Weighted training on unweighted + weighted test events
    #Row 3: ROC curve of unweighted and weighted training on unweighted events(left) vs weighted events(right)
    y_test = np.asarray(y_test)
    score_unweighted = np.asarray(score_unweighted)
    score_weighted = np.asarray(score_weighted)

    if w_test is None:
        w_test = np.ones_like(y_test, dtype=float)
    else:
        w_test = np.asarray(w_test, dtype=float)

    sig = (y_test == 1)
    bkg = (y_test == 0)

    fig, axes = plt.subplots(3, 2, figsize=(12,15))

    # -------------------------
    # Row 1: unweighted-trained model score
    # -------------------------
    ax = axes[0, 0]
    ax.hist(
        score_unweighted[bkg],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        label="Background",
    )
    ax.hist(
        score_unweighted[sig],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        label="Signal",
    )
    ax.set_title("Unweighted-training | unweighted test events")
    ax.set_xlabel("Model score")
    ax.set_ylabel("A.U.")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.hist(
        score_unweighted[bkg],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        weights=w_test[bkg],
        label="Background",
    )
    ax.hist(
        score_unweighted[sig],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        weights=w_test[sig],
        label="Signal",
    )
    ax.set_title("Unweighted-training | weighted test events")
    ax.set_xlabel("Model score")
    ax.set_ylabel("A.U.")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # -------------------------
    # Row 2: weighted-trained model score
    # -------------------------
    ax = axes[1, 0]
    ax.hist(
        score_weighted[bkg],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        label="Background",
    )
    ax.hist(
        score_weighted[sig],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        label="Signal",
    )
    ax.set_title("Weighted-training | unweighted test events")
    ax.set_xlabel("Model score")
    ax.set_ylabel("A.U.")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.hist(
        score_weighted[bkg],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        weights=w_test[bkg],
        label="Background",
    )
    ax.hist(
        score_weighted[sig],
        bins=bins,
        range=(0, 1),
        density=True,
        histtype="step",
        linewidth=2,
        weights=w_test[sig],
        label="Signal",
    )
    ax.set_title("Weighted-training | weighted test events")
    ax.set_xlabel("Model score")
    ax.set_ylabel("A.U.")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    # -------------------------
    # Row 3: ROC
    # -------------------------
    fpr_u_unw, tpr_u_unw, _ = roc_curve(y_test, score_unweighted)
    auc_u_unw = auc(fpr_u_unw, tpr_u_unw)

    fpr_w_unw, tpr_w_unw, _ = roc_curve(y_test, score_weighted)
    auc_w_unw = auc(fpr_w_unw, tpr_w_unw)

    ax = axes[2, 0]
    ax.plot(fpr_u_unw, tpr_u_unw, linewidth=2, label=f"Unweighted-training (AUC={auc_u_unw:.4f})")
    ax.plot(fpr_w_unw, tpr_w_unw, linewidth=2, label=f"Weighted-training (AUC={auc_w_unw:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("Unweighted test events ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    fpr_u_w, tpr_u_w, _ = roc_curve(y_test, score_unweighted, sample_weight=w_test)
    auc_u_w = auc(fpr_u_w, tpr_u_w)

    fpr_w_w, tpr_w_w, _ = roc_curve(y_test, score_weighted, sample_weight=w_test)
    auc_w_w = auc(fpr_w_w, tpr_w_w)

    ax = axes[2, 1]
    ax.plot(fpr_u_w, tpr_u_w, linewidth=2, label=f"Unweighted-training (AUC={auc_u_w:.4f})")
    ax.plot(fpr_w_w, tpr_w_w, linewidth=2, label=f"Weighted-training (AUC={auc_w_w:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title("Weighted test events ROC")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)

    plt.tight_layout()
    
    return fig


def plot_score_and_roc_2x2(y_true, score_unweighted, score_weighted, weights, output_dir, bins=50, save_png=True):
    fig = make_score_and_roc_figure(y_true, score_unweighted, score_weighted, weights, bins=bins)
    _savefig(fig, output_dir, "score_and_roc_2x2.png", save_png=save_png)
    plt.close(fig)


def make_significance_scan_figure(y_true, score_unweighted, score_weighted, weights, n_thresholds=250):
    thr_u, S_u, B_u, Z_u = scan_significance(
        y_true, score_unweighted, weights=weights, n_thresholds=n_thresholds
    )
    thr_w, S_w, B_w, Z_w = scan_significance(
        y_true, score_weighted, weights=weights, n_thresholds=n_thresholds
    )

    sum_u = summarize_best_significance(thr_u, S_u, B_u, Z_u)
    sum_w = summarize_best_significance(thr_w, S_w, B_w, Z_w)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # -----------------------------------
    # Left: weighted significance overlay
    # -----------------------------------
    axes[0].plot(thr_u, Z_u, linewidth=2, label="Unweighted-training")
    axes[0].plot(thr_w, Z_w, linewidth=2, label="Weighted-training")
    axes[0].axvline(sum_u["best_threshold"], linestyle="--", linewidth=1.2)
    axes[0].axvline(sum_w["best_threshold"], linestyle="--", linewidth=1.2)

    axes[0].set_title("significance scan (weighted test events)")
    axes[0].set_xlabel("Score threshold")
    axes[0].set_ylabel(r"$Z$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    if sum_u["best_Z"] > 0:
        rel_gain = 100.0 * (sum_w["best_Z"] / sum_u["best_Z"] - 1.0)
        txt = (
            f"Unweighted-training:\n"
            f"  best thr = {sum_u['best_threshold']:.3f}\n"
            f"  S = {sum_u['best_S']:.2f}, B = {sum_u['best_B']:.2f}\n"
            f"  Z = {sum_u['best_Z']:.3f}\n\n"
            f"Weighted-training:\n"
            f"  best thr = {sum_w['best_threshold']:.3f}\n"
            f"  S = {sum_w['best_S']:.2f}, B = {sum_w['best_B']:.2f}\n"
            f"  Z = {sum_w['best_Z']:.3f}\n\n"
            f"Relative gain = {rel_gain:.1f}%"
        )
    else:
        txt = (
            f"Unweighted-training:\n"
            f"  best thr = {sum_u['best_threshold']:.3f}\n"
            f"  S = {sum_u['best_S']:.2f}, B = {sum_u['best_B']:.2f}\n"
            f"  Z = {sum_u['best_Z']:.3f}\n\n"
            f"Weighted-training:\n"
            f"  best thr = {sum_w['best_threshold']:.3f}\n"
            f"  S = {sum_w['best_S']:.2f}, B = {sum_w['best_B']:.2f}\n"
            f"  Z = {sum_w['best_Z']:.3f}"
        )

    axes[0].text(
        0.03, 0.97,
        txt,
        transform=axes[0].transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    # -----------------------------------
    # Right: weighted S and B vs threshold
    # -----------------------------------
    axes[1].plot(thr_u, S_u, linewidth=2, label="S | Unweighted-training")
    axes[1].plot(thr_u, B_u, linewidth=2, linestyle="--", label="B | Unweighted-training")
    axes[1].plot(thr_w, S_w, linewidth=2, label="S | Weighted-training")
    axes[1].plot(thr_w, B_w, linewidth=2, linestyle="--", label="B | Weighted-training")

    axes[1].set_title("Weighted yields vs threshold")
    axes[1].set_xlabel("Score threshold")
    axes[1].set_ylabel("Weighted yield")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, ncol=2)

    plt.tight_layout()
    return fig    


def plot_significance_scan(y_true, score_unweighted, score_weighted, weights, output_dir, n_thresholds=250, save_png=True):
    fig = make_significance_scan_figure(
        y_true, score_unweighted, score_weighted, weights, n_thresholds=n_thresholds
    )
    _savefig(fig, output_dir, "significance_scan.png", save_png=save_png)
    plt.close(fig)


def make_decision_boundaries_figure(model_unweighted, model_weighted, X, y, weights, bins_2d=60, grid_points=250):
    x1 = X[:, 0]
    x2 = X[:, 1]

    bkg = (y == 0)
    sig = (y == 1)

    xx, yy, grid = _make_meshgrid(X, n_points=grid_points)
    zz_u = _predict_on_grid(model_unweighted, grid).reshape(xx.shape)
    zz_w = _predict_on_grid(model_weighted, grid).reshape(xx.shape)

    x1_min, x1_max = xx.min(), xx.max()
    x2_min, x2_max = yy.min(), yy.max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    cf = axes[0].contourf(
        xx, yy, zz_u,
        levels=np.linspace(0.0, 1.0, 21),
        alpha=0.85,
    )
    axes[0].contour(xx, yy, zz_u, levels=[0.5], linewidths=2.0, linestyles="--")
    axes[0].scatter(x1[bkg], x2[bkg], s=6, alpha=0.20, label="Background")
    axes[0].scatter(x1[sig], x2[sig], s=16, facecolors="none", edgecolors="red", linewidths=1.0, label="Signal")
    axes[0].set_title("Decision boundary: unweighted model")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].set_xlim(x1_min, x1_max)
    axes[0].set_ylim(x2_min, x2_max)
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    fig.colorbar(cf, ax=axes[0], label="Model score")

    cf = axes[1].contourf(
        xx, yy, zz_w,
        levels=np.linspace(0.0, 1.0, 21),
        alpha=0.85,
    )
    axes[1].contour(xx, yy, zz_w, levels=[0.5], linewidths=2.0, linestyles="--")
    axes[1].scatter(x1[bkg], x2[bkg], s=6, alpha=0.20, label="Background")
    sig_sizes = 12.0 * (0.6 + 1.8 * weights[sig] / np.max(weights[sig]))
    axes[1].scatter(x1[sig], x2[sig], s=sig_sizes, facecolors="none", edgecolors="red", linewidths=1.0, label="Signal")
    axes[1].set_title("Decision boundary: weighted model")
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].set_xlim(x1_min, x1_max)
    axes[1].set_ylim(x2_min, x2_max)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.colorbar(cf, ax=axes[1], label="Model score")

    plt.tight_layout()
    return fig


def plot_decision_boundaries(model_unweighted, model_weighted, X, y, weights, output_dir, bins_2d=60, grid_points=250, save_png=True):
    fig = make_decision_boundaries_figure(
        model_unweighted, model_weighted, X, y, weights,
        bins_2d=bins_2d, grid_points=grid_points
    )
    _savefig(fig, output_dir, "decision_boundaries.png", save_png=save_png)
    plt.close(fig)
