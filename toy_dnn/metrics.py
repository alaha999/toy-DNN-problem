import numpy as np
from sklearn.metrics import roc_curve, auc


def compute_roc(y_true, y_score, sample_weight=None):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, sample_weight=sample_weight)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def summarize_yields(y_train, y_test, w_train=None, w_test=None):
    def _compute(y, w):
        if w is None:
            w = np.ones_like(y, dtype=float)

        out = {}
        for cls, name in [(0, "Background"), (1, "Signal")]:
            mask = (y == cls)
            out[name] = {
                "count": int(np.sum(mask)),
                "weight": float(np.sum(w[mask])),
            }

        out["Total"] = {
            "count": int(len(y)),
            "weight": float(np.sum(w)),
        }
        return out

    return {
        "Train": _compute(y_train, w_train),
        "Test": _compute(y_test, w_test),
    }


def format_yield_text(results):
    lines = []
    header = f"{'Dataset':<10} {'Class':<12} {'Unweighted':>12} {'Weighted':>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for dataset in ["Train", "Test"]:
        for cls in ["Background", "Signal", "Total"]:
            vals = results[dataset][cls]
            lines.append(
                f"{dataset:<10} {cls:<12} {vals['count']:>12d} {vals['weight']:>14.3f}"
            )

    return "\n".join(lines)

def _significance(S, B, mode="soversqrtb"):
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)

    if mode == "soversqrtb":
        Z = np.zeros_like(S)
        mask = B > 0
        Z[mask] = S[mask] / np.sqrt(B[mask])
        return Z

    if mode == "soversqrtsplusb":
        Z = np.zeros_like(S)
        mask = (S + B) > 0
        Z[mask] = S[mask] / np.sqrt(S[mask] + B[mask])
        return Z

    if mode == "asimov":
        Z = np.zeros_like(S)
        mask = (S > 0) & (B > 0)
        Z[mask] = np.sqrt(
            2.0 * ((S[mask] + B[mask]) * np.log(1.0 + S[mask] / B[mask]) - S[mask])
        )
        return Z

    raise ValueError(f"Unknown significance mode: {mode}")


def scan_significance(y_true, y_score, weights=None, n_thresholds=200, mode="soversqrtb"):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if weights is None:
        weights = np.ones_like(y_score, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)

    sig_mask = (y_true == 1)
    bkg_mask = (y_true == 0)

    #Vectorized computation (no loop over thresholds)
    score_matrix = y_score[:, None] >= thresholds[None, :]

    S_vals = np.sum(weights[:, None] * score_matrix * sig_mask[:, None], axis=0)
    B_vals = np.sum(weights[:, None] * score_matrix * bkg_mask[:, None], axis=0)

    Z_vals = _significance(S_vals, B_vals, mode=mode)

    return thresholds, S_vals, B_vals, Z_vals


def summarize_best_significance(thresholds, S, B, Z):
    i = int(np.argmax(Z))
    return {
        "best_threshold": float(thresholds[i]),
        "best_S": float(S[i]),
        "best_B": float(B[i]),
        "best_Z": float(Z[i]),
    }
