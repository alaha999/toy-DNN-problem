import numpy as np
from sklearn.model_selection import train_test_split


def make_gaussian_toy_data(
    n_bkg=12000,
    n_sig=2500,
    random_state=42,
    bkg_mean=(0.0, 0.0),
    bkg_cov=((1.6, 0.4), (0.4, 1.2)),
    sig_mean=(1.8, 1.6),
    sig_cov=((0.45, 0.10), (0.10, 0.35)),
    background_mode="radial",
    signal_mode="mild",
):
    rng = np.random.default_rng(random_state)

    X_bkg = rng.multivariate_normal(np.array(bkg_mean), np.array(bkg_cov), size=n_bkg)
    X_sig = rng.multivariate_normal(np.array(sig_mean), np.array(sig_cov), size=n_sig)

    y_bkg = np.zeros(n_bkg, dtype=int)
    y_sig = np.ones(n_sig, dtype=int)

    w_bkg = _make_background_weights(X_bkg, mode=background_mode)
    w_sig = _make_signal_weights(X_sig, mode=signal_mode)

    X = np.vstack([X_bkg, X_sig])
    y = np.concatenate([y_bkg, y_sig])
    w = np.concatenate([w_bkg, w_sig])

    return X, y, w


def make_hep_tail_toy_data(
    n_bkg=10000,
    n_sig_main=1000,
    n_sig_tail=150,
    random_state=42,
    tail_weight=15.0,
):
    rng = np.random.default_rng(random_state)

    # Background: broad cloud near origin
    xb1 = rng.normal(loc=[0.0, 0.0], scale=[1.2, 1.0], size=(n_bkg, 2))

    # Main signal: partially overlaps with background
    xs1 = rng.normal(loc=[1.8, 1.8], scale=[0.8, 0.8], size=(n_sig_main, 2))

    # Rare tail signal: farther away, more discovery-like
    xs2 = rng.normal(loc=[3.2, 0.2], scale=[0.35, 0.35], size=(n_sig_tail, 2))

    X = np.vstack([xb1, xs1, xs2])
    y = np.hstack([
        np.zeros(len(xb1), dtype=int),
        np.ones(len(xs1), dtype=int),
        np.ones(len(xs2), dtype=int),
    ])

    w = np.hstack([
        np.ones(len(xb1)),
        np.ones(len(xs1)),
        np.full(len(xs2), tail_weight),
    ])

    return X, y, w


def generate_toy_data(cfg):
    toy_mode = cfg["data"].get("toy_mode", "HEP tail-enhanced signal")

    if toy_mode == "Gaussian weighted density":
        return make_gaussian_toy_data(
            n_bkg=cfg["data"]["n_background"],
            n_sig=cfg["data"]["n_signal"],
            random_state=cfg["seed"],
            bkg_mean=cfg["data"]["background"]["mean"],
            bkg_cov=cfg["data"]["background"]["cov"],
            sig_mean=cfg["data"]["signal"]["mean"],
            sig_cov=cfg["data"]["signal"]["cov"],
            background_mode=cfg["data"]["weights"]["background_mode"],
            signal_mode=cfg["data"]["weights"]["signal_mode"],
        )

    if toy_mode == "HEP tail-enhanced signal":
        return make_hep_tail_toy_data(
            n_bkg=cfg["data"]["n_background"],
            n_sig_main=cfg["data"]["n_signal_main"],
            n_sig_tail=cfg["data"]["n_signal_tail"],
            random_state=cfg["seed"],
            tail_weight=cfg["data"]["tail_weight"],
        )

    raise ValueError(f"Unknown toy_mode: {toy_mode}")


def split_data(X, y, w, test_size=0.3, seed=42):
    return train_test_split(
        X, y, w,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )


def _make_background_weights(X, mode="radial"):
    x1 = X[:, 0]
    x2 = X[:, 1]

    if mode == "radial":
        r = np.sqrt(x1**2 + x2**2)
        w = 0.8 + 1.8 * (r / (np.max(r) + 1e-12))
    else:
        w = np.ones(len(X))

    return w.astype(float)


def _make_signal_weights(X, mode="mild"):
    x1 = X[:, 0]
    x2 = X[:, 1]

    if mode == "mild":
        s = x1 + x2
        w = 0.9 + 0.4 * (s - np.min(s)) / (np.max(s) - np.min(s) + 1e-12)
    else:
        w = np.ones(len(X))

    return w.astype(float)
