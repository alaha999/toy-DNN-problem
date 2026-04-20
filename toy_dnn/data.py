import numpy as np
from sklearn.model_selection import train_test_split


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
        w = 0.9 + 0.4 * (x1 + x2 - np.min(x1 + x2)) / (
            np.max(x1 + x2) - np.min(x1 + x2) + 1e-12
        )
    else:
        w = np.ones(len(X))

    return w.astype(float)


def generate_toy_data(cfg):
    n_bkg = cfg["data"]["n_background"]
    n_sig = cfg["data"]["n_signal"]

    bkg_mean = np.array(cfg["data"]["background"]["mean"], dtype=float)
    bkg_cov = np.array(cfg["data"]["background"]["cov"], dtype=float)

    sig_mean = np.array(cfg["data"]["signal"]["mean"], dtype=float)
    sig_cov = np.array(cfg["data"]["signal"]["cov"], dtype=float)

    X_bkg = np.random.multivariate_normal(bkg_mean, bkg_cov, size=n_bkg)
    X_sig = np.random.multivariate_normal(sig_mean, sig_cov, size=n_sig)

    y_bkg = np.zeros(n_bkg, dtype=int)
    y_sig = np.ones(n_sig, dtype=int)

    w_bkg = _make_background_weights(
        X_bkg,
        mode=cfg["data"]["weights"]["background_mode"],
    )
    w_sig = _make_signal_weights(
        X_sig,
        mode=cfg["data"]["weights"]["signal_mode"],
    )

    X = np.vstack([X_bkg, X_sig])
    y = np.concatenate([y_bkg, y_sig])
    w = np.concatenate([w_bkg, w_sig])

    return X, y, w


def split_data(X, y, w, test_size=0.3, seed=42):
    return train_test_split(
        X, y, w,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
