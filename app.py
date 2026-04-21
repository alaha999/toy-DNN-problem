import hashlib
import json
import numpy as np
import streamlit as st
import tensorflow as tf

from toy_dnn.data import generate_toy_data, split_data
from toy_dnn.model import build_dnn, train_model, predict_scores
from toy_dnn.metrics import summarize_yields, format_yield_text
from toy_dnn.plots import (
    make_input_space_figure,
    make_score_and_roc_figure,
    make_significance_scan_figure,
    make_decision_boundaries_figure,
)

st.set_page_config(page_title="Toy Weighted vs Unweighted DNN", layout="wide")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_hash(obj) -> str:
    return hashlib.md5(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def build_data_config(
    toy_mode,
    seed,
    test_size,
    n_background,
    n_signal,
    n_signal_main,
    n_signal_tail,
    tail_weight,
    bkg_mean1,
    bkg_mean2,
    bkg_cov_11,
    bkg_cov_12,
    bkg_cov_22,
    sig_mean1,
    sig_mean2,
    sig_cov_11,
    sig_cov_12,
    sig_cov_22,
    background_mode,
    signal_mode,
):
    return {
        "seed": seed,
        "data": {
            "toy_mode": toy_mode,
            "test_size": test_size,
            "n_background": n_background,
            "n_signal": n_signal,
            "n_signal_main": n_signal_main,
            "n_signal_tail": n_signal_tail,
            "tail_weight": tail_weight,
            "background": {
                "mean": [bkg_mean1, bkg_mean2],
                "cov": [
                    [bkg_cov_11, bkg_cov_12],
                    [bkg_cov_12, bkg_cov_22],
                ],
            },
            "signal": {
                "mean": [sig_mean1, sig_mean2],
                "cov": [
                    [sig_cov_11, sig_cov_12],
                    [sig_cov_12, sig_cov_22],
                ],
            },
            "weights": {
                "background_mode": background_mode,
                "signal_mode": signal_mode,
            },
        },
    }


def build_train_config(
    seed,
    epochs,
    batch_size,
    learning_rate,
    hidden_layers,
):
    return {
        "seed": seed,
        "model": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_layers": hidden_layers,
        },
    }


def build_plot_config(
    bins_1d,
    bins_2d,
    n_thresholds,
):
    return {
        "plots": {
            "bins_1d": bins_1d,
            "bins_2d": bins_2d,
            "n_thresholds": n_thresholds,
        }
    }


@st.cache_data(show_spinner=False)
def cached_generate_data(data_cfg):
    set_seed(data_cfg["seed"])
    X, y, w = generate_toy_data(data_cfg)
    return X, y, w


@st.cache_resource(show_spinner=False)
def cached_train_models(data_cfg, train_cfg):
    set_seed(train_cfg["seed"])

    X, y, w = generate_toy_data(data_cfg)
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(
        X, y, w,
        test_size=data_cfg["data"]["test_size"],
        seed=data_cfg["seed"],
    )

    model_u = build_dnn(
        input_dim=X.shape[1],
        hidden_layers=train_cfg["model"]["hidden_layers"],
        learning_rate=train_cfg["model"]["learning_rate"],
    )
    model_w = build_dnn(
        input_dim=X.shape[1],
        hidden_layers=train_cfg["model"]["hidden_layers"],
        learning_rate=train_cfg["model"]["learning_rate"],
    )

    hist_u = train_model(
        model_u,
        X_train,
        y_train,
        epochs=train_cfg["model"]["epochs"],
        batch_size=train_cfg["model"]["batch_size"],
        sample_weight=None,
    )
    hist_w = train_model(
        model_w,
        X_train,
        y_train,
        epochs=train_cfg["model"]["epochs"],
        batch_size=train_cfg["model"]["batch_size"],
        sample_weight=w_train,
    )

    score_u = predict_scores(model_u, X_test)
    score_w = predict_scores(model_w, X_test)

    yields = summarize_yields(y_train, y_test, w_train=w_train, w_test=w_test)

    return {
        "X": X,
        "y": y,
        "w": w,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "w_train": w_train,
        "w_test": w_test,
        "model_u": model_u,
        "model_w": model_w,
        "score_u": score_u,
        "score_w": score_w,
        "hist_u": hist_u.history,
        "hist_w": hist_w.history,
        "yields": yields,
    }


# ----------------------------
# session state
# ----------------------------
defaults = {
    "data_result": None,
    "train_result": None,
    "data_hash": None,
    "train_hash": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


st.title("Toy weighted vs unweighted DNN")
st.caption(
    "First generate the toy data and inspect diagnostics. "
    "Then explicitly train the models only when you are satisfied."
)

# ----------------------------
# sidebar: data form
# ----------------------------
with st.sidebar:
    st.header("1. Toy data")

    with st.form("data_form"):
        toy_mode = st.selectbox(
            "Toy problem type",
            ["HEP tail-enhanced signal", "Gaussian weighted density"],
            index=0,
        )

        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        test_size = st.slider("Test fraction", 0.1, 0.5, 0.3, step=0.05)

        if toy_mode == "HEP tail-enhanced signal":
            st.subheader("HEP tail toy")
            n_background = st.slider("Background events", 1000, 50000, 10000, step=500)
            n_signal_main = st.slider("Main signal events", 100, 10000, 950, step=50)
            n_signal_tail = st.slider("Tail signal events", 20, 2000, 50, step=10)
            tail_weight = st.slider("Tail signal weight", 1.0, 30.0, 15.0, step=1.0)

            n_signal = n_signal_main + n_signal_tail
            bkg_mean1, bkg_mean2 = 0.0, 0.0
            bkg_cov_11, bkg_cov_12, bkg_cov_22 = 1.6, 0.4, 1.2
            sig_mean1, sig_mean2 = 1.8, 1.6
            sig_cov_11, sig_cov_12, sig_cov_22 = 0.45, 0.10, 0.35
            background_mode, signal_mode = "none", "none"

        else:
            st.subheader("Gaussian toy")
            n_background = st.slider("Background events", 1000, 50000, 12000, step=500)
            n_signal = st.slider("Signal events", 200, 10000, 2500, step=100)

            st.subheader("Background Gaussian")
            bkg_mean1 = st.number_input("Bkg mean x1", value=0.0)
            bkg_mean2 = st.number_input("Bkg mean x2", value=0.0)
            bkg_cov_11 = st.number_input("Bkg cov 11", value=1.6)
            bkg_cov_12 = st.number_input("Bkg cov 12", value=0.4)
            bkg_cov_22 = st.number_input("Bkg cov 22", value=1.2)

            st.subheader("Signal Gaussian")
            sig_mean1 = st.number_input("Sig mean x1", value=1.8)
            sig_mean2 = st.number_input("Sig mean x2", value=1.6)
            sig_cov_11 = st.number_input("Sig cov 11", value=0.45)
            sig_cov_12 = st.number_input("Sig cov 12", value=0.10)
            sig_cov_22 = st.number_input("Sig cov 22", value=0.35)

            st.subheader("Weights")
            background_mode = st.selectbox("Background weighting", ["radial", "none"], index=0)
            signal_mode = st.selectbox("Signal weighting", ["mild", "none"], index=0)

            n_signal_main = n_signal
            n_signal_tail = 0
            tail_weight = 1.0

        generate_data_clicked = st.form_submit_button("Generate toy data")

    st.header("2. Training")

    with st.form("train_form"):
        epochs = st.slider("Epochs", 3, 50, 20)
        batch_size = st.select_slider("Batch size", options=[64, 128, 256, 512], value=256)
        learning_rate = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
        hidden_layers_text = st.text_input("Hidden layers", value="64,32,16")
        run_model_clicked = st.form_submit_button("Run model", type="primary")

    st.header("3. Plot settings")
    bins_1d = st.slider("1D bins", 20, 100, 40)
    bins_2d = st.slider("2D bins", 20, 100, 60)
    n_thresholds = st.slider("Threshold scan points", 50, 500, 250, step=25)


hidden_layers = [int(x.strip()) for x in hidden_layers_text.split(",") if x.strip()]

data_cfg = build_data_config(
    toy_mode=toy_mode,
    seed=seed,
    test_size=test_size,
    n_background=n_background,
    n_signal=n_signal,
    n_signal_main=n_signal_main,
    n_signal_tail=n_signal_tail,
    tail_weight=tail_weight,
    bkg_mean1=bkg_mean1,
    bkg_mean2=bkg_mean2,
    bkg_cov_11=bkg_cov_11,
    bkg_cov_12=bkg_cov_12,
    bkg_cov_22=bkg_cov_22,
    sig_mean1=sig_mean1,
    sig_mean2=sig_mean2,
    sig_cov_11=sig_cov_11,
    sig_cov_12=sig_cov_12,
    sig_cov_22=sig_cov_22,
    background_mode=background_mode,
    signal_mode=signal_mode,
)

train_cfg = build_train_config(
    seed=seed,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    hidden_layers=hidden_layers,
)

plot_cfg = build_plot_config(
    bins_1d=bins_1d,
    bins_2d=bins_2d,
    n_thresholds=n_thresholds,
)

data_hash = make_hash(data_cfg)
train_hash = make_hash({"data": data_cfg, "train": train_cfg})

# ----------------------------
# generate data
# ----------------------------
if generate_data_clicked:
    with st.spinner("Generating toy data..."):
        X, y, w = cached_generate_data(data_cfg)

    st.session_state.data_result = {"X": X, "y": y, "w": w}
    st.session_state.data_hash = data_hash
    st.session_state.train_result = None
    st.session_state.train_hash = None

# ----------------------------
# no data yet
# ----------------------------
if st.session_state.data_result is None:
    st.info("Set up the toy problem in the sidebar and click **Generate toy data**.")
    st.stop()

data_result = st.session_state.data_result

# ----------------------------
# diagnostics
# ----------------------------
st.subheader("Toy-data diagnostics")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total events", f"{len(data_result['X'])}")
with c2:
    st.metric("Background", f"{int(np.sum(data_result['y'] == 0))}")
with c3:
    st.metric("Signal", f"{int(np.sum(data_result['y'] == 1))}")
with c4:
    st.metric("Toy mode", data_cfg["data"]["toy_mode"])

fig_diag = make_input_space_figure(
    data_result["X"],
    data_result["y"],
    data_result["w"],
    bins_1d=plot_cfg["plots"]["bins_1d"],
    bins_2d=plot_cfg["plots"]["bins_2d"],
)
st.pyplot(fig_diag, clear_figure=True)

# ----------------------------
# train only on button
# ----------------------------
if run_model_clicked:
    if st.session_state.data_hash != data_hash:
        st.warning("Current toy-data settings differ from the generated sample. Click **Generate toy data** first.")
    else:
        with st.spinner("Training weighted and unweighted models..."):
            st.session_state.train_result = cached_train_models(data_cfg, train_cfg)
            st.session_state.train_hash = train_hash

# ----------------------------
# training state checks
# ----------------------------
if st.session_state.train_result is None:
    st.warning("Diagnostics are ready. Click **Run model** when you are satisfied with the toy data.")
    st.stop()

if st.session_state.data_hash != data_hash:
    st.warning("Toy-data controls changed after training. Regenerate toy data, then rerun the model.")
    st.stop()

if st.session_state.train_hash != train_hash:
    st.warning("Training controls changed after the last training. Click **Run model** to retrain.")
    st.stop()

result = st.session_state.train_result

# ----------------------------
# results
# ----------------------------
st.subheader("Model results")
st.code(format_yield_text(result["yields"]))

r1, r2 = st.columns(2)
with r1:
    st.metric("Train events", f"{len(result['X_train'])}")
with r2:
    st.metric("Test events", f"{len(result['X_test'])}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Decision boundaries",
    "Score and ROC",
    "Significance",
    "Training history",
])

with tab1:
    fig = make_decision_boundaries_figure(
        result["model_u"],
        result["model_w"],
        result["X_test"],
        result["y_test"],
        result["w_test"],
        bins_2d=plot_cfg["plots"]["bins_2d"],
        grid_points=250,
    )
    st.pyplot(fig, clear_figure=True)

with tab2:
    fig = make_score_and_roc_figure(
        result["y_test"],
        result["score_u"],
        result["score_w"],
        result["w_test"],
        bins=50,
    )
    st.pyplot(fig, clear_figure=True)

with tab3:
    fig = make_significance_scan_figure(
        result["y_test"],
        result["score_u"],
        result["score_w"],
        result["w_test"],
        n_thresholds=plot_cfg["plots"]["n_thresholds"],
    )
    st.pyplot(fig, clear_figure=True)

with tab4:
    a, b = st.columns(2)
    with a:
        st.markdown("**Unweighted history**")
        st.json(result["hist_u"])
    with b:
        st.markdown("**Weighted history**")
        st.json(result["hist_w"])
