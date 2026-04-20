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


def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_config(
    seed,
    n_background,
    n_signal,
    test_size,
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
    epochs,
    batch_size,
    learning_rate,
    hidden_layers,
    bins_1d,
    bins_2d,
    n_thresholds,
):
    return {
        "seed": seed,
        "data": {
            "n_background": n_background,
            "n_signal": n_signal,
            "test_size": test_size,
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
        "model": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_layers": hidden_layers,
        },
        "plots": {
            "bins_1d": bins_1d,
            "bins_2d": bins_2d,
            "n_thresholds": n_thresholds,
        },
    }


@st.cache_data(show_spinner=False)
def cached_generate_data(cfg):
    set_seed(cfg["seed"])
    X, y, w = generate_toy_data(cfg)
    return X, y, w


@st.cache_resource(show_spinner=False)
def cached_train_models(cfg):
    set_seed(cfg["seed"])

    X, y, w = generate_toy_data(cfg)
    X_train, X_test, y_train, y_test, w_train, w_test = split_data(
        X, y, w,
        test_size=cfg["data"]["test_size"],
        seed=cfg["seed"],
    )

    model_u = build_dnn(
        input_dim=X.shape[1],
        hidden_layers=cfg["model"]["hidden_layers"],
        learning_rate=cfg["model"]["learning_rate"],
    )
    model_w = build_dnn(
        input_dim=X.shape[1],
        hidden_layers=cfg["model"]["hidden_layers"],
        learning_rate=cfg["model"]["learning_rate"],
    )

    hist_u = train_model(
        model_u,
        X_train,
        y_train,
        epochs=cfg["model"]["epochs"],
        batch_size=cfg["model"]["batch_size"],
        sample_weight=None,
    )
    hist_w = train_model(
        model_w,
        X_train,
        y_train,
        epochs=cfg["model"]["epochs"],
        batch_size=cfg["model"]["batch_size"],
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


st.title("Toy weighted vs unweighted DNN")
st.caption("Interactive demo for how event weights reshape the classifier, score, ROC, and significance.")

with st.sidebar:
    st.header("Dataset")

    seed = st.number_input("Seed", min_value=0, value=42, step=1)
    n_background = st.slider("Background events", 1000, 50000, 12000, step=500)
    n_signal = st.slider("Signal events", 200, 10000, 2500, step=100)
    test_size = st.slider("Test fraction", 0.1, 0.5, 0.3, step=0.05)

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

    st.subheader("Model")
    epochs = st.slider("Epochs", 3, 50, 20)
    batch_size = st.select_slider("Batch size", options=[64, 128, 256, 512], value=256)
    learning_rate = st.select_slider("Learning rate", options=[1e-4, 3e-4, 1e-3, 3e-3], value=1e-3)
    hidden_layers_text = st.text_input("Hidden layers", value="64,32,16")

    st.subheader("Plots")
    bins_1d = st.slider("1D bins", 20, 100, 40)
    bins_2d = st.slider("2D bins", 20, 100, 60)
    n_thresholds = st.slider("Threshold scan points", 50, 500, 250, step=25)

hidden_layers = [int(x.strip()) for x in hidden_layers_text.split(",") if x.strip()]

cfg = build_config(
    seed,
    n_background,
    n_signal,
    test_size,
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
    epochs,
    batch_size,
    learning_rate,
    hidden_layers,
    bins_1d,
    bins_2d,
    n_thresholds,
)

with st.spinner("Generating toy data and training models..."):
    result = cached_train_models(cfg)

st.subheader("Yield summary")
st.code(format_yield_text(result["yields"]))

col1, col2 = st.columns(2)
with col1:
    st.metric("Train events", f"{len(result['X_train'])}")
with col2:
    st.metric("Test events", f"{len(result['X_test'])}")

tab1, tab2, tab3, tab4 = st.tabs([
    "Input space",
    "Decision boundaries",
    "Score and ROC",
    "Significance",
])

with tab1:
    fig = make_input_space_figure(
        result["X"],
        result["y"],
        result["w"],
        bins_1d=bins_1d,
        bins_2d=bins_2d,
    )
    st.pyplot(fig, clear_figure=True)

with tab2:
    fig = make_decision_boundaries_figure(
        result["model_u"],
        result["model_w"],
        result["X_test"],
        result["y_test"],
        result["w_test"],
        bins_2d=bins_2d,
        grid_points=250,
    )
    st.pyplot(fig, clear_figure=True)

with tab3:
    fig = make_score_and_roc_figure(
        result["y_test"],
        result["score_u"],
        result["score_w"],
        result["w_test"],
        bins=50,
    )
    st.pyplot(fig, clear_figure=True)

with tab4:
    fig = make_significance_scan_figure(
        result["y_test"],
        result["score_u"],
        result["score_w"],
        result["w_test"],
        n_thresholds=n_thresholds,
    )
    st.pyplot(fig, clear_figure=True)
