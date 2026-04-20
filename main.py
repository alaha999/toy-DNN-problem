import os

from toy_dnn.utils import load_config, ensure_dir, set_global_seed, setup_logger, save_json
from toy_dnn.data import generate_toy_data, split_data
from toy_dnn.model import build_dnn, train_model, predict_scores
from toy_dnn.metrics import summarize_yields, format_yield_text
from toy_dnn.plots import plot_input_space, plot_score_and_roc_2x2, plot_significance_scan


def main():
    cfg = load_config("config.yaml")
    output_dir = cfg["output"]["dir"]

    ensure_dir(output_dir)
    logger = setup_logger(output_dir)

    seed = cfg["seed"]
    set_global_seed(seed)
    logger.info("Loaded config and set random seed = %d", seed)

    # Toy problem generation
    X, y, w = generate_toy_data(cfg)
    logger.info("Generated toy dataset with %d events", len(X))

    X_train, X_test, y_train, y_test, w_train, w_test = split_data(
        X, y, w,
        test_size=cfg["data"]["test_size"],
        seed=seed,
    )
    logger.info("Train size = %d, Test size = %d", len(X_train), len(X_test))

    # Yield summary
    yields = summarize_yields(y_train, y_test, w_train=w_train, w_test=w_test)
    yield_text = format_yield_text(yields)
    logger.info("\n%s", yield_text)
    save_json(yields, os.path.join(output_dir, "yields.json"))

    # Input plots
    plot_input_space(
        X, y, w,
        output_dir=output_dir,
        bins_1d=cfg["plots"]["bins_1d"],
        bins_2d=cfg["plots"]["bins_2d"],
        save_png=cfg["output"]["save_png"],
    )
    logger.info("Saved input space plots")

    # Models
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
    logger.info("Finished unweighted training")

    hist_w = train_model(
        model_w,
        X_train,
        y_train,
        epochs=cfg["model"]["epochs"],
        batch_size=cfg["model"]["batch_size"],
        sample_weight=w_train,
    )
    logger.info("Finished weighted training")

    # Scores
    score_u = predict_scores(model_u, X_test)
    score_w = predict_scores(model_w, X_test)
    logger.info("Computed model scores on test set")

    # Main evaluation plots
    plot_score_and_roc_2x2(
        y_true=y_test,
        score_unweighted=score_u,
        score_weighted=score_w,
        weights=w_test,
        output_dir=output_dir,
        save_png=cfg["output"]["save_png"],
    )
    logger.info("Saved score and ROC plots")

    plot_significance_scan(
        y_true=y_test,
        score_unweighted=score_u,
        score_weighted=score_w,
        weights=w_test,
        output_dir=output_dir,
        n_thresholds=cfg["plots"]["n_thresholds"],
        save_png=cfg["output"]["save_png"],
    )
    logger.info("Saved significance scan plots")

    # Training summaries
    training_summary = {
        "unweighted_final": {
            k: float(v[-1]) for k, v in hist_u.history.items()
        },
        "weighted_final": {
            k: float(v[-1]) for k, v in hist_w.history.items()
        },
    }
    save_json(training_summary, os.path.join(output_dir, "training_summary.json"))
    logger.info("Saved training summary JSON")

    logger.info("Done.")


if __name__ == "__main__":
    main()
