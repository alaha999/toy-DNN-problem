import tensorflow as tf


def build_dnn(input_dim, hidden_layers, learning_rate=1e-3):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation="relu"))

    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def train_model(model, X_train, y_train, epochs, batch_size, sample_weight=None):
    history = model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=0,
    )
    return history


def predict_scores(model, X):
    return model.predict(X, verbose=0).ravel()
