from tensorflow import keras


def build_model(hp):
    model = keras.Sequential()
    model.add(
        keras.layers.Conv2D(
            hp.Choice("filter", values=[16, 32, 64]),
            3,
            activation=hp.Choice("activation", values=["tanh", "softplus", "relu"]),
        )
    )
    model.add(
        keras.layers.Conv2D(
            2 * hp.Choice("filter", values=[16, 32, 64]),
            3,
            activation=hp.Choice("activation", values=["tanh", "softplus", "relu"]),
        )
    )
    if hp.Boolean("batchnorm"):
        model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.MaxPooling2D(pool_size=2))
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(
            hp.Int("units", min_value=32, max_value=512, step=32), activation="relu"
        )
    )
    model.add(
        keras.layers.Dropout(
            hp.Float("dropout", min_value=0.25, max_value=0.75, step=0.25)
        )
    )
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["acc"],
    )
    return model
