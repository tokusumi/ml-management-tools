from tensorflow import keras


def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(32, 32, 3)))
    model.add(
        keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
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