from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

from data_generator import DataGenerator
from util_ae import evaluate_ae, plot_training_ae, plot_reconstructions

DATASET = "mnist"        
EPOCHS = 10
BATCH_SIZE = 128
LATENT_DIMS = [2, 16, 64]
SEED = 0


def build_autoencoder(input_shape, latent_dim: int):
    """Convolutional autoencoder for MNIST-like images.

    For MNIST (28x28x1), we downsample to 7x7 feature maps and then encode to latent_dim.
    Decoder upsamples back to original resolution.
    """
    inputs = keras.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D(2, padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)

    shape_before_flatten = keras.backend.int_shape(x)[1:]
    x = layers.Flatten()(x)
    latent = layers.Dense(latent_dim, name="latent")(x)

    # Decoder
    x = layers.Dense(np.prod(shape_before_flatten), activation="relu")(latent)
    x = layers.Reshape(shape_before_flatten)(x)

    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.UpSampling2D(2)(x)
    x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)

    outputs = layers.Conv2D(input_shape[-1], 3, activation="sigmoid", padding="same")(x)

    model = keras.Model(inputs, outputs, name=f"autoencoder_latent{latent_dim}")
    return model


def main():
    np.random.seed(SEED)
    keras.utils.set_random_seed(SEED)

    # Load data
    data = DataGenerator(verbose=True)
    data.generate(dataset=DATASET)

    assert data.x_train.min() >= 0.0 - 1e-6 and data.x_train.max() <= 1.0 + 1e-6, \
        "Expected normalized data in [0,1]. Check data_generator.normalize()."

    input_shape = data.x_train.shape[1:]

    results = []

    for latent_dim in LATENT_DIMS:
        print("\n" + "="*80)
        print(f"Training autoencoder with latent_dim={latent_dim}")
        print("="*80)

        keras.backend.clear_session()
        model = build_autoencoder(input_shape, latent_dim)
        model.compile(optimizer=keras.optimizers.Adam(), loss="mse")
        model.summary()

        history = model.fit(
            data.x_train, data.x_train,
            validation_data=(data.x_valid, data.x_valid),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2
        )

        train_mse, val_mse = evaluate_ae(model, data, final=False)
        results.append((latent_dim, float(train_mse), float(val_mse)))

        plot_training_ae(history)

        cmap = "gray" if input_shape[-1] == 1 else None
        plot_reconstructions(model, data.x_test, n=10, cmap=cmap if cmap else "viridis")

    print("\nSummary (lower is better):")
    print("latent_dim\ttrain_mse\tval_mse")
    for latent_dim, train_mse, val_mse in results:
        print(f"{latent_dim}\t\t{train_mse:.6f}\t{val_mse:.6f}")


if __name__ == "__main__":
    main()
