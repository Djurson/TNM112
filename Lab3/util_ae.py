import numpy as np
import matplotlib.pyplot as plt

def evaluate_ae(model, dataset, final=False, verbose=False):
    """Evaluate autoencoder reconstruction loss (MSE) on train/val or train/test."""
    print("Autoencoder performance (MSE):")
    train_mse = model.evaluate(dataset.x_train, dataset.x_train, verbose=0 if not verbose else 1)
    print(f"\tTrain MSE:       {train_mse:0.6f}")

    if final:
        test_mse = model.evaluate(dataset.x_test, dataset.x_test, verbose=0 if not verbose else 1)
        print(f"\tTest MSE:        {test_mse:0.6f}")
        return train_mse, test_mse
    else:
        val_mse = model.evaluate(dataset.x_valid, dataset.x_valid, verbose=0 if not verbose else 1)
        print(f"\tValidation MSE:  {val_mse:0.6f}")
        return train_mse, val_mse

def plot_training_ae(history):
    """Plot training/validation loss curves for an autoencoder."""
    if history is None:
        return
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(8,4))
    plt.plot(loss)
    if len(val_loss) > 0:
        plt.plot(val_loss)
        plt.legend(["Train", "Validation"])
    else:
        plt.legend(["Train"])
    plt.title("Autoencoder loss (MSE)")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.grid(True)
    plt.show()

def plot_reconstructions(model, x, n=10, cmap="gray"):
    """Plot original vs reconstructed images."""
    x = x[:n]
    x_hat = model.predict(x, verbose=0)

    plt.figure(figsize=(2*n, 4))
    for i in range(n):
        plt.subplot(2, n, i+1)
        plt.imshow(np.squeeze(x[i]), cmap=cmap)
        plt.axis("off")

        plt.subplot(2, n, i+1+n)
        plt.imshow(np.squeeze(x_hat[i]), cmap=cmap)
        plt.axis("off")
    plt.tight_layout()
    plt.show()
