import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time

device = "cpu"
def plot_loss(history):
    fig = plt.figure(1, figsize=(12, 4))

    # raw loss
    plt.plot(history["iter"], history["train_loss"], label="Train")
    plt.plot(history["iter"], history["val_loss"],   label="Val")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("loss_curve_regularization.png", dpi=150)
    plt.show()

history = torch.load('training_history_2.pt', map_location=device)
plot_loss(history)
