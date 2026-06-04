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

    # error in cm
    # train_cm = [torch.sqrt(torch.tensor(l)).item() * 100 for l in history["train_loss"]]
    # val_cm   = [torch.sqrt(torch.tensor(l)).item() * 100 for l in history["val_loss"]]
    # ax2.plot(history["iter"], train_cm, label="Train")
    # ax2.plot(history["iter"], val_cm,   label="Val")
    # ax2.set_xlabel("Iteration")
    # ax2.set_ylabel("RMSE (cm)")
    # ax2.set_title("Position Error")
    # ax2.legend()
    # ax2.grid(True)

    plt.tight_layout()
    plt.savefig("loss_curve_regularization.png", dpi=150)
    plt.show()

history = torch.load('training_history_2.pt', map_location=device)
plot_loss(history)
