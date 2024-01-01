"""Selects the model parameters corresponding to the epoch with highest
validation IoU and evaluates the model with them on the test dataset.
"""
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import ImagesTransforms, ModelConstants
from unet import UNet
from utils import CustomDataset, calculate_iou

RANDOM_STATE = 30224
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    df_test = pd.read_csv("test.csv", usecols=["image_path", "mask_path"])
    results = pd.read_csv("results.csv", index_col="epoch").sort_index()

    # get the epoch for which the validation IoU is largest
    max_val_iou = results["validation_iou"].max()
    best_epoch = results[results["validation_iou"] == max_val_iou].index.values[0]
    print(f"Best epoch is {best_epoch}")
    model_path = f"./models/epoch_{best_epoch}.pt"

    # create test dataset
    test_dataset = CustomDataset(
        df_test,
        image_transform=ImagesTransforms.IMAGE_TRANSFORM,
        mask_transform=ImagesTransforms.MASK_TRANSFORM,
    )

    # check if CUDA is available and use it, otherwise stick with CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(ModelConstants.N_CHANNELS, ModelConstants.N_CLASSES)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = DataLoader(
        test_dataset,
        batch_size=ModelConstants.BATCH_SIZE,
        shuffle=False,
    )
    model.eval()
    test_loss = 0
    test_iou = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            test_iou += calculate_iou(output, target).item()

    print(
        f"Test Loss: {test_loss / len(test_loader)}, Test IoU: {test_iou / len(test_loader)}"
    )
