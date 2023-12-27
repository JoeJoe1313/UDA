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


MODEL = "./models/epoch_88.pt"


if __name__ == "__main__":
    df_test = pd.read_csv("test.csv", usecols=["image_path", "mask_path"])

    # create dataset
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
    checkpoint = torch.load(
        MODEL, map_location=torch.device("cpu")
    )  # model was trained on GPU, but tested on CPU
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

# Test Loss: 0.01436641849935628, Test IoU: 0.85662147632012
