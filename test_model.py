import random

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from unet import UNet

RANDOM_STATE = 30224
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BATCH_SIZE = 32
N_CHANNELS = 3  # RGB images
N_CLASSES = 1
MODEL = "./models/epoch_88.pt"


def extract_patient(path: str) -> str:
    return path.split("/")[-2]


def calculate_iou(pred, target, threshold=0.5):
    pred = torch.sigmoid(pred)
    pred = pred > threshold
    target = target > threshold

    intersection = (pred & target).float().sum((1, 2, 3))
    union = (pred | target).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean()


class CustomDataset(Dataset):

    def __init__(self, dataframe, image_transform=None, mask_transform=None):
        self.dataframe = dataframe
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx, 0]
        mask_path = self.dataframe.iloc[idx, 1]

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


if __name__ == "__main__":

    df_test = pd.read_csv("test.csv", usecols=["image_path", "mask_path"])
    print(df_test.values.shape)

    # transforms for the MRI images
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transforms for the masks
    mask_transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((256, 256))])

    # create datasets
    test_dataset = CustomDataset(
        df_test,
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    # check if CUDA is available and use it, otherwise stick with CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(N_CHANNELS, N_CLASSES)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    checkpoint = torch.load(MODEL, map_location=torch.device('cpu')) # model was trained on GPU, but tested on CPU
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
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
