import os
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import ImagesTransforms, ModelConstants
from unet import UNet
from utils import CustomDataset, calculate_iou, extract_patient

RANDOM_STATE = 30224
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
torch.cuda.manual_seed(RANDOM_STATE)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_PATH = "./lgg-mri-segmentation/kaggle_3m"


if __name__ == "__main__":
    images = []
    masks = glob(os.path.join(DATA_PATH, "*/*_mask*"))

    for image_file in masks:
        images.append(image_file.replace("_mask", ""))

    df = pd.DataFrame(data={"image_path": images, "mask_path": masks})
    df["patient"] = df["image_path"].apply(extract_patient)
    df = df.set_index("patient")
    unique_patients = df.index.unique()

    train_patients, test_patients = train_test_split(
        unique_patients,
        test_size=0.1,
        random_state=RANDOM_STATE,
    )
    train_patients, validation_patients = train_test_split(
        train_patients,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    df_train = df[df.index.isin(train_patients)]
    df_test = df[df.index.isin(test_patients)]
    df_val = df[df.index.isin(validation_patients)]

    print(
        f"Train dataset shape {df_train.shape} with {len(df_train.index.unique())} unique patients."
    )
    print(
        f"Validation dataset shape {df_val.shape} with {len(df_val.index.unique())} unique patients."
    )
    print(
        f"Test dataset shape: {df_test.shape} with {len(df_test.index.unique())} unique patients."
    )

    # Create datasets
    train_dataset = CustomDataset(
        df_train,
        image_transform=ImagesTransforms.IMAGE_TRANSFORM,
        mask_transform=ImagesTransforms.MASK_TRANSFORM,
    )
    val_dataset = CustomDataset(
        df_val,
        image_transform=ImagesTransforms.IMAGE_TRANSFORM,
        mask_transform=ImagesTransforms.MASK_TRANSFORM,
    )
    df_test.to_csv("test.csv", index=False)  # for use in test_model.py & results.ipynb

    model = UNet(ModelConstants.N_CHANNELS, ModelConstants.N_CLASSES)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=ModelConstants.LEARNING_RATE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=ModelConstants.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=ModelConstants.BATCH_SIZE, shuffle=False
    )

    # check if CUDA is available and use it, otherwise stick with CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in tqdm(range(ModelConstants.NUM_EPOCHS)):
        model.train()
        train_loss = 0
        train_iou = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_iou += calculate_iou(output, target).item()

        # validation
        model.eval()
        val_loss = 0
        val_iou = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                val_iou += calculate_iou(output, target).item()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "training_loss": train_loss / len(train_loader),
                "training_iou": train_iou / len(train_loader),
                "validation_loss": val_loss / len(val_loader),
                "validation_iou": val_iou / len(val_loader),
            },
            f"./models/epoch_{epoch}.pt",
        )
