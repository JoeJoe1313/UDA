import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from constants import ImagesTransforms


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


def diagnosis(mask_path):
    if np.max(cv2.imread(mask_path)) > 0:
        return True  # tumor
    else:
        return False  # no tumor


def preprocess_image(image_path):
    transform = ImagesTransforms.IMAGE_TRANSFORM
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)

    return image


def predict_mask(model, image_path, device):
    model.eval()
    image = preprocess_image(image_path)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        probs = torch.sigmoid(output)
        predicted_mask = probs > 0.5

    return predicted_mask.squeeze(0).squeeze(0).cpu().numpy()
    # remove batch and channel dimensions


def print_res(image, mask, predict_mask):
    """Displays the result along with original, grayscale, and seeded region."""
    _, ax = plt.subplots(1, 3, figsize=(20, 6))

    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title("Mask")
    ax[1].axis("off")

    ax[2].imshow(predict_mask, cmap="gray")
    ax[2].set_title("Predicted Mask")
    ax[2].axis("off")

    plt.show()


class CustomDataset(Dataset):
    """Creates a PyTorch dataset from Pandas dataframe
    with columns image_path and mask_path."""

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
