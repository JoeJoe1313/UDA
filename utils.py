import torch
from PIL import Image
from torch.utils.data import Dataset


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
