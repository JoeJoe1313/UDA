"""Takes the checkpoints saved in the `models` folder, processses them 
and produces a `results.csv` file containing information about each 
epoch such as training loss, training iou, validation loss, validation iou.
"""

import os
from glob import glob

import pandas as pd
import torch
from tqdm import tqdm

if __name__ == "__main__":
    models = glob(os.path.join("./models", "*"))
    results = []
    for model in tqdm(models):
        checkpoint = torch.load(model)
        results.append(
            {
                "epoch": checkpoint["epoch"],
                "training_loss": checkpoint["training_loss"],
                "training_iou": checkpoint["training_iou"],
                "validation_loss": checkpoint["validation_loss"],
                "validation_iou": checkpoint["validation_iou"],
            }
        )

    results = pd.DataFrame(results)
    results = results.set_index("epoch")
    results = results.sort_index()
    results.to_csv("results.csv")
