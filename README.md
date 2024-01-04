# Unstructured Data Analysis Final Project

Brain MRI Segmentation with U-Net

# Introduction

This project focuses on the topic of Brain MRI segmentation with the help of Deep Learning techniques. More precisely, it explores the application of a U-Net (a convolutional neural network developed for biomedical image segmentation) on Brain MRIs. The aim is to train a model with the help of manually created fluid attenuated inversion recovery (FLAIR) abnormality segmentation masks,capable of detecting a lower-grade glioma (LGG - a type of brain tumor) in a 2D Brain MR image.

# Data

The image data for this project is taken from [Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation/data). The LGG Segmentation Dataset is used in "Mateusz Buda, AshirbaniSaha, Maciej A. Mazurowski "Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019." and "Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi, Katherine B. Peters, Ashirbani Saha "Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017.".

The dataset contains preoperative brain MRIs together with manual FLAIR abnormality segmentation masks showing the presence (or absence) of LGG. The images were obtained from The Cancer Imaging Archive (TCIA). They correspond to 110 patients, with number of slices varying among patients from 20 to 88, included in The Cancer Genome Atlas (TCGA) lower-grade glioma collection with at least FLAIR sequence and genomic cluster data available. The 

All images are provided in `.tif` format with 3 channels per image corresponding to the following 3 MRI sequences in the given order: pre-contrast, FLAIR, post-contrast. For 101 cases, the 3 sequences are available, for 9 cases, post-contrast sequence is missing and for 6 cases, pre-contrast sequence is missing. The missing sequences are replaced with FLAIR sequence to make all images 3-channel. Masks are binary, 1-channel images, and they segment FLAIR abnormality present in the FLAIR sequence, and are available for all cases.

The dataset has 110 folders named after case ID that contains information about source institution. Each folder contains MRI images with the naming convention `TCGA_<institution>_<patient-id>_<slice-number>.tif`. Corresponding masks have a `_mask` suffix.

# Repository and Realted Technical Details

### GitHub Repository

The code of the project is stored in the following public GitHub [repository](https://github.com/JoeJoe1313/UDA). 

The folders: 

- `lgg-mri-segmentation/kaggle_3m` contains a subset of the data of size 100 MB corresponding to 12 patients in total. 
- `models` is kept empty and is for saving checkpoints of the model during the training procedure.

The files:

- `unet.py` contains the implementation of the U-Net model
- `constants.py` contains the constants shared between the files: model parameters and image transformations
- `utils.py` stores multiple functions used in multiple files
- `train_model.py` performs the creation of train, validation, and test datasets, the training and validation of the model while saving checkpoints for each epoch in the `models` folder
- `process_models.py` takes the checkpoints saved in the `models` folder, processses them and produces a `results.csv` file containing information about each epoch such as training loss, training iou, validation loss, validation iou
- `test_model.py` selects the model parameters corresponding to the epoch with highest validation IoU and evaluates the model with them on the test dataset

The notebooks:

- `train_test_100mb.ipynb` clones the project GitHub repository, executes `train_model.py`, `process_models.py`, `test_model.py`, and visualises the results and predictions for the 100 MB subset of the data. No credentials are needed for executing the notebook.
- `train_test_1gb.ipynb` downloads the full dataset from Kaggle, clones the project GitHub repository, executes `train_model.py`, `process_models.py`, `test_model.py`, and visualises the results and predictions for the whole 1 GB dataset

### Environment

A list of the needed libraries with their versions:

```
Package                          Version
-------------------------------- ---------------------
matplotlib                       3.7.1
numpy                            1.23.5
opencv-python                    4.8.0.76
pandas                           1.5.3
Pillow                           9.4.0
scikit-learn                     1.2.2
torch                            2.1.0+cu121
torchaudio                       2.1.0+cu121
tqdm                             4.66.1
```
