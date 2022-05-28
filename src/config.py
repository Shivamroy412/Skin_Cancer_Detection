import os
import torch

#Defining Device for GPU operations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#When training locally the folder sturctures are different
training_local = True

#Image Size suitable for models to be run in preprocessing
IMAGE_SIZE = (512, 512)

#Defining absolute Paths for directories, the os.path.sep method below adds the forward slash
#required for defining absolute Paths

DATA_PATH = os.path.join(os.path.sep, "kaggle","input","siim-isic-melanoma-classification")

INPUT_PATH = os.path.join(os.path.sep, "kaggle","working","Skin_Cancer_Detection","input")

SRC_PATH = os.path.join(os.path.sep, "kaggle","working","Skin_Cancer_Detection","src")

if training_local:
    LOCAL_ROOT_PATH = os.path.join(os.path.sep, "Users", "ShivamRoy", "Development", "Python")

    INPUT_PATH = os.path.join(LOCAL_ROOT_PATH, INPUT_PATH[1:])
    DATA_PATH = os.path.join(LOCAL_ROOT_PATH, DATA_PATH[1:])
    SRC_PATH = os.path.join(LOCAL_ROOT_PATH, SRC_PATH[1:])
    #Sliced the path above to remove the leading '/' seperator

