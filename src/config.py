import os

#Defining absolute Paths for directories, the os.path.sep method below adds the forward slash
#required for defining absolute Paths

DATA_PATH = os.path.join(os.path.sep, "kaggle","input","siim-isic-melanoma-classification")

INPUT_PATH = os.path.join(os.path.sep, "kaggle","working","Skin_Cancer_Detection","input")

#When training locally the folder sturctures are different
training_local = True
LOCAL_ROOT_PATH = os.path.join(os.path.sep, "Users", "ShivamRoy", "Development", "Python")