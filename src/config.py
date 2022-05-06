import os

#When training locally the folder sturctures are different
training_local = True


#Defining absolute Paths for directories, the os.path.sep method below adds the forward slash
#required for defining absolute Paths

DATA_PATH = os.path.join(os.path.sep, "kaggle","input","siim-isic-melanoma-classification")

INPUT_PATH = os.path.join(os.path.sep, "kaggle","working","Skin_Cancer_Detection","input")



if training_local:
    LOCAL_ROOT_PATH = os.path.join(os.path.sep, "Users", "ShivamRoy", "Development", "Python")

    INPUT_PATH = os.path.join(LOCAL_ROOT_PATH, INPUT_PATH[1:])
    DATA_PATH = os.path.join(LOCAL_ROOT_PATH, DATA_PATH[1:])
    #Sliced the path above to remove the leading '/' seperator