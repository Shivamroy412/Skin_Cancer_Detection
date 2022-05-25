import pandas as pd
import config
import os
from PIL import Image


class SkinCancerDataset:

    def __init__(self, folds: list):
        
        df = pd.read_csv(config.INPUT_PATH)[["image_name", "target", "kfold"]]
        df = df.loc[df.kfold.isin(folds)].reset_index(drop= True)

        image_names = df.image_name.to_numpy()
        self.image_paths = [os.path.join([config.INPUT_PATH, "processed_img_train", 
                                 file_name + ".jpg"]) for file_name in image_names]

        self.target = df.target.to_numpy()


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        target = self.target[idx]

        image = Image.open(self.image_paths[idx])
        


