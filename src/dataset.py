import pandas as pd
import numpy as np
import config
import os
from PIL import Image
import torch


class SkinCancerDataset:

    def __init__(self, folds: list, augmentations=None, channel_first = True):
        
        df = pd.read_csv(os.path.join(config.INPUT_PATH,"train_folds.csv"))[["image_name", "target", "kfold"]]
        df = df.loc[df.kfold.isin(folds)].reset_index(drop= True)

        image_names = df.image_name.to_numpy()
        self.image_paths = [os.path.join(config.INPUT_PATH, "processed_img_train", 
                                 file_name + ".jpg") for file_name in image_names]

        self.target = df.target.to_numpy()
        self.augmentations = augmentations
        self.channel_first = channel_first 
        #Whether channel dimension is mentioned at the beginning of the image array


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        target = self.target[idx]

        image = Image.open(self.image_paths[idx])
        image = np.array(image)

        if self.augmentations:
            augmentation_dict = self.augmentations(image = image)
            #This returns a dictionary with image as one of the keys
            image = augmentation_dict["image"]

        if self.channel_first:
            image = np.transpose(image, (2, 0 , 1)).astype(np.float32)

        return {"image": torch.tensor(image), 
                "target": torch.tensor(target)}
