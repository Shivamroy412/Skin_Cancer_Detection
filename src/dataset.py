import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as T
import config


class SkinCancerDataset:

    def __init__(self, folds: list, transforms=None, channel_first = True):
        
        df = pd.read_csv(os.path.join(config.INPUT_PATH,"train_folds.csv"))[["image_name", "target", "kfold"]]
        df = df.loc[df.kfold.isin(folds)].reset_index(drop= True)

        image_names = df.image_name.to_numpy()
        if config.resize_on_load:
            self.image_paths = [os.path.join(config.DATA_PATH, "jpeg", "train", 
                                    file_name + ".jpg") for file_name in image_names]
        else:
            self.image_paths = [os.path.join(config.INPUT_PATH, "processed_img_train", 
                                    file_name + ".jpg") for file_name in image_names]

        self.target = torch.tensor(df.target.to_numpy())
        self.transforms = transforms
        self.channel_first = channel_first 
        #Whether channel dimension is mentioned at the beginning of the image array


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

    
        target = self.target[idx]

        image = Image.open(self.image_paths[idx])

        if config.resize_on_load:
            image = T.Resize(config.IMAGE_SIZE)(image)

        if not self.transforms:
            self.transforms = T.ToTensor()
        
        image = self.transforms(image)
        return image, target





#Tests
#data = SkinCancerDataset([0])
#print(data)
#print(data[0][1].shape)