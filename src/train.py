import torch
from torch.utils.data import DataLoader

import config
from dataset import SkinCancerDataset
import models


def train(train_folds: list, val_fold: list):

    #Defining Train Dataset and DataLoader
    train_dataset = SkinCancerDataset(train_folds)
    train_loader = DataLoader(dataset= train_dataset, 
                                batch_size = config.TRAIN_BATCH_SIZE, 
                                shuffle = True, 
                                num_workers = 4)

    #Defining Validation Dataset and DataLoader
    val_dataset = SkinCancerDataset(val_fold)
    val_loader = DataLoader(dataset= val_dataset, 
                                batch_size = config.VAL_BATCH_SIZE, 
                                shuffle = False, 
                                num_workers = 4)

    

    #Get mean and standard deviation over the entire dataset
    mean, std = models.get_mean_std(train_loader)

    print(mean, std)



if __name__ == "__main__":



    for validation_fold in range(config.N_FOLDS):
        train_folds = [i for i in range(10) if i != validation_fold]
        #Gives a list containing all folds except the particular validation_fold

        train(train_folds = train_folds, val_fold = [validation_fold])