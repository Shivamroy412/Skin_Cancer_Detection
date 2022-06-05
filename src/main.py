import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

import config
from dataset import SkinCancerDataset
import models


def main(train_folds: list, val_fold: list, mean = (0.485, 0.456, 0.406), std = (229, 224, 225)):

    #Definig the model
    model = models.ResNext_model(False)
    #Mounting model in GPU if available
    model.to(config.DEVICE)


    #Define Transformations
    train_tranforms = T.Compose([T.RandomHorizontalFlip(), 
                                 T.RandomVerticalFlip(),
                                 T.Normalize(mean, std), 
                                 T.ToTensor()])

    val_tranforms = T.Compose([ T.Normalize(mean, std), 
                                 T.ToTensor()])
    

    #Defining Train Dataset and DataLoader
    train_dataset = SkinCancerDataset(train_folds, transforms= train_tranforms)
    train_loader = DataLoader(dataset= train_dataset, 
                                batch_size = config.TRAIN_BATCH_SIZE, 
                                shuffle = True, 
                                num_workers = 4)

    #Defining Validation Dataset and DataLoader
    val_dataset = SkinCancerDataset(val_fold, transforms= val_tranforms)
    val_loader = DataLoader(dataset= val_dataset, 
                                batch_size = config.VAL_BATCH_SIZE, 
                                shuffle = False, 
                                num_workers = 4)
 

    #Get mean and standard deviation over the entire dataset
    mean, std = models.get_mean_std(train_loader)

    #Defining the optimizer, leaving the default hyperparameters like lr
    optimizer = torch.optim.Adam(model.parameters())

    #Defining a scheduler that reduces the LR as the model starts plateuing 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = "min", 
                                                            patience=5, factor = 0.3, verbose= True)


    #Running Epochs 
    for epoch in range(config.EPOCHS):

        #Training
        model.train() #Activating the train mode for the model

        for batch_idx, data in tqdm(enumerate(train_loader), 
                                    total = int(len(train_dataset) / train_loader.batch_size)):
            images, labels = data

            #Mounting data to GPU if available
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            #Empyting gradients
            optimizer.zero_grad()

            #Calculating outputs from the model
            outputs = model(images)

            #Calculating loss
            #The BCEWithLogitsLoss compensates for the absence of Sigmoid function around outputs
            #and then works same as Cross Entropy Loss
            loss = torch.nn.BCEWithLogitsLoss()(outputs, labels.reshape(-1, 1).as_type(outputs))

            #Backward Propagation Step
            loss.backward()
            optimizer.step()
        #___________________________________________________________________

        #Validation
        model.eval() #Activating the eval mode for the model

        final_loss = 0 #Initiating final loss
        counter = 0 # Used as a denominator to calculate the average loss

        for batch_idx, data in tqdm(enumerate(val_loader), 
                                    total = int(len(val_dataset) / val_loader.batch_size)):
            couner += 1
            images, labels = data

            #Mounting data to GPU if available
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            #Calculating outputs from the model
            outputs = model(images)

            #Calculating loss
            #The BCEWithLogitsLoss compensates for the absence of Sigmoid function around outputs
            #and then works same as Cross Entropy Loss
            loss = torch.nn.BCEWithLogitsLoss()(outputs, labels.reshape(-1, 1).as_type(outputs))

            final_loss += loss
        
        return final_loss / counter


if __name__ == "__main__":

    #Get mean and standard deviation over the entire dataset
    data_mean, data_std = models.get_mean_std()

    for validation_fold in range(config.N_FOLDS):

        train_folds = [i for i in range(config.N_FOLDS) if i != validation_fold]
        #Gives a list containing all folds except the particular validation_fold

        #Call training function and start training the model
        mean_loss = main(train_folds = train_folds, val_fold = [validation_fold], mean = data_mean, std = data_std )