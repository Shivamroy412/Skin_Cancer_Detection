import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.models 
import torch
from tqdm import tqdm
import os
import pickle
from dataset import SkinCancerDataset
import config


class ResNext_model(nn.Module):
    def __init__(self, freeze_pretrained_weights = True):

        super(ResNext_model, self).__init__()

        self.model = torchvision.models.resnext50_32x4d(pretrained = False)
        last_layer_input_dim = self.model.fc.in_features #Getting input dimensions for last layer
        self.model.fc = nn.Linear(last_layer_input_dim, 1) #Modifying the last layer

        self.freeze_pretrained_weights = freeze_pretrained_weights

        #Freezing all layer weights except the last layer for training
        if self.freeze_pretrained_weights:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.fc.requires_grad = True

        def forward(self, images):
            return self.model(images)

        
#Normalize using the mean and standard deviation of this particular dataset or imagenet values
def get_mean_std():

    mean_std_file_name = "mean_std.pkl"
    mean_std_file_path = os.path.join(config.MODEL_PATH, mean_std_file_name)

    if os.path.exists(mean_std_file_path):
        with open(mean_std_file_path, 'rb') as file:
            mean, std = pickle.load(file)

    else:
        all_folds = [i for i in range(config.N_FOLDS)]

        dataset = SkinCancerDataset(all_folds)
        data_loader = DataLoader(dataset= dataset, 
                                    batch_size = config.TRAIN_BATCH_SIZE, 
                                    shuffle = False, 
                                    num_workers = 4)

        channel_wise_sum, channel_wise_squared_sum, num_batches = 0, 0, 0

        print("Calculating mean and standard deviation over the entire dataset")

        for image_batch, _ in tqdm(data_loader):
            channel_wise_sum += torch.mean(image_batch, dim = [0, 2, 3])
            channel_wise_squared_sum += torch.mean(image_batch**2, dim = [0, 2, 3])
            #The [0, 2, 3] implies the dimensions of batchsize, height and width
            #This would give the mean for each channel

            num_batches += 1

        mean = channel_wise_sum / num_batches
        std = ( (channel_wise_squared_sum / num_batches) - mean**2 ) ** 0.5

        print(f"Dataset Mean {mean}  Standard Deviation  {std}")

        if not os.path.exists(config.MODEL_PATH):
            os.makedirs(config.MODEL_PATH)

        with open(mean_std_file_path, 'wb') as file:
            pickle.dump((mean, std), file)

    return mean, std


#Tests
# from dataset import SkinCancerDataset
# from torchsummary import summary

# model = ResNext_model()
# model = torchvision.models.resnext50_32x4d(pretrained = False)
# print(model.fc.in_features)
# print(summary(model, (3, 512, 512)))

# data = SkinCancerDataset([0])
# image , _ = data[0]
# print(image.shape)
# print(model(image.unsqueeze(0)))
# print(model)
