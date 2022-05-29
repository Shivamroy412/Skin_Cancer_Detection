import torch.nn as nn
from torch.nn import functional as F
import torchvision.models 
import torch
from tqdm import tqdm


class ResNext_model(nn.Module):
    def __init__(self, freeze_pretrained_weights = True, 
                mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)):

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


        self.mean = mean
        self.std = std


        def forward(self, images):
            return self.model(images)

        
#Normalize using the mean and standard deviation of this particular dataset or imagenet values
def get_mean_std(data_loader):
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
