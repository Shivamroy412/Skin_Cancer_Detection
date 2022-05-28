import torch.nn as nn
from torch.nn import functional as F
import torchvision.models 


class ResNext_model(nn.Module):
    def __init__(self):
        super(ResNext_model, self).__init__()

        self.model = torchvision.models.resnext50_32x4d(pretrained = False)
        last_layer_input_dim = self.model.fc.in_features #Getting input dimensions for last layer
        self.model.fc = nn.Linear(last_layer_input_dim, 1) #Modifying the last layer

    def forward(self, images):
        return self.model(images)



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
