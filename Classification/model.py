# This defines the model
import torch.nn as nn

from torchvision import models


import torch.nn as nn
import torch


class MyEnsemble(nn.Module):

    def __init__(self, modelA, modelB, input):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.fc1 = nn.Linear(input, 2)

    def forward(self, x):
        out1 = self.modelA(x)
        out2 = self.modelB(x)
        out = out1 + out2
        x = self.fc1(out)
        return torch.softmax(x, dim=1)
        
"""
def build_model( num_classes=2):
    model1 = models.resnet101(pretrained=True)
    num_ftrs1 = model1.fc.in_features
    model1.fc = nn.Linear(num_ftrs1, num_classes)

    model2 = models.resnet152(pretrained=True)
    num_ftrs2 = model2.fc.in_features
    model2.fc = nn.Linear(num_ftrs2, num_classes)

    model = MyEnsemble(model1, model2,num_classes)

    return model

"""
def build_model(pretrained=True, fine_tune=True, num_classes=10):
    model=models.resnet152(pretrained=True)
    num_ftrs1 = model.fc.in_features
    model.fc = nn.Linear(num_ftrs1, num_classes)
    return model


    