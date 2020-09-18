import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import cv2
import pickle

import torchvision.models 

'''
OPTIONS
-ResNet18 classification
-ResNet50 classification
-ResNet50 segmentation (CityScapes oder aus torchvision)
-SemSeg
'''

#TODO: evaluate a bigger network and/or pre-trained for segmentation (first: better pairs?)

#[1, 512, 23, 40]
def get_encoder_resnet18():
    model=torchvision.models.resnet18(pretrained=True)
    extractor = nn.Sequential(model.conv1,model.bn1,model.relu,model.maxpool,model.layer1,model.layer2,model.layer3,model.layer4)    
    return extractor

def get_encoder_fcn_resnet101():
    model=torchvision.models.segmentation.fcn_resnet101(pretrained=True)

