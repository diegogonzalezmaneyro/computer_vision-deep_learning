## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # self.conv1 = nn.Conv2d(1, 32, 4)
        
        #### Following the paper recommendation ###
        # Convolutional layers:
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)        
        
        # Pooling layers: use a pool shape of (2, 2), with non-overlapping strides and no zero padding
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout layers: probability is increased from 0.1 to 0.6 from Dropout1 to Dropout6, with a step size of 0.1
        self.conv1_drop = nn.Dropout(p=0.1)
        self.conv2_drop = nn.Dropout(p=0.2)
        self.conv3_drop = nn.Dropout(p=0.3)
        self.conv4_drop = nn.Dropout(p=0.4)
        self.conv5_drop = nn.Dropout(p=0.5)
        self.conv6_drop = nn.Dropout(p=0.6)
        
        # Dense layers: Using as input 224x224
        # Layer shape conv1-pool (32, 110, 110)
        # Layer shape conv2-pool (64, 54, 54)
        # Layer shape conv3-pool (128, 26, 26)
        # Layer shape conv4-pool (256, 13, 13)
        self.fc1 = nn.Linear(256*13*13, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        x = self.pool(F.elu(self.conv1(x)))
        x = self.conv1_drop(x)
        
        x = self.pool(F.elu(self.conv2(x)))
        x = self.conv2_drop(x)
                            
        x = self.pool(F.elu(self.conv3(x)))
        x = self.conv3_drop(x)
        
        x = self.pool(F.elu(self.conv4(x)))
        x = self.conv4_drop(x)

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # three linear layers with dropout in between
        x = F.elu(self.fc1(x))
        x = self.conv5_drop(x)

        x = F.relu(self.fc2(x))
        x = self.conv6_drop(x)

        x = self.fc3(x)



        return x
