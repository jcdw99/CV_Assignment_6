# exposes PyTorch Object
from typing_extensions import Self
from torch.nn import Module
# PyTorch's Convolutional Layers
from torch.nn import Conv2d
# PyTorch's Fully Connected Layers
from torch.nn import Linear
# 2D MaxPooling to reduce spacial dimensions
from torch.nn import MaxPool2d
# Relu Activation
from torch.nn import ReLU
# Softmax to convert to probabilities
from torch.nn import LogSoftmax
# Flatten output of a 2D layer (CONV or POOL) so that fully connected can be applied
from torch import flatten

import torch

class Q1_model(Module):
	def __init__(self, numChannels, classes):
		# call the parent constructor
		super(Q1_model, self).__init__()
		
        # initialize first set of CONV => RELU => POOL layers
		self.conv1 = Conv2d(in_channels=numChannels, out_channels=10,
			kernel_size=(5, 5))
		self.batch1 = torch.nn.BatchNorm2d(10)
		self.relu1 = ReLU()
		self.maxpool1 = MaxPool2d(kernel_size=(2, 2))
		
        # initialize second set of CONV => RELU => POOL layers
		self.conv2 = Conv2d(in_channels=10, out_channels=30,
			kernel_size=(5, 5))
		self.batch2 = torch.nn.BatchNorm2d(30)
		
		self.relu2 = ReLU()
		self.maxpool2 = MaxPool2d(kernel_size=(2, 2))

        # initialize third set of CONV => RELU => POOL layers
		self.conv3 = Conv2d(in_channels=30, out_channels=50,
			kernel_size=(3, 3))
		self.relu3 = ReLU()
		self.maxpool3 = MaxPool2d(kernel_size=(2, 2))
		self.batch3 = torch.nn.BatchNorm2d(50)

        # initialize first (and only) set of FC => RELU layers
		self.fc1 = Linear(in_features=31250, out_features=500)
		self.relu4 = ReLU()
		# initialize our softmax classifier
		self.fc2 = Linear(in_features=500, out_features=classes)
		self.logSoftmax = LogSoftmax(dim=1)
		self.dropout = torch.nn.Dropout(0.5)  # 50% Probability


	def forward(self, x):
		# pass the input through our first set of CONV => RELU =>
		# POOL layers
		x = self.conv1(x)
		x = self.batch1(x)
		x = self.relu1(x)
		x = self.maxpool1(x)
		# pass the output from the previous layer through the second
		# set of CONV => RELU => POOL layers
		x = self.conv2(x)
		x = self.batch2(x)
		x = self.relu2(x)
		x = self.maxpool2(x)

		x = self.dropout(x)
        # pass the output from the previous layer through the third
		# set of CONV => RELU => POOL layers
		x = self.conv3(x)
		x = self.batch3(x)
		x = self.relu3(x)
		x = self.maxpool3(x)

		# flatten the output from the previous layer and pass it
		# through our only set of FC => RELU layers
		x = flatten(x, 1)
		x = self.fc1(x)
		x = self.relu4(x)

		# pass the output to our softmax classifier to get our output
		# predictions
		x = self.fc2(x)
		output = self.logSoftmax(x)
		# return the output predictions
		return output