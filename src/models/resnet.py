import torch 
import torch.nn as nn
import torch.nn.functional as F 

from src.utils.weight_standardization import WSConv2d

class BasicBlock(nn.Module):
	def __init__(self, inplanes, planes, strides =1, downsample = None, use_ws = False, num_groups = 32):
		super().__init__()
		conv_layer = WSConv2d if use_ws else nn.Conv2d

		self.conv1 = conv_layer(inplanes, planes, kernel_size=3, stride=strides, padding=1, bias=False)
		self.gn1 = nn.GroupNorm(num_groups, planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv_layer(planes, planes, kernel_size=3, padding=1, bias=False)
		self.gn2 = nn.GroupNorm(num_groups, planes)
		self.downsample = downsample

	def forward(self, x):
		out = self.conv1(x)
		out = self.gn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.gn2(out)

		if downsample is not None:
			x = self.downsample(x)

		out += x

		return self.relu(out)

class Resnet(nn.Module):
	def __init__(self, block, layers, num_classes=1000, use_ws=False):
		super().__init__()
		self.inplanes = 64
		conv_layer = WSConv2d if use_ws else nn.Conv2d

		self.conv1 = conv_layer(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, paddding=1)


	def _make_layer(self, ):
		pass

	def forward(self,):
		pass
