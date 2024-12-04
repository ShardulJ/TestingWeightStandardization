import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Conv2d):
	def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
		super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

	def forward(self, x):
		weight = self.weight
		weight_mean = weight.mean(dim=(1,2,3), keep_dim=True)
		weight = weight - weight_mean
		return x #temporary
