import torch
import torch.nn as nn
import torch.nn.functional as F

class WSConv2d(nn.Conv2d):
	def __init__(self,in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
		super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

	def forward(self, x):
		weight = self.weight
		weight_mean = weight.mean(dim=(1,2,3), keep_dim=True)
		weight = weight - weight_mean
		std = weight.view(weight.size(0), -1).std(dim=1, keepdim=True) + 1e-5
		weight = weight/std.view(-1,1,1,1)

		return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
