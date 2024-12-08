import torch
import torch.nn as nn
import torch.optim as optim

from .models.resnet import resnet18
from .data.dataloaders import get_imagenet_data

def train_model(model, train_loader, val_loader, learning_rate=0.1):
	device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
	model = model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
	scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=30, gamma=0.1)

	#Training loop to be implemented

def main():
	model = resnet18(num_classes=10, use_ws=True)
	print(model)


if __name__ == "__main__":
	model = resnet18(num_classes=1000, use_ws=True)
