import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

def preprocess_val(data_dir):
	val_dir = os.path.join(data_dir, "val")
	val_images = os.path.join(val_dir, "images")
	val_annotations = os.path.join(val_dir, "val_annotations.txt")

	with open(val_annotations, "r") as f:
		for line in f:
			parts = line.split("\t")
			img_name, class_name = parts[0], parts[1]
			class_dir = os.path.join(val_images, class_name)
			if not os.path.exists(class_dir):
				os.makedirs(class_dir)
			os.rename(os.path.join(val_images, img_name), os.path.join(class_dir, img_name))

def get_imagenet_data(data_dir, batch_size=256):
	transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=transform_train)
    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform_val)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def TinyImageNetTestDataset(Dataset):
	def __init__(self, data_dir, transform=None):
		self.data_dir = data_dir
		self.transform = transform
		self.image_paths = [os.path.join(data_dir,fname) for fname in os.listdir(data_dir) if fname.endswith(".JPEG")]

	def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
    	img_path = self.image_paths[idx]
    	image = Image.open(img_path).convert("RGB")
    	if self.transform:
    		image = self.transform(image)
    	return image, img_path