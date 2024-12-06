from .models.resnet import resnet18

def main():
	model = resnet18(num_classes=10, use_ws=True)
	print(model)


if __name__ == "__main__":
	main()

