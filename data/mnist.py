from torchvision import datasets

train_dataset = datasets.MNIST('data', train=True, download=True)
val_dataset = datasets.MNIST('data', train=False, download=True)

