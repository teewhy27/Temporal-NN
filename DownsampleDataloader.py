
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import DataLoader, Subset
class Downsample(object):
    def __init__(self, size=(10, 10)):
        self.size = size

    def __call__(self, img):
        # Add a batch and channel dimension, resize, then remove batch dimension
        return F.interpolate(img.unsqueeze(0), size=self.size, mode='area').squeeze(0)
    


# Define the transformation pipeline
transform_pipeline = transforms.Compose([
    transforms.ToTensor(),
    Downsample(size=(10, 10))
])


# Load MNIST dataset with the custom transform
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_pipeline)
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_pipeline)



#Function creates a new dataset taking batch size and subset of the dataset as input

def train_loader(batch_size,subset=False,subset_indices=None):
    if subset:
        subset_indices = torch.arange(0,subset_indices)
        mnist_subset = Subset(mnist_trainset, subset_indices)
        train_loader = DataLoader(mnist_subset, batch_size=100, shuffle=True)
    else:
        train_loader = DataLoader(mnist_trainset, batch_size, shuffle=True)
    return train_loader
def test_loader(batch_size):
    test_loader = DataLoader(mnist_testset, batch_size, shuffle=False)
    return test_loader
