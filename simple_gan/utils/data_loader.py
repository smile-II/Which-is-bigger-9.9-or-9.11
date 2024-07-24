import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    mnist = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    data_loader = DataLoader(dataset=mnist, batch_size=batch_size, shuffle=True)
    
    return data_loader
