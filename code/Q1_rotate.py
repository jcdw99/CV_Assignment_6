
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np
import torch


class rotateDataset(Dataset):

    def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y
        self.label_names =  ['Coast', 'Forest', 'Highway', 'Kitchen', 'Mountain', 'Office', 'Store', 'Street', 'Suburb']
        self.directory_to_class = {'Coast':0, 'Forest':1, 'Highway':2, 'Kitchen':3, 'Mountain':4, 'Office':5, 'Store':6, 'Street':7, 'Suburb':8}
        self.class_to_directory = {0:'Coast', 1:'Forest', 2:'Highway', 3:'Kitchen', 4:'Mountain', 5:'Office', 6:'Store', 7:'Street', 8:'Suburb'}

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.X[index]
        X = self.transform(np.array(image))
        y = torch.tensor(self.label_names.index(self.y[index]))
        return X, y

    transform = T.Compose([
        T.ToPILImage(),
        T.RandomRotation(90),
        T.ToTensor()])

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    
def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)   
        break 

