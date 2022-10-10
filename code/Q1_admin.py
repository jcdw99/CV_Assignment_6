import torch
from PIL import Image, ImageOps
import numpy as np
import os

# PREPROCESSING
def crop_faces(imgsize):
    """ Crop the images in resources/images/* to a fixed size. This size is specified by provided argument. 
        Cropping is implemented using the resize() functionality provided by the PIL image library. """
    directories = ['Coast', 'Forest', 'Highway', 'Kitchen', 'Mountain', 'Office', 'Store', 'Street', 'Suburb']
    for directory in directories:
        for traintest in ["train", "test"]:
            files = os.listdir(f'resources/images/{directory}/{traintest}/')
            for pic in files:
                img = Image.open(f'resources/images/{directory}/{traintest}/{pic}')
                img = img.resize(imgsize)
                img.save(f'resources/images/{directory}/{traintest}/{pic}')


# get either training or testing data
def get_data(traintest):
    directories = ['Coast', 'Forest', 'Highway', 'Kitchen', 'Mountain', 'Office', 'Store', 'Street', 'Suburb']
    results = []
    labels = []
    for directory in directories:
        files = os.listdir(f'../resources/images/{directory}/{traintest}/')
        for pic in files:
            img = Image.open(f'../resources/images/{directory}/{traintest}/{pic}')
            results.append(np.array(img).astype(np.uint8))
            labels.append(directory)
    return results, labels

if __name__ == "__main__":
    pass