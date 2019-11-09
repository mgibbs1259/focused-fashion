import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms, datasets


IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"


def obtain_smallest_image_size(img_dir):
    """Returns the dimensions of the smallest image in terms of area."""
    image_sizes = {}
    for image in os.listdir(img_dir):
        try:
            with Image.open(image) as img:
                width, height = img.size
                image_sizes[image] = (width, height, width*height)
        except:
            image_sizes[image] = (1000, 1000, 1000)
    smallest_image = min(image_sizes, key=lambda k: image_sizes[k][2])
    return image_sizes[smallest_image]


def create_data_loader(img_dir):
    """Create an image loader for the model."""



# data_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
# hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                            transform=data_transform)
# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                              batch_size=4, shuffle=True,
#                                              num_workers=4)
