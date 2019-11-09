import os
from collections import OrderedDict

import pandas as pd
from PIL import Image
from torchvision import transforms, datasets


IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"


def obtain_image_sizes(img_dir):
    """Obtain the size of each image in a given image directory."""
    image_sizes = {}
    for image in os.listdir(img_dir):
        try:
            with Image.open(image) as img:
                width, height = img.size
                image_sizes[image] = (width, height)
        except:
            image_sizes[image] = (0, 0)
    return image_sizes


def determine_smallest_image_size(img_size_dict):
    """Determine the smallest nonzero image size from a given image size dictionary."""



def create_data_loader(img_dir):
    """Create an image loader for the model."""




# data_transform = transforms.Compose([
#         transforms.RandomSizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
#     ])
# hymenoptera_dataset = datasets.ImageFolder(root='hymenoptera_data/train',
#                                            transform=data_transform)
# dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
#                                              batch_size=4, shuffle=True,
#                                              num_workers=4)