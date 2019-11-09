import os

import pandas as pd
from PIL import Image
from torchvision import transforms, datasets


IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"


def obtain_smallest_image_size(img_dir):
    """Determine the smallest image size in a given image directory."""
    image_sizes = {}


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