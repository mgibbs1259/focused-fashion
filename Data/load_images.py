import os

import torch
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader


IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/output_validation"


def obtain_smallest_image_size(img_dir):
    """Returns the dimensions of the smallest image in terms of area."""
    image_sizes = {}
    for image in os.listdir(img_dir):
        print(image)
        try:
            with Image.open(os.path.join(IMG_DIR, image)) as img:
                width, height = img.size
                image_sizes[image] = (width, height, width * height)
                print(image_sizes[image])
        except:
            pass
    smallest_image = min(image_sizes, key=lambda k: image_sizes[k][2])
    return image_sizes[smallest_image]






# def create_data_loader(img_dir):
#     """Returns an image loader for the model."""
#     transform = transforms.Compose([transforms.Resize((32, 32), interpolation=Image.BILINEAR),
#                                     transforms.ToTensor()])
#     dataset = datasets.ImageFolder(root=IMG_DIR,
#                                    transform=transform)
#
#     loader = DataLoader(dataset, batch_size=512, shuffle=True)
#     return loader
