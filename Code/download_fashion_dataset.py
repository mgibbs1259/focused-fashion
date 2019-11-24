from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import sys
import json
import urllib3
import multiprocessing

import PIL
from PIL import Image
from tqdm import tqdm
from urllib3.util import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


TRAIN_JSON = "/home/ubuntu/Final-Project-Group8/Data/train.json"
TRAIN_IMG_DIR = "/home/ubuntu/Final-Project-Group8/Data/train"


def parse_dataset(_dataset, _outdir):
    """Parse the dataset to create a list of tuple containing absolute path and url of image."""
    _fnames_urls = []
    with open(_dataset, 'r') as f:
        data = json.load(f)
        for image in data["images"]:
            url = image["url"]
            fname = os.path.join(_outdir, "{}.jpg".format(image["imageId"]))
            _fnames_urls.append((fname, url))
    return _fnames_urls[:500000]


def download_image(fnames_and_urls):
    """Download image and save its with 90% quality as JPG format.
    Skip image downloading if image already exists at given path."""
    fname, url = fnames_and_urls
    if not os.path.exists(fname):
        http = urllib3.PoolManager(retries=Retry(connect=3, read=2, redirect=3))
        response = http.request("GET", url)
        image = Image.open(io.BytesIO(response.data))
        image_rgb = image.convert("RGB")
        image_rgb.save(fname, format='JPEG', quality=90)


if __name__ == '__main__':
    # Parse train json
    fnames_urls = parse_dataset(TRAIN_JSON, TRAIN_IMG_DIR)
    # Download images
    pool = multiprocessing.Pool(processes=12)
    with tqdm(total=len(fnames_urls)) as progress_bar:
        for _ in pool.imap_unordered(download_image, fnames_urls):
            progress_bar.update(1)
    sys.exit(1)
