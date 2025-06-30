from urllib import request

import numpy as np
import skimage


def download_image(url):
    if url:
        image = request.urlopen(url)
        return skimage.io.imread(image, plugin='pil')
    else:
        return None


def remove_alpha_channel(original_image):
    image = np.array(original_image)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    return image


def apply_grayfilter(original_image):
    return skimage.color.rgb2gray(original_image)


def preprocess_img(data: np.ndarray | str):
    image = download_image(data) if isinstance(data, str) else data
    image = remove_alpha_channel(image)
    image = apply_grayfilter(image)
    return image
