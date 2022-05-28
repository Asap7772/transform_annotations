import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

folder_path = './images/'
os.makedirs(folder_path, exist_ok=True)

def generate_random_images(n_images = 5, n_pixels=128):
    """
    Generate n_images images with n_pixels pixels.
    """
    size = n_images, n_pixels, n_pixels, 3
    images = np.random.randint(low=0, high=255, size=size, dtype=np.uint8)

    for i in range(n_images):
        path = folder_path + 'image_' + str(i) + '.png'
        im = Image.fromarray(images[i])
        im.save(path)
        
    return images

generate_random_images()