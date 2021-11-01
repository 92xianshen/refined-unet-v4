import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

task_name = '../result/a=100, b=.25, r=3/'

""" Apply the given mask to the image. """

def apply_mask(image, mask, mask_value, color, alpha=.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == mask_value, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])

    return image

image_path = '../datasets/false_colorized/'

for f in os.listdir(task_name):
    if os.path.splitext(f)[-1] == '.npz':
        print(f)
        
        mask_name = os.path.join(task_name, f)
        image_name = os.path.join(image_path, f.replace('rfn.npz', 'sr_bands.png'))
        save_name = os.path.join(task_name, f.replace('rfn.npz', 'sr_bands_masked.png'))

        image = np.array(Image.open(image_name), dtype=np.float32)
        mask = np.load(mask_name)['arr_0']

        print(mask.shape, mask.max(), mask.min())

        cloud, cloud_color = 3, [115, 223, 255]
        shadow, shadow_color = 2, [38, 115, 0]

        mask_image = apply_mask(image, mask, cloud, cloud_color)
        mask_image = apply_mask(mask_image, mask, shadow, shadow_color)

        mask_image = np.uint8(mask_image)

        plt.imsave(save_name, mask_image)