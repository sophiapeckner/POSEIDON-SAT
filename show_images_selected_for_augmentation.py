import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

def show_image(image_file):
    print(f'Showing image {image_file}')
    img = Image.open(os.path.join('ShipRSImageNet_V1', 'VOC_Format', 'JPEGImages', image_file))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        show_image(sys.argv[1])
        sys.exit(0)

    with open('selected_for_augmentation.txt', 'r') as f:
        selected_images = f.read().splitlines()

    for image_file in selected_images:
        show_image(image_file)