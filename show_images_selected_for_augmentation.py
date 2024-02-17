import sys

from shiprsimagenet import ShipRSImageNet

def show_image(dataset: ShipRSImageNet, image_file):
    print(f'Showing image {image_file}')
    dataset.get_image(image_file).show()

if __name__ == '__main__':
    """
    Usage:
    python show_images_selected_for_augmentation.py [image_file] [dataset_path]
    
    If no image file is specified, it will show all the images in the 'selected_for_augmentation.txt' file.
    
    If no dataset path is specified, it will use the default 'ShipRSImageNet_V1' dataset.
    """
    
    if len(sys.argv) > 2:
        dataset = ShipRSImageNet(sys.argv[2])
    else:
        dataset = ShipRSImageNet('ShipRSImageNet_V1')

    if len(sys.argv) > 1:
        show_image(dataset, sys.argv[1])
        sys.exit(0)
    
    with open('selected_for_augmentation.txt', 'r') as f:
        selected_images = f.read().splitlines()

    for image_file in selected_images:
        show_image(dataset, image_file)