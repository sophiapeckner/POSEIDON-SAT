import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from utils.misc import ignore_extended_attributes, NpEncoder


# Function to normalize an image
def normalize_image(img_row, min_width, train_annotations, images_path):
    # Get the annotations for this image
    annotations = pd.DataFrame(train_annotations['annotations'])
    instances = annotations[annotations['image_id'] == img_row['id']]

    # Calculate the normalization factor
    normalize_factor = min_width / img_row['width']

    # Open the image and resize it
    img_path = os.path.join(images_path, 'train', img_row['file_name'])
    image = Image.open(img_path)
    img_row['width'] *= normalize_factor
    img_row['height'] *= normalize_factor
    image.resize((int(img_row['width']), int(img_row['height'])))
    image.save(img_path)

    # Normalize the bounding boxes
    for i, instance in instances.iterrows():
        x, y, w, h = instance['bbox']
        center = (x + w / 2, y + h / 2)
        w = w * normalize_factor
        h = h * normalize_factor
        x = center[0] - w / 2
        y = center[1] - h / 2
        instance['bbox'] = [x, y, w, h]

        """
        instance["bbox"][0] *= normalize_factor
        instance["bbox"][1] *= normalize_factor
        instance["bbox"][2] *= normalize_factor
        instance["bbox"][3] *= normalize_factor
        instance["area"] *= normalize_factor
        """

        train_annotations['annotations'] = annotations.to_dict('records')

# Function to normalize the dataset
def normalize(output_path, base_path):
    # Remove the previous dataset if it exists
    if os.path.exists(output_path):
        print("Removing previous dataset in the specified path")
        shutil.rmtree(output_path, onerror=ignore_extended_attributes)

    # Copy the original dataset
    print("Copying the original dataset")
    shutil.copytree(base_path, 
            os.path.join(output_path),
            ignore=shutil.ignore_patterns('.*'))

    # Update the base path
    base_path = output_path

    # Load the annotations
    train_annotations_path = os.path.join(base_path, "annotations", "instances_train.json")
    images_path = os.path.join(base_path, "images")

    with open(train_annotations_path) as f:
        train_annotations = json.load(f)

    # Normalize the images
    images = pd.DataFrame(train_annotations['images'])
    min_width = images['width'].min()

    tqdm.pandas()

    print("Normalizing images from the train set")
    images.progress_apply(lambda x: normalize_image(x, min_width=min_width, train_annotations=train_annotations, images_path=images_path), axis=1)

    # Save the annotations
    with open(train_annotations_path, 'w') as f:
        json.dump(train_annotations, f,  cls=NpEncoder)

    # Return the path to the normalized dataset
    return output_path
