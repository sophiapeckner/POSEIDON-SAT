"""
This file is from the original POSEIDON implementation, with minor tweaks to remove the use of a class and to
use the adjusted paths from the ShipRSImageNet dataset.

-------------------------------------------------------------------------------

MIT License

Copyright (c) 2022 Pablo Ruiz Ponce

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import os
import json
import shutil
import pandas as pd
from tqdm import tqdm
from PIL import Image
from poseidon.utils.misc import ignore_extended_attributes, NpEncoder


# Function to normalize an image
def normalize_image(img_row, min_width, train_annotations, images_path):
    # Get the annotations for this image
    annotations = pd.DataFrame(train_annotations['annotations'])
    instances = annotations[annotations['image_id'] == img_row['id']]

    # Calculate the normalization factor
    normalize_factor = min_width / img_row['width']

    # Open the image and resize it
    img_path = os.path.join(images_path, img_row['file_name'])
    image = Image.open(img_path)
    img_row['width'] = round(img_row['width'] * normalize_factor)
    img_row['height'] = round(img_row['height'] * normalize_factor)
    image = image.resize((int(img_row['width']), int(img_row['height'])))
    image.save(img_path)

    # Normalize the bounding boxes
    for idx, instance in instances.iterrows():
        x, y, w, h = instance['bbox']
        x *= normalize_factor
        y *= normalize_factor
        w *= normalize_factor
        h *= normalize_factor
        
        instance['bbox'] = [int(x), int(y), int(w), int(h)]
        instance['area'] = w * h
        
        instance['segmentation'][0] = [float(round(x * normalize_factor)) for x in instance['segmentation'][0]]
        
        annotations.loc[idx] = instance
        
    train_annotations['annotations'] = annotations.to_dict('records')

# Function to normalize the dataset
def normalize(source_path, output_path, images_to_normalize, image_set):
    annotation_file = f"ShipRSImageNet_bbox_{image_set}.json"

    # Remove the previous dataset if it exists
    if os.path.exists(output_path):
        print("Removing previous dataset in the specified output path...")
        shutil.rmtree(output_path, onerror=ignore_extended_attributes)

    # Copy the original dataset
    print("Copying the original dataset...")
    os.mkdir(output_path)
    shutil.copy(os.path.join(source_path, "COCO_Format", annotation_file), os.path.join(output_path, annotation_file))
    shutil.copytree(os.path.join(source_path, "VOC_Format", "JPEGImages"),
            os.path.join(output_path, 'images'),
            ignore=shutil.ignore_patterns('.*'))

    # Update the base path
    source_path = output_path

    # Load the annotations
    train_annotations_path = os.path.join(source_path, annotation_file)
    images_path = os.path.join(source_path, 'images')

    with open(train_annotations_path) as f:
        train_annotations = json.load(f)

    # Normalize the images
    images = pd.DataFrame(train_annotations['images'])
    selected_images = images[images['file_name'].isin(images_to_normalize)]
    min_width = selected_images['width'].min()

    tqdm.pandas()

    print("Normalizing images from the train set")
    selected_images.progress_apply(lambda x: normalize_image(x, min_width=min_width, train_annotations=train_annotations, images_path=images_path), axis=1)

    # Save the annotations
    with open(train_annotations_path, 'w') as f:
        json.dump(train_annotations, f,  cls=NpEncoder)

    # Return the path to the normalized dataset
    return output_path
