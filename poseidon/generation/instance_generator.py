"""
Modified instance generator implementation.

Copyright (c) 2024 Kyler Nelson

Several portions of this file are from the original POSEIDON implementation in
virtually unmodified form, though most of this file has been significantly modified
from its original. The modified version of this file is subject to the above copyright
and is made available under the license defined for the full source code repository
in which this file originates.

The license and copyright for the original POSEIDON implementation is included below
as per MIT license requirements.

-------------------------------------------------------------------------------

Original POSEIDON code implementation license

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
import numpy as np
from PIL import Image
from tqdm import tqdm
from random import Random
from pathlib import Path
from typing import Optional

from shiprsimagenet import ShipRSImageNet, LabeledImage


_INSTANCE_RESOLUTION = 0.3


class InstanceGenerator():

    def __init__(self, instances_class_name: str, instances_path: str, instances_resolution: float = _INSTANCE_RESOLUTION, seed: Optional[int] = None):
        self.instances_class = instances_class_name
        self.instances_path = instances_path
        self.instances_resolution = instances_resolution
        
        self.random = Random(seed)
        self._orig_random_state = self.random.getstate()

        self._instance_images = list(map(lambda path: Image.open(path).convert('RGBA'), Path(instances_path).iterdir()))
        self._repopulate_instance_random_select_pool()


    def _repopulate_instance_random_select_pool(self):
        self._instance_random_select_pool = list(range(len(self._instance_images)))
        self.random.shuffle(self._instance_random_select_pool)


    def reset(self, new_seed: Optional[int] = None):
        if new_seed is not None:
            self.random.seed(new_seed)
            self._orig_random_state = self.random.getstate()
        else:
            self.random.setstate(self._orig_random_state)

        self._repopulate_instance_random_select_pool()


    def _select_random_instance(self, rescale_to_spatial_resolution: Optional[float] = None):
        if (len(self._instance_random_select_pool) == 0):
            self._repopulate_instance_random_select_pool()

        image = self._instance_images[self._instance_random_select_pool.pop()]

        flip_horizontal, flip_vertical = self.random.choices([True, False], k=2)
        if flip_horizontal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_vertical:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

        if rescale_to_spatial_resolution is None:
            return image
        
        new_width = int(image.width * self.instances_resolution / rescale_to_spatial_resolution)
        new_height = int(image.height * self.instances_resolution / rescale_to_spatial_resolution)
        
        if new_width == 0 or new_height == 0:
            # Since all our extracted instances are at the same resolution, there's no sense in trying again for that image. Bail out.
            # If we had different resolutions, we could just try again with a different instance or use separate buckets for low/high resolution instances.
            # Not necessary in this specific case, so we'll just raise an error.
            raise ValueError(f"Rescaling randomly selected instance to {rescale_to_spatial_resolution} m/pixel resolution results in an instance with a 0 width or height: cannot provide instances for desired resolution")

        return image.resize((new_width, new_height))


    def _add_instances_to_image(self, source_image: LabeledImage, image_out_dir: str, image_id: int, new_instance_class_id: int, num_instances: int, annotations: pd.DataFrame) -> pd.DataFrame:
        existing_instances_in_image = annotations[annotations['image_id'] == image_id]
        next_free_id: int = annotations['id'].max() + 1
        image = Image.open(source_image.file_path).convert('RGBA')

        for iter_idx in range(num_instances):
            instance = self._select_random_instance(source_image.spatial_resolution)
            instance_id = next_free_id + iter_idx
                    
            # Just to debug the number of collisions in case of very long loop for a particular image
            iters_colliding = 0

            while True:
                instance_x = self.random.randint(0, source_image.width - instance.width)
                instance_y = self.random.randint(0, source_image.height - instance.height)

                instance_bbox = [
                                    instance_x,
                                    instance_y,
                                    instance.width,
                                    instance.height
                                ]

                instance_bbox_df = pd.DataFrame([instance_bbox], columns=['x', 'y', 'w', 'h'])
                bboxs = (existing_instances_in_image['bbox']).copy()

                if len(bboxs) == 0:
                    break

                if not _is_instance_colliding(instance_bbox_df, bboxs):
                    break

                iters_colliding += 1
                #if iters_colliding == 10:
                    #print(bboxs)
                    #print(instance_bbox)
                    #image.show()
                
            instance_area = instance.width * instance.height
            instance_series = pd.Series([instance_id, image_id, instance_bbox, instance_area, new_instance_class_id, 0, 0, []],
                                         index=['id', 'image_id', 'bbox', 'area', 'category_id', 'ignore', 'iscrowd', 'segmentation'])

            image.paste(instance, box=(instance_x, instance_y), mask=instance)
            annotations = pd.concat([annotations, pd.DataFrame([instance_series])], ignore_index=True, copy=False)

        image.save(os.path.join(image_out_dir, os.path.basename(source_image.file_path)))
        return annotations


    def augment(self, dataset: ShipRSImageNet, output_path: str, images_to_augment: 'list[str]', total_instances_to_add: int, min_instances_per_image: int, max_instances_per_image: int):
        if max_instances_per_image * len(images_to_augment) < total_instances_to_add:
            raise ValueError(f"Cannot add {total_instances_to_add} instances to {len(images_to_augment)} images when the maximum number of instances per image is {max_instances_per_image}")
        
        self._repopulate_instance_random_select_pool()

        augmented_dataset = ShipRSImageNet(output_path)
        
        print(f"Copying original dataset to {output_path}...")

        out_path = Path(output_path)
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True)
        shutil.copytree(dataset.image_path, augmented_dataset.image_path)

        annotation_file = dataset.get_coco_annotation_file_name('train')
        with open(os.path.join(dataset.coco_root_dir, annotation_file)) as f:
            labels = json.load(f)

        annotations = pd.DataFrame(labels['annotations'])
        image_names_to_ids: dict[str, int] = {img['file_name']: img['id'] for img in labels['images']}
        generated_instances_class_id: int = next(cat['id'] for cat in labels['categories'] if cat['name'] == self.instances_class)

        instances_add_count = 0
        instances_to_add_for_image = [None] * len(images_to_augment)
        image_idx = 0
        while instances_add_count < total_instances_to_add:
            current_instances_to_add_for_image = instances_to_add_for_image[image_idx] or 0

            # If assigning a number of new instances to each image does not result in the desired total number of instances to add, we'll need to loop through the images again
            # and at that point, we need to drop the minimum number of additional instances to add to 1 so that we are at least adding something while avoiding adding more than the maximum
            # number of instances for a single image.
            max_instances_to_add = max_instances_per_image - current_instances_to_add_for_image
            min_instances_to_add = min_instances_per_image if instances_to_add_for_image[image_idx] is not None else 1  
            if max_instances_to_add <= 0:
                # This image already has the maximum number of instances added
                image_idx += 1
                continue

            num_instances_to_add = self.random.randint(min_instances_to_add, max_instances_to_add)
            if num_instances_to_add + instances_add_count > total_instances_to_add:
                num_instances_to_add = total_instances_to_add - instances_add_count

            instances_to_add_for_image[image_idx] = num_instances_to_add + current_instances_to_add_for_image
            instances_add_count += num_instances_to_add

            image_idx += 1
            if image_idx >= len(instances_to_add_for_image):
                image_idx = 0

        instances_to_add_for_image: list[int] = [i for i in instances_to_add_for_image if i is not None]
        self.random.shuffle(instances_to_add_for_image) # Ensures that images listed first in the augment list do not get more instances than those listed later due to how the instances are assigned above
        
        print(f"Adding {instances_add_count} instances to {len(instances_to_add_for_image)} images...")

        for idx, image_name in enumerate(tqdm(images_to_augment[:len(instances_to_add_for_image)], desc='Adding instances')):
            if image_name not in image_names_to_ids:
                new_image_meta = _create_new_image_metadata(dataset, image_name, labels['images'])
                image_id = new_image_meta['id']
                labels['images'].append(new_image_meta)
            else:
                image_id = image_names_to_ids[image_name]

            num_instances_to_add = instances_to_add_for_image[idx]
            annotations = self._add_instances_to_image(dataset.get_image(image_name), augmented_dataset.image_path, image_id, generated_instances_class_id, num_instances_to_add, annotations)
        
        labels['annotations'] = annotations.to_dict('records')
        
        with open(os.path.join(augmented_dataset.coco_root_dir, annotation_file), 'w') as f:
            json.dump(labels, f)
        
        # Validation set is not augmented, so just copy the annotation file
        val_annotation_filename = dataset.get_coco_annotation_file_name('val')
        shutil.copyfile(os.path.join(dataset.coco_root_dir, val_annotation_filename), os.path.join(augmented_dataset.coco_root_dir, val_annotation_filename))

        return augmented_dataset


def _create_new_image_metadata(dataset: ShipRSImageNet, image_name: str, image_metadata: 'list[dict[str,str|int]]'):
    img = Image.open(dataset.get_image(image_name).file_path)
    new_id = max([i['id'] for i in image_metadata], default=0) + 1
    return {
        'id': new_id,
        'file_name': image_name,
        'width': img.width,
        'height': img.height
    }


def _test_box_collision(x1, x2):
    return(x1.x < x2.x + x2.w and
           x1.x + x1.w > x2.x and
           x1.y < x2.y + x2.h and
           x1.h + x1.y > x2.y)


def _is_instance_colliding(new_instance_bbox, bboxs):
    bboxs = np.array([i for i in bboxs])
    bboxs = pd.DataFrame(bboxs, columns=['x', 'y', 'w', 'h'])
    window_has_collision = bboxs.apply(lambda x: _test_box_collision(new_instance_bbox.iloc[0], x), axis=1)
    return window_has_collision.any()
