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

    def __init__(self, instances_class_name: str, instances_path: str, instances_resolution: float, seed: Optional[int] = None):
        self.instances_class = instances_class_name
        self.instances_path = instances_path
        self.instances_resolution = instances_resolution
        
        self.random = Random(seed)
        self._orig_random_state = self.random.getstate()

        self._instance_images = list(map(Image.open, Path(instances_path).iterdir()))
        self._repopulate_instance_random_select_pool()


    def _repopulate_instance_random_select_pool(self):
        self._instance_random_select_pool = list(range(len(self._instance_images)))
        self.random.shuffle(self._instance_random_select_pool)


    def reset(self, new_seed: Optional[int] = None):
        if new_seed is not None:
            self.random.seed(new_seed)
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
            raise ValueError(f"Rescaling randomly selected instance to {rescale_to_spatial_resolution} m/pixel results in an image with a 0 width or height: cannot reliably provide rescaled instance for desired resolution")

        return image.resize((new_width, new_height))


    def _add_instances_to_image(self, source_image: LabeledImage, image_out_dir: str, image_id: int, new_instance_class_id: int, num_instances: int, annotations: pd.DataFrame) -> pd.DataFrame:
        existing_instances_in_image = annotations[annotations['image_id'] == image_id]
        next_free_id: int = annotations['id'].max() + 1
        image = Image.open(source_image.file_path)

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
            instance_series = pd.Series([instance_id, image_id, instance_bbox, instance_area, new_instance_class_id],
                                         index=['id', 'image_id', 'bbox', 'area', 'category_id'])

            image.paste(instance, box=(instance_x, instance_y))
            annotations = pd.concat(annotations, [instance_series], ignore_index=True, copy=False)

        image.save(os.path.join(image_out_dir, os.path.basename(source_image.file_path)))
        return annotations


    def augment(self, dataset: ShipRSImageNet, output_path: str, images_to_augment: 'list[str]', total_instances_to_add: int, min_instances_per_image: int, max_instances_per_image: int):
        if max_instances_per_image * len(images_to_augment) < total_instances_to_add:
            raise ValueError(f"Cannot add {total_instances_to_add} instances to {len(images_to_augment)} images when the maximum number of instances per image is {max_instances_per_image}")

        augmented_dataset = ShipRSImageNet(output_path)
        
        out_path = Path(output_path)
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True)
        shutil.copytree(dataset.image_path, augmented_dataset.image_path)
        annotation_file = dataset.get_coco_annotation_file_name('train')

        with open(dataset.coco_root_dir / annotation_file) as f:
            labels = json.load(f)

        annotations = pd.DataFrame(labels['annotations'])
        image_names_to_ids: dict[str, int] = {img['file_name']: img['id'] for img in labels['images']}
        generated_instances_class_id: int = next(cat['id'] for cat in labels['categories'] if cat['name'] == self.instances_class)

        instances_add_count = 0
        instances_to_add_for_image = [None] * len(images_to_augment)
        image_idx = 0
        while instances_add_count < total_instances_to_add:
            current_instances_to_add_for_image = instances_to_add_for_image[image_idx] or 0

            num_instances_to_add = self.random.randint(min_instances_per_image, max_instances_per_image - current_instances_to_add_for_image)
            if num_instances_to_add + instances_add_count > total_instances_to_add:
                num_instances_to_add = total_instances_to_add - instances_add_count

            instances_to_add_for_image[image_idx] = num_instances_to_add + current_instances_to_add_for_image
            instances_add_count += num_instances_to_add

            image_idx += 1
            if image_idx >= len(instances_to_add_for_image):
                image_idx = 0

        instances_to_add_for_image: list[int] = [i for i in instances_to_add_for_image if i is not None]
        
        print(f"Adding {instances_add_count} instances to {len(instances_to_add_for_image)} images...")

        for idx, image_name in enumerate(tqdm(images_to_augment[:len(instances_to_add_for_image)], desc='Augmenting Images')):
            num_instances_to_add = instances_to_add_for_image[idx]
            image_id = image_names_to_ids[image_name]
            annotations = self._add_instances_to_image(dataset.get_image(image_name), augmented_dataset.image_path, image_id, generated_instances_class_id, num_instances_to_add, annotations)
        
        labels['annotations'] = annotations.to_dict('records')
        
        with open(augmented_dataset.coco_root_dir / annotation_file, 'w') as f:
            json.dump(labels, f)

        return augmented_dataset


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
