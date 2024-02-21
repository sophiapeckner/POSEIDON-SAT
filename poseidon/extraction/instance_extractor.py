import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path

from shiprsimagenet import ShipRSImageNet, LabeledObject


class InstanceExtractor():
    def __init__(self, source_dataset: ShipRSImageNet, image_set: str = 'train'):
        self.source_dataset = source_dataset
        self.images = { os.path.basename(img.file_path): img for img in source_dataset.get_image_set(image_set) }


    def _extract_instance(self, instance: LabeledObject, instance_id: int, output_path: Path):
        img = Image.open(self.images[instance.image_name].file_path)
        
        # Mask out everything outside the horizontal bounding box
        mask = Image.new('1', img.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(instance.rotated_bndbox.to_polygon(), outline=1, fill=1, width=1) 

        # Apply the mask
        img.putalpha(mask)

        img = img.crop(instance.bndbox.to_tuple())
        img.save(output_path / f'{Path(instance.image_name).stem}_{instance_id}.png')
    

    def extract_instances(self, class_name: str, output_path: str):
        out_path = Path(output_path)
        shutil.rmtree(str(out_path), ignore_errors=True)
        out_path.mkdir(parents=True)

        instances = [ instance for image in self.images.values() for instance in image.objects if instance.name == class_name and not instance.truncated ]

        for idx, instance in enumerate(tqdm(instances, desc=f'Extracting non-truncated instances of {class_name}s from dataset...')):
            self._extract_instance(instance, idx, out_path)
