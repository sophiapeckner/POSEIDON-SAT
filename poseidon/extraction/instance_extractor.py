import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
#from rembg import remove
from pathlib import Path

from shiprsimagenet import ShipRSImageNet, LabeledObject


class InstanceExtractor():
    def __init__(self, source_dataset: ShipRSImageNet, image_set: str = 'train'):
        self.source_dataset = source_dataset
        self.images = { os.path.basename(img.file_path): img for img in source_dataset.get_image_set(image_set) }


    # Extract and save a particular instance from an image
    def extract_instance_image(self, img, bbox, output_path, angle_camera, angle_divisions=8):
        output_path = os.path.split(output_path)
        # Create output directory
        if not os.path.exists(os.path.join(output_path[0], str(bbox['category_id']),)):
            os.mkdir(os.path.join(output_path[0], str(bbox['category_id']),))
        
        if angle_camera is not None:
            angle_camera_bin = int(angle_camera) % int(angle_divisions)
            if not os.path.exists(os.path.join(output_path[0], str(bbox['category_id']), str(angle_camera_bin))):
                os.mkdir(os.path.join(output_path[0], str(bbox['category_id']), str(angle_camera_bin)))
            # Save instance on the path 'base_path/outputs/category_id/instance_id.png'
            output_path = os.path.join(output_path[0], str(bbox['category_id']), str(angle_camera_bin), output_path[1])
        else:
            # Save instance on the path 'base_path/outputs/category_id/instance_id.png'
            output_path = os.path.join(output_path[0], str(bbox['category_id']), output_path[1])
        output_path = output_path + "_" + str(bbox['id']) + ".png"
        # Extract bounding box
        bbox = bbox['bbox']
        x, y, w, h = bbox
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        instance = Image.fromarray(img[y:y+h, x:x+w])
        instance.save(output_path)
        return 


    # Extract all instances from an image
    def extract_instances_image(self, annotations, img_row, output_path):
        output_path = os.path.join(output_path, str(img_row['id']))
        angle_camera = None
        #print(img_row)
        if img_row["meta"] is not None and "gimbal_heading(degrees)" in img_row["meta"]:
            angle_camera = img_row["meta"]["gimbal_heading(degrees)"]
            bboxs =  annotations[annotations['image_id'] == img_row['id']]
            img_path = os.path.join(self.images_path, 'train', img_row['file_name'])
            img = Image.open(img_path) 
            img = np.array(img)
            bboxs.apply(lambda x: self.extract_instance_image(img, x, output_path, angle_camera) ,axis=1)
        return


    # Extract and save all the intances from all the images in the training set 
    def extract(self, output_path='./outputs'):
        # Create output directory
        if not os.path.exists(output_path):
            os.mkdir(output_path)
            print("Output directory creted: ", output_path)
        # Get annotations and images information
        annotations = pd.DataFrame(self.train_annotations['annotations'])
        images = pd.DataFrame(self.train_annotations['images'])
        # Fancier
        print("Extracting Instances:")
        tqdm.pandas()
        # Extraction
        images.progress_apply(lambda x: self.extract_instances_image(annotations, x, output_path), axis=1)
        return
    

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

        instances = [ instance for image in self.images.values() for instance in image.objects if instance.name == class_name ]

        for idx, instance in enumerate(tqdm(instances, desc=f'Extracting instances of {class_name}s from dataset...')):
            self._extract_instance(instance, idx, out_path)
