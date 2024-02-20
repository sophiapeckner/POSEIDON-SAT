import os
import json
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from typing import Optional
from PIL import Image, ImageDraw
from pathlib import Path


class HorizontalBoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
    
    def to_polygon(self):
        return [(self.xmin, self.ymin), (self.xmax, self.ymin), (self.xmax, self.ymax), (self.xmin, self.ymax)]
    
    def to_array(self):
        return [self.xmin, self.ymin, self.xmax, self.ymax]
    
    def to_tuple(self):
        return (self.xmin, self.ymin, self.xmax, self.ymax)


class OrientedBoundingBox:
    def __init__(self, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int, y4: int):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3
        self.x4 = x4
        self.y4 = y4

    def to_polygon(self):
        return [(self.x1, self.y1), (self.x2, self.y2), (self.x3, self.y3), (self.x4, self.y4)]

    def to_array(self):
        return [self.x1, self.y1, self.x2, self.y2, self.x3, self.y3, self.x4, self.y4]


class LabeledObject:
    def __init__(self, image_name: str, name: str, truncated: 'Optional[bool]', difficult: 'Optional[bool]', bndbox: HorizontalBoundingBox, rotated_box: OrientedBoundingBox, levels: 'Optional[tuple[int, int, int, int]]', location: 'Optional[bool]'):
        self.image_name = image_name
        self.name = name
        self.truncated = truncated
        self.difficult = difficult
        self.bndbox = bndbox
        self.rotated_bndbox = rotated_box
        self.level_group = levels
        self.location = location


class LabeledImage:
    def __init__(self, path: str, width: int, height: int, objects: 'list[LabeledObject]', source_dataset: 'Optional[str]' = None, spatial_resolution: 'Optional[float]' = None):
        self.file_path = path
        self.width = width
        self.height = height
        self.objects = objects
        self.source_dataset = source_dataset
        self._spatial_resolution = spatial_resolution

    @property
    def spatial_resolution(self):
        if self._spatial_resolution is not None:
            return self._spatial_resolution
        return None # TODO: Dynamically implement this based on the source_dataset property

    def show(self):
        # Load the image
        img = Image.open(self.file_path)

        # Draw each bounding box
        imgDraw = ImageDraw.Draw(img)
        for obj in self.objects:
            rotated_box = obj.rotated_bndbox
            imgDraw.polygon(obj.bndbox.to_polygon(), outline='red', width=3)
            imgDraw.polygon(rotated_box.to_polygon(), outline='lime', width=5)

        # Display the image
        plt.imshow(img)
        plt.axis('off')
        plt.show()


class ShipRSImageNet:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.voc_root_path = os.path.join(root_path, 'VOC_Format')
        self.is_original_dataset = os.path.exists(self.voc_root_path)
        self.image_path = os.path.join(self.root_path, 'images') if not self.is_original_dataset else os.path.join(self.voc_root_path, 'JPEGImages')
        self._coco_image_to_annotation_map = None
        self._coco_category_map = {}


    def get_image_set(self, image_set: str):
        # Processed image sets are in COCO format only. Original dataset should parse VOC format since it has more metadata
        return self._get_voc_image_set(image_set) if self.is_original_dataset else self._get_coco_image_set(image_set)


    def _get_coco_image_set(self, image_set: str):
        annotation_file = os.path.join(self.root_path, f'ShipRSImageNet_bbox_{image_set}.json')
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        return [self._get_coco_image(image['file_name']) for image in annotations['images']]


    def _get_voc_image_set(self, image_set: str):
        image_set_path = os.path.join(self.voc_root_path, 'ImageSets', f'{image_set}.txt')
        with open(image_set_path, 'r') as f:
            image_names = f.read().splitlines()

        return [self.get_image(image_name) for image_name in image_names]


    def get_image(self, image_name: str):
        return self._get_voc_image(image_name) if self.is_original_dataset else self._get_coco_image(image_name)


    def _get_coco_image(self, image_name: str):
        if not self._coco_image_to_annotation_map:
            self._coco_image_to_annotation_map = {}
            metadata_files = glob.glob(os.path.join(self.root_path, '*.json'))
            for metadata_file in metadata_files:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    for idx, image in enumerate(metadata['images']):
                        self._coco_image_to_annotation_map[image['file_name']] = (metadata_file, image['id'], idx)
                    for category in metadata['categories']:
                        self._coco_category_map[category['id']] = category['name']
        
        metadata_file, image_id, img_idx = self._coco_image_to_annotation_map[image_name]
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        image_metadata = metadata['images'][img_idx]
        image_annotations = [annotation for annotation in metadata['annotations'] if annotation['image_id'] == image_id]
        
        labeled_objects: list[LabeledObject] = []
        for annotation in image_annotations:
            category_name = self._coco_category_map[annotation['category_id']]
            bndbox = HorizontalBoundingBox(
                xmin=annotation['bbox'][0],
                ymin=annotation['bbox'][1],
                xmax=annotation['bbox'][0] + annotation['bbox'][2],
                ymax=annotation['bbox'][1] + annotation['bbox'][3])
            rotated_box = OrientedBoundingBox(*annotation['segmentation'][0])
            labeled_object = LabeledObject(
                image_name=image_metadata['file_name'],
                name=category_name,
                truncated=None,
                difficult=None,
                bndbox=bndbox,
                rotated_box=rotated_box,
                levels=None,
                location=None)
            labeled_objects.append(labeled_object)
        
        return LabeledImage(os.path.join(self.image_path, image_name), int(image_metadata['width']), int(image_metadata['height']), labeled_objects)


    def _get_voc_image(self, image_name: str):
        annotation_path = os.path.join(self.voc_root_path, 'Annotations', f'{Path(image_name).stem}.xml')
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        
        source_dataset = root.find('source').find('dataset_source').text
        spatial_resolution = root.find('Img_Resolution').text

        labeled_objects: list[LabeledObject] = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            truncated = obj.find('truncted').text == '1'
            difficult = obj.find('difficult').text == '1'
            
            bndbox = obj.find('bndbox')
            bndbox = {child.tag: int(child.text) for child in bndbox}
            
            rotated_box_poly = obj.find('polygon')
            rotated_box_poly = {child.tag: _parse_to_int(child.text) for child in rotated_box_poly}
            
            ship_location = obj.find('Ship_location').text
            levels = tuple([int(obj.find(f'level_{i}').text) for i in range(4)])

            labeled_object = LabeledObject(filename, name, truncated, difficult, HorizontalBoundingBox(**bndbox), OrientedBoundingBox(**rotated_box_poly), levels, ship_location)
            labeled_objects.append(labeled_object)
        
        return LabeledImage(os.path.join(self.image_path, image_name), width, height, labeled_objects, source_dataset, spatial_resolution)


def _parse_to_int(s: str):
    if '.' in s:
        i, dec = s.split('.')
        if int(dec) != 0:
            raise ValueError(f'Expected integer, got {s}')
        return int(i)
    return int(s)
