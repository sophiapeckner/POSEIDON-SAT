import os
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from pathlib import Path


class HorizontalBoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


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


class LabeledObject:
    def __init__(self, filename: str, name: str, truncated: bool, difficult: bool, bndbox: HorizontalBoundingBox, rotated_box: OrientedBoundingBox, levels: 'tuple[int, int, int, int]', location: 'str | None'):
        self.filename = filename
        self.name = name
        self.truncated = truncated
        self.difficult = difficult
        self.bndbox = bndbox
        self.rotated_bndbox = rotated_box
        self.level_group = levels
        self.location = location


class LabeledImage:
    def __init__(self, path: str, objects: 'list[LabeledObject]'):
        self.file_path = path
        self.objects = objects
    

    def show(self):
        # Load the image
        img = Image.open(self.file_path)

        # Draw each bounding box
        imgDraw = ImageDraw.Draw(img)
        for obj in self.objects:
            rotated_box = obj.rotated_bndbox
            imgDraw.polygon([(rotated_box.x1, rotated_box.y1), (rotated_box.x2, rotated_box.y2), (rotated_box.x3, rotated_box.y3), (rotated_box.x4, rotated_box.y4)], outline='lime', width=5)

        # Display the image
        plt.imshow(img)
        plt.axis('off')
        plt.show()


class ShipRSImageNet:
    def __init__(self, root_path: str):
        self.root_path = root_path


    def get_image_set(self, image_set: str):
        image_set_path = os.path.join(self.root_path, 'ImageSets', f'{image_set}.txt')
        with open(image_set_path, 'r') as f:
            image_names = f.read().splitlines()

        return [self.get_image(image_name) for image_name in image_names]


    def get_image(self, image_name: str):
        annotation_path = os.path.join(self.root_path, 'Annotations', f'{Path(image_name).stem}.xml')
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        filename = root.find('filename').text

        labeled_objects: list[LabeledObject] = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            truncated = obj.find('truncted').text == '1'
            difficult = obj.find('difficult').text == '1'
            
            bndbox = obj.find('bndbox')
            bndbox = {child.tag: int(child.text) for child in bndbox}
            
            rotated_box_poly = obj.find('polygon')
            rotated_box_poly = {child.tag: _parse_to_int(child.text) for child in rotated_box_poly}
            
            ship_location = obj.find('Ship_location')
            levels = tuple([int(obj.find(f'level_{i}').text) for i in range(4)])

            labeled_object = LabeledObject(filename, name, truncated, difficult, HorizontalBoundingBox(**bndbox), OrientedBoundingBox(**rotated_box_poly), levels, ship_location)
            labeled_objects.append(labeled_object)
        
        return LabeledImage(os.path.join(self.root_path, 'JPEGImages', image_name), labeled_objects)


def _parse_to_int(s: str):
    if '.' in s:
        i, dec = s.split('.')
        if int(dec) != 0:
            raise ValueError(f'Expected integer, got {s}')
        return int(i)
    return int(s)
