import os
import xml.etree.ElementTree as ET
from pathlib import Path


class HorizontalBoundingBox:
    def __init__(self, xmin: int, ymin: int, xmax: int, ymax: int):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class OrientedBoundingBox:
    def __init__(self, cx: float, cy: float, width: float, height: float, rot: float):
        self.center_x = cx
        self.center_y = cy
        self.width = width
        self.height = height
        self.rotation = rot
    

class LabeledObject:
    def __init__(self, filename: str, name: str, truncated: bool, difficult: bool, bndbox: HorizontalBoundingBox, rotated_box: OrientedBoundingBox, levels: 'tuple[int, int, int, int]'):
        self.source_file = filename
        self.name = name
        self.truncated = truncated
        self.difficult = difficult
        self.bndbox = bndbox
        self.rotated_bndbox = rotated_box
        self.level_group = levels


class LabeledImage:
    def __init__(self, filename: str, objects: 'list[LabeledObject]'):
        self.filename = filename
        self.objects = objects


def parse_voc(root_path: str, image_set: str):
    labeled_images: list[LabeledImage] = []
    image_set_path = os.path.join(root_path, 'ImageSets', f'{image_set}.txt')
    with open(image_set_path, 'r') as f:
        image_names = f.read().splitlines()

    for image_name in image_names:
        annotation_path = os.path.join(root_path, 'Annotations', f'{Path(image_name).stem}.xml')
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
            
            rotated_box = obj.find('rotated_box')
            rotated_box = {child.tag: float(child.text) for child in rotated_box}

            levels = tuple([int(obj.find(f'level_{i}').text) for i in range(4)])

            labeled_object = LabeledObject(filename, name, truncated, difficult, HorizontalBoundingBox(**bndbox), OrientedBoundingBox(**rotated_box), levels)
            labeled_objects.append(labeled_object)
        
        labeled_images.append(LabeledImage(filename, labeled_objects))

    return labeled_images
