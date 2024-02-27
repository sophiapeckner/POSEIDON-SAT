# Adapted from https://github.com/ultralytics/ultralytics/blob/f8e681c2be251562633d424f4a1a1cc7da526bed/ultralytics/data/converter.py - AGPL-3.0 License
                                                                                                                                                          
import re
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from ultralytics.utils import LOGGER, TQDM


def convert_coco(
    labels_dir="../coco/annotations/",
    images_dir="../coco/images/",
    save_dir="coco_converted/",
    class_remap={},
):
    # Create dataset directory
    save_dir = Path(save_dir)
    for p in save_dir / "labels", save_dir / "images":
        p.mkdir(parents=True, exist_ok=True)  # make dir

    # Import json
    for split in "train", "val":
        json_file = Path(labels_dir).resolve() / f"ShipRSImageNet_bbox_{split}_level_3.json"
        fn = Path(save_dir) / "labels" / re.sub('_level_\d$', '', json_file.stem.replace("ShipRSImageNet_bbox_", ""))  # folder name
        fn.mkdir(parents=True, exist_ok=True)
        with open(json_file) as f:
            data = json.load(f)

        # Create image directory
        image_out_path = Path(save_dir) / "images" / split
        image_out_path.mkdir(parents=True, exist_ok=True)

        # Create image dict
        images = {f'{x["id"]:d}': x for x in data["images"]}
        # Create image-annotations dict
        imgToAnns = defaultdict(list)
        for ann in data["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
        # Copy Images
        for img in data["images"]:
            shutil.copyfile(Path(images_dir) / img["file_name"], image_out_path / img["file_name"])

        # Write labels file
        for img_id, anns in TQDM(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:d}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            for ann in anns:
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue

                coco_cls = ann["category_id"]
                if coco_cls in class_remap:
                    cls = class_remap[coco_cls] - 1  # class
                else:
                    cls = coco_cls - 1
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            with open((fn / f).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (
                        *(bboxes[i]),
                    )  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")

        LOGGER.info(f"COCO data converted successfully.\nResults saved to {save_dir.resolve()}")
