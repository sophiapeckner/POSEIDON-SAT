import torch
import numpy as np
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data import build_yolo_dataset
from ultralytics.utils import RANK

from yolo.class_weighted_yolo import ClassWeightedDetectionModel


class ClassWeightedDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None
    
    def get_model(self, cfg=None, weights=None, verbose=True):
        dataset_label_reader = build_yolo_dataset(self.args, self.get_dataset(self.data)[0], data=self.data, mode="train", batch=1, rect=False)
        cls_pos_weight = labels_to_class_pos_weights(dataset_label_reader.get_labels(), self.data["nc"]).to(self.device)

        model = ClassWeightedDetectionModel(cfg, cls_pos_weight, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model


def labels_to_class_pos_weights(labels, nc=80):
    """Calculates class weights from labels to handle class imbalance in training"""
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    classes = np.concatenate([label['cls'].astype(int) for label in labels]).squeeze() # classes.shape = (num_instances,)
    occurrences_per_class = np.bincount(classes, minlength=nc)  # occurrences per class

    instance_count = classes.shape[0]
    weights = (instance_count - occurrences_per_class) / occurrences_per_class  # Weight of positive instances of each class = negative_examples_of_class / positive_examples_of_class
    weights[occurrences_per_class == 0] = 1  # Set weights of classes with no instances to 1 to avoid weighting the loss of positive examples for these classes
    return torch.from_numpy(weights).float()
