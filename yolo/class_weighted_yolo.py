from torch.nn import BCEWithLogitsLoss
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss


class ClassWeightedDetectionModel(DetectionModel):
    def __init__(self, cfg, cls_pos_weight, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.class_weights = cls_pos_weight


    def init_criterion(self):
        loss = v8DetectionLoss(self)
        # Modify classification loss for the model to use a BCE loss with class weights instead of an unwieghted BCE loss
        loss.bce = BCEWithLogitsLoss(reduction='none', pos_weight=self.class_weights)
        return loss
