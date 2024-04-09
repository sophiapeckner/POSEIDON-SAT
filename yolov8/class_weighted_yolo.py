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


    def loss(self, batch, preds=None):
        # Need to convert predictions to float32 during loss computation so that loss computation is always done as float32
        # to avoid floating point overflow When using a larger batch size.

        if isinstance(preds, tuple):
            preds = list(preds)
            if isinstance(preds[1], list):
                preds[1] = [p.float() for p in preds[1]]
            else:
                preds[1] = preds[1].float()
            preds = tuple(preds)
        elif preds is not None:
            preds = preds.float()

        results = super().loss(batch, preds)
        return results
