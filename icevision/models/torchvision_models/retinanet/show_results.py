__all__ = ["show_results", "interp"]

from icevision.models.torchvision_models.faster_rcnn.show_results import *
from icevision.models.interpretation import Interpretation
from icevision.models.torchvision_models.retinanet.dataloaders import (
    valid_dl,
    infer_dl,
)
from icevision.models.torchvision_models.retinanet.prediction import (
    predict_dl,
)

_LOSSES_DICT = {
    "classification": [],
    "bbox_regression": [],
    "loss_total": [],
}

interp = Interpretation(
    losses_dict=_LOSSES_DICT,
    valid_dl=valid_dl,
    infer_dl=infer_dl,
    predict_dl=predict_dl,
)
