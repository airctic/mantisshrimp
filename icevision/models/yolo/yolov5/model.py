__all__ = ["model"]

from icevision.imports import *
from icevision.utils.data_dir import get_root_dir
import yaml

import yolov5
from yolov5.models.yolo import Model
from yolov5.utils.google_utils import attempt_download
from yolov5.utils.torch_utils import intersect_dicts
from yolov5.utils.general import check_img_size

yolo_dir = get_root_dir() / "yolo"
yolo_dir.mkdir(exist_ok=True)

def model(
    num_classes: int,
    img_size: int, #must be multiple of 32
    model_name: str = "yolov5s",
    pretrained: bool = True,
) -> nn.Module:

    cfg_filepath = Path(yolov5.__file__).parent / f"models/{model_name}.yaml"
    if pretrained:
        weights_path = yolo_dir / f"{model_name}.pt"

        with open(Path(yolov5.__file__).parent / "data/hyp.finetune.yaml") as f: 
            hyp = yaml.load(f, Loader=yaml.SafeLoader)

        attempt_download(weights_path)  # download if not found locally 
        sys.path.insert(0, str(Path(yolov5.__file__).parent))
        ckpt = torch.load(weights_path) #, map_location=device)  # load checkpoint
        sys.path.remove(str(Path(yolov5.__file__).parent))
        if hyp.get('anchors'): ckpt['model'].yaml['anchors'] = round(hyp['anchors'])  # force autoanchor
        model = Model(cfg_filepath or ckpt['model'].yaml, ch=3, nc=num_classes)#.to(device)  # create
        exclude = []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load    
    else:
        with open(Path(yolov5.__file__).parent / "data/hyp.scratch.yaml") as f: 
            hyp = yaml.load(f, Loader=yaml.SafeLoader)

        model = Model(cfg_filepath, ch=3, nc=num_classes, anchors=hyp.get('anchors'))#.to(device)  # create

    gs = int(model.stride.max())  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(img_size, gs)  # verify imgsz are gs-multiples

    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= num_classes / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    model.nc = num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)

    return model