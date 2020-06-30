__all__ = ["VOC_CATEGORIES", "VocXmlParser", "VocMaskParser"]

import xml.etree.ElementTree as ET
from mantisshrimp.imports import *
from mantisshrimp.utils import *
from mantisshrimp.core import *
from mantisshrimp.parsers.parser import *
from mantisshrimp.parsers.defaults import *
from mantisshrimp.parsers.mixins import *


VOC_CATEGORIES = {
    "person",
    "bird",
    "cat",
    "cow",
    "dog",
    "horse",
    "sheep",
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "motorbike",
    "train",
    "bottle",
    "chair",
    "diningtable",
    "pottedplant",
    "sofa",
    "tvmonitor",
}
VOC_CATEGORIES = sorted(VOC_CATEGORIES)


class VocXmlParser(DefaultImageInfoParser, LabelParserMixin, BBoxParserMixin):
    def __init__(
        self,
        annotations_dir: Union[str, Path],
        images_dir: Union[str, Path],
        categories: List[str],
    ):
        self.images_dir = Path(images_dir)

        self.annotations_dir = Path(annotations_dir)
        self.annotation_files = get_files(self.annotations_dir, extensions=[".xml"])

        self.categories = categories
        self.category2id = {cat: i + 1 for i, cat in enumerate(self.categories)}

    def __len__(self):
        return len(self.annotation_files)

    def __iter__(self):
        yield from self.annotation_files

    def prepare(self, o):
        tree = ET.parse(str(o))
        self._root = tree.getroot()
        self._filename = self._root.find("filename").text
        self._size = self._root.find("size")

    def imageid(self, o) -> Hashable:
        return str(Path(self._filename).stem)

    def filepath(self, o) -> Union[str, Path]:
        return self.images_dir / self._filename

    def height(self, o) -> int:
        return int(self._size.find("height").text)

    def width(self, o) -> int:
        return int(self._size.find("width").text)

    def label(self, o) -> List[int]:
        labels = []
        for object in self._root.iter("object"):
            label = object.find("name").text
            label_id = self.category2id[label]
            labels.append(label_id)

        return labels

    def bbox(self, o) -> List[BBox]:
        def to_int(x):
            return int(float(x))

        bboxes = []
        for object in self._root.iter("object"):
            xml_bbox = object.find("bndbox")
            xmin = to_int(xml_bbox.find("xmin").text)
            ymin = to_int(xml_bbox.find("ymin").text)
            xmax = to_int(xml_bbox.find("xmax").text)
            ymax = to_int(xml_bbox.find("ymax").text)

            bbox = BBox.from_xyxy(xmin, ymin, xmax, ymax)
            bboxes.append(bbox)

        return bboxes


class VocMaskParser(Parser, ImageidParserMixin, MaskParserMixin):
    def __init__(self, masks_dir: Union[str, Path]):
        self.mask_files = get_image_files(masks_dir)

    def __len__(self):
        return len(self.mask_files)

    def __iter__(self):
        yield from self.mask_files

    def imageid(self, o) -> Hashable:
        return str(Path(o).stem)

    def mask(self, o) -> List[Mask]:
        return [VocMaskFile(o)]
