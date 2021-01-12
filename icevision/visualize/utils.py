__all__ = ["draw_label", "bbox_polygon", "draw_mask"]

from icevision.imports import *
from matplotlib import patches
from PIL import Image, ImageFont, ImageDraw
import PIL


def draw_label(ax, x, y, name, color, fontsize=18):
    ax.text(
        x + 1,
        y - 2,
        name,
        fontsize=fontsize,
        color="white",
        va="bottom",
        bbox=dict(facecolor=color, edgecolor=color, pad=2, alpha=0.9),
    )


def bbox_polygon(bbox):
    bx, by, bw, bh = bbox.xywh
    poly = np.array([[bx, by], [bx, by + bh], [bx + bw, by + bh], [bx + bw, by]])
    return patches.Polygon(poly)


def draw_mask(ax, mask, color):
    color_mask = np.ones((*mask.shape, 3)) * color
    ax.imshow(np.dstack((color_mask, mask * 0.5)))
    ax.contour(mask, colors=[color_mask[0, 0, :]], alpha=0.4)


def as_rgb_tuple(x: Union[np.ndarray, tuple, list, str]) -> tuple:
    "Convert np RGB values -> tuple for PIL compatibility"
    if isinstance(x, (np.ndarray, tuple, list)):
        if not len(x) == 3:
            raise ValueError(f"Expected 3 (RGB) numbers, got {len(x)}")
        if isinstance(x, np.ndarray):
            return tuple(x.astype(np.int))
        elif isinstance(x, tuple):
            return x
        elif isinstance(x, list):
            return tuple(x)
    elif isinstance(x, str):
        return PIL.ImageColor.getrgb(x)
    else:
        raise ValueError(f"Expected {{np.ndarray|list|tuple}}, got {type(x)}")
