import numpy as np
import numpy as xp

import six
import torch
from six import __init__
from PIL import Image, ImageDraw
import matplotlib.patches as patche
import matplotlib.pyplot as plt
from data.dataset import inverse_normalize
from visdom import Visdom


# rpn bbox回归权值映射到实际bbox中
def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """

    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 3] - src_bbox[:, 1]
    src_width = src_bbox[:, 2] - src_bbox[:, 0]
    src_ctr_y = src_bbox[:, 1] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 0] + 0.5 * src_width

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    ctr_y = dy * src_height[:, None] + src_ctr_y[:, None]
    ctr_x = dx * src_width[:, None] + src_ctr_x[:, None]
    h = xp.exp(dh) * src_height[:, None]
    w = xp.exp(dw) * src_width[:, None]

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w

    return dst_bbox


# 实际bbox映射到边界框回归
def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """

    height = src_bbox[:, 3] - src_bbox[:, 1]
    width = src_bbox[:, 2] - src_bbox[:, 0]
    ctr_y = src_bbox[:, 1] + 0.5 * height
    ctr_x = src_bbox[:, 0] + 0.5 * width

    base_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_ctr_y = dst_bbox[:, 1] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 0] + 0.5 * base_width

    eps = xp.finfo(height.dtype).eps
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    if height.any() < 0:
        raise IndexError
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dx, dy, dw, dh)).transpose()
    return loc


def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # 为了两数组可以广播操作，对bbox_a在axis=1增加一个维度,这个维度用于放置广播计算多出的数组
    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    # (tl < br).all(axis=2) 用于判断锚框是否相交
    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.

    Generate anchors that are scaled and modified to the given aspect ratios.
    Area of a scaled anchor is preserved when modifying to the given aspect
    ratio.

    :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
    function.
    The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
    generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.

    For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
    the width and the height of the base window will be stretched by :math:`8`.
    For modifying the anchor to the given aspect ratio,
    the height is halved and the width is doubled.

    Args:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    Returns:
        ~numpy.ndarray:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

    """
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])
            # w = np.sqrt(base_size * anchor_scales[j]) * np.sqrt(ratios[i])
            # h = np.sqrt(base_size * anchor_scales[j]) * np.sqrt(1. / ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base[:, [1, 0, 3, 2]]


# anchor xyxy
def plot_anchor_matplot(anchor_box, img, *, normal=False):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    if img.ndim > 3:
        img = img.squeeze(0)
    if isinstance(anchor_box, torch.Tensor):
        anchor_box = anchor_box.cpu().numpy()
    if anchor_box.ndim > 2:
        anchor_box = anchor_box.squeeze(0)
    img = img.transpose((2, 1, 0))

    axis = plt.figure()
    ax = axis.add_subplot(1, 1, 1)
    if normal:
        ax.imshow(np.uint8(inverse_normalize(img)))
    else:
        ax.imshow(img)
    color = ['blue', 'green', 'red']

    for idx, bbox in enumerate(anchor_box):
        leftTop_x, leftTop_y = bbox[0], bbox[1]
        bbx_w = bbox[2] - bbox[0]
        bbx_h = bbox[3] - bbox[1]
        rec = patche.Rectangle((leftTop_x, leftTop_y), bbx_w, bbx_h, fill=False, edgecolor=color[idx%3], linewidth=2)
        ax.add_patch(rec)
    ax.spines['right'].set_color('None')
    ax.spines['top'].set_color('None')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))
    # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    # Visdom.matplot(axis)
    plt.show()


if __name__ == '__main__':
    anchor = generate_anchor_base()
    plot_anchor_matplot(anchor)
