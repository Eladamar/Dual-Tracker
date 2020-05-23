import numpy as np


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two or more bounding boxes
    """
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]
    if not x1y1x2y2:
        # Transform from xywh
        b1_x1, b1_x2 = box1[:, 0], box1[:, 0] + box1[:, 2]
        b1_y1, b1_y2 = box1[:, 1], box1[:, 1] + box1[:, 3]
        b2_x1, b2_x2 = box2[:, 0], box2[:, 0] + box2[:, 2]
        b2_y1, b2_y2 = box2[:, 1], box2[:, 1] + box2[:, 3]
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)
    # Intersection area
    inter_area = np.clip(inter_rect_x2 - inter_rect_x1, a_min=0, a_max=None) * np.clip(
        inter_rect_y2 - inter_rect_y1, a_min=0, a_max=None
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def bbox_distance(box1, box2, x1y1x2y2=False):
    """
    Returns the distance of two or more bounding boxes
    """
    if box1.ndim == 1:
        box1 = box1[np.newaxis, :]
    if box2.ndim == 1:
        box2 = box2[np.newaxis, :]
    if not x1y1x2y2:
        # Transform from xywh to get centers
        b1_xctr = box1[:, 0] + box1[:, 2] / 2
        b1_yctr = box1[:, 1] + box1[:, 3] / 2
        b2_xctr = box2[:, 0] + box2[:, 2] / 2
        b2_yctr = box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the centers of bounding boxes
        b1_xctr = box1[:, 0] + (box1[:, 2] - box1[:, 0]) / 2
        b1_yctr = box1[:, 1] + (box1[:, 3] - box1[:, 1]) / 2
        b2_xctr = box2[:, 0] + (box2[:, 2] - box2[:, 0]) / 2
        b2_yctr = box2[:, 1] + (box2[:, 3] - box2[:, 1]) / 2

    # l2 distance
    dist = (b1_xctr - b2_xctr)**2 + (b1_yctr - b2_yctr)**2
    dist = np.sqrt(dist)

    return dist
