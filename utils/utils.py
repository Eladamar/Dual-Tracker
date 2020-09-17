import os
import numpy as np
import logging
from shapely.geometry import Polygon
import matplotlib.pyplot as plt

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


def get_logger(mode='info', logger_file='logger.txt'):
    if mode == 'info':
        level = logging.INFO
    elif mode == 'debug':
        level = logging.DEBUG
    else:
        raise Exception("No such logger mode")
    
    if os.path.exists(logger_file):
        os.remove(logger_file)
    logging.basicConfig(filename=logger_file,
                            filemode='a',
                            format='%(asctime)s, %(levelname)s \n%(message)s',
                            datefmt='%H:%M:%S',
                            level=level)
    logger = logging.getLogger()
    return logger


def iou(pt1, pt2):
    if pt1 != pt1 or pt2 != pt2 or any(None in l for l in pt1) or any(None in l for l in pt2):
        return 0
    poly1 = Polygon(pt1)
    poly2 = Polygon(pt2)
    return poly1.intersection(poly2).area / poly1.union(poly2).area


def center_distance(pt1, pt2):
    if pt1 != pt1 or pt2 != pt2 or any(None in l for l in pt1) or any(None in l for l in pt2):
        return np.inf
    center1 = Polygon(pt1).centroid
    center2 = Polygon(pt2).centroid
    return center1.distance(center2)


def save_figure(x, y, title, x_title, y_title, loc, scatter=False):
    success_fig = plt.figure()
    if scatter:
        plt.scatter(x, y)
        plt.hlines(np.mean(y), 0, len(y), 'r')
        plt.hlines(np.mean([yy for yy in y if yy != 0]), 0, len(y), 'b')

    else:
        plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    success_fig.savefig(loc)
    
    
def convert_df_bbox(df):
    # convert to points # x,y,w,h
    gt_array = df.to_numpy()
    new_array = np.empty((gt_array.shape[0], 8))
    new_array[:,0], new_array[:,1] = gt_array[:,0], gt_array[:,1]
    new_array[:,2], new_array[:,3] = gt_array[:,0] + gt_array[:,2], gt_array[:,1]
    new_array[:,4], new_array[:,5] = gt_array[:,0] + gt_array[:,2], gt_array[:,1] + gt_array[:,3]
    new_array[:,6], new_array[:,7] = gt_array[:,0], gt_array[:,1] + gt_array[:,3]
    new_array = new_array.reshape(-1,4,2).tolist()
    return new_array


def calc_robustness(ious):
    return 1 - float(ious.count(0))/float(len(ious))


def calc_eao(ious):
    eao = [np.mean(ious[:x+1]) for x in range(len(ious))]
    return np.mean(eao)


def calc_precision(ious):
    return np.mean(ious)