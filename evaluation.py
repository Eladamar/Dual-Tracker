import json
import os
import argparse
import numpy as np
from shapely.geometry import Polygon


def iou(pt1, pt2):
    poly1 = Polygon(pt1)
    poly2 = Polygon(pt2)
    return poly1.intersection(poly2).area / poly1.union(poly2).area


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Evaluating Hybrid Tracker on VOT')
    parser.add_argument('--gt', type=str,
                        help='Path to the ground truth file')
    
    parser.add_argument('--file', type=str,
                        help='Path to the ground truth file')
    args = parser.parse_args()
    
    
    with open(args.file, 'r') as f:
        track_annotations = json.load(f)
  

    gt_array = np.loadtxt(args.gt, delimiter=',').reshape(-1,4,2).tolist()
#     gt = []
#     with open(args.gt, 'r') as f:
#         for line in f:
#             gt.append(line.split()) for line in

    ious = []
    for gt, tracked in zip(gt_array, track_annotations):
#         gt_list = list(gt)
        if tracked is None:
            ious.append(0)
        else:
            ious.append(iou(gt, tracked))
    
    print(np.mean(ious))