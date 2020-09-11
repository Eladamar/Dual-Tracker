import json
import os
import argparse
import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, make_interp_spline


from utils import *

"""
def iou(pt1, pt2):
    poly1 = Polygon(pt1)
    poly2 = Polygon(pt2)
    return poly1.intersection(poly2).area / poly1.union(poly2).area


def center_distance(pt1, pt2):
    center1 = Polygon(pt1).centroid
    center2 = Polygon(pt2).centroid
    return center1.distance(center2)

def save_figure(x, y, title, x_title, y_title, loc):
    success_fig = plt.figure()
    plt.plot(x, y)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.title(title)
    success_fig.savefig(loc)
""" 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Evaluating Hybrid Tracker on VOT')
    parser.add_argument('--gt', type=str,
                        help='Path to the ground truth file')
    
    parser.add_argument('--file', type=str,
                        help='Path to the tracler annotations file')
    args = parser.parse_args()
    # load annotations
    with open(args.file, 'r') as f:
        track_annotations = json.load(f)
    
    print(len(track_annotations))
    gt_array = np.loadtxt(args.gt, delimiter=',')
    if gt_array.shape[1] == 4: # x,y,w,h
        new_array = np.empty((gt_array.shape[0], 8))
        new_array[:,0], new_array[:,1] = gt_array[:,0], gt_array[:,1]
        new_array[:,2], new_array[:,3] = gt_array[:,0] + gt_array[:,2], gt_array[:,1]
        new_array[:,4], new_array[:,5] = gt_array[:,0] + gt_array[:,2], gt_array[:,1] + gt_array[:,3]
        new_array[:,6], new_array[:,7] = gt_array[:,0], gt_array[:,1] + gt_array[:,3]
        gt_array = new_array.reshape(-1,4,2).tolist()
    elif gt_array.shape[1] == 8: # x1,y1,x2,y2...
        gt_array = gt_array.reshape(-1,4,2).tolist()
    else:
        raise Exception("Unknown groundtruth.txt format")
    
    ious = []
    center_distances = []
    for gt, tracked in zip(gt_array, track_annotations):
        frame_number, location = tracked[0], tracked[1:]
        if np.isnan(gt).any():
            continue
        elif location is None or any(None in point for point in location):
            ious.append(0)
            center_distances.append(np.inf)
        else:
            ious.append(iou(gt, location))
            center_distances.append(center_distance(gt, location))
    
    if len(gt_array) != len(track_annotations):
        print(f"tracking history: {len(track_annotations)} and groundtruth: {len(gt_array)} doesnt have same length")
        # case tracker failed and deleted object
        for gt in gt_array[len(track_annotations):]:
            if not np.isnan(gt).any():
                ious.append(0)
                center_distances.append(np.inf)
        
        
    robustness = float(ious.count(0))/float(len(ious))
    eao = [np.mean(ious[:x+1]) for x in range(len(ious))]
    print("precision:", np.mean(ious))
    print("robustness:", robustness)
    print("Expected Average Overlap:", np.mean(eao))
    
    ious = np.array(ious)
    center_distances = np.array(center_distances)
    success_rate = []
    overlap_thres = np.linspace(0,1,100)
    # success plot 
    for rate in overlap_thres:
        success_rate.append((ious > rate).sum())
    
    success_rate = np.array(success_rate)/ious.size
    
    precision_rate = []
    precision_thres = np.linspace(0,50,100)
    # precision plot
    for rate in precision_thres:
        precision_rate.append((center_distances < rate).sum())
    
    precision_rate = np.array(precision_rate)/center_distances.size
    
    base_folder = args.gt.rsplit('/',1)[0]
    loc = os.path.join(base_folder, 'success_plot.png')
    save_figure(overlap_thres, success_rate, 'Success Plot', 'Overlap threshold', 'Success rate', loc)
    loc = os.path.join(base_folder, 'precision_plot.png')
    save_figure(precision_thres, precision_rate, 'Precision Plot', 'Location error threshold', 'Precision', loc)
