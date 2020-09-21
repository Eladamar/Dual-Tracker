import json
import os
import argparse
import numpy as np
import pandas as pd
import glob
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, make_interp_spline

from utils import *
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Evaluating Hybrid Tracker on VisDrone')
    parser.add_argument('--gt', type=str,
                        help='Path to the ground truth file')
    
    parser.add_argument('--folder', type=str,
                        help='Path to the tracker annotations folder')
    args = parser.parse_args()
    # load annotations
    dfs = []
    all_annotaions = glob.glob(args.folder + "/*.txt")
    for annotation in sorted(all_annotaions):
        with open(annotation, 'r') as f:
            track_annotations = json.load(f)
            frame_num = [tracked[0] for tracked in track_annotations]
            points = [tracked[1:] for tracked in track_annotations]
            dfs.append(pd.DataFrame(data={'frame_index': frame_num, 'tracked_points':points}))
    
    vis_cols = ['frame_index', 'target_id', 'bbox_left', 'bbox_top' , 'bbox_width', 'bbox_height', 'score', 'object_category', 'truncation', 'occlusion']
    gt = pd.read_csv(args.gt, delimiter=",", header=None, names=vis_cols)
    gt = gt.apply(pd.to_numeric)
    gt['gt_points'] = convert_df_bbox(gt[['bbox_left', 'bbox_top' , 'bbox_width', 'bbox_height']])

    gt_ids = gt.groupby(['target_id'])
    gt_tracked_iou = []
    for gt_id, df_group in gt_ids:
        for tracked_id, tracked in enumerate(dfs):
            mergedStuff = pd.merge(tracked, df_group, on=['frame_index'], how='inner')
            if mergedStuff.empty:
                continue
            first_gt_points = mergedStuff['gt_points'].loc[:5]
            first_tracked_points = mergedStuff['tracked_points'].loc[:5]
            
            ious = [iou(p1,p2) for p1,p2 in zip(first_gt_points,first_tracked_points)]
            if not ious:
                mean_iou = 0
            else:
                mean_iou = np.mean(ious)
            gt_tracked_iou.append((gt_id, tracked_id, mean_iou))
    
    
    gt_tracked_iou.sort(key = lambda x: x[2], reverse=True)
    gt_ignore = []
    tracked_ignore = []
    max_gt_tracked_iou = []
    for comb in gt_tracked_iou:
        if comb[0] in gt_ignore or comb[1] in tracked_ignore:
            continue
        gt_ignore.append(comb[0])
        tracked_ignore.append(comb[1])
        max_gt_tracked_iou.append(comb)
    
    max_id = gt['target_id'].max() + 1
    eaos = [0 for _ in range(max_id)]
    precisions = [0 for _ in range(max_id)]
    robusts = [0 for _ in range(max_id)]
    
    overlap_thres = np.linspace(0,1,100)
    precision_thres = np.linspace(0,50,100)

    success_rate_fig, success_rate_ax = plt.subplots()
    precision_rate_fig, precision_rate_ax = plt.subplots()

    for i, gt_tracked in enumerate(max_gt_tracked_iou):
        gt_index, tracked_index, _ = gt_tracked
        mergedStuff = pd.merge(dfs[tracked_index], gt_ids.get_group(gt_index), on=['frame_index'], how='inner')
        ious = list(map(iou, mergedStuff['gt_points'], mergedStuff['tracked_points']))
        center_distances = list(map(center_distance, mergedStuff['gt_points'], mergedStuff['tracked_points']))

        eaos[gt_index] = calc_eao(ious)
        precisions[gt_index] = calc_precision(ious)
        robusts[gt_index] = calc_robustness(ious)
        
        precision_rate = []
        success_rate = []
        for rate in overlap_thres:
            success_rate.append((ious > rate).sum())
        success_rate = np.array(success_rate)/len(ious)
        for rate in precision_thres:
            precision_rate.append((center_distances < rate).sum())
        precision_rate = np.array(precision_rate)/len(center_distances)
        success_rate_ax.plot(overlap_thres, success_rate)
        precision_rate_ax.plot(precision_thres, precision_rate)
    
    success_rate_ax.set_xlabel('Overlap threshold')
    success_rate_ax.set_ylabel('Success rate')
    success_rate_ax.set_title('Success Plot')
    
    precision_rate_ax.set_xlabel('Location error threshold')
    precision_rate_ax.set_ylabel('Precision')
    precision_rate_ax.set_title('Precision Plot')
    
    success_rate_fig.savefig(args.folder + '/success.png')
    precision_rate_fig.savefig(args.folder + '/precision_plot.png')
    
    save_figure(range(len(eaos)), eaos, "Expected Average Overlap Per Object", "ID", 'EAO', args.folder + '/EAO.png', scatter=True)
    save_figure(range(len(precisions)), precisions, "Precision Per Object", 'ID', "precision", args.folder + '/precision.png', scatter=True )
    save_figure(range(len(robusts)), robusts, "Robustness Per Object", 'ID', "robustness", args.folder + '/robustness.png', scatter=True )