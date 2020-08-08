import sys
sys.path.append('/home/eladamar/tracking/detector')
sys.path.append('/home/eladamar/tracking/detector/yolo_utils')
sys.path.append('/home/eladamar/tracking/utils')
import os
import cv2
import time
import numpy as np
from random import randint
import json
import argparse

import torch
import torchvision

from detector.detector import YOLOv3
from trackers.multi_tracker import MultiTracker
from utils.utils import get_logger
from utils.frame_loader import get_frames_loader

import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (14,14)
# import matplotlib.patches as patches

# from IPython.display import clear_output, Image, display

def main(detector_params, tracker_params, classes, logger_mode, frame_loader):
    #classes = ['person', 'car', 'bicycle', 'motor', 'bus', 'truck']

    # create logger
    logger = get_logger(mode=logger_mode)
    begin_time = time.time()
    
    # initiate detector
    detector_type = detector_params.pop("type")
    if detector_type == "YOLO":
        detector = YOLOv3(classes=classes, **detector_params)
        logger.info(f"Loaded detector:\n{detector}")
    else:
        raise ValueError(f'Detector type: {model_type} is not implemented')

    frame = frame_loader.read_first_frame()
    
    # get first detections to initiate tracker
    detections = detector.detect(frame, nms_thres=0.4, conf_thres=0.6)

    if detections is None or not len(detections):
        logger.info("No detections for initla image")
    else:
        frame = detector.draw(detections, frame)
        detections = detections.cpu().numpy()

    plt.imsave("initial_frame.png", frame)

    multiTracker = MultiTracker(logger=logger, classes=classes, **tracker_params)
    logger.info(f"Loaded multi tracker: {vars(multiTracker)}")
    
    # Initialize MultiTracker
    multiTracker.initialize(frame, detections)

    # video parameters
    fps = frame_loader.fps
    logger.info(f"input video FPS: {fps}")
    width  = frame_loader.width
    height = frame_loader.height 
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('demo.mp4', fourcc, fps, (width,height), isColor=True)

    frame_generator = frame_loader.get_next_frame()

    for kth, frame in enumerate(frame_generator):
        k = kth + 1
        thickness = round(1e-3 * (frame.shape[0] + frame.shape[1]) / 2) + 1

        # Start timer
        timer = cv2.getTickCount()

        # get updated location of objects in subsequent frames
        # success, boxes = multiTracker.update(frame)

        # update all objects
        start = time.time()
        success = multiTracker.update(frame)

#         success = multiTracker.update(frame)
        end = time.time()
        print("update ", end-start)
        if success and k % 60 != 0:
            for i, obj in enumerate(multiTracker.objects):
                if obj.frames_without_detection > 0:
                    continue
                p1 = int(obj.bbox[0]), int(obj.bbox[1]) 
                p2 = (int(obj.bbox[0] + obj.bbox[2]), int(obj.bbox[1] + obj.bbox[3]))
                cv2.rectangle(frame, p1, p2, obj.color, 10, thickness)
                cv2.putText(frame, f"Id:{obj.id}", p1, 0, thickness, obj.color, thickness)
        else:
    #         cv2.putText(frame, "Tracking failure detected", (100, 80), 0, thickness, (0, 0, 255), thickness)
            detections = detector.detect(frame, nms_thres=0.4, conf_thres=0.6)
            multiTracker.new_detection(frame, detections)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {(int(fps))}", (75, 75), 0, thickness/3, (50, 170, 50), thickness)

        resize = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
#         if k % 3 == 0:
        output.write(cv2.cvtColor(resize, cv2.COLOR_RGB2BGR))

    output.release()
    logger.debug(f"Total time: {time.time()-begin_time}")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Activating Hybrid Tracker')
    parser.add_argument('--cfg', type=str, default='cfg/config.json',
                        help='Path to the config file')
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        cfg = json.load(f)
    detector_params = cfg['Detector']
    tracker_params = cfg['Tracker']
    classes = cfg['classes']
    logger_mode = cfg['logger_mode']
    
    frames_path = cfg.get('frames_path', 'drone_neigh.mp4')
    loader_name = cfg.get('frame_loader')
    loader = get_frames_loader(loader_name)
    frame_loader = loader(frames_path)
    main(detector_params, tracker_params, classes, logger_mode, frame_loader)