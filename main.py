import sys
sys.path.append('/home/eladamar/tracking/detector')
sys.path.append('/home/eladamar/tracking/detector/yolo_utils')
sys.path.append('/home/eladamar/tracking/utils')

import cv2
import time
import numpy as np
from random import randint

import torch
import torchvision

from detector.detector import YOLOv3
from tracker.multi_tracker import MultiTracker
from utils.utils import get_logger

import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (14,14)
# import matplotlib.patches as patches

# from IPython.display import clear_output, Image, display

def main():
    classes = ['person', 'car', 'bicycle', 'motor', 'bus', 'truck']

    weights_path = '/home/eladamar/tracking/detector/weights/spp_background_anchors.pt'
    config_file = '/home/eladamar/tracking/detector/cfg/yolov3-spp.cfg'
    
    # create logger
    logger = get_logger(mode='debug')

    # initiate detector
    detector = YOLOv3(weights_path, config_file, classes)

    # Set video to load
    videoPath = "drone-neigh.mp4"

    # Create a video capture object to read videos
    cap = cv2.VideoCapture(videoPath)

    # Read first frame
    success, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # quit if unable to read the video file
    if not success:
        print('Failed to read video')
    
    # get first detections to initiate tracker
    detections = detector.detect(frame, nms_thres=0.4, conf_thres=0.6)
#     print('detections:\n\t', detections)
#     print(frame.shape[:2])
    frame = detector.draw(detections, frame)
    plt.imsave("initial_frame.png", frame)

    # Specify the tracker type
    trackerType = "CSRT"

    # Create MultiTracker object
    # multiTracker = cv2.MultiTracker_create()
    multiTracker = MultiTracker(logger=logger, classes=classes)

    detections = detections.cpu().numpy()

    # Initialize MultiTracker
    multiTracker.initialize(frame, detections)

    # video parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(3)) // 4 # width
    height = int(cap.get(4)) // 4 # height
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('demo.mp4', fourcc, fps, (width,height), isColor=True)

    k=0
    while cap.isOpened():
        k +=1 
        success, frame = cap.read()
        if not success:
            cap.release()
            output.release()
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        thickness = round(1e-3 * (frame.shape[0] + frame.shape[1]) / 2) + 1

        # Start timer
        timer = cv2.getTickCount()

        # get updated location of objects in subsequent frames
        # success, boxes = multiTracker.update(frame)

        # update all objects
        success = multiTracker.update(frame)

        if success and k % 30 != 0:
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
            # Initialize MultiTracker
            multiTracker.new_detection(frame, detections)

        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        # Display FPS on frame
        cv2.putText(frame, f"FPS: {(int(fps))}", (75, 75), 0, thickness/3, (50, 170, 50), thickness)

        resize = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        output.write(cv2.cvtColor(resize, cv2.COLOR_RGB2BGR))
        
        
if __name__ == "__main__":
    main()