from __future__ import print_function
import sys
from utils.utils import bbox_iou, bbox_distance
import itertools
import cv2
import numpy as np
from random import randint


def create_tracker_by_type(tracker_type):
    # Create a tracker based on tracker name
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    else:
        raise Exception(f"There is no tracker name: {tracker_type}")

    return tracker


class Object:
    def __init__(self, id, frame, type, bbox, tracker_type):
        self.id = id
        self.type = type
        self.bbox = bbox
        self.tracker_type = tracker_type
        self.frames_without_detection = 0
        self.tracker = create_tracker_by_type(tracker_type)
        success = self.tracker.init(frame, bbox)
        if not success:
            raise Exception("could not init tracker")

    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        if not ok:
            self.frames_without_detection += 1
        else:
            self.frames_without_detection = 0
            self.bbox = bbox
        return ok

    def reinitialize(self, frame, bbox):
        self.tracker = create_tracker_by_type(self.tracker_type)
        success = self.tracker.init(frame, bbox)
        if not success:
            raise Exception("could not init tracker")
        self.bbox = bbox
        self.frames_without_detection = 0


class MultiTracker:
    def __init__(self, default_tracker='CSRT',
                 failures_threshold=0.5,
                 iou_thres=0.1,
                 dist_thres=100,
                 no_detection_thres=2):
        self.objects = []
        self.default_tracker = default_tracker
        self.new_id = itertools.count().__next__
        self.failures_threshold = failures_threshold
        self.iou_thres = iou_thres
        self.dist_thres = dist_thres
        self.no_detection_thres = no_detection_thres

    def add(self, tracker_type, frame, bbox):
        obj = Object(self.new_id(), frame, "check", bbox, tracker_type)
        self.objects.append(obj)

    def update(self, frame):
        number_of_fails = 0
        for i, obj in enumerate(self.objects):
            ok = obj.update(frame)
            number_of_fails += 1 - int(ok)
        fails_percentage = float(number_of_fails)/float(len(self.objects))

        if fails_percentage < self.failures_threshold:
            return True
        return False

    def get_bboxes_ids(self):
        bboxes_ids = []
        for obj in self.objects:
            if obj.frames_without_detection == 0:
                bboxes_ids.append((obj.bbox, obj.id))
        return bboxes_ids

    def new_detection(self, frame, bboxes):
        bboxes_array = np.array(bboxes)
        to_remove = []
        for i, obj in enumerate(self.objects):
            if bboxes_array.size == 0:
                return
            cur_bbox = np.array(obj.bbox)
            ious = bbox_iou(cur_bbox, bboxes_array)
            greatest_overlap = np.argmax(ious)

            # check for greatest intersection over union detection
            if ious[greatest_overlap] > self.iou_thres:
                obj.reinitialize(frame, bboxes[greatest_overlap])
                bboxes_array = np.delete(bboxes_array, greatest_overlap, axis=0)
                continue
            # check for closest detection
            distances = bbox_distance(cur_bbox, bboxes_array)
            closest_box = np.argmin(distances)
            if distances[closest_box] < self.dist_thres:
                obj.reinitialize(frame, bboxes[closest_box])
                bboxes_array = np.delete(bboxes_array, closest_box, axis=0)
                continue
            if obj.frames_without_detection > self.no_detection_thres:
                to_remove.append(i)

        # remove undetected objects
        self.objects = [self.objects[i] for i in range(len(self.objects)) if i not in to_remove]

        # add new detections
        for bbox in bboxes_array:
            self.add(self.default_tracker, frame, tuple(bbox))

