from __future__ import print_function
import sys
import cv2
import numpy as np
from random import randint
from multi_tracker import MultiTracker



# Set video to load
videoPath = "ships.mp4"

# Create a video capture object to read videos
cap = cv2.VideoCapture(videoPath)

# Read first frame
success, frame = cap.read()
# quit if unable to read the video file
if not success:
    print('Failed to read video')
    sys.exit(1)

# Select boxes
colors = []


def select_bboxes():
    # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
    # So we will call this function in a loop till we are done selecting all objects
    bboxes = []
    while True:
        # draw bounding boxes over objects
        # selectROI's default behaviour is to draw box starting from the center
        # when fromCenter is set to false, you can draw box starting from top left corner
        bbox = cv2.selectROI('MultiTracker', frame)
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if k == 113:  # q is pressed
            print("q")
            break
    return bboxes


bboxes = select_bboxes()
print('Selected bounding boxes {}'.format(bboxes))


# Specify the tracker type
trackerType = "CSRT"

# Create MultiTracker object
# multiTracker = cv2.MultiTracker_create()
multiTracker = MultiTracker()

# Initialize MultiTracker
for bbox in bboxes:
    multiTracker.add(trackerType, frame, bbox)

# Process video and track objects
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    # Start timer
    timer = cv2.getTickCount()

    # get updated location of objects in subsequent frames
    # success, boxes = multiTracker.update(frame)

    # update all objects
    success = multiTracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    if success:
        # draw tracked objects
        bboxes_ids = multiTracker.get_bboxes_ids()
        for i, (newbox, id) in enumerate(bboxes_ids):
            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)
            cv2.putText(frame, f"Id:{id}", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[i])
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        bboxes = select_bboxes()
        # Initialize MultiTracker
        multiTracker.new_detection(frame, bboxes)


    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

    # show frame
    cv2.imshow('MultiTracker', frame)

    k = cv2.waitKey(1)
    # quit on ESC button
    if k & 0xFF == 27:  # Esc pressed
        break
    if k == ord('a'):
        bboxes = select_bboxes()
        # Initialize MultiTracker
        multiTracker.new_detection(frame, bboxes)
