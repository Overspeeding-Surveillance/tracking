import cv2
import torch
from custom_utils.sort import *
import numpy as np
from custom_utils.custom_utils import get_detections

"""
This demo tracks the cars based on the sort algorithm (version 1)
"""

# Text Parameters
FONT_FACE = 0
FONT_SIZE = 0.5
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

useful_classes = [2, 3, 5, 7]

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
video_path = "highway.mp4"
cap = cv2.VideoCapture(video_path)

sort = Sort(max_age=5, min_hits=3, iou_threshold=0.1)

while True:
    ret, frame = cap.read()
    boxes, classes, scores = get_detections(model(frame))

    detections_in_frame = len(boxes)
    if detections_in_frame:
        idxs = np.where(detected_class == useful_class for detected_class in classes for useful_class in useful_classes)[0]
        boxes = boxes[idxs]
        scores = scores[idxs]
        classes = classes[idxs]
    else:
        boxes = np.empty((0, 5))

    dets = np.hstack((boxes, scores[:, np.newaxis]))
    res = sort.update(dets)

    boxes_track = res[:, :-1]
    boxes_ids = res[:, -1].astype(int)

    print(boxes_track)
    for i in range(0, len(boxes_track)):
        cv2.rectangle(frame, (int(boxes_track[i][0]), int(boxes_track[i][1])), (int(boxes_track[i][2]), int(boxes_track[i][3])), BLUE, -1)
        cv2.putText(frame, str(boxes_ids[i]), (int(boxes_track[i][0]), int(boxes_track[i][1])), FONT_FACE, FONT_SIZE, WHITE, THICKNESS)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(0)
    if key == 27:
        break
