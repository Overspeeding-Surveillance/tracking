import cv2
import torch
from custom_utils.euclidean import EuclideanDistanceTracker
from custom_utils.capture import capture_vehicle


"""
This demo tracks the cars based on the euclidean distance (version 2)
"""

# Text Parameters
FONT_FACE = 0
FONT_SIZE = 0.5
THICKNESS = 1

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (255, 178, 50)
RED = (0, 0, 255)
YELLOW = (0, 100, 100)
GREEN = (0, 255, 0)

colors = [RED, YELLOW, BLUE, GREEN]

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
video_path = "highway.mp4"
cap = cv2.VideoCapture(video_path)

tracker = EuclideanDistanceTracker()

MAX_SPEED = 50

while True:
    ret, frame = cap.read()
    results = model(frame)

    if not ret:
        continue

    response = tracker.update(results)

    for vehicle_id, info in response.items():  # info: {'box': [[], ...], 'class': 1}
        current_color = colors[int(vehicle_id) % len(colors)]
        x1 = info['box'][0]
        y1 = info['box'][1]
        x2 = info['box'][2]
        y2 = info['box'][3]
        speed = info['speed']
        # for bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 1)
        # for text
        cv2.rectangle(frame, (x1, y1 - 15), (x1 + 50, y1), current_color, -1)
        cv2.putText(frame, str(vehicle_id), (x1, y1), FONT_FACE, FONT_SIZE, WHITE, THICKNESS)
        if speed and speed > MAX_SPEED:
            capture_vehicle(frame, x1, y1, x2, y2)
        cv2.putText(frame, str(speed), (x2, y2), FONT_FACE, FONT_SIZE, WHITE, THICKNESS)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
