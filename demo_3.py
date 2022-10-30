import cv2
import torch
from custom_utils import EuclideanDistanceTracker

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

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
video_path = "highway.mp4"
cap = cv2.VideoCapture(video_path)

tracker = EuclideanDistanceTracker()

while True:
    ret, frame = cap.read()
    results = model(frame)

    if not ret:
        continue

    response = tracker.update(results)

    for vehicle_id, coord in response.items():
        cv2.rectangle(frame, (coord[0], coord[1] - 15), (coord[0] + 20, coord[1] + 5), BLUE, -1)
        cv2.putText(frame, str(vehicle_id), coord, FONT_FACE, FONT_SIZE, WHITE, THICKNESS)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
