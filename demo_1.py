import cv2
import torch

# Text Parameters
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 10

# Colors
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
video_path = "highway.mp4"
cap = cv2.VideoCapture(video_path)

center_points = []

while True:
    ret, frame = cap.read()
    results = model(frame)
    readable_results = results.pandas().xyxy  # xmin, ymin, xmax, ymax, confidence, class, name
    new_points = []

    if not ret:
        continue

    print(readable_results)
    for result in readable_results:
        for i in range(0, len(result)):
            xmin = int(result['xmin'][i])
            ymin = int(result['ymin'][i])
            xmax = int(result['xmax'][i])
            ymax = int(result['ymax'][i])
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
            center_points.append(center)
            cv2.rectangle(frame, start_point, end_point, BLUE, THICKNESS)
            # cv2.circle(frame, center, 5, RED, -1)
    for center in center_points:
        cv2.circle(frame, center, 5, RED, -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
