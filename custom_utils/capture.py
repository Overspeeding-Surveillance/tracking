import cv2
import uuid


def capture_vehicle(frame, x1, y1, x2, y2):
    roi = frame[y1: y2, x1:x2]
    path = "captures/" + str(uuid.uuid4()) + ".jpg"
    cv2.imwrite(path, roi)

