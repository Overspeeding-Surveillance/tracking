import math
import cv2
import numpy as np


def get_points_within_distance(new_points: list[int], prev_points: list[dict[str, tuple[int, int] | int]]):
    """
        only used in demo_2.py
    """

    radius = 20  # pixels
    res_points = []
    existing_ids = []

    for prev_point in prev_points:
        for new_point in new_points:
            # print(prev_point)
            new_point_x = int(new_point[0])
            new_point_y = int(new_point[1])
            prev_point_x = int(prev_point["coord"][0])
            prev_point_y = int(prev_point["coord"][1])
            if abs(math.hypot(new_point_x - prev_point_x, new_point_y - prev_point_y)) <= radius:
                if not prev_point["id"] in existing_ids:
                    existing_ids.append(prev_point["id"])
                    res_points.append({"id": prev_point["id"], "coord": new_point})

    return res_points


class EuclideanDistanceTracker:
    """
        used only for demo_3.py
        only works for results from  yolov5 model using pytorch
    """
    def __init__(self):
        self.center_points = {}  # {"23": (123, 122), "12": (134, 433)}
        self.max_distance = 20  # radius
        self.count = 0

    def update(self, results):
        res_points = {}

        new_points: list[tuple[int, int]] = []
        readable_results = results.pandas().xyxy  # xmin, ymin, xmax, ymax, confidence, class, name
        for result in readable_results:
            for i in range(0, len(result)):
                xmin = int(result['xmin'][i])
                ymin = int(result['ymin'][i])
                xmax = int(result['xmax'][i])
                ymax = int(result['ymax'][i])
                center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
                new_points.append(center)

        for new_point in new_points:
            was_detected_in_previous_frame = False
            for point_id, coord in self.center_points.items():
                if abs(math.hypot(coord[0] - new_point[0], coord[1] - new_point[1])) < self.max_distance:
                    was_detected_in_previous_frame = True
                    res_points[point_id] = new_point
                    break

            if was_detected_in_previous_frame is False:
                self.count = self.count + 1
                new_id = str(self.count)
                res_points[new_id] = new_point

        self.center_points = res_points.copy()
        return res_points


class KalmanFilter:
    kf = cv2.KalmanFilter(4, 2)  # 4: dimensionality of state, 2: dimensionality of measurement
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def predict(self, coordX, coordY):
        # This function estimates the position of the object
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y


