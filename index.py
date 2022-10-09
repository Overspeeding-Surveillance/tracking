import torch
import cv2

# TODO: add label to the bounding-boxes
# TODO: add a condition check to see if its a vehicle

# Text Parameters
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 10

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)


def track():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        results = model(frame)
        readable_results = results.pandas().xyxy  # returns pandas' DataFrame
        for result in readable_results:
            xmin = int(result['xmin'][0])  # result['xmin'] returns pandas' series
            ymin = int(result['ymin'][0])
            xmax = int(result['xmax'][0])
            ymax = int(result['ymax'][0])
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            cv2.rectangle(frame, start_point, end_point, BLUE, THICKNESS)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    track()
