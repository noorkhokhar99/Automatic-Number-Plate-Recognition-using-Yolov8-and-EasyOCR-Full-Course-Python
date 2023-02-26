from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2

model = YOLO("yolov8n.pt")

results = model.predict(source=0, show=True)
print(results)