import cv2
import numpy as np
from typing import Optional
from ultralytics import YOLO
from models.ocr import config

class YoloDetector:
    def __init__(self, weights_path: str | None = None):
        self.weights_path = weights_path or config.YOLO_WEIGHTS
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        # Lazy-load mode
        if self._model is None:
            print(f"[YOLO] Loading weights: {self.weights_path}")
            self._model = YOLO(self.weights_path)
        return self._model

    # image: Ảnh BGR (numpy array)
    def detect(self, image: np.ndarray) -> list[dict]:
        results = self.model(
            image,
            conf=config.YOLO_CONF_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            imgsz=1280,
            verbose=False,
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy   = box.xyxy[0].cpu().numpy()          # (x1, y1, x2, y2)
            conf   = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, xyxy)

            pad = config.BBOX_PADDING
            x1_c = max(0, x1 - pad)
            y1_c = max(0, y1 - pad)
            x2_c = min(image.shape[1], x2 + pad)
            y2_c = min(image.shape[0], y2 + pad)

            crop = image[y1_c:y2_c, x1_c:x2_c]

            detections.append({
                'box':        (x1, y1, x2, y2),
                'confidence':  conf,
                'crop':       crop,
            })

        return detections

    def detect_from_file(self, image_path: str) -> list[dict]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.detect(image)
