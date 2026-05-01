import re
import cv2
import numpy as np
from typing import Optional
# import models.ocr.config as config
from models.ocr import config

# Image Enhancement BGR
def enhance_plate_image(plate_crop: np.ndarray) -> np.ndarray:
    # Resize ×2
    res_img = cv2.resize(
        plate_crop,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrasted = clahe.apply(gray)
    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)

