import os
import sys
import time
import cv2
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.ocr import config
from models.ocr.yolo_detector import YoloDetector
from models.ocr.paddleocr_engine import PaddleOCREngine
from models.ocr.preprocessing import enhance_plate_image

def run_benchmark():
    # Load 10 images
    valid_exts = ('.jpg', '.jpeg', '.png')
    images_names = [f for f in os.listdir(config.TEST_IMAGE_DIR) if f.lower().endswith(valid_exts)]
    images_names.sort()
    
    if len(images_names) < 10:
        print(f"Không đủ 10 ảnh trong {config.TEST_IMAGE_DIR}")
        return
        
    images_paths = [os.path.join(config.TEST_IMAGE_DIR, f) for f in images_names[:10]]
    images = [cv2.imread(p) for p in images_paths]
    
    print("Khởi tạo mô hình...")
    yolo = YoloDetector()
    ocr = PaddleOCREngine()
    
    # Warm-up (Khởi động GPU để thời gian đo được chính xác)
    print("Đang chạy khởi động (Warm-up GPU)...")
    _ = yolo.detect(images[0])
    _ = ocr.readtext(images[0])
    
    print("\n--- 1. CHẠY TUẦN TỰ TỪNG ẢNH (10 ẢNH RIÊNG LẺ) ---")
    yolo_seq_time = 0.0
    ocr_seq_time = 0.0
    
    t_seq_start = time.perf_counter()
    for img in images:
        t_y = time.perf_counter()
        detections = yolo.detect(img)
        yolo_seq_time += time.perf_counter() - t_y
        
        if detections:
            crop = detections[0]['crop']
            enhanced = enhance_plate_image(crop)
            t_o = time.perf_counter()
            _ = ocr.readtext(enhanced)
            ocr_seq_time += time.perf_counter() - t_o
            
    total_seq_time = time.perf_counter() - t_seq_start
    
    print("\n--- 2. CHẠY BATCH (GỘP CHUNG BATCH 10 ẢNH) ---")
    t_batch_start = time.perf_counter()
    
    # Đo thời gian YOLO xử lý cả batch 10
    t_y_batch = time.perf_counter()
    batch_detections = yolo.detect_batch(images, batch_size=10)
    yolo_batch_time = time.perf_counter() - t_y_batch
    
    # OCR tuần tự các crop (do PaddleOCR C++ backend không cho phép chạy đa luồng an toàn trên Windows)
    ocr_batch_time = 0.0
    for detections in batch_detections:
        if detections:
            crop = detections[0]['crop']
            enhanced = enhance_plate_image(crop)
            t_o = time.perf_counter()
            _ = ocr.readtext(enhanced)
            ocr_batch_time += time.perf_counter() - t_o
            
    total_batch_time = time.perf_counter() - t_batch_start

    # Tạo bảng báo cáo
    data = [
        {
            "Phương pháp": "Chạy 10 ảnh riêng lẻ (Tuần tự)",
            "Tổng thời gian YOLO (s)": round(yolo_seq_time, 3),
            "Tổng thời gian OCR (s)": round(ocr_seq_time, 3),
            "Tổng thời gian Toàn bộ (s)": round(total_seq_time, 3),
            "FPS (Ảnh/giây)": round(10 / total_seq_time, 2)
        },
        {
            "Phương pháp": "Chạy Batch (Batch size = 10)",
            "Tổng thời gian YOLO (s)": round(yolo_batch_time, 3),
            "Tổng thời gian OCR (s)": round(ocr_batch_time, 3),
            "Tổng thời gian Toàn bộ (s)": round(total_batch_time, 3),
            "FPS (Ảnh/giây)": round(10 / total_batch_time, 2)
        }
    ]
    
    df = pd.DataFrame(data)
    
    print("\n" + "="*80)
    print("BẢNG SO SÁNH HIỆU NĂNG (10 ẢNH) TRÊN GPU RTX 4060".center(80))
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    print(f"Tiết kiệm thời gian tổng cộng: {round(total_seq_time - total_batch_time, 3)} giây")
    print(f"Tốc độ tăng trưởng FPS: +{round((10/total_batch_time) - (10/total_seq_time), 2)} Ảnh/giây")

if __name__ == "__main__":
    run_benchmark()
