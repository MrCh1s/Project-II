# python batch_process.py [--engine {easyocr,paddleocr,both}] [--batch_size 8] [--debug]
import os
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
import sys
import time
import argparse
import numpy as np
import pandas as pd
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from yolo_detector import YoloDetector
from easyocr_engine import EasyOCREngine
from paddleocr_engine import PaddleOCREngine
from preprocessing import enhance_plate_image
from run_pipeline import load_ground_truth, apply_ocr_correction, build_ocr_engine, save_results, _print_comparison
from metrics import OcrResult, TimingResult, aggregate_confidence, component_accuracy, compute_aggregate_metrics, print_metrics_report

def process_batch(
    images_paths: list[str],
    yolo_detector: YoloDetector,
    ocr_engine,
    engine_name: str,
    fallback_engine=None,
    ground_truth: dict = None,
    batch_size: int = 8,
    debug: bool = False,
) -> list[OcrResult]:
    
    results: list[OcrResult] = []
    
    # Chia danh sách ảnh thành các lô (batch)
    for i in range(0, len(images_paths), batch_size):
        batch_paths = images_paths[i:i + batch_size]
        batch_imgs = []
        valid_paths = []
        
        # Đọc ảnh
        for p in batch_paths:
            img = cv2.imread(p)
            if img is not None:
                batch_imgs.append(img)
                valid_paths.append(p)
            else:
                print(f"[WARN] Cannot read: {p}")
                
        if not batch_imgs:
            continue
            
        # 1. YOLO BATCH INFERENCE
        t0 = time.perf_counter()
        batch_detections = yolo_detector.detect_batch(batch_imgs, batch_size=batch_size)
        yolo_time_total = time.perf_counter() - t0
        yolo_time_per_img = yolo_time_total / len(batch_imgs)
        
        # 2. Xử lý OCR cho từng ảnh trong batch (OCR tuần tự hoặc trên các crop)
        for j, detections in enumerate(batch_detections):
            img_path = valid_paths[j]
            image_name = os.path.basename(img_path)
            image = batch_imgs[j]
            
            final_plate = "NO_TEXT"
            confidence = 0.0
            ocr_time = 0.0
            
            if detections:
                det = detections[0]
                crop = det['crop']
                enhanced = enhance_plate_image(crop)

                raw_data = ocr_engine.readtext(enhanced)
                items = raw_data.get('items', [])
                ocr_time = raw_data.get('elapsed', 0.0)

                lines = []
                scores = []
                if items:
                    if engine_name == 'easyocr':
                        items_sorted = sorted(items, key=lambda x: sum(pt[1] for pt in x['bbox'])/len(x['bbox']))
                    else: # paddleocr
                        items_sorted = sorted(items, key=lambda x: x['y_center'])
                    
                    lines = [it['text'] for it in items_sorted]
                    scores = [it['confidence'] for it in items_sorted]
                    
                    raw_plate = " ".join([str(l).strip() for l in lines if str(l).strip()])
                    final_plate = apply_ocr_correction(raw_plate)
                    confidence = aggregate_confidence(scores)

            # FALLBACK LOGIC
            if (not detections or final_plate == "NO_TEXT") and fallback_engine:
                if debug:
                    reason = "YOLO không tìm thấy biển" if not detections else "OCR trên vùng cắt thất bại"
                    print(f"    [Fallback] {reason}. Đang chuyển ảnh gốc cho PaddleOCR...")

                fallback_data = fallback_engine.readtext(image) 
                f_items = fallback_data.get('items', [])
                ocr_time += fallback_data.get('elapsed', 0.0) 

                if f_items:
                    f_items_sorted = sorted(f_items, key=lambda x: x['y_center'])
                    f_lines = [it['text'] for it in f_items_sorted]
                    f_scores = [it['confidence'] for it in f_items_sorted]
                    
                    f_raw_plate = " ".join([str(l).strip() for l in f_lines if str(l).strip()])
                    f_final = apply_ocr_correction(f_raw_plate)
                    
                    if f_final != "NO_TEXT":
                        final_plate = f_final
                        confidence = aggregate_confidence(f_scores)
                        if debug: print(f"    [Success] Fallback cứu thành công: {final_plate}")

            # ĐÁNH GIÁ METRICS
            total_time = yolo_time_per_img + ocr_time
            timing = TimingResult(image_name=image_name, yolo_time=yolo_time_per_img, ocr_time=ocr_time, total_time=total_time)
            
            gt_dict = ground_truth.get(image_name, {}) if ground_truth else {}
            gt_full = gt_dict.get('full_plate', '')
            ca_province = ca_series = ca_number = ca_full = False

            if gt_full:
                ca_province = component_accuracy(final_plate, gt_full, 'province')
                ca_series   = component_accuracy(final_plate, gt_full, 'series')
                ca_number   = component_accuracy(final_plate, gt_full, 'number')
                ca_full     = component_accuracy(final_plate, gt_full, 'full')

            if debug:
                print(f"  [OK] {image_name} → {final_plate} (conf={confidence:.3f})")

            results.append(OcrResult(
                image_name=image_name, predicted_plate=final_plate, confidence=confidence, timing=timing,
                ca_province=ca_province, ca_series=ca_series, ca_number=ca_number, ca_full=ca_full, engine=engine_name
            ))
            
    return results

def run_batch_pipeline(
    engine_name: str,
    ground_truth: dict | None = None,
    batch_size: int = 8,
    debug: bool = False,
) -> list[OcrResult]:
    
    print(f"\n{'='*50}")
    print(f"  BATCH PIPELINE: YOLO11 + {engine_name.upper()} (Batch Size: {batch_size})")
    print(f"{'='*50}")

    yolo = YoloDetector()
    ocr  = build_ocr_engine(engine_name)
    fallback = PaddleOCREngine() if engine_name == 'easyocr' else ocr

    valid_exts = ('.jpg', '.jpeg', '.png')
    images_names = [f for f in os.listdir(config.TEST_IMAGE_DIR) if f.lower().endswith(valid_exts)]
    images_names.sort()
    images_paths = [os.path.join(config.TEST_IMAGE_DIR, f) for f in images_names]
    print(f"[Data] {len(images_paths)} images found in: {config.TEST_IMAGE_DIR}\n")

    results = process_batch(
        images_paths=images_paths,
        yolo_detector=yolo,
        ocr_engine=ocr,
        engine_name=engine_name,
        fallback_engine=fallback,
        ground_truth=ground_truth,
        batch_size=batch_size,
        debug=debug,
    )

    print()
    agg = compute_aggregate_metrics(results)
    print_metrics_report(agg)
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch Pipeline cho YOLO11 + OCR")
    parser.add_argument('--engine', '-e', choices=['easyocr', 'paddleocr', 'both'], default='both')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Kích thước batch cho YOLO")
    parser.add_argument('--debug', '-d', action='store_true')
    args = parser.parse_args()

    gt = load_ground_truth(config.GROUND_TRUTH_CSV)
    all_results = {}

    if args.engine in ('easyocr', 'both'):
        results_easy = run_batch_pipeline('easyocr', gt, batch_size=args.batch_size, debug=args.debug)
        save_results(results_easy, config.OUTPUT_CSV_EASY)
        all_results['easyocr'] = results_easy

    if args.engine in ('paddleocr', 'both'):
        results_paddle = run_batch_pipeline('paddleocr', gt, batch_size=args.batch_size, debug=args.debug)
        save_results(results_paddle, config.OUTPUT_CSV_PADDLE)
        all_results['paddleocr'] = results_paddle

    if args.engine == 'both' and 'easyocr' in all_results and 'paddleocr' in all_results:
        _print_comparison(all_results['easyocr'], all_results['paddleocr'])

if __name__ == '__main__':
    main()
