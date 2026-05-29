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
from concurrent.futures import ThreadPoolExecutor

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

    # =====================================================
    # CHIA BATCH
    # =====================================================

    for i in range(0, len(images_paths), batch_size):

        batch_paths = images_paths[i:i + batch_size]

        batch_imgs = []
        valid_paths = []

        # =====================================================
        # LOAD ẢNH
        # =====================================================

        for p in batch_paths:

            img = cv2.imread(p)

            if img is not None:
                batch_imgs.append(img)
                valid_paths.append(p)
            else:
                print(f"[WARN] Cannot read: {p}")

        if not batch_imgs:
            continue

        # =====================================================
        # YOLO BATCH DETECTION
        # =====================================================

        t0 = time.perf_counter()

        batch_detections = yolo_detector.detect_batch(
            batch_imgs,
            batch_size=batch_size
        )

        yolo_time_total = time.perf_counter() - t0

        yolo_time_per_img = yolo_time_total / len(batch_imgs)

        # =====================================================
        # THU THẬP TẤT CẢ CROP
        # =====================================================

        all_crops = []

        crop_metadata = []

        for j, detections in enumerate(batch_detections):

            img_path = valid_paths[j]

            image_name = os.path.basename(img_path)

            image = batch_imgs[j]

            if detections:

                det = detections[0]

                crop = det['crop']

                enhanced = enhance_plate_image(crop)

                all_crops.append(enhanced)

                crop_metadata.append({
                    "image_name": image_name,
                    "image": image,
                    "detections": detections
                })

        # =====================================================
        # OCR BATCH
        # =====================================================

        ocr_outputs = []

        t_ocr = time.perf_counter()

        # =====================================================
        # OCR (TUẦN TỰ)
        # =====================================================
        # LƯU Ý: PaddleOCR C++ backend không hỗ trợ đa luồng (not thread-safe). 
        # Chạy ThreadPoolExecutor sẽ gây ra lỗi crash C++.
        # Do đó, ta duyệt qua các crop một cách tuần tự.
        
        ocr_outputs = []
        for crop in all_crops:
            ocr_outputs.append(ocr_engine.readtext(crop))


        ocr_time_total = time.perf_counter() - t_ocr

        ocr_time_per_img = (
            ocr_time_total / max(len(all_crops), 1)
        )

        # =====================================================
        # XỬ LÝ KẾT QUẢ OCR
        # =====================================================

        processed_images = set()

        for meta, raw_data in zip(crop_metadata, ocr_outputs):

            image_name = meta["image_name"]

            image = meta["image"]

            detections = meta["detections"]

            processed_images.add(image_name)

            final_plate = "NO_TEXT"

            confidence = 0.0

            items = raw_data.get('items', [])

            # =====================================================
            # OCR SUCCESS
            # =====================================================

            if items:

                if engine_name == 'easyocr':

                    items_sorted = sorted(
                        items,
                        key=lambda x:
                        sum(pt[1] for pt in x['bbox']) / len(x['bbox'])
                    )

                else:

                    items_sorted = sorted(
                        items,
                        key=lambda x: x['y_center']
                    )

                lines = [
                    it['text']
                    for it in items_sorted
                ]

                scores = [
                    it['confidence']
                    for it in items_sorted
                ]

                raw_plate = " ".join(
                    [
                        str(l).strip()
                        for l in lines
                        if str(l).strip()
                    ]
                )

                final_plate = apply_ocr_correction(raw_plate)

                confidence = aggregate_confidence(scores)

            # =====================================================
            # FALLBACK
            # =====================================================

            if final_plate == "NO_TEXT" and fallback_engine:

                if debug:
                    print(
                        f"    [Fallback] "
                        f"{image_name} -> PaddleOCR toàn ảnh"
                    )

                fallback_data = fallback_engine.readtext(image)

                f_items = fallback_data.get('items', [])

                if f_items:

                    f_items_sorted = sorted(
                        f_items,
                        key=lambda x: x['y_center']
                    )

                    f_lines = [
                        it['text']
                        for it in f_items_sorted
                    ]

                    f_scores = [
                        it['confidence']
                        for it in f_items_sorted
                    ]

                    f_raw_plate = " ".join(
                        [
                            str(l).strip()
                            for l in f_lines
                            if str(l).strip()
                        ]
                    )

                    f_final = apply_ocr_correction(f_raw_plate)

                    if f_final != "NO_TEXT":

                        final_plate = f_final

                        confidence = aggregate_confidence(f_scores)

            # =====================================================
            # METRICS
            # =====================================================

            total_time = (
                yolo_time_per_img +
                ocr_time_per_img
            )

            timing = TimingResult(
                image_name=image_name,
                yolo_time=yolo_time_per_img,
                ocr_time=ocr_time_per_img,
                total_time=total_time
            )

            gt_dict = (
                ground_truth.get(image_name, {})
                if ground_truth else {}
            )

            gt_full = gt_dict.get('full_plate', '')

            ca_province = False
            ca_series = False
            ca_number = False
            ca_full = False

            if gt_full:

                ca_province = component_accuracy(
                    final_plate,
                    gt_full,
                    'province'
                )

                ca_series = component_accuracy(
                    final_plate,
                    gt_full,
                    'series'
                )

                ca_number = component_accuracy(
                    final_plate,
                    gt_full,
                    'number'
                )

                ca_full = component_accuracy(
                    final_plate,
                    gt_full,
                    'full'
                )

            if debug:

                print(
                    f"[OK] {image_name} "
                    f"→ {final_plate} "
                    f"(conf={confidence:.3f})"
                )

            results.append(
                OcrResult(
                    image_name=image_name,
                    predicted_plate=final_plate,
                    confidence=confidence,
                    timing=timing,
                    ca_province=ca_province,
                    ca_series=ca_series,
                    ca_number=ca_number,
                    ca_full=ca_full,
                    engine=engine_name
                )
            )

        # =====================================================
        # HANDLE YOLO FAIL
        # =====================================================

        for j, detections in enumerate(batch_detections):

            image_name = os.path.basename(valid_paths[j])

            if image_name in processed_images:
                continue

            # =====================================================
            # FALLBACK CHO ẢNH BỊ YOLO BỎ QUA
            # =====================================================
            final_plate = "NO_TEXT"
            confidence = 0.0
            image = batch_imgs[j]
            ocr_time_f = 0.0
            
            if fallback_engine:
                if debug:
                    print(f"    [Fallback] YOLO không tìm thấy biển {image_name}. Đang chuyển ảnh gốc cho PaddleOCR...")
                
                t_f = time.perf_counter()
                fallback_data = fallback_engine.readtext(image)
                ocr_time_f = time.perf_counter() - t_f
                f_items = fallback_data.get('items', [])
                
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
            
            # Tính Metrics
            total_time = yolo_time_per_img + ocr_time_f
            timing = TimingResult(
                image_name=image_name, yolo_time=yolo_time_per_img, ocr_time=ocr_time_f, total_time=total_time
            )
            
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

def run_batch_pipeline(   #hàm để chạy toàn bộ pipeline
    #khai báo tham số đầu vào của hàm run_batch_pipeline
    engine_name: str,   #tên của mô hình OCR
    ground_truth: dict | None = None, #ground truth
    batch_size: int = 8, #kích thước batch
    debug: bool = False, #chế độ debug
) -> list[OcrResult]:
    
    print(f"\n{'='*50}") #In ra đường kẻ ngang
    print(f"  BATCH PIPELINE: YOLO11 + {engine_name.upper()} (Batch Size: {batch_size})") #In ra tên của pipeline và kích thước batch
    print(f"{'='*50}") #In ra đường kẻ ngang

    yolo = YoloDetector() #Khởi tạo YOLO
    ocr  = build_ocr_engine(engine_name) #Khởi tạo OCR
    fallback = PaddleOCREngine() if engine_name == 'easyocr' else ocr #Nếu là easyocr thì dùng PaddleOCR làm fallback, ngược lại thì dùng OCR mặc định

    #Khâu thu thập và chuẩn bị dữ liệu (Data Loading) trước khi đưa vào máy chấm điểm
    valid_exts = ('.jpg', '.jpeg', '.png') #Các định dạng ảnh được phép
    images_names = [f for f in os.listdir(config.TEST_IMAGE_DIR) if f.lower().endswith(valid_exts)] #Lấy danh sách các tệp ảnh
    images_names.sort() #Sắp xếp danh sách ảnh theo thứ tự bảng chữ cái
    images_paths = [os.path.join(config.TEST_IMAGE_DIR, f) for f in images_names] #Tạo danh sách đường dẫn tuyệt đối của các tệp ảnh
    print(f"[Data] {len(images_paths)} images found in: {config.TEST_IMAGE_DIR}\n") #In ra số lượng tệp ảnh và đường dẫn

    results = process_batch( #Gọi hàm process_batch để xử lý từng ảnh
        images_paths=images_paths, #Danh sách đường dẫn ảnh
        yolo_detector=yolo, #YOLO detector
        ocr_engine=ocr, #OCR engine
        engine_name=engine_name, #Tên OCR engine
        fallback_engine=fallback, #Fallback OCR engine
        ground_truth=ground_truth, #Ground truth
        batch_size=batch_size,  #Kích thước batch
        debug=debug, #Chế độ debug
    )

    # khâu "Tổng kết và Báo cáo" 
    print()
    agg = compute_aggregate_metrics(results) #Tính toán metrics
    print_metrics_report(agg) #In báo cáo metrics
    return results #Trả về kết quả

def main(): #hàm chính nơi sẽ thiết lập các câu lệnh, cờ (flags)
    parser = argparse.ArgumentParser(description="Batch Pipeline cho YOLO11 + OCR") #Khởi tạo argparse thư viện giúp user có thể truyền trực tiếp các tham số cấu hình từ màn hình Terminal mà không cần phải mở file code ra để sửa
    #Tạo các tùy chọn cho người dùng khi chạy script
    parser.add_argument('--engine', '-e', choices=['easyocr', 'paddleocr', 'both'], default='both') #Tùy chọn OCR engine
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Kích thước batch cho YOLO") #Kích thước batch cho YOLO
    parser.add_argument('--debug', '-d', action='store_true') #Chế độ debug
    args = parser.parse_args() #Parse các đối số

    gt = load_ground_truth(config.GROUND_TRUTH_CSV) #Load ground truth
    all_results = {} #Khởi tạo danh sách kết quả

    if args.engine in ('easyocr', 'both'): #Nếu là easyocr hoặc cả hai 
        results_easy = run_batch_pipeline('easyocr', gt, batch_size=args.batch_size, debug=args.debug) #Gọi hàm run_batch_pipeline để xử lý từng ảnh
        save_results(results_easy, config.OUTPUT_CSV_EASY) #Lưu kết quả vào file CSV 
        all_results['easyocr'] = results_easy #Lưu kết quả vào danh sách kết quả 

    if args.engine in ('paddleocr', 'both'): #Nếu là paddleocr hoặc cả hai
        results_paddle = run_batch_pipeline('paddleocr', gt, batch_size=args.batch_size, debug=args.debug) #Gọi hàm run_batch_pipeline để xử lý từng ảnh
        save_results(results_paddle, config.OUTPUT_CSV_PADDLE) #Lưu kết quả vào file CSV 
        all_results['paddleocr'] = results_paddle #Lưu kết quả vào danh sách kết quả

    if args.engine == 'both' and 'easyocr' in all_results and 'paddleocr' in all_results: #Nếu là cả hai và đã có kết quả của cả hai
        _print_comparison(all_results['easyocr'], all_results['paddleocr']) #In so sánh các metrics giữa hai engine

if __name__ == '__main__':
    main()
