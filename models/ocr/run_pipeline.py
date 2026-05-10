# python run_pipeline.py [--engine {easyocr,paddleocr,both}] [--debug]
import os
# --- PADDLE WINDOWS FIXES ---
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
import sys
import time
import argparse
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from yolo_detector import YoloDetector
from easyocr_engine import EasyOCREngine
from paddleocr_engine import PaddleOCREngine
from preprocessing import enhance_plate_image
from metrics import (OcrResult, TimingResult, component_accuracy, aggregate_confidence, compute_aggregate_metrics, print_metrics_report, _parse_gt_components,_parse_pred_components)
import cv2

# Ground Truth Loader
def load_ground_truth(csv_path: str) -> dict[str, dict]:
    if not os.path.exists(csv_path):
        print(f"[WARNING] Ground truth CSV not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path, dtype=str).fillna('')
    result = {}
    for _, row in df.iterrows():
        fname = str(row.get('filename', '')).strip()
        if not fname:
            continue
        result[fname] = {
            'full_plate': str(row.get('full_plate', '')).strip(),
            'province':   str(row.get('province',   '')).strip(),
            'series':     str(row.get('series',     '')).strip(),
            'number':     str(row.get('number',     '')).strip(),
        }
    print(f"[GT] Loaded {len(result)} ground truth entries.")
    return result

def build_ocr_engine(engine_name: str):
    if engine_name == 'easyocr':
        return EasyOCREngine()
    elif engine_name == 'paddleocr':
        return PaddleOCREngine()
    else:
        raise ValueError(f"Unknown engine: {engine_name}")
    

def apply_ocr_correction(text: str) -> str:
    """
    Hàm Post-processing: Ép kiểu OCR dựa trên luật định dạng vị trí biển số xe máy.
    Định dạng mục tiêu: [2 Tỉnh (Số)] - [1 Chữ][1 Số] [4-5 Số]
    """
    # Xóa các ký tự đặc biệt, khoảng trắng, dấu gạch ngang để lấy chuỗi thô (VD: "60855723")
    clean_text = "".join(c for c in text.upper() if c.isalnum())
    
    # Biển số xe máy thông thường có 8 hoặc 9 ký tự. 
    # Nếu OCR đọc thiếu quá nhiều, trả về nguyên bản để tránh cắt chuỗi bị lỗi index.
    if len(clean_text) < 8:
        return text 
        
    # --- TỪ ĐIỂN MAPPING ---
    # Ép sang CHỮ (Áp dụng cho ký tự Series 1)
    NUM_TO_CHAR = {
        '8': 'B', '5': 'S', '0': 'D', '2': 'Z', '6': 'G', 
        '7': 'T', '1': 'V', '3': 'E', '4': 'A', '9': 'P'
    }
    
    # Ép sang SỐ (Áp dụng cho Mã Tỉnh, Series 2, và dãy Số cuối)
    CHAR_TO_NUM = {v: k for k, v in NUM_TO_CHAR.items()}
    # Bổ sung thêm các luật nhiễu thường gặp nhất:
    CHAR_TO_NUM.update({
        'O': '0', 'Q': '0', 'D': '0',
        'T': '1',  # Theo rule của bạn: FT -> F1
        'I': '1', 'L': '4',
        'B': '8', 'S': '5', 'G': '6', 'Z': '2'
    })
    
    # --- BÓC TÁCH VỊ TRÍ ---
    province = clean_text[:2]   # 2 ký tự đầu
    series   = clean_text[2:4]  # 2 ký tự tiếp theo
    number   = clean_text[4:]   # 4 hoặc 5 ký tự cuối cùng
    
    # 1. SỬA TỈNH: Ép 2 ký tự đầu BẮT BUỘC phải là SỐ
    corr_province = "".join([CHAR_TO_NUM.get(c, c) for c in province])
    
    # 2. SỬA SERIES: Ép ký tự đầu là CHỮ, ký tự sau là SỐ
    s1 = NUM_TO_CHAR.get(series[0], series[0]) 
    s2 = CHAR_TO_NUM.get(series[1], series[1]) 
    corr_series = f"{s1}{s2}"
    
    # 3. SỬA DÃY SỐ: Ép toàn bộ phần đuôi BẮT BUỘC phải là SỐ
    corr_number = "".join([CHAR_TO_NUM.get(c, c) for c in number])

    if len(corr_number) > 5:
        corr_number = corr_number[-5:] 
    
    # Format lại thành chuỗi chuẩn: XX-XX XXXXX (Khớp với file Ground Truth)
    return f"{corr_province}-{corr_series} {corr_number}"

def process_image(
    image_path: str,
    yolo_detector: YoloDetector,
    ocr_engine,
    engine_name: str,
    fallback_engine: PaddleOCREngine = None, # Thêm engine dự phòng
    ground_truth: dict | None = None,
    debug: bool = False,
) -> OcrResult | None:
    
    image_name = os.path.basename(image_path)
    image = cv2.imread(image_path)
    if image is None: return None

    final_plate = "NO_TEXT"
    confidence = 0.0
    yolo_time = 0.0
    ocr_time = 0.0

    # --- BƯỚC 1: CHẠY PIPELINE CHÍNH (YOLO + CROP) ---
    t0 = time.perf_counter()
    detections = yolo_detector.detect(image)
    yolo_time = time.perf_counter() - t0

    if detections:
        det = detections[0]
        crop = det['crop']
        enhanced = enhance_plate_image(crop)

        raw_data = ocr_engine.readtext(enhanced)
        items = raw_data.get('items', [])
        ocr_time = raw_data.get('elapsed', 0.0)

        # Trích xuất text (giữ nguyên logic cũ của bạn)
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

    # --- BƯỚC 2: LUỒNG FALLBACK (NẾU NO_PLATE HOẶC NO_TEXT) ---
    # Nếu kết quả vẫn là NO_PLATE hoặc NO_TEXT, cho PaddleOCR quét toàn ảnh gốc
    if (not detections or final_plate == "NO_TEXT") and fallback_engine:
        if debug:
            reason = "YOLO không tìm thấy biển" if not detections else "OCR trên vùng cắt thất bại"
            print(f"    [Fallback] {reason}. Đang chuyển ảnh gốc cho PaddleOCR...")

        fallback_data = fallback_engine.readtext(image) # Quét trên ảnh gốc
        f_items = fallback_data.get('items', [])
        ocr_time += fallback_data.get('elapsed', 0.0) # Cộng dồn thời gian

        if f_items:
            f_items_sorted = sorted(f_items, key=lambda x: x['y_center'])
            f_lines = [it['text'] for it in f_items_sorted]
            f_scores = [it['confidence'] for it in f_items_sorted]
            
            f_raw_plate = " ".join([str(l).strip() for l in f_lines if str(l).strip()])
            f_final = apply_ocr_correction(f_raw_plate)
            
            # Chỉ lấy kết quả fallback nếu nó ra được biển số hợp lệ
            if f_final != "NO_TEXT":
                final_plate = f_final
                confidence = aggregate_confidence(f_scores)
                if debug: print(f"    [Success] Fallback cứu thành công: {final_plate}")

    # --- PHẦN CÒN LẠI (Timing, Metrics...) GIỮ NGUYÊN ---
    timing = TimingResult(image_name=image_name, yolo_time=yolo_time, ocr_time=ocr_time, total_time=yolo_time + ocr_time)
    
    # Evaluate against ground truth
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

    return OcrResult(image_name=image_name, predicted_plate=final_plate, confidence=confidence, timing=timing,
                     ca_province=ca_province, ca_series=ca_series, ca_number=ca_number, ca_full=ca_full, engine=engine_name)

# YOLO detect → crop → enhance → OCR → metrics.
# def process_image(
#     image_path: str,
#     yolo_detector: YoloDetector,
#     ocr_engine,
#     engine_name: str,
#     ground_truth: dict | None = None,
#     debug: bool = False,
# ) -> OcrResult | None:
    
#     image_name = os.path.basename(image_path)
#     image = cv2.imread(image_path)

#     if image is None:
#         print(f"[WARN] Cannot read: {image_path}")
#         return None

#     # YOLO Detection 
#     t0 = time.perf_counter()
#     detections = yolo_detector.detect(image)
#     yolo_time = time.perf_counter() - t0

#     if not detections:
#         result = OcrResult(
#             image_name=image_name,
#             predicted_plate='NO_PLATE',
#             confidence=0.0,
#             timing=TimingResult(image_name=image_name, yolo_time=yolo_time, total_time=time.perf_counter() - t0),
#             engine=engine_name,
#         )
#         return result

#     det  = detections[0]
#     crop = det['crop']
#     enhanced = enhance_plate_image(crop)

#     raw_data = ocr_engine.readtext(enhanced)
#     items = raw_data.get('items', [])
#     ocr_time = raw_data.get('elapsed', 0.0)

#     lines = []
#     scores = []

#     # OCR
#     if engine_name == 'easyocr':
#         if items:
#             # Xử lý ảnh nghiêng
#             items_sorted = sorted(items, key=lambda x: sum(pt[1] for pt in x['bbox'])/len(x['bbox']))
#             lines = [it['text'] for it in items_sorted]
#             scores = [it['confidence'] for it in items_sorted]

#     elif engine_name == 'paddleocr':
#         if items:
#             items_sorted = sorted(items, key=lambda x: x['y_center'])
#             lines = [it['text'] for it in items_sorted]
#             scores = [it['confidence'] for it in items_sorted]

#     if not lines:
#         final_plate = "NO_TEXT"
#     # else:
#     #     # Ví dụ: lines = ['29A1', '12345'] -> "29A1 12345"
#     #     final_plate = " ".join([str(l).strip() for l in lines if str(l).strip()])
        
#     #     if not final_plate:
#     #         final_plate = "NO_TEXT"
#     else:
#         # Ghép lines thành chuỗi thô ban đầu (VD: "60-85 5723" hoặc "77 FT 30541")
#         raw_plate = " ".join([str(l).strip() for l in lines if str(l).strip()])
        
#         if not raw_plate:
#             final_plate = "NO_TEXT"
#         else:
#             # GỌI HÀM POST-PROCESSING TẠI ĐÂY
#             final_plate = apply_ocr_correction(raw_plate)

#     # Aggregate metrics
#     confidence = aggregate_confidence(scores)
#     total_time = yolo_time + ocr_time

#     timing = TimingResult(
#         image_name=image_name,
#         yolo_time=yolo_time,
#         ocr_time=ocr_time,
#         total_time=total_time,
#     )

#     # Evaluate against ground truth
#     gt_dict = ground_truth.get(image_name, {}) if ground_truth else {}
#     gt_full = gt_dict.get('full_plate', '')

#     ca_province = False
#     ca_series    = False
#     ca_number    = False
#     ca_full      = False

#     if gt_full:
#         ca_province = component_accuracy(final_plate, gt_full, 'province')
#         ca_series   = component_accuracy(final_plate, gt_full, 'series')
#         ca_number   = component_accuracy(final_plate, gt_full, 'number')
#         ca_full     = component_accuracy(final_plate, gt_full, 'full')

#     if debug:
#         tag = "[OK]"
#         print(f"  {tag} {image_name} → {final_plate}  (conf={confidence:.3f})")

#     return OcrResult(
#         image_name=image_name,
#         predicted_plate=final_plate,
#         confidence=confidence,
#         timing=timing,
#         ca_province=ca_province,
#         ca_series=ca_series,
#         ca_number=ca_number,
#         ca_full=ca_full,
#         engine=engine_name,
#     )

# Batch pipeline
def run_pipeline(
    engine_name: str,
    ground_truth: dict | None = None,
    debug: bool = False,
) -> list[OcrResult]:
    
    print(f"\n{'='*50}")
    print(f"  PIPELINE: YOLO11 + {engine_name.upper()}")
    print(f"{'='*50}")

    yolo = YoloDetector()
    ocr  = build_ocr_engine(engine_name)

    # SUA DAY
    fallback = PaddleOCREngine() if engine_name == 'easyocr' else ocr

    # Danh sách ảnh
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [
        f for f in os.listdir(config.TEST_IMAGE_DIR)
        if f.lower().endswith(valid_exts)
    ]
    images.sort()
    print(f"[Data] {len(images)} images found in: {config.TEST_IMAGE_DIR}\n")

    results: list[OcrResult] = []
    # flipped_images: list[dict] = []

    for fname in images:
        img_path = os.path.join(config.TEST_IMAGE_DIR, fname)
        res = process_image(
            image_path=img_path,
            yolo_detector=yolo,
            ocr_engine=ocr,
            engine_name=engine_name,
            fallback_engine=fallback, # Truyền fallback vào đây
            ground_truth=ground_truth,
            debug=debug,
        )
        if res:
            results.append(res)

    print()
    agg = compute_aggregate_metrics(results)
    print_metrics_report(agg)
    return results

# Save results
def save_results(results: list[OcrResult], output_csv: str):
    """Lưu kết quả ra CSV."""
    rows = []
    for r in results:
        rows.append({
            'filename':          r.image_name,
            'predicted_plate':   r.predicted_plate,
            'confidence':        round(r.confidence, 4),
            'ca_province':       r.ca_province,
            'ca_series':         r.ca_series,
            'ca_number':         r.ca_number,
            'ca_full':           r.ca_full,
            'yolo_ms':           round(r.timing.yolo_ms,  2),
            'ocr_ms':            round(r.timing.ocr_ms,   2),
            'total_ms':          round(r.timing.total_ms, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n[Output] Results saved → {output_csv}")

def main():
    parser = argparse.ArgumentParser(
        description="OCR Pipeline: YOLO11 + EasyOCR/PaddleOCR cho xe máy VN"
    )
    parser.add_argument(
        '--engine', '-e',
        choices=['easyocr', 'paddleocr', 'both'],
        default='both',
        help="Chọn OCR engine (mặc định: both để so sánh)",
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help="In log chi tiết từng ảnh",
    )
    args = parser.parse_args()

    gt = load_ground_truth(config.GROUND_TRUTH_CSV)
    all_results = {}

    if args.engine in ('easyocr', 'both'):
        results_easy = run_pipeline('easyocr', gt, debug=args.debug)
        save_results(results_easy, config.OUTPUT_CSV_EASY)
        all_results['easyocr'] = results_easy

    if args.engine in ('paddleocr', 'both'):
        results_paddle = run_pipeline('paddleocr', gt, debug=args.debug)
        save_results(results_paddle, config.OUTPUT_CSV_PADDLE)
        all_results['paddleocr'] = results_paddle

    if args.engine == 'both' and 'easyocr' in all_results and 'paddleocr' in all_results:
        _print_comparison(all_results['easyocr'], all_results['paddleocr'])


def _print_comparison(easy_results: list, paddle_results: list):
    print("\n" + "=" * 60)
    print("                SO SÁNH: EasyOCR vs PaddleOCR")
    print("=" * 60)

    agg_easy   = compute_aggregate_metrics(easy_results)
    agg_paddle = compute_aggregate_metrics(paddle_results)

    def row(label, easy_v, paddle_v, higher_is_better=True):
        if easy_v == 'N/A' or paddle_v == 'N/A':
            winner = ""
        elif easy_v == paddle_v:
            winner = "≈ tie"
        else:
            is_easy_better = (easy_v > paddle_v) if higher_is_better else (easy_v < paddle_v)
            winner = "EasyOCR" if is_easy_better else "PaddleOCR"
            
        return f"  {label:<30} Easy={easy_v:<10} Paddle={paddle_v:<10} {winner}"

    print(row("CA Province (%)", agg_easy.get('ca_province_%', 0), agg_paddle.get('ca_province_%', 0)))
    print(row("CA Series (%)",   agg_easy.get('ca_series_%',   0), agg_paddle.get('ca_series_%',   0)))
    print(row("CA Number (%)",   agg_easy.get('ca_number_%',   0), agg_paddle.get('ca_number_%',   0)))
    print(row("CA Full (%)",     agg_easy.get('ca_full_%',     0), agg_paddle.get('ca_full_%',     0)))
    print(row("Mean Confidence", agg_easy.get('mean_confidence', 0), agg_paddle.get('mean_confidence', 0)))
    print(row("Mean Total ms",   agg_easy.get('mean_total_ms', 0), agg_paddle.get('mean_total_ms', 0), higher_is_better=False))
    print(row("Mean OCR ms",     agg_easy.get('mean_ocr_ms',   0), agg_paddle.get('mean_ocr_ms',   0), higher_is_better=False))

if __name__ == '__main__':
    main()
