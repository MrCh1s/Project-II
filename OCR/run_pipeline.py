"""
run_pipeline.py – Main pipeline: YOLO + EasyOCR / PaddleOCR cho xe máy VN

So sánh 2 OCR engine (EasyOCR vs PaddleOCR) trên ảnh xe máy,
dùng chung YOLO detector đã train.
Kết quả + metrics được lưu ra CSV.

Usage:
    python run_pipeline.py [--engine {easyocr,paddleocr,both}] [--debug]

Example:
    python run_pipeline.py --engine both --debug
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import Levenshtein
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from yolo_detector       import YoloDetector
from easyocr_engine      import EasyOCREngine
from paddleocr_engine    import PaddleOCREngine
from preprocessing       import (
    enhance_plate_image,
    # postprocess_plate_text,
    parse_plate_components,
)
from metrics import (
    OcrResult,
    TimingResult,
    component_accuracy,
    aggregate_confidence,
    compute_aggregate_metrics,
    print_metrics_report,
)


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

# OCR Engine Selector
def build_ocr_engine(engine_name: str):
    if engine_name == 'easyocr':
        return EasyOCREngine()
    elif engine_name == 'paddleocr':
        return PaddleOCREngine()
    else:
        raise ValueError(f"Unknown engine: {engine_name}")


# Single-image processing: YOLO detect → crop → enhance → OCR → postprocess → metrics.
def process_image(
    image_path: str,
    yolo_detector: YoloDetector,
    ocr_engine,
    engine_name: str,
    ground_truth: dict | None = None,
    debug: bool = False,
) -> OcrResult | None:
    """
    Args:
        image_path:    Đường dẫn ảnh.
        yolo_detector: YoloDetector instance.
        ocr_engine:    EasyOCREngine hoặc PaddleOCREngine instance.
        engine_name:   Tên engine ('easyocr' | 'paddleocr').
        ground_truth:  Dict ground truth (optional).
        debug:         In log chi tiết.

    Returns:
        OcrResult hoặc None nếu ảnh không đọc được.
    """
    image_name = os.path.basename(image_path)
    image = __import__('cv2').imread(image_path)

    if image is None:
        print(f"[WARN] Cannot read: {image_path}")
        return None

    # ── 1) YOLO Detection ──
    t0 = time.perf_counter()
    detections = yolo_detector.detect(image)
    yolo_time = time.perf_counter() - t0

    if not detections:
        result = OcrResult(
            image_name=image_name,
            predicted_plate='NO_PLATE',
            confidence=0.0,
            timing=TimingResult(image_name=image_name, yolo_time=yolo_time, total_time=time.perf_counter() - t0),
            engine=engine_name,
        )
        return result

    # Lấy detection đầu tiên (thường 1 plate / ảnh)
    det  = detections[0]
    crop = det['crop']

    # Enhance
    enhanced = enhance_plate_image(crop)

    raw_data = ocr_engine.readtext(enhanced)

    items = raw_data.get('items', [])
    ocr_time = raw_data.get('elapsed', 0.0)

    lines = []
    scores = []

    # OCR
    if engine_name == 'easyocr':
        # Logic EasyOCR giữ nguyên như cũ của bạn
        if items:
            items_sorted = sorted(items, key=lambda x: sum(pt[1] for pt in x['bbox'])/len(x['bbox']))
            lines = [it['text'] for it in items_sorted]
            scores = [it['confidence'] for it in items_sorted]

    elif engine_name == 'paddleocr':
        # KHÔNG gọi lại ocr_engine.readtext ở đây nữa
        if items:
            # Sắp xếp theo y_center đã được engine chuẩn hóa từ Polygon
            items_sorted = sorted(items, key=lambda x: x['y_center'])
            lines = [it['text'] for it in items_sorted]
            scores = [it['confidence'] for it in items_sorted]


    # ── 4) Postprocess ──
    # final_plate, was_flipped = postprocess_plate_text(lines)
    if not lines:
        final_plate = "NO_TEXT"
    else:
        # Làm sạch khoảng trắng 2 đầu của từng dòng và nối lại bằng dấu cách
        # Ví dụ: lines = ['29A1', '12345'] -> "29A1 12345"
        final_plate = " ".join([str(l).strip() for l in lines if str(l).strip()])
        
        # Đề phòng trường hợp các dòng toàn chứa khoảng trắng
        if not final_plate:
            final_plate = "NO_TEXT"

    # Vì không dùng hàm lật biển số nữa nên mặc định là False
    was_flipped = False

    # ── 5) Aggregate metrics ──
    confidence = aggregate_confidence(scores)
    total_time = yolo_time + ocr_time

    timing = TimingResult(
        image_name=image_name,
        yolo_time=yolo_time,
        ocr_time=ocr_time,
        total_time=total_time,
    )

    # ── 6) Evaluate against ground truth ──
    gt_dict = ground_truth.get(image_name, {}) if ground_truth else {}
    gt_full = gt_dict.get('full_plate', '')

    ca_province = False
    ca_series    = False
    ca_number    = False
    ca_full      = False
    ed_full      = -1
    ned_full     = -1.0
    ed_province  = -1
    ed_series    = -1
    ed_number    = -1

    if gt_full:
        ca_province = component_accuracy(final_plate, gt_full, 'province')
        ca_series   = component_accuracy(final_plate, gt_full, 'series')
        ca_number   = component_accuracy(final_plate, gt_full, 'number')
        ca_full     = component_accuracy(final_plate, gt_full, 'full')

        ed_full     = Levenshtein.distance(final_plate, gt_full)
        ned_full    = Levenshtein.ratio(final_plate, gt_full) # Dùng ratio làm normalized distance

        gt_parsed = parse_plate_components(gt_full)
        pred_comp = _pred_components(final_plate)
        ed_province = Levenshtein.distance(pred_comp['province'], gt_parsed['province']) if gt_parsed['province'] else -1
        ed_series   = Levenshtein.distance(pred_comp['series'],   gt_parsed['series'])   if gt_parsed['series']   else -1
        ed_number   = Levenshtein.distance(pred_comp['number'],   gt_parsed['number'])   if gt_parsed['number']   else -1

    if debug:
        tag = "[FLIP FIXED]" if was_flipped else "[OK]"
        print(f"  {tag} {image_name} → {final_plate}  (conf={confidence:.3f}, ed={ed_full})")

    return OcrResult(
        image_name=image_name,
        predicted_plate=final_plate,
        confidence=confidence,
        timing=timing,
        ca_province=ca_province,
        ca_series=ca_series,
        ca_number=ca_number,
        ca_full=ca_full,
        edit_dist_full=ed_full,
        norm_edit_dist=ned_full,
        edit_dist_province=ed_province,
        edit_dist_series=ed_series,
        edit_dist_number=ed_number,
        engine=engine_name,
        was_flipped=was_flipped,
    )


def _pred_components(pred_str: str) -> dict:
    """Parse predicted plate (postprocessed) thành components."""
    import re
    cleaned = re.sub(r'[\s-]', '', str(pred_str).upper())
    province, series, number = '', '', ''
    if len(cleaned) >= 2 and cleaned[:2].isdigit():
        province = cleaned[:2]
        rest = cleaned[2:]
        if len(rest) <= 2:
            series = rest
        else:
            series = rest[:2]
            number = rest[2:]
    else:
        series = cleaned
    return {'province': province, 'series': series, 'number': number}


# Batch pipeline
def run_pipeline(
    engine_name: str,
    ground_truth: dict | None = None,
    debug: bool = False,
) -> list[OcrResult]:
    """
    Chạy pipeline cho toàn bộ ảnh trong TEST_IMAGE_DIR.

    Args:
        engine_name:  'easyocr' | 'paddleocr'
        ground_truth: dict GT
        debug:        Log chi tiết từng ảnh

    Returns:
        List[OcrResult]
    """
    print(f"\n{'='*50}")
    print(f"  PIPELINE: YOLO11 + {engine_name.upper()}")
    print(f"{'='*50}")

    # Khởi tạo models
    print("[Init] Loading YOLO detector...")
    yolo = YoloDetector()

    print(f"[Init] Loading {engine_name} engine...")
    ocr  = build_ocr_engine(engine_name)

    # Danh sách ảnh
    valid_exts = ('.jpg', '.jpeg', '.png')
    images = [
        f for f in os.listdir(config.TEST_IMAGE_DIR)
        if f.lower().endswith(valid_exts)
    ]
    images.sort()
    print(f"[Data] {len(images)} images found in: {config.TEST_IMAGE_DIR}\n")

    results: list[OcrResult] = []
    flipped_images: list[dict] = []

    for fname in images:
        img_path = os.path.join(config.TEST_IMAGE_DIR, fname)
        res = process_image(
            image_path=img_path,
            yolo_detector=yolo,
            ocr_engine=ocr,
            engine_name=engine_name,
            ground_truth=ground_truth,
            debug=debug,
        )
        if res:
            results.append(res)
            if res.was_flipped:
                flipped_images.append({'file': fname, 'plate': res.predicted_plate})

    print()
    agg = compute_aggregate_metrics(results)
    print_metrics_report(agg)

    if flipped_images:
        print(f"\n[Flip] {len(flipped_images)} images corrected:")
        for item in flipped_images:
            print(f"  {item['file']} → {item['plate']}")

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
            'engine':            r.engine,
            'was_flipped':       r.was_flipped,
            # CA
            'ca_province':       r.ca_province,
            'ca_series':         r.ca_series,
            'ca_number':         r.ca_number,
            'ca_full':           r.ca_full,
            # Edit distance
            'edit_dist_full':    r.edit_dist_full,
            'norm_edit_dist':    round(r.norm_edit_dist, 4),
            'edit_dist_province': r.edit_dist_province,
            'edit_dist_series':   r.edit_dist_series,
            'edit_dist_number':   r.edit_dist_number,
            # Timing
            'yolo_ms':           round(r.timing.yolo_ms,  2),
            'ocr_ms':            round(r.timing.ocr_ms,   2),
            'total_ms':          round(r.timing.total_ms, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n[Output] Results saved → {output_csv}")


# CLI entrypoint
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

    # Load ground truth
    gt = load_ground_truth(config.GROUND_TRUTH_CSV)

    all_results = {}

    if args.engine in ('easyocr', 'both'):
        results_easy = run_pipeline('easyocr',   gt, debug=args.debug)
        save_results(results_easy, config.OUTPUT_CSV_EASY)
        all_results['easyocr'] = results_easy

    if args.engine in ('paddleocr', 'both'):
        results_paddle = run_pipeline('paddleocr', gt, debug=args.debug)
        save_results(results_paddle, config.OUTPUT_CSV_PADDLE)
        all_results['paddleocr'] = results_paddle

    # ── So sánh 2 engine ──
    if args.engine == 'both' and 'easyocr' in all_results and 'paddleocr' in all_results:
        _print_comparison(all_results['easyocr'], all_results['paddleocr'])


def _print_comparison(easy_results: list, paddle_results: list):
    """So sánh nhanh 2 engine."""
    print("\n" + "=" * 60)
    print("       COMPARISON: EasyOCR vs PaddleOCR")
    print("=" * 60)

    agg_easy   = compute_aggregate_metrics(easy_results)
    agg_paddle = compute_aggregate_metrics(paddle_results)

    def row(label, easy_v, paddle_v):
        winner = "✓ EasyOCR" if easy_v > paddle_v else ("✓ PaddleOCR" if paddle_v > easy_v else "≈ tie")
        return f"  {label:<30} Easy={easy_v:<10} Paddle={paddle_v:<10} {winner}"

    print(row("CA Province (%)", agg_easy.get('ca_province_%','N/A'), agg_paddle.get('ca_province_%','N/A')))
    print(row("CA Series (%)",   agg_easy.get('ca_series_%',  'N/A'), agg_paddle.get('ca_series_%',  'N/A')))
    print(row("CA Number (%)",   agg_easy.get('ca_number_%',  'N/A'), agg_paddle.get('ca_number_%',  'N/A')))
    print(row("CA Full (%)",     agg_easy.get('ca_full_%',    'N/A'), agg_paddle.get('ca_full_%',    'N/A')))
    print(row("Mean Edit Dist",  agg_easy.get('mean_edit_dist','N/A'), agg_paddle.get('mean_edit_dist','N/A')))
    print(row("Mean Confidence", agg_easy.get('mean_confidence','N/A'), agg_paddle.get('mean_confidence','N/A')))
    print(row("Mean Total ms",   agg_easy.get('mean_total_ms','N/A'), agg_paddle.get('mean_total_ms','N/A')))
    print(row("Mean OCR ms",     agg_easy.get('mean_ocr_ms',  'N/A'), agg_paddle.get('mean_ocr_ms',  'N/A')))
    print("=" * 60)


if __name__ == '__main__':
    main()
