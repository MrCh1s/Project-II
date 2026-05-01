import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
# import models.ocr.config as config
from models.ocr import config
import re

# Chia biển số thành các component để tính CA riêng cho từng phần
def component_accuracy(
    pred_str: str,
    gt_str:   str,
    component: str,   
) -> bool:
    
    gt_parsed = _parse_gt_components(gt_str)
    pred_parsed = _parse_pred_components(pred_str)

    gt_val   = gt_parsed.get(component, '')
    pred_val = pred_parsed.get(component, '')

    gt_norm   = gt_val.upper().replace(' ', '')
    pred_norm = pred_val.upper().replace(' ', '')

    return gt_norm == pred_norm

# Tách ground truth thành 4 phần: province, series, number, full
# Ví dụ: "29A1 12345" -> province="29", series="A1", number="12345", full="29A12345"
def _parse_gt_components(plate_str: str) -> dict[str, str]:
    raw = re.sub(r'[\s]', '', plate_str.strip().upper())
    if not raw:
        return {'province': '', 'series': '', 'number': '', 'full': ''}

    province = ''
    series   = ''
    number   = ''

    if len(raw) >= 2 and raw[:2] in config.PROVINCE_CODES:
        province = raw[:2]
        rest = raw[2:]
    else:
        rest = raw

    if len(rest) <= 2:
        series = rest
    else:
        series = rest[:2]
        number = rest[2:]

    full = province + series + number
    return {'province': province, 'series': series, 'number': number, 'full': full}

def _parse_pred_components(pred_str: str) -> dict[str, str]:
    cleaned = re.sub(r'[^A-Z0-9]', '', pred_str.upper())
    
    province = ''
    series   = ''
    number   = ''

    if len(cleaned) >= 2 and cleaned[:2].isdigit():
        province = cleaned[:2]
        s_char1 = cleaned[2]
        s_char2 = cleaned[3]
        num_to_char_map = {'8': 'B', '5': 'S', '0': 'D', '2': 'Z', '6': 'G', '7' : 'T', '1': 'I', '3': 'E', '4': 'A', '9': 'P'}
        
        if s_char1.isdigit() and s_char1 in num_to_char_map:
            s_char1 = num_to_char_map[s_char1] 
            
        series = s_char1 + s_char2
        number = cleaned[4:]
    else:
        series = cleaned

    full = province + series + number
    return {'province': province, 'series': series, 'number': number, 'full': full}

# Inference Time
@dataclass
class TimingResult:
    image_name: str
    yolo_time:   float = 0.0   # seconds – YOLO detection
    ocr_time:    float = 0.0   # seconds – OCR inference
    total_time:  float = 0.0   # seconds – tổng cộng

    @property
    def yolo_ms(self)  -> float: return self.yolo_time  * 1000
    @property
    def ocr_ms(self)   -> float: return self.ocr_time   * 1000
    @property
    def total_ms(self) -> float: return self.total_time * 1000

# Confidence Score cho từng ảnh
def aggregate_confidence(ocr_scores: list[float], aggregate: str = 'mean') -> float:
    return float(np.mean(ocr_scores))

# Per-image Result Record
@dataclass
class OcrResult:
    """Kết quả OCR cho một ảnh."""
    image_name:     str
    predicted_plate: str
    confidence:      float
    timing:          TimingResult
    ca_province: bool = False
    ca_series:   bool = False
    ca_number:   bool = False
    ca_full:     bool = False
    engine:       str = 'unknown'

def compute_aggregate_metrics(results: list[OcrResult]) -> dict:
    if not results:
        return {}

    n = len(results)
    ca_province_cnt = sum(1 for r in results if r.ca_province)
    ca_series_cnt   = sum(1 for r in results if r.ca_series)
    ca_number_cnt   = sum(1 for r in results if r.ca_number)
    ca_full_cnt     = sum(1 for r in results if r.ca_full)
    mean_conf  = sum(r.confidence for r in results) / n
    mean_total = sum(r.timing.total_ms for r in results) / n
    mean_ocr   = sum(r.timing.ocr_ms  for r in results) / n

    return {
        'n_images':         n,
        'ca_province_%':    round(ca_province_cnt / n * 100, 2),
        'ca_series_%':      round(ca_series_cnt   / n * 100, 2),
        'ca_number_%':      round(ca_number_cnt   / n * 100, 2),
        'ca_full_%':        round(ca_full_cnt     / n * 100, 2),
        'mean_confidence':  round(mean_conf, 4),
        'mean_total_ms':    round(mean_total, 2),
        'mean_ocr_ms':      round(mean_ocr,  2),
    }

def print_metrics_report(metrics: dict):
    print("=" * 50)
    print("\t\tOCR METRICS REPORT")
    print("=" * 50)
    print(f"  Tổng số ảnh:            {metrics.get('n_images', 'N/A')}")
    print("-" * 50)
    print(f"    Đúng Tỉnh:            {metrics.get('ca_province_%', 'N/A')}%")
    print(f"    Đúng Series:          {metrics.get('ca_series_%', 'N/A')}%")
    print(f"    Đúng Number:          {metrics.get('ca_number_%', 'N/A')}%")
    print(f"    Đúng Toàn Biển:       {metrics.get('ca_full_%', 'N/A')}%")
    print("-" * 50)
    print(f"    Trung bình total / ảnh:    {metrics.get('mean_total_ms', 'N/A')} ms")
    print(f"    Trung bình OCR / ảnh:      {metrics.get('mean_ocr_ms', 'N/A')} ms")
    print("-" * 50)
    print(f"    Mean confidence:       {metrics.get('mean_confidence', 'N/A')}")
    print("=" * 50)