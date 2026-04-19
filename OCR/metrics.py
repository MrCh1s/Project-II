import time
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import config
import Levenshtein
import re
# Character Accuracy
def character_accuracy(pred_text: str, gt_text: str) -> float:
    """
    Tính tỉ lệ ký tự nhận diện đúng trên tổng số ký tự (ground truth).

    CA = (# chars correctly matched) / len(gt_text)

    Dùng alignment để so sánh từng ký tự,
    không chỉ đơn thuần so sánh chuỗi bằng ==.
    """
    if not gt_text:
        return 1.0 if not pred_text else 0.0
    if not pred_text:
        return 0.0

    # Dùng Levenshtein alignment để đếm matched chars
    # Cách đơn giản: so sánh từng vị trí tương ứng (cắt ngắn nhất)
    # Hoặc dùng ratio: Levenshtein ratio ≈ 1 - (distance / max_len)
    ratio = Levenshtein.ratio(gt_text, pred_text)
    return ratio


def component_accuracy(
    pred_str: str,
    gt_str:   str,
    component: str,   
) -> bool:
    """
    Kiểm tra xem một thành phần của biển số có đúng hoàn toàn không.

    Args:
        pred_str:   Chuỗi predicted (đã postprocess, đã format)
        gt_str:     Ground truth chuỗi
        component:  Loại component cần so sánh
                    'province' | 'series' | 'number' | 'full'

    Returns:
        True nếu component đó khớp chính xác (sau khi normalize).
    """
    gt_parsed = _parse_gt_components(gt_str)
    pred_parsed = _parse_pred_components(pred_str)

    gt_val   = gt_parsed.get(component, '')
    pred_val = pred_parsed.get(component, '')

    # Normalize: bỏ khoảng trắng, viết hoa
    gt_norm   = gt_val.upper().replace(' ', '')
    pred_norm = pred_val.upper().replace(' ', '')

    return gt_norm == pred_norm

#  Tách ground truth thành 4 phần:
def _parse_gt_components(plate_str: str) -> dict[str, str]:
    raw = re.sub(r'[\s-]', '', plate_str.strip().upper())
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
    cleaned = re.sub(r'[\s-]', '', pred_str.upper())

    province = ''
    series   = ''
    number   = ''

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


# Confidence Score
def aggregate_confidence(
    ocr_scores: list[float],
    aggregate: str = 'mean',
) -> float:
    """
    Tổng hợp confidence scores từ OCR engine.

    Args:
        ocr_scores: List confidence scores (0–1) của từng ký tự/dòng.
        aggregate:  'mean' | 'min' | 'max' | 'product'

    Returns:
        Scalar confidence score.
    """
    if not ocr_scores:
        return 0.0

    if aggregate == 'mean':
        return float(np.mean(ocr_scores))
    elif aggregate == 'min':
        return float(np.min(ocr_scores))
    elif aggregate == 'max':
        return float(np.max(ocr_scores))
    elif aggregate == 'product':
        result = 1.0
        for s in ocr_scores:
            result *= s
        return float(result ** (1.0 / len(ocr_scores)))  # geometric mean
    else:
        return float(np.mean(ocr_scores))


# Per-image Result Record
@dataclass
class OcrResult:
    """Kết quả OCR cho một ảnh."""
    image_name:     str
    predicted_plate: str
    confidence:      float
    timing:          TimingResult
    # CA flags
    ca_province: bool = False
    ca_series:   bool = False
    ca_number:   bool = False
    ca_full:     bool = False
    # Edit distances
    edit_dist_full:    int = -1
    norm_edit_dist:    float = -1.0
    edit_dist_province: int = -1
    edit_dist_series:   int = -1
    edit_dist_number:   int = -1
    # Meta
    engine:       str = 'unknown'
    was_flipped:  bool = False


# Aggregate Metrics Report
def compute_aggregate_metrics(results: list[OcrResult]) -> dict:
    """
    Tính các chỉ số tổng hợp từ danh sách OcrResult.

    Returns:
        dict chứa:
          - n_images:        int – tổng số ảnh
          - ca_province:     float – % đúng province
          - ca_series:       float – % đúng series
          - ca_number:       float – % đúng number
          - ca_full:         float – % đúng toàn biển
          - mean_edit_dist:  float – khoảng cách Levenshtein trung bình
          - mean_confidence: float – confidence trung bình
          - mean_total_ms:   float – thời gian xử lý trung bình (ms)
          - mean_ocr_ms:     float – thời gian OCR trung bình (ms)
          - flipped_count:   int   – số ảnh bị flip đã sửa
    """
    if not results:
        return {}

    n = len(results)

    ca_province_cnt = sum(1 for r in results if r.ca_province)
    ca_series_cnt   = sum(1 for r in results if r.ca_series)
    ca_number_cnt   = sum(1 for r in results if r.ca_number)
    ca_full_cnt     = sum(1 for r in results if r.ca_full)

    valid_edit = [r.edit_dist_full for r in results if r.edit_dist_full >= 0]
    mean_edit  = sum(valid_edit) / len(valid_edit) if valid_edit else 0.0

    mean_conf  = sum(r.confidence for r in results) / n
    mean_total = sum(r.timing.total_ms for r in results) / n
    mean_ocr   = sum(r.timing.ocr_ms  for r in results) / n

    flipped_cnt = sum(1 for r in results if r.was_flipped)

    return {
        'n_images':         n,
        'ca_province_%':    round(ca_province_cnt / n * 100, 2),
        'ca_series_%':      round(ca_series_cnt   / n * 100, 2),
        'ca_number_%':      round(ca_number_cnt   / n * 100, 2),
        'ca_full_%':        round(ca_full_cnt     / n * 100, 2),
        'mean_edit_dist':   round(mean_edit, 2),
        'mean_confidence':  round(mean_conf, 4),
        'mean_total_ms':    round(mean_total, 2),
        'mean_ocr_ms':      round(mean_ocr,  2),
        'flipped_count':    flipped_cnt,
    }


def print_metrics_report(metrics: dict):
    """In báo cáo metrics ra console."""
    print("=" * 50)
    print("       OCR METRICS REPORT")
    print("=" * 50)
    print(f"  Total images:            {metrics.get('n_images', 'N/A')}")
    print("-" * 50)
    print("  Character Accuracy (CA):")
    print(f"    ✓ Đúng Tỉnh:            {metrics.get('ca_province_%', 'N/A')}%")
    print(f"    ✓ Đúng Series:          {metrics.get('ca_series_%', 'N/A')}%")
    print(f"    ✓ Đúng Number:          {metrics.get('ca_number_%', 'N/A')}%")
    print(f"    ✓ Đúng Toàn Biển:       {metrics.get('ca_full_%', 'N/A')}%")
    print("-" * 50)
    print(f"  Edit Distance (Levenshtein):")
    print(f"    Mean distance:          {metrics.get('mean_edit_dist', 'N/A')}")
    print("-" * 50)
    print(f"  Inference Time:")
    print(f"    Mean total / image:    {metrics.get('mean_total_ms', 'N/A')} ms")
    print(f"    Mean OCR / image:      {metrics.get('mean_ocr_ms', 'N/A')} ms")
    print("-" * 50)
    print(f"  Confidence Score:")
    print(f"    Mean confidence:       {metrics.get('mean_confidence', 'N/A')}")
    print(f"  Flipped images fixed:   {metrics.get('flipped_count', 'N/A')}")
    print("=" * 50)