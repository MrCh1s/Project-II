# preprocessing.py – Tiền xử lý ảnh biển số & postprocess kết quả OCR
import re
import cv2
import numpy as np
from typing import Optional
import config

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

# Post-processing helpers
def clean_text(text: str) -> str:
    """Loại bỏ ký tự không phải alphanumeric, viết hoa."""
    return re.sub(r'[^A-Za-z0-9]', '', text).upper()


def apply_flip_correction(line1_raw: str, line2_raw: str) -> tuple[str, str, bool]:
    """
    Phát hiện và sửa biển số bị chụp ngược (lật 180°).

    Điều kiện FLIP phải thỏa ĐỒNG THỜI cả 2:
      1. line1 bắt đầu bằng CHỮ IN HOA (E, S, G, ...)
         mà không phải số → bất thường (dòng 1 phải bắt đầu số)
      2. line2 bắt đầu bằng ĐÚNG 2 CHỮ SỐ (province code)
         trong khi line1 KHÔNG bắt đầu bằng 2 số

    Trường hợp KHÔNG flip dù line2 bắt đầu bằng số:
      - Province ở dòng 2  (ví dụ: "48FA 1856" → dòng 2 = "48FA")
      - Một số biển đặc biệt

    Khi lật:
      - Dòng trên <-> Dòng dưới
      - Thứ tự ký tự bị đảo ngược
      - Mỗi ký tự bị flip qua FLIP_MAP

    Returns:
        (corrected_line1, corrected_line2, was_flipped)
    """
    flip_map = config.FLIP_MAP

    def _flip(s: str) -> str:
        return ''.join(flip_map.get(c, c) for c in s[::-1])

    l1 = line1_raw.strip()
    l2 = line2_raw.strip()

    # ── Bug fix #1: Flip chỉ trigger khi thỏa ĐỒNG THỜI cả 2 điều kiện ──
    # Điều kiện A: line1 bắt đầu bằng chữ IN HOA (không phải số)
    cond_A = bool(l1) and l1[0].isalpha()

    # Điều kiện B: line2 bắt đầu bằng ĐÚNG 2 chữ số (province code)
    #              VÀ line1 KHÔNG bắt đầu bằng 2 số
    l1_starts_2digits = len(l1) >= 2 and l1[:2].isdigit()
    l2_starts_2digits = len(l2) >= 2 and l2[:2].isdigit()
    cond_B = l2_starts_2digits and not l1_starts_2digits

    is_flipped = cond_A and cond_B

    if is_flipped:
        c1 = _flip(l2)
        c2 = _flip(l1)
    else:
        c1, c2 = l1, l2

    return c1, c2, is_flipped


def _enforce_line1_format(raw: str) -> str:
    """
    Ép dòng 1 về định dạng XY-ZT:
      X  (pos 0): số 1-9
      Y  (pos 1): số 0-9
      Z  (pos 2): CHỮ
      T  (pos 3): chữ HOẶC số

    Dùng CHAR_TO_NUM / NUM_TO_CHAR để fix ký tự nhầm.
    """
    c2n = config.CHAR_TO_NUM
    n2c = config.NUM_TO_CHAR

    chars = list(raw)

    # pos 0: phải là số 1-9
    if len(chars) >= 1:
        if not chars[0].isdigit() or chars[0] == '0':
            mapped = c2n.get(chars[0], chars[0])
            if mapped == '0':
                mapped = '8'
            chars[0] = mapped

    # pos 1: phải là số
    if len(chars) >= 2:
        if not chars[1].isdigit():
            chars[1] = c2n.get(chars[1], chars[1])

    # pos 2: phải là CHỮ
    if len(chars) >= 3:
        if chars[2].isdigit():
            chars[2] = n2c.get(chars[2], chars[2])
        elif chars[2] not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            chars[2] = 'S'

    # pos 3: ưu tiên số nếu là ký tự dễ nhầm
    if len(chars) >= 4:
        if chars[3] in ('S', 'E', 'G', 'B'):
            chars[3] = c2n.get(chars[3], chars[3])

    result = ''.join(chars)

    # Format XY-ZT
    if len(result) >= 2:
        result = f"{result[:2]}-{result[2:]}"

    return result


def _enforce_line2_format(raw: str) -> str:
    """Ép dòng 2 về định dạng số seri (ví dụ: 66480 → 66.480)."""
    c2n = config.CHAR_TO_NUM

    chars = [c2n.get(c, c) if not c.isdigit() else c for c in raw]
    result = ''.join(chars)

    # Format NNN.NN (3 chữ số + dấu . + 2 chữ số)
    if len(result) == 5:
        result = f"{result[:3]}.{result[3:]}"

    return result


# ──────────────────────────────────────────────
# 3. Main postprocess
# ──────────────────────────────────────────────

def postprocess_plate_text(lines: list[str]) -> tuple[str, bool]:
    """
    Hàm chính postprocess kết quả OCR → biển số xe máy VN hoàn chỉnh.

    Args:
        lines: List các dòng text thô từ OCR engine
               (đã sort theo tọa độ Y, từ trên xuống dưới).

    Returns:
        (final_plate_str, was_flipped)
        Ví dụ: ("59P1 66480", False)
               ("51-V2 39.74", True)
    """
    if not lines:
        return "NO_TEXT", False

    # Clean tất cả lines
    cleaned = [clean_text(l) for l in lines if clean_text(l)]
    if not cleaned:
        return "NO_TEXT", False

    l1_raw = cleaned[0]
    l2_raw = cleaned[1] if len(cleaned) >= 2 else ""

    # ── Bước 1: Flip correction ──
    l1_corr, l2_corr, was_flipped = apply_flip_correction(l1_raw, l2_raw)

    # ── Bước 2: Ép định dạng dòng 1 (XY-ZT) ──
    l1_fmt = _enforce_line1_format(l1_corr)

    # ── Bước 3: Ép định dạng dòng 2 (số seri) ──
    l2_fmt = _enforce_line2_format(l2_corr)

    final = f"{l1_fmt} {l2_fmt}".strip()
    return final, was_flipped


# ──────────────────────────────────────────────
# 4. Plate component parsing (cho ground truth)
# ──────────────────────────────────────────────

def parse_plate_components(plate_str: str) -> dict[str, str]:
    """
    Tách biển số thành 3 phần: province, series, number.

    Args:
        plate_str: Chuỗi plate đã được format
                  (ví dụ: "59P1 66480" hoặc "59 P1 66480")

    Returns:
        dict với keys: province, series, number, full_plate
    """
    # Normalize space
    raw = re.sub(r'\s+', ' ', plate_str.strip())
    tokens = raw.split(' ')

    province = ''
    series   = ''
    number   = ''

    first = tokens[0] if tokens else ''

    # Province: 2 chữ số đầu
    if len(first) >= 2 and first[:2].isdigit() and first[:2] in config.PROVINCE_CODES:
        province = first[:2]
        rest = first[2:]
        if rest:
            tokens = [rest] + tokens[1:]
        else:
            tokens = tokens[1:]

    # Series + number
    if tokens:
        series = tokens[0].upper()
    if len(tokens) > 1:
        number = ''.join(tokens[1:])

    return {
        'province': province,
        'series':   series,
        'number':   number,
        'full_plate': ' '.join(p for p in [province, series, number] if p),
    }
