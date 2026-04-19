from email.mime import image
import time
import numpy as np
import easyocr
from typing import Optional
import config

class EasyOCREngine:
    def __init__(
        self,
        languages: list[str] | None = None,
        gpu: bool | None = None,
        batch_size: int | None = None,
    ):
        self.languages  = languages  or config.EASYOCR_LANGUAGES
        self.gpu       = gpu       if gpu       is not None else config.EASYOCR_GPU
        self._reader: Optional[easyocr.Reader] = None

    @property
    def reader(self) -> easyocr.Reader:
        """Lazy-load reader."""
        if self._reader is None:
            print(f"EasyOCR Initializing languages={self.languages}, gpu={self.gpu}")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
            )
        return self._reader

    def readtext(
        self,
        image: np.ndarray,
    ) -> list[dict]:
        """
        Nhận diện toàn bộ text trong ảnh đã crop (biển số).

        Args:
            image: Ảnh BGR numpy array (đã crop biển số).

        Returns:
            List of dict, mỗi dict chứa:
              - text:      str – ký tự nhận diện được
              - confidence: float – confidence score (0–1)
              - bbox:      list – bounding box [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        start = time.perf_counter()
        # Thêm dấu gạch ngang vào cuối chuỗi
        allowed_chars = '0123456789ABCDEFGHKLMNPSTUVXYZ-'
        results = self.reader.readtext(image, allowlist=allowed_chars)
        elapsed = time.perf_counter() - start

        parsed = []
        for item in results:
            # EasyOCR format: (bbox, text, confidence)
            bbox = item[0]
            text = str(item[1]).strip()
            conf = float(item[2]) if len(item) > 2 else 0.0
            parsed.append({
                'text':       text,
                'confidence': conf,
                'bbox':       bbox,
            })

        return {
            'items':   parsed,
            'elapsed': elapsed,
        }

    def extract_text_lines(
        self,
        image: np.ndarray,
    ) -> tuple[list[str], float]:
        """
        Trả về list các dòng text đã sắp xếp từ trên xuống dưới.

        Returns:
            (sorted_lines, elapsed_seconds)
        """
        data = self.readtext(image)
        items = data['items']

        if not items:
            return [], data['elapsed']

        # Sắp xếp theo tọa độ Y trung bình của bbox (từ trên xuống)
        def avg_y(item):
            bbox = item['bbox']
            ys = [pt[1] for pt in bbox]
            return sum(ys) / len(ys)

        sorted_items = sorted(items, key=avg_y)
        texts = [it['text'] for it in sorted_items]
        return texts, data['elapsed']
