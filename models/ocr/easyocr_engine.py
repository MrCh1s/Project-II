from email.mime import image
import time
import numpy as np
import easyocr
from typing import Optional
from models.ocr import config

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
            print(f"EasyOCR Initializing language={self.languages}, gpu={self.gpu}")
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
            )
        return self._reader

    def readtext(
        self,
        image: np.ndarray,
    ) -> list[dict]:
        
        allowed_chars = '0123456789ABCDEFGHKLMNPSTUVXYZ-'
        start = time.perf_counter()
        results = self.reader.readtext(image, allowlist=allowed_chars)
        elapsed = time.perf_counter() - start

        parsed = []
        # EasyOCR trả theo khối văn bản
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
