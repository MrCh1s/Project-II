import os
# --- PADDLE WINDOWS FIXES ---
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

import time
import numpy as np
from paddleocr import PaddleOCR
from typing import Optional
from models.ocr import config
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

class PaddleOCREngine:
    def __init__(
        self,
        language: str | None = None,
        use_angle_cls: bool | None = None,
    ):
        self.lang = language or config.PADDLE_LANG
        self.use_angle_cls = use_angle_cls if use_angle_cls is not None else config.PADDLE_USE_ANGLE
        self._engine: Optional[PaddleOCR] = None

    @property
    def engine(self) -> PaddleOCR:
        """Lazy-load PaddleOCR engine."""
        if self._engine is None:
            print(f"[PaddleOCR] Initializing lang={self.lang}")
            self._engine = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
            )
        return self._engine

    def readtext(
        self, 
        image: np.ndarray
    ) -> dict:
        
        start = time.perf_counter()
        results = self.engine.ocr(image)
        elapsed = time.perf_counter() - start

        parsed = []
        # print("Nội dung của mỗi item:", results[0] if len(results) > 0 else "N/A")

        if isinstance(results, list) and len(results) > 0:
            res = results[0]
            if isinstance(res, dict) and 'rec_texts' in res:
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                polys = res.get('dt_polys', []) 
                
                for i in range(len(texts)):
                    # print(f"polys[{i}] type: {type(polys[i])}, value: {polys[i]}")
                    poly = polys[i].tolist() if hasattr(polys[i], 'tolist') else polys[i]
                    
                    # Tính y_center từ poly (tọa độ 4 điểm)
                    all_y = [pt[1] for pt in poly]
                    y_center = sum(all_y) / len(all_y)

                    parsed.append({
                        'text': str(texts[i]).strip(),
                        'confidence': float(scores[i]),
                        'bbox': poly,
                        'y_center': y_center
                    })

        return {
            'items': parsed,
            'elapsed': elapsed,
        }