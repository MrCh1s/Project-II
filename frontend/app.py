import cv2
import gradio as gr
import numpy as np
from models.ocr.yolo_detector import YoloDetector  
from models.ocr.paddleocr_engine import PaddleOCREngine 
from models.ocr.preprocessing import enhance_plate_image 

detector = YoloDetector() 
ocr_engine = PaddleOCREngine()

def process_license_plate(image):
    try:
        # Safety check
        if image is None:
            return None, "Vui lòng tải ảnh lên", 0.0

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_draw = img_bgr.copy()

        # 1. Try to detect the plate with YOLO first
        detections = detector.detect(img_bgr)

        best_plate = "OCR_FAILED"
        best_conf = 0.0
        
        # FALLBACK STRATEGY: YOLO failed, use OCR on the whole image
        if not detections:
            print("YOLO failed to find plate. Triggering OCR Fallback...")
            # Pass the entire original image to PaddleOCR
            ocr_result = ocr_engine.readtext(img_bgr)
            items = ocr_result['items']
            
            # If even OCR fails to find text, then give up
            if not items:
                return image, "Không phát hiện biển số", 0.0
                
            texts = [it['text'] for it in items]
            confidences = [it['confidence'] for it in items]
            
            best_plate = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences)
            best_conf = round(float(avg_conf), 4)

            # Draw ORANGE boxes so you know the Fallback worked
            for it in items:
                poly = np.array(it['bbox'], np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_draw, [poly], isClosed=True, color=(0, 165, 255), thickness=3)

            return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), best_plate, best_conf

        # NORMAL STRATEGY: YOLO successfully found the plate
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = [int(v) for v in box]
            
            padding = 10
            h, w = img_bgr.shape[:2]
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w, x2 + padding)
            py2 = min(h, y2 + padding)
            
            plate_crop_padded = img_bgr[py1:py2, px1:px2]
            processed_crop = enhance_plate_image(plate_crop_padded)
            ocr_result = ocr_engine.readtext(processed_crop)
            items = ocr_result['items']
            
            if items:
                texts = [it['text'] for it in items]
                confidences = [it['confidence'] for it in items]
                
                best_plate = " ".join(texts)
                avg_conf = sum(confidences) / len(confidences)
                best_conf = round(float(avg_conf), 4)

            # Draw GREEN box so you know YOLO worked
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
            break 

        return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), best_plate, best_conf

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return image, "ERROR", 0.0

demo = gr.Interface(
    fn=process_license_plate,
    inputs=gr.Image(label="Tải ảnh xe lên"),
    outputs=[
        gr.Image(label="Kết quả phát hiện"),
        gr.Textbox(label="Biển số dự đoán (Raw Text)"),
        gr.Number(label="Độ tin cậy")
    ],
    title="Nhận Diện Biển Số - OCR Baseline",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()