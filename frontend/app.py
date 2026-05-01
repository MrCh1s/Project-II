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
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_draw = img_bgr.copy()

        # Phát hiện biển số[cite: 5]
        detections = detector.detect(img_bgr)

        if not detections:
            return image, "Không phát hiện biển số", 0.0

        best_plate = "OCR_FAILED"
        best_conf = 0.0
        
        for det in detections:
            box = det['box']
            plate_crop = det['crop'] 
            
            processed_crop = enhance_plate_image(plate_crop)
            
            ocr_result = ocr_engine.readtext(processed_crop)
            items = ocr_result['items']
            
            if items:
                texts = [it['text'] for it in items]
                confidences = [it['confidence'] for it in items]
                
                best_plate = " ".join(texts)
                avg_conf = sum(confidences) / len(confidences)
                best_conf = round(float(avg_conf), 4)

            x1, y1, x2, y2 = box
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