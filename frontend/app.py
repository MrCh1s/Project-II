import cv2
import gradio as gr
import numpy as np
import pandas as pd
from models.ocr.yolo_detector import YoloDetector  
from models.ocr.paddleocr_engine import PaddleOCREngine 
from models.ocr.preprocessing import enhance_plate_image 

detector = YoloDetector() 
ocr_engine = PaddleOCREngine()

# ================= SINGLE IMAGE =================
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
            px2 = min(w, x2 + padding + 10)
            py2 = min(h, y2 + padding + 10)
            
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


# ================= BATCH MODE =================
def process_batch(files):
    results = []

    if not files:
        return pd.DataFrame()

    for file in files:
        # Lấy tên file để hiển thị trên bảng
        filename = file.name.replace("\\", "/").split("/")[-1] 
        
        # Đọc ảnh từ đường dẫn file
        image = cv2.imread(file.name)
        if image is None:
            continue
            
        # Do cv2.imread trả về BGR, nhưng hàm process_license_plate cần RGB (giống gradio image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Gọi hàm xử lý 1 ảnh
        _, plate, conf = process_license_plate(image_rgb)

        results.append({
            "Tên ảnh (Filename)": filename,
            "Biển số dự đoán (Predicted Plate)": plate,
            "Độ tin cậy (Confidence)": conf
        })

    df = pd.DataFrame(results)
    return df


# ================= UI =================
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Nhận Diện Biển Số Xe")

    with gr.Tab("Chế độ 1 ảnh"):
        image_input = gr.Image(label="Tải ảnh xe lên")
        image_output = gr.Image(label="Kết quả phát hiện")
        plate_output = gr.Textbox(label="Biển số dự đoán")
        conf_output = gr.Number(label="Độ tin cậy")

        btn_single = gr.Button("Chạy")

        btn_single.click(
            fn=process_license_plate,
            inputs=image_input,
            outputs=[image_output, plate_output, conf_output]
        )

    with gr.Tab("Chế độ Xử lý theo lô"):
        # Dùng gr.File để tải lên nhiều ảnh cùng lúc
        batch_input = gr.File(
            file_count="multiple",
            file_types=["image"],
            label="Kéo thả / Chọn nhiều ảnh cùng lúc"
        )

        batch_output = gr.Dataframe(
            label="Kết quả xử lý hàng loạt",
            headers=["Tên ảnh", "Biển số dự đoán", "Độ tin cậy"]
        )

        btn_batch = gr.Button("Chạy")

        btn_batch.click(
            fn=process_batch,
            inputs=batch_input,
            outputs=batch_output
        )

if __name__ == "__main__":
    demo.launch()
