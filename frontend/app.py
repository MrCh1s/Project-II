import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        
        # FALLBACK STRATEGY
        if not detections:
            print("YOLO failed to find plate. Triggering OCR Fallback...")
            ocr_result = ocr_engine.readtext(img_bgr)
            items = ocr_result['items']
    
            if not items:
                return image, "Không phát hiện biển số", 0.0
                
            texts = [it['text'] for it in items]
            confidences = [it['confidence'] for it in items]
            
            best_plate = " ".join(texts)
            avg_conf = sum(confidences) / len(confidences) 
            best_conf = round(float(avg_conf), 4)

            for it in items:
                poly = np.array(it['bbox'], np.int32).reshape((-1, 1, 2))
                cv2.polylines(img_draw, [poly], isClosed=True, color=(0, 165, 255), thickness=3)

            return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), best_plate, best_conf

        # NORMAL STRATEGY
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
            processed_crop = enhance_plate_image(plate_crop_padded) #Tiền xử lý ảnh biển số
            ocr_result = ocr_engine.readtext(processed_crop) # Nhận diện biển số
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
    results = [] # Khởi tạo danh sách chứa kết quả

    if not files: # Nếu không có ảnh nào được tải lên
        return pd.DataFrame() # Trả về DataFrame rỗng

    filenames = [] # Tạo danh sách chứa tên ảnh
    images_bgr = [] # Tạo danh sách chứa ảnh
    
    # 1. Đọc và gom tất cả ảnh vào một list
    for file in files: 
        filename = file.name.replace("\\", "/").split("/")[-1]  # Tách tên ảnh từ đường dẫn
        image = cv2.imread(file.name) # Đọc ảnh
        if image is not None: # Kiểm tra xem ảnh có đọc được không
            filenames.append(filename) 
            images_bgr.append(image)

    if not images_bgr:
        return pd.DataFrame()

    # 2. YOLO nhận diện theo lô 
    # Tối ưu tốc độ bằng cách xử lý nhiều ảnh cùng lúc  
    batch_size = 8
    batch_detections = detector.detect_batch(images_bgr, batch_size=batch_size) 

    # 3. Duyệt qua kết quả của từng ảnh để chạy OCR
    for i, detections in enumerate(batch_detections): 
        filename = filenames[i]
        image = images_bgr[i]
        
        best_plate = "OCR_FAILED"
        best_conf = 0.0

        if not detections: 
            print(f"YOLO failed to find plate for {filename}. Triggering OCR Fallback...")
            # FALLBACK STRATEGY
            ocr_result = ocr_engine.readtext(image)
            items = ocr_result.get('items', [])
            
            if items:
                texts = [it['text'] for it in items]
                confidences = [it['confidence'] for it in items]
                best_plate = " ".join(texts)
                avg_conf = sum(confidences) / len(confidences)
                best_conf = round(float(avg_conf), 4)
            else:
                best_plate = "Không phát hiện biển số"
        else:
            # NORMAL STRATEGY: Lấy bounding box đầu tiên
            det = detections[0]
            box = det['box']
            x1, y1, x2, y2 = [int(v) for v in box]
            
            padding = 10
            h, w = image.shape[:2]
            px1 = max(0, x1 - padding)
            py1 = max(0, y1 - padding)
            px2 = min(w, x2 + padding + 10)
            py2 = min(h, y2 + padding + 10)
            
            plate_crop_padded = image[py1:py2, px1:px2]
            processed_crop = enhance_plate_image(plate_crop_padded) # Tiền xử lý
            
            ocr_result = ocr_engine.readtext(processed_crop) # OCR
            items = ocr_result.get('items', [])
            
            if items:
                texts = [it['text'] for it in items]
                confidences = [it['confidence'] for it in items]
                best_plate = " ".join(texts)
                avg_conf = sum(confidences) / len(confidences)
                best_conf = round(float(avg_conf), 4)
            else:
                best_plate = "Không đọc được chữ"

        results.append({
            "Tên ảnh (Filename)": filename,
            "Biển số dự đoán (Predicted Plate)": best_plate,
            "Độ tin cậy (Confidence)": best_conf
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
