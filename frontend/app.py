# import cv2
# import gradio as gr
# import numpy as np
# from models.ocr.yolo_detector import YoloDetector  
# from models.ocr.paddleocr_engine import PaddleOCREngine 
# from models.ocr.preprocessing import enhance_plate_image 

# detector = YoloDetector() 
# ocr_engine = PaddleOCREngine()

# # def process_license_plate(image):
# #     try:
# #         img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# #         img_draw = img_bgr.copy()

# #         # Phát hiện biển số[cite: 5]
# #         detections = detector.detect(img_bgr)

# #         if not detections:
# #             return image, "Không phát hiện biển số", 0.0

# #         best_plate = "OCR_FAILED"
# #         best_conf = 0.0
        
# #         for det in detections:
# #             box = det['box']
# #             plate_crop = det['crop'] 
            
# #             processed_crop = enhance_plate_image(plate_crop)
            
# #             ocr_result = ocr_engine.readtext(processed_crop)
# #             items = ocr_result['items']
            
# #             if items:
# #                 texts = [it['text'] for it in items]
# #                 confidences = [it['confidence'] for it in items]
                
# #                 best_plate = " ".join(texts)
# #                 avg_conf = sum(confidences) / len(confidences)
# #                 best_conf = round(float(avg_conf), 4)

# #             x1, y1, x2, y2 = box
# #             cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
# #             break 

# #         return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), best_plate, best_conf

# #     except Exception as e:
# #         print(f"Lỗi hệ thống: {e}")
# #         return image, "ERROR", 0.0

# def process_license_plate(image):
#     # Khởi tạo mặc định
#     best_plate = "Không xác định"
#     best_conf = 0.0
    
#     try:
#         img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#         img_draw = img_bgr.copy()

#         # 1. Thử dùng YOLO trước
#         detections = detector.detect(img_bgr)

#         if detections:
#             for det in detections:
#                 box = det['box']
#                 plate_crop = det['crop']
#                 processed_crop = enhance_plate_image(plate_crop)
                
#                 ocr_result = ocr_engine.readtext(processed_crop)
#                 items = ocr_result.get('items', [])
                
#                 if items:
#                     texts = [it['text'] for it in items]
#                     confidences = [it['confidence'] for it in items]
#                     best_plate = " ".join(texts)
#                     best_conf = round(float(sum(confidences) / len(confidences)), 4)

#                 x1, y1, x2, y2 = box
#                 cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                 break 
#         else:
#             # 2. Fallback: YOLO không thấy -> Dùng PaddleOCR trên ảnh gốc
#             ocr_result = ocr_engine.readtext(img_bgr) 
#             items = ocr_result.get('items', [])
            
#             if items:
#                 all_texts = []
#                 all_confs = []
                
#                 for it in items:
#                     # FIX SHAPE Ở ĐÂY: Chuyển từ [[x,y],...] sang (n, 1, 2)
#                     pts = np.array(it['points'], dtype=np.int32)
#                     pts = pts.reshape((-1, 1, 2)) 
                    
#                     # Vẽ đa giác màu xanh dương (để phân biệt với YOLO màu xanh lá)
#                     cv2.polylines(img_draw, [pts], isClosed=True, color=(255, 0, 0), thickness=2)
                    
#                     all_texts.append(it['text'])
#                     all_confs.append(it['confidence'])
                    
#                     # Lấy tọa độ điểm đầu tiên để ghi text
#                     text_pos = tuple(pts[0][0])
#                     cv2.putText(img_draw, it['text'], (text_pos[0], text_pos[1] - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
#                 best_plate = " ".join(all_texts)
#                 best_conf = round(float(sum(all_confs) / len(all_confs)), 4)
#             else:
#                 return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), "Không phát hiện biển số", 0.0

#         return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), best_plate, best_conf

#     except Exception as e:
#         print(f"Lỗi hệ thống: {e}")
#         return image, f"Lỗi: {str(e)}", 0.0

# demo = gr.Interface(
#     fn=process_license_plate,
#     inputs=gr.Image(label="Tải ảnh xe lên"),
#     outputs=[
#         gr.Image(label="Kết quả phát hiện"),
#         gr.Textbox(label="Biển số dự đoán (Raw Text)"),
#         gr.Number(label="Độ tin cậy")
#     ],
#     title="Nhận Diện Biển Số - OCR Baseline",
#     theme=gr.themes.Soft()
# )

# if __name__ == "__main__":
#     demo.launch()


import cv2
import gradio as gr
import numpy as np
import re

from models.ocr.yolo_detector import YoloDetector  
from models.ocr.paddleocr_engine import PaddleOCREngine 
from models.ocr.preprocessing import enhance_plate_image 

detector = YoloDetector() 
ocr_engine = PaddleOCREngine()


# ==============================
# Regex lọc biển số (Việt Nam cơ bản)
# ==============================
def is_valid_plate(text):
    text = text.replace(" ", "")
    pattern = r"[0-9]{2}[A-Z0-9]{1,2}-?[0-9]{4,5}"
    return re.match(pattern, text)


# ==============================
# MAIN FUNCTION
# ==============================
def process_license_plate(image):
    best_plate = "Không xác định"
    best_conf = 0.0
    
    try:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_draw = img_bgr.copy()

        # =========================
        # 1. YOLO DETECTION
        # =========================
        detections = detector.detect(img_bgr)

        if detections:
            for det in detections:
                box = det['box']
                plate_crop = det['crop']

                processed_crop = enhance_plate_image(plate_crop)
                ocr_result = ocr_engine.readtext(processed_crop)

                items = ocr_result.get('items', [])

                if items:
                    texts = [it['text'] for it in items]
                    confidences = [it['confidence'] for it in items]

                    best_plate = " ".join(texts)
                    best_conf = round(float(sum(confidences) / len(confidences)), 4)

                # vẽ bbox YOLO
                x1, y1, x2, y2 = box
                cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)

                break  # chỉ lấy 1 biển

        else:
            # =========================
            # 2. FALLBACK OCR
            # =========================
            ocr_result = ocr_engine.readtext(img_bgr)
            items = ocr_result.get('items', [])

            print("Fallback OCR items:", items)  # debug

            if not items:
                return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), "Không phát hiện biển số", 0.0

            all_texts = []
            all_confs = []

            for it in items:
                # ===== FIX QUAN TRỌNG =====
                pts = np.array(it['bbox']).astype(np.int32).reshape((-1, 1, 2))

                # vẽ bbox polygon
                cv2.polylines(img_draw, [pts], True, (255, 0, 0), 2)

                # lấy tọa độ text
                x, y = pts[0][0]
                y = max(y - 10, 0)

                text = it['text']
                conf = it['confidence']

                # vẽ text
                cv2.putText(img_draw, text, (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # lưu kết quả
                all_texts.append(text)
                all_confs.append(conf)

            # =========================
            # 3. POST-PROCESS
            # =========================
            # ưu tiên text giống biển số
            valid_texts = [t for t in all_texts if is_valid_plate(t)]

            if valid_texts:
                best_plate = " ".join(valid_texts)
            else:
                best_plate = " ".join(all_texts)

            best_conf = round(float(sum(all_confs) / len(all_confs)), 4)

        return cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB), best_plate, best_conf

    except Exception as e:
        print(f"Lỗi hệ thống: {e}")
        return image, f"Lỗi: {str(e)}", 0.0


# ==============================
# GRADIO UI
# ==============================
demo = gr.Interface(
    fn=process_license_plate,
    inputs=gr.Image(label="Tải ảnh xe lên"),
    outputs=[
        gr.Image(label="Kết quả phát hiện"),
        gr.Textbox(label="Biển số dự đoán (Raw Text)"),
        gr.Number(label="Độ tin cậy")
    ],
    title="Nhận Diện Biển Số - OCR Baseline (YOLO + PaddleOCR Fallback)",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch()