import os
import time
import cv2
import numpy as np
import gradio as gr
import re
from ultralytics import YOLO

# --- PADDLE WINDOWS FIXES ---
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"
from paddleocr import PaddleOCR

# --- 1. LOAD MODELS ---
print("Loading Models...") 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_WEIGHTS = os.path.join(BASE_DIR, "runs", "detect", "runs_yolo11", "plate_detection", "weights", "best.pt")

yolo_model = YOLO(YOLO_WEIGHTS)                                             #tải mô hình yolo
print(f"YOLO Model Classes: {yolo_model.names}")

ocr_model = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)  #tải mô hình ocr

print("Warming up OCR model (this may take a minute)...")
try:
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr_model.predict(dummy_img)
    print("Warm-up complete! Models loaded successfully!") 
except Exception as e:
    print(f"Warm-up notice: {e}")

# --- 2.CUSTOM FUNCTIONS ---
def deskew_plate(img, points):   #sửa lại góc của biển số để ocr đọc dễ hơn
    pts = np.array(points, dtype="float32")       #chuyển tọa độ điểm sang dạng float32
    rect = np.zeros((4, 2), dtype="float32")      #tạo ma trận 4x2 để lưu tọa độ điểm
    s = pts.sum(axis=1)                             #tính tổng tọa độ điểm
    rect[0] = pts[np.argmin(s)]                     #tìm điểm có tổng tọa độ nhỏ nhất
    rect[2] = pts[np.argmax(s)]                     #tìm điểm có tổng tọa độ lớn nhất
    diff = np.diff(pts, axis=1)                     #tính hiệu tọa độ điểm
    rect[1] = pts[np.argmin(diff)]                  #tìm điểm có hiệu tọa độ nhỏ nhất
    rect[3] = pts[np.argmax(diff)]                  #tìm điểm có hiệu tọa độ lớn nhất

    (tl, tr, br, bl) = rect                         #gán tọa độ điểm
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))  #tính chiều rộng của biển số
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))  #tính chiều rộng của biển số
    maxWidth = max(int(widthA), int(widthB))                          #lấy chiều rộng lớn nhất

    heightA = np.sqrt(((tr[1] - br[1]) ** 2) + ((tr[0] - br[0]) ** 2))  #tính chiều cao của biển số
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  #tính chiều cao của biển số
    maxHeight = max(int(heightA), int(heightB))                       #lấy chiều cao lớn nhất

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")  #tạo ma trận 4x2 để lưu tọa độ điểm
    M = cv2.getPerspectiveTransform(rect, dst)                        #tạo ma trận chuyển đổi
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))         #chuyển đổi ảnh

def smart_sort_plate_lines(texts):                                    #sắp xếp biển số
    line1, line2 = "", ""                                           #khởi tạo biến line1 và line2
    for t in texts:
        t_clean = str(t).replace(" ", "").replace("-", "").replace(".", "")  #xóa khoảng trắng, dấu gạch ngang, dấu chấm
        if any(c.isalpha() for c in t_clean): line1 = t                   #nếu có ký tự chữ thì gán vào line1
        elif any(c.isdigit() for c in t_clean): line2 = t                   #nếu có ký tự số thì gán vào line2
    return [line1, line2] if line1 and line2 else texts                   #trả về line1 và line2 nếu có cả hai, ngược lại trả về texts

def sort_multiline_text(texts, dt_polys):                               #sắp xếp biển số nhiều dòng
    if not texts or not dt_polys or len(texts) == 0: return []             #trả về rỗng nếu không có biển số
    combined = []                                                       #khởi tạo biến combined
    for i in range(len(texts)):                                         #lặp qua từng dòng biển số
        try:
            y_min = np.min(dt_polys[i][:, 1])                           #tìm tọa độ y nhỏ nhất của biển số
        except Exception:                                               #nếu có lỗi thì gán y_min = 0
            y_min = 0
        combined.append({"text": str(texts[i]).upper(), "y": y_min})    #gán tọa độ y vào biển số
    sorted_data = sorted(combined, key=lambda x: x["y"])                #sắp xếp biển số theo tọa độ y
    return [item["text"] for item in sorted_data]                       #trả về biển số đã sắp xếp

def enhance_plate_image(plate_crop):                                    #tăng cường ảnh biển số
    # KIỂM TRA AN TOÀN
    if plate_crop is None or plate_crop.size == 0 or plate_crop.shape[0] < 2 or plate_crop.shape[1] < 2:
        return plate_crop
    res_img = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)        #tăng kích thước ảnh lên 2 lần
    gray_res = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)                #chuyển ảnh sang grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))          #tạo bộ lọc CLAHE
    contrasted = clahe.apply(gray_res)                                  #áp dụng bộ lọc CLAHE
    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)                 #chuyển ảnh sang BGR

def postprocess_plate_text(lines):                                      #xử lý text sau ocr
    if not lines: return "NO_TEXT", False   #trả về rỗng nếu không có biển số
    cleaned = [re.sub(r'[^A-Z0-9]', '', str(l)).upper() for l in lines if l]  #xóa ký tự không phải chữ hoặc số
    if not cleaned: return "NO_TEXT", False
    
    l1_raw = cleaned[0]
    l2_raw = cleaned[1] if len(cleaned) >= 2 else ""
    
    # Logic kiểm tra xem có cần đảo ngược biển số không (Rescue Mission)
    # Nếu dòng 1 chỉ toàn số và dòng 2 có chữ -> Có thể ảnh bị ngược
    has_alpha_l1 = any(c.isalpha() for c in l1_raw)
    has_alpha_l2 = any(c.isalpha() for c in l2_raw)
    
    was_flipped = False
    if not has_alpha_l1 and has_alpha_l2:
        was_flipped = True
        cleaned = cleaned[::-1] # Đảo ngược danh sách dòng
        
    full_plate = "-".join(cleaned)
    return full_plate, was_flipped

def check_line1_format(plate_str):                                                      #kiểm tra định dạng dòng đầu tiên
    if not plate_str or plate_str == "NO_PLATE" or "-" not in plate_str:                #kiểm tra dòng đầu tiên có rỗng hoặc bằng NO_PLATE hoặc không có dấu cách
        return False
    line1 = plate_str.split('-')[0]                                                     #tách dòng đầu tiên
    clean_l1 = line1.replace('-', '')                                                   #xóa dấu gạch ngang
    pattern = r'^[1-9][0-9][A-Z][A-Z0-9]$'                                               #kiểm tra định dạng dòng đầu tiên
    return bool(re.match(pattern, clean_l1))


# --- 3. THE CORE PIPELINE FUNCTION ---
def process_license_plate(image):                                                           #hàm xử lý biển số
    start_total = time.time()
    try:
        # Ép kiểu ảnh về chuẩn uint8 và 3 kênh màu (RGB)
        image = np.array(image).astype(np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                                    #chuyển ảnh sang BGR
        img_draw = img_bgr.copy()                                                           #tạo bản sao của ảnh

        # Dùng imgsz=1280 và conf=0.2 để nhận diện nhạy hơn và chính xác hơn
        yolo_results = yolo_model(image, conf=0.2, imgsz=1280, verbose=False)                      #yolo phát hiện biển số
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()     #ouput=tọa dộ bounding box: (x1, y1, x2, y2)
        scores = yolo_results[0].boxes.conf.cpu().numpy()
        
        yolo_time = time.time() - start_total
        print(f"DEBUG: YOLO tìm thấy {len(boxes)} box. Scores: {scores}. Time: {yolo_time:.3f}s")

        if len(boxes) == 0:                                                                 #nếu yolo không phát hiện biển số
            # RESCUE MISSION (Pass 3): Thử chạy OCR trên toàn bộ ảnh đề phòng biển quá to (chụp sát)
            print("DEBUG: YOLO 0 box, thử chạy OCR toàn bộ ảnh...")
            processed_full = enhance_plate_image(img_bgr)
            ocr_res_full = ocr_model.predict(processed_full)
            
            if ocr_res_full and ocr_res_full[0] is not None:
                data_f = ocr_res_full[0]
                if isinstance(data_f, dict) and 'rec_texts' in data_f and len(data_f['rec_texts']) > 0:
                    sorted_f = sort_multiline_text(data_f['rec_texts'], data_f.get('dt_polys', []))
                    text_f, _ = postprocess_plate_text(sorted_f)
                    conf_f = float(np.mean(data_f['rec_scores']))
                    
                    if text_f != "NO_TEXT" and conf_f > 0.4:
                        print(f"DEBUG: Rescue thành công: {text_f}")
                        total_time = time.time() - start_total
                        return img_draw, text_f, round(conf_f, 4), round(total_time, 2)

            return image, "Không phát hiện biển số", 0.0, 0.0

        best_plate = "OCR_FAILED"                                                         #gán giá trị mặc định cho biển số
        best_conf = 0.0                                                                     #gán giá trị mặc định cho điểm số
        
        for i, box in enumerate(boxes):                                                     #lặp qua các bounding box
            x1, y1, x2, y2 = map(int, box)                                                  #tọa dộ bounding box
            padding = 5                                                                     #padding
            plate_crop = img_bgr[max(0, y1-padding):min(img_bgr.shape[0], y2+padding),    #cắt ảnh biển số để lấy riêng phần biển số để ocr dễ đọc hơn
                                 max(0, x1-padding):min(img_bgr.shape[1], x2+padding)]
            
            if plate_crop.size == 0 or plate_crop.shape[0] < 2 or plate_crop.shape[1] < 2:
                continue

            # PASS 1: Xử lý bình thường
            processed_crop = enhance_plate_image(plate_crop)
            ocr_res = ocr_model.predict(processed_crop)
            
            full_text = "TEXT_NOT_FOUND"
            avg_conf = 0.0
            
            if ocr_res and len(ocr_res) > 0 and ocr_res[0] is not None:
                data = ocr_res[0]
                if isinstance(data, dict) and 'rec_texts' in data and len(data['rec_texts']) > 0:
                    raw_texts = data['rec_texts']
                    raw_polys = data.get('dt_polys', [])
                    
                    sorted_lines = sort_multiline_text(raw_texts, raw_polys)
                    full_text, was_flipped = postprocess_plate_text(sorted_lines)
                    
                    scores_ocr = data.get('rec_scores', [0.0] * len(raw_texts))
                    avg_conf = float(np.mean(scores_ocr))

            # PASS 2: Nếu Pass 1 nghi ngờ bị ngược hoặc kết quả kém
            if full_text == "TEXT_NOT_FOUND" or avg_conf < 0.7:
                # Căn chỉnh lại góc biển số (Deskew)
                results_yolo = yolo_model(image, conf=0.1, imgsz=1280, verbose=False)
                masks = getattr(results_yolo[0], 'masks', None)
                if masks is not None and len(masks) > 0:
                    pass 
                
                # Cố gắng xoay 180 độ và đọc lại
                plate_crop_flip = cv2.rotate(plate_crop, cv2.ROTATE_180)
                processed_flip = enhance_plate_image(plate_crop_flip)
                res_flip = ocr_model.predict(processed_flip)
                
                if res_flip and res_flip[0] is not None and 'rec_texts' in res_flip[0] and len(res_flip[0]['rec_texts']) > 0:
                    text_f, _ = postprocess_plate_text(res_flip[0]['rec_texts'])
                    conf_f = float(np.mean(res_flip[0]['rec_scores']))
                    if conf_f > avg_conf:
                        avg_conf = conf_f
                        full_text = text_f

            best_plate = full_text
            best_conf = round(float(avg_conf), 4)
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
            break 
        
        total_time = time.time() - start_total
        return img_draw, best_plate, best_conf, round(total_time, 2)

    except Exception as e:
        print("ERROR:", e)
        return image, f"LỖI: {str(e)}", 0.0, 0.0

# --- 4. GIAO DIỆN WEB ---
demo = gr.Interface(
    fn=process_license_plate,
    inputs=gr.Image(label="Tải ảnh xe lên"),
    outputs=[
        gr.Image(label="Kết quả phát hiện"),
        gr.Textbox(label="Biển số dự đoán"),
        gr.Number(label="Độ tin cậy"),
        gr.Number(label="Thời gian xử lý (giây)")
    ],
    title="Nhận Diện Biển Số Xe",
    description="Hệ thống sử dụng YOLOv11 và PaddleOCR",
    theme=gr.themes.Soft(),  
)

if __name__ == "__main__":
    demo.launch(share=False)