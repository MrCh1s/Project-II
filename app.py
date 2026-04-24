import os
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
YOLO_WEIGHTS = r"C:\Users\DELL\Project-II\runs\detect\runs_yolo11\plate_detection11\weights\best.pt"  #trọng số của mô hình yolo
yolo_model = YOLO(YOLO_WEIGHTS)                                             #tải mô hình yolo
ocr_model = PaddleOCR(use_angle_cls=False, lang='en', enable_mkldnn=False)  #tải mô hình ocr
print("Models loaded successfully!") 


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

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))  #tính chiều cao của biển số
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))  #tính chiều cao của biển số
    maxHeight = max(int(heightA), int(heightB))                       #lấy chiều cao lớn nhất

    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")  #tạo ma trận 4x2 để lưu tọa độ điểm
    M = cv2.getPerspectiveTransform(rect, dst)                        #tạo ma trận chuyển đổi
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))         #chuyển đổi ảnh

def smart_sort_plate_lines(texts):                                    #sắp xếp biển số
    line1, line2 = "", ""                                           #khởi tạo biến line1 và line2
    for t in texts:
        t_clean = t.replace(" ", "").replace("-", "").replace(".", "")  #xóa khoảng trắng, dấu gạch ngang, dấu chấm
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
    res_img = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)        #tăng kích thước ảnh lên 2 lần
    gray_res = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)                #chuyển ảnh sang grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))          #tạo bộ lọc CLAHE
    contrasted = clahe.apply(gray_res)                                  #áp dụng bộ lọc CLAHE
    return cv2.cvtColor(contrasted, cv2.COLOR_GRAY2BGR)                 #chuyển ảnh sang BGR

def postprocess_plate_text(lines):   #xóa ký tự sai, sửa lỗi ocr cho đúng format, o-0, z-2, s-5
    if not lines: return "NO_TEXT", False   #trả về rỗng nếu không có biển số
    cleaned = [re.sub(r'[^A-Z0-9]', '', str(l)).upper() for l in lines if l]  #xóa ký tự không phải chữ hoặc số
    if not cleaned: return "NO_TEXT", False     

    flip_map = {'9':'6', '6':'9', 'L':'1', 'I':'1', 'A':'V', 'V':'A', 'E':'3', '3':'E', 'S':'5', '5':'S', '7':'2', '2':'7', 'J':'C', 'C':'J', 'Z':'2'}  #đảo ngược ký tự
    char_to_num = {'E':'6', 'G':'6', 'S':'5', 'B':'8', 'O':'0', 'D':'0', 'Z':'2', 'L':'1', 'I':'1', 'T':'7', 'W':'7', 'A':'4'}  #chuyển ký tự sang số
    num_to_char = {'6':'G', '5':'S', '8':'B', '0':'D', '2':'Z', '1':'L', '7':'T', '4':'A', '3':'E'}     #chuyển số sang ký tự

    is_flipped = False      #kiểm tra biển số bị lật hay không
    l1_raw = cleaned[0]     #lấy dòng đầu tiên
    l2_raw = cleaned[1] if len(cleaned) >= 2 else ""  #lấy dòng thứ hai nếu có

    # FLIP LOGIC RESTORED
    if (len(l1_raw) >= 1 and l1_raw[0].isalpha()) or (len(l2_raw) >= 2 and l2_raw[:2].isdigit() and not l1_raw[:2].isdigit()):  
        is_flipped = True   #đảo ngược biển số
        line1 = "".join([flip_map.get(c, c) for c in l2_raw[::-1]]) if l2_raw else l1_raw  #đảo ngược dòng đầu tiên
        line2 = "".join([flip_map.get(c, c) for c in l1_raw[::-1]])  #đảo ngược dòng thứ hai
    else:                                                                               #không đảo ngược biển số
        line1 = l1_raw                                                                  #gán dòng đầu tiên
        line2 = l2_raw                                                                  #gán dòng thứ hai

    l1 = list(line1)                                                                    #chuyển dòng đầu tiên sang list
    if len(l1) >= 1:                                                                    #kiểm tra dòng đầu tiên có ít nhất 1 ký tự
        if not l1[0].isdigit() or l1[0] == '0':                                         #kiểm tra ký tự đầu tiên có phải số hoặc bằng 0
            l1[0] = char_to_num.get(l1[0], l1[0])                                       #chuyển ký tự đầu tiên sang số
            if l1[0] == '0': l1[0] = '8'                                                #nếu ký tự đầu tiên bằng 0 thì chuyển sang 8
    if len(l1) >= 2:                                                                    #kiểm tra dòng đầu tiên có ít nhất 2 ký tự
        if not l1[1].isdigit():                                                         #kiểm tra ký tự thứ hai có phải số
            l1[1] = char_to_num.get(l1[1], l1[1])                                       #chuyển ký tự thứ hai sang số
    if len(l1) >= 3:                                                                    #kiểm tra dòng đầu tiên có ít nhất 3 ký tự
        if l1[2].isdigit():                                                             #kiểm tra ký tự thứ ba có phải số
            l1[2] = num_to_char.get(l1[2], l1[2])                                       #chuyển ký tự thứ ba sang ký tự
        elif l1[2] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":                               #kiểm tra ký tự thứ ba có phải chữ cái
            l1[2] = 'S'                                                                 #nếu ký tự thứ ba không phải chữ cái thì chuyển sang S
    if len(l1) >= 4:                                                                    #kiểm tra dòng đầu tiên có ít nhất 4 ký tự
        if l1[3] in ['S', 'E', 'G', 'B']:                                               #kiểm tra ký tự thứ tư có phải S, E, G, B
            l1[3] = char_to_num.get(l1[3], l1[3])                                       #chuyển ký tự thứ tư sang số

    l1_res = "".join(l1)                                                               #chuyển dòng đầu tiên sang chuỗi
    if len(l1_res) > 2: l1_res = f"{l1_res[:2]}-{l1_res[2:]}"                            #nếu dòng đầu tiên có nhiều hơn 2 ký tự thì thêm dấu gạch ngang

    l2_res = "".join([char_to_num.get(c, c) if not c.isdigit() else c for c in line2]) #chuyển dòng thứ hai sang chuỗi
    if len(l2_res) == 5:                                                                #nếu dòng thứ hai có 5 ký tự
        l2_res = f"{l2_res[:3]}.{l2_res[3:]}"                                           #thêm dấu chấm vào dòng thứ hai

    return f"{l1_res} {l2_res}".strip(), is_flipped                                   #trả về dòng đầu tiên và dòng thứ hai đã được định dạng

def check_line1_format(plate_str):                                                      #kiểm tra định dạng dòng đầu tiên
    if not plate_str or plate_str == "NO_PLATE" or " " not in plate_str:                #kiểm tra dòng đầu tiên có rỗng hoặc bằng NO_PLATE hoặc không có dấu cách
        return False
    line1 = plate_str.split(' ')[0]                                                     #tách dòng đầu tiên
    clean_l1 = line1.replace('-', '')                                                   #xóa dấu gạch ngang
    pattern = r'^[1-9][0-9][A-Z][A-Z0-9]$'                                               #kiểm tra định dạng dòng đầu tiên
    return bool(re.match(pattern, clean_l1))


# --- 3. THE CORE PIPELINE FUNCTION (Pass 1 & Pass 2 Restored) ---
def process_license_plate(image):                                                           #hàm xử lý biển số
    try:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)                                    #chuyển ảnh sang BGR
        img_draw = img_bgr.copy()                                                           #tạo bản sao của ảnh

        yolo_results = yolo_model(img_bgr, verbose=False)                                   #yolo phát hiện biển số
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()     #ouput=tọa dộ bounding box: (x1, y1, x2, y2)

        if len(boxes) == 0:                                                                 #nếu yolo không phát hiện biển số
            return image, "Không phát hiện biển số", 0.0

        best_plate = "OCR_FAILED"                                                         #gán giá trị mặc định cho biển số
        best_conf = 0.0                                                                     #gán giá trị mặc định cho điểm số
        
        for i, box in enumerate(boxes):                                                     #lặp qua các bounding box
            x1, y1, x2, y2 = map(int, box)                                                  #tọa dộ bounding box
            padding = 5                                                                     #padding
            plate_crop = img_bgr[max(0, y1-padding):min(img_bgr.shape[0], y2+padding),    #cắt ảnh biển số để lấy riêng phần biển số để ocr dễ đọc hơn
                                 max(0, x1-padding):min(img_bgr.shape[1], x2+padding)]
            
            # PASS 1
            processed_crop = enhance_plate_image(plate_crop)  #resize, chuyển xám, tảng tương phản để giúp ocr đọc chinh xac hơn
            ocr_res = ocr_model.predict(processed_crop)       #ocr đọc biển số, output: rec_texts(text đọc đc), rec_scores(điểm số), dt_polys(vị trí chữ)
            
            full_text = "TEXT_NOT_FOUND"                                                    #gán giá trị mặc định cho biển số
            avg_conf = 0.0                                                                      #gán giá trị mặc định cho điểm số
            
            if ocr_res and len(ocr_res) > 0 and ocr_res[0] is not None:                         #kiểm tra ocr có trả về kết quả không
                data = ocr_res[0]                                                               #lấy kết quả ocr
                if 'rec_texts' in data and len(data['rec_texts']) > 0:                          #kiểm tra ocr có trả về text không
                    raw_texts = data['rec_texts']                                               #lấy text
                    raw_polys = data.get('dt_polys', [])                                        #lấy vị trí chữ
                    
                    sorted_lines = sort_multiline_text(raw_texts, raw_polys)   #sắp xếp các dòng text theo thứ tự từ trên xuống dưới
                    full_text, was_flipped = postprocess_plate_text(sorted_lines)             #xử lý text
                    
                    scores = data.get('rec_scores', [0.0] * len(raw_texts))                     #lấy điểm số
                    avg_conf = sum(scores) / len(scores)                                        #tính điểm số trung bình

            # PASS 2: RESCUE MISSION RESTORED
            try:                                                                            #thử kiểm tra định dạng dòng đầu tiên
                is_valid = check_line1_format(full_text)                                    #kiểm tra định dạng dòng đầu tiên
            except:                                                                         #nếu có lỗi
                is_valid = True                                                             #gán giá trị mặc định cho dòng đầu tiên
                
            if not is_valid or avg_conf < 0.90:                                             #nếu dòng đầu tiên không hợp lệ hoặc điểm số trung bình nhỏ hơn 0.90
                res_pass2 = ocr_model.predict(img_bgr)                                      #ocr đọc biển số
                if res_pass2 and len(res_pass2[0]['dt_polys']) > 0:                           #kiểm tra ocr có trả về kết quả không
                    pts = np.array(res_pass2[0]['dt_polys']).reshape(-1, 2).astype(np.float32) #tọa dộ bounding box
                    rect = cv2.minAreaRect(pts)                                               #tạo bounding box
                    deskew_box = cv2.boxPoints(rect).astype(np.float32)                       #tạo bounding box
                    img_warped = deskew_plate(img_bgr, deskew_box)                            #căn chỉnh ảnh biển số
                    img_warped_flip = cv2.rotate(img_warped, cv2.ROTATE_180)                  #xoay ảnh biển số 180 độ
                    
                    for img_version in [img_warped, img_warped_flip]:                         #lặp qua ảnh biển số đã căn chỉnh và xoay
                        processed = enhance_plate_image(img_version)                          #tăng cường ảnh biển số
                        res_version = ocr_model.predict(processed)                            #ocr đọc biển số
                        
                        if res_version and len(res_version[0]['rec_texts']) >= 2:               #kiểm tra ocr có trả về kết quả không
                            sorted_lines2 = smart_sort_plate_lines(res_version[0]['rec_texts']) #sắp xếp các dòng text theo thứ tự từ trên xuống dưới
                            text2, _ = postprocess_plate_text(sorted_lines2)                    #xử lý text
                            
                            try:                                                                #thử kiểm tra định dạng dòng đầu tiên
                                valid2 = check_line1_format(text2)                              #kiểm tra định dạng dòng đầu tiên
                            except:                                                             #nếu có lỗi
                                valid2 = True                                                             #gán giá trị mặc định cho dòng đầu tiên
                                
                            if valid2:                                                          #nếu dòng đầu tiên hợp lệ
                                conf2 = np.mean(res_version[0]['rec_scores'])                   #tính điểm số trung bình
                                if conf2 > avg_conf:                                              #nếu điểm số trung bình lớn hơn
                                    avg_conf = conf2                                              #gán giá trị mặc định cho điểm số trung bình
                                    full_text = text2                                             #gán giá trị mặc định cho text

            best_plate = full_text                                                            #gán giá trị mặc định cho biển số
            best_conf = round(float(avg_conf), 4)                                             #làm tròn điểm số trung bình
            
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)   #vẽ hình chữ nhật quanh biển số
            img_draw = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)                              #chuyển đổi ảnh sang định dạng RGB
            break 

        return img_draw, best_plate, best_conf                                             #trả về ảnh đã vẽ, biển số và điểm số trung bình

    except Exception as e:                                                                #nếu có lỗi
        print("ERROR:", e)                                                                 #in ra lỗi
        return image, "ERROR", 0.0                                                        #trả về ảnh gốc, "ERROR" và 0.0

# --- 4. GIAO DIỆN WEB ---
demo = gr.Interface(              #tạo giao diện web
    fn=process_license_plate,     #hàm xử lý
    inputs=gr.Image(label="Tải ảnh xe lên"),   #input là ảnh
    outputs=[
        gr.Image(label="Kết quả phát hiện"),   #output là ảnh
        gr.Textbox(label="Biển số dự đoán"),   #output là text
        gr.Number(label="Độ tin cậy")        #output là số
    ],
    title="Nhận Diện Biển Số Xe",
    description="Upload ảnh xe",
    theme=gr.themes.Soft(),  
  
)

if __name__ == "__main__":
    demo.launch(share=False)   #chạy giao diện web