# Hướng Dẫn Huấn Luyện Dò Tìm Biển Số (Detection) Từ A-Z với YOLO11

Mục tiêu của phần này là lấy file `location.txt` và thư mục ảnh `Bike`, xử lý chúng để huấn luyện ra một mô hình AI cho phép nhận diện và vẽ khung (Bounding Box) bao quanh biển số xe xuất hiện trong bất cứ ảnh/video nào bạn cung cấp.

Tất cả các đoạn code bên dưới có thể chạy theo từng Cell trong file `notebooks/data_preprocessing.ipynb`.

## BƯỚC 1: Cài đặt Môi Trường (Environment Setup)
Khởi động Terminal/Command Prompt trong thư mục Project (hoặc chạy trực tiếp trong Jupyter Notebook bằng cách thêm dấu `!` đầu tiên).

```bash
# Cài đặt YOLO11 (thông qua gói ultralytics mới nhất) và các thư viện cần thiết
pip install ultralytics opencv-python scikit-learn matplotlib
```

## BƯỚC 2: Tiền xử lý dữ liệu - Đọc File & Chuyển sang chuẩn YOLO
(Gộp bước Tạo Label `.txt` và Chia thư mục Train/Val/Test vào chung một quy trình).

Đảm bảo sau bước này bạn sẽ có cấu trúc thư mục sau:
```text
yolo_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

**Mã Python xử lý (Chạy trong Jupyter):**
```python
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split

# --- THIẾT LẬP ĐƯỜNG DẪN GỐC ---
bike_dir = r"e:\OneDrive\Desktop\Project II\data\vietnam-car-license-plate\Bike\GreenParking"
bike_annotation = r"e:\OneDrive\Desktop\Project II\data\vietnam-car-license-plate\Bike\GreenParking\location.txt"
output_labels_dir = r"e:\OneDrive\Desktop\Project II\data\temp_labels" # Thư mục nháp
final_dataset = r"e:\OneDrive\Desktop\Project II\yolo_dataset"       # Thư mục đích chuẩn YOLO

# 1. Chuyển đổi định dạng Toạ Độ (To YOLO txt format)
os.makedirs(output_labels_dir, exist_ok=True)
with open(bike_annotation, 'r') as f:
    lines = f.readlines()
    
valid_images = [] # Chỉ giữ lại các file ảnh Tồn Tại và Hợp lệ
for line in lines:
    line = line.strip()
    if not line: continue
    
    parts = line.split(' ')
    image_name = parts[0]
    class_id = 0 # 0 là class Mặc định (Biển số / License Plate)
    x_min, y_min, w_box, h_box = map(float, parts[2:6])
    
    img_path = os.path.join(bike_dir, image_name)
    if not os.path.exists(img_path): continue
        
    img = cv2.imread(img_path)
    if img is None: continue
    img_h, img_w, _ = img.shape
    
    # Tính tâm và chuẩn hoá (Normalize) về [0, 1]
    # Dataset location.txt đã CÓ SẴN toạ độ là (x_center, y_center), nên KHÔNG CỘNG THÊM w_box/2.0
    x_center = (x_min + w_box / 2.0) / img_w
    y_center = (y_min + h_box / 2.0) / img_h
    w_norm = w_box / img_w
    h_norm = h_box / img_h
    
    if w_norm > 1.0 or h_norm > 1.0: continue # Lọc dữ liệu lỗi (box to hơn ảnh)
    
    txt_name = image_name.replace('.jpg', '.txt').replace('.png', '.txt')
    txt_path = os.path.join(output_labels_dir, txt_name)
    
    with open(txt_path, 'w') as out_f:
        out_f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    valid_images.append(image_name)

print(f"Đã tạo {len(valid_images)} file labels YOLO.")

# 2. Xóa Dataset cũ nếu có để tránh ghi đè rác
if os.path.exists(final_dataset):
    shutil.rmtree(final_dataset)

# 3. Chia tập dữ liệu (80-10-10)
train_imgs, temp_imgs = train_test_split(valid_images, test_size=0.2, random_state=42)
val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

splits = {'train': train_imgs, 'val': val_imgs, 'test': test_imgs}

for split_name, imgs in splits.items():
    img_dest = os.path.join(final_dataset, 'images', split_name)
    lbl_dest = os.path.join(final_dataset, 'labels', split_name)
    os.makedirs(img_dest, exist_ok=True)
    os.makedirs(lbl_dest, exist_ok=True)
    
    for img_name in imgs:
        lbl_name = img_name.replace('.jpg', '.txt').replace('.png', '.txt')
        # Copy file
        shutil.copy(os.path.join(bike_dir, img_name), os.path.join(img_dest, img_name))
        shutil.copy(os.path.join(output_labels_dir, lbl_name), os.path.join(lbl_dest, lbl_name))

# Xoá file nháp
shutil.rmtree(output_labels_dir)
print(f"XUYÊN SUỐT: Train({len(train_imgs)}) - Val({len(val_imgs)}) - Test({len(test_imgs)})")
```

## BƯỚC 3: Tạo tệp Cấu hình (data.yaml)
YOLO11 cần một tệp config khai báo cho nó biết ảnh nằm ở đâu và bạn có bao nhiêu labels.

**Bạn tạo một file mới tên là `data.yaml` ở thư mục `Project II` với nội dung:**
*(Mở text editor hoặc VS Code để tạo)*
```yaml
path: e:/OneDrive/Desktop/Project II/yolo_dataset
train: images/train
val: images/val
test: images/test

# Số lượng class (chúng ta chỉ dò 1 class là cái biển số)
nc: 1

# Tên các class
names: ['license_plate']
```

## BƯỚC 4: Tiến hành Huấn Luyện (Training) với YOLO11
Bạn tạo một Cell mới trong Jupyter Notebook để chạy Training. Ở bước này Model sẽ chạy tính toán nhiều (có thể tốn thời gian tùy chất lượng CPU/GPU của bạn).

```python
from ultralytics import YOLO

# 1. Khởi tạo Mô hình YOLO11 Bản Nhỏ (n)
# yolo11n.pt là pre-trained weight mới nhất của Hãng, sẽ tự động được Download nếu chưa có
model = YOLO('yolo11n.pt')

# 2. Chạy hàm Train
results = model.train(
    data=r'e:\OneDrive\Desktop\Project II\data.yaml',   # Trỏ tới file Yaml vừa cấu hình 
    epochs=60,                  # Huấn luyện lặp qua lặp lại 60 vòng
    imgsz=640,                  # Resize ảnh về chuẩn 640x640 của YOLO11
    batch=32,                   # Tối ưu cho Card RTX 4060 8GB
    device='0',                 # Sử dụng Card rời NVIDIA (GPU)
    project='runs_yolo11',      # Tên thư mục lưu kết quả 
    name='plate_detection',     # Tên phiên làm việc
    workers=0                   # RẤT QUAN TRỌNG TRÊN WINDOWS: Khắc phục lỗi sập CUDA
)

print("Đã Training xong YOLO11!")
```

## BƯỚC 5: Chạy Dự Đoán Lên Hình Ảnh Mới (Inference)
Xong bước 4, mô hình tốt nhất của bạn sẽ được tự động lưu vào thư mục: `runs_yolo11/plate_detection/weights/best.pt`.
Bạn có thể mang đoạn code sau đi bất cứ đâu để test lên 1 hình ảnh hoặc video tuỳ ý.

```python
from ultralytics import YOLO
import cv2

# Load mô hình "Ngôi sao" mà bạn vừa Train ra
best_model = YOLO(r'runs_yolo11\plate_detection\weights\best.pt')

# Chạy test lên một cái ảnh bất kỳ để phát hiện biển số
test_image_url = r"e:\OneDrive\Desktop\Project II\yolo_dataset\images\test\0021_06361_b.jpg"

results = best_model.predict(source=test_image_url, save=True, show=True)

# Tới thư mục runs_yolo11\plate_detection\predict để xem kết quả!
```
