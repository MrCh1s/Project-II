# Tiền Xử Lý Dữ Liệu (Data Preprocessing) Dự Án Đọc Biển Số Xe Tự Động (ALPR)

Trong một dự án Đọc Biển Số Xe Tự Động (ALPR - Automatic License Plate Recognition), **Tiền xử lý dữ liệu (Data Preprocessing)** là một trong những bước quan trọng nhất quyết định độ chính xác của toàn bộ hệ thống. 

Dưới đây là tất tần tật các bước tiền xử lý dữ liệu cần thiết được áp dụng trong project đọc biển số xe, chia theo luồng công việc thực tế:

## 1. Hiểu và Làm Sạch Dữ Liệu (Data Cleaning & Exploration)
Trước khi xử lý, bạn cần dọn dẹp các dữ liệu "rác" trong thư mục:
- **Kiểm tra ảnh hỏng:** Lọc ra những ảnh không thể đọc hoặc có dung lượng bằng 0 KB.
- **Kiểm tra nhãn (Annotations) lỗi:** Đọc tệp `location.txt` hoặc `_report.csv` để đối chiếu với ảnh thật. Nếu một dòng tham chiếu đến một ảnh không tồn tại, hoặc ảnh có nhưng thiếu nhãn, cần phải loại bỏ hoặc báo lỗi để xử lý.
- **Quan sát phân phối:** Tính xem bạn có bao nhiêu ảnh xe máy (Bike), bao nhiêu ảnh ô tô (Car). Điều này giúp chọn tỉ lệ chia tập dữ liệu cân bằng ở bước sau.

## 2. Phân Tích Cú Pháp Nhãn (Annotation Parsing & Format Conversion)
Mỗi thuật toán AI (như YOLO, Faster R-CNN) lại yêu cầu định dạng nhãn (label) khác nhau. File `location.txt` thường chứa định dạng đặc thù:
- **Thông tin trích xuất:** Tên file ảnh (VD: `0001.jpg`), tọa độ hộp bao quanh biển số (Bounding Box: `xmin, ymin, xmax, ymax`), và chuỗi ký tự trên biển số (VD: `29A12345`).
- **Chuyển đổi Format:** 
  - Nếu bạn dùng YOLO để dò tìm vị trí biển (Detection), bạn cần chuyển vị trí khung Bounding Box về định dạng `[class_id center_x center_y width height]` và chuẩn hóa các giá trị tọa độ về khoảng `[0, 1]` chia theo chiều rộng và chiều cao ảnh gốc.
  - Lưu mỗi file nhãn `.txt` tương ứng với một file ảnh `.jpg`.

## 3. Xử Lý Ảnh Mức Cơ Bản (Image Processing)
Để đưa ảnh vào mô hình Deep Learning chuẩn, bạn cần thực hiện:
- **Resize:** Đưa mọi ảnh về một kích thước chuẩn (ví dụ: `416x416` hoặc `640x640`). 
  > *Lưu ý quan trọng:* Trong biển số, tỉ lệ chiều ngang/dọc rất quan trọng. Khi resize nên dùng **Letterbox Padding** (thêm viền/padding) để giữ nguyên được tỉ lệ khung hình (aspect ratio) gốc, tránh làm biển số bị méo xệch.
- **Chuẩn hóa (Normalization):** Đưa giá trị các pixel từ `0-255` về dải `0-1` (chia cho 255.0) hoặc về `[-1, 1]`, điều này giúp cho model học nhanh và ổn định hơn.
- **Chuyển sang ảnh xám (Grayscale):** Thường được áp dụng trong mô hình nhận diện ký tự (OCR) vì màu sắc của xe đôi khi không quan trọng, chuyển sang ảnh xám giúp giảm kích thước tham số.

## 4. Cắt Biển Số (Cropping & Alignment) - Nhánh OCR
Hệ thống đọc biển số hiện đại thường chia làm 2 phần: Mô hình dò tìm biển (Detection) và Mô hình đọc chữ (OCR).
- **Cắt (Crop):** Từ tọa độ bounding box có sẵn trong tập nhãn, viết hàm cắt riêng vùng ảnh chứa phần biển số xe lưu thành một thư mục riêng (ví dụ `plates/`). Thư mục này dùng để huấn luyện phần đọc chữ phía sau (OCR).
- **Căn chỉnh (Alignment):** Nếu biển số bị nghiêng lệch do góc camera, cần thực hiện **Perspective Transform** để kéo phẳng lăng kính về thành hình chữ nhật chuẩn trước khi đưa vào mô hình OCR.

## 5. Tăng Cường Dữ Liệu (Data Augmentation)
Camera quay biển số ngoài đường có thể gặp rất nhiều điều kiện khắc nghiệt, do đó hệ thống cần tạo ra thêm dữ liệu mô phỏng nhờ *Data Augmentation*:
- **Thay đổi ánh sáng / Tương phản:** Mô phỏng điều kiện chụp ban ngày, chói đèn ban đêm, sương mù.
- **Thêm nhiễu loạn (Noise & Blur):** Mô phỏng camera cao tốc khi chụp vật di chuyển nhanh bị mờ nhòe.
- **Xoay méo nhẹ (Rotation / Shear):** Phóng tạo các Camera lắp đặt hơi nghiêng.
- **Khảm ảnh (Mosaic):** Kỹ thuật đắc lực trên các mô hình YOLO (ghép 4 tấm ảnh khác nhau vào chung một khung hình).
- 🚫 **Lưu ý đặc biệt (TRONG MÔ HÌNH OCR):** Tuyệt đối HẠN CHẾ VÀ CẨN THẬN khi làm **Lật ảnh (Horizontal/Vertical Flip)** vì chữ E, số 3, số 9, chữ b/d lật ngược sẽ ra ký tự sai làm rối dữ liệu.

## 6. Chia Tập Dữ Liệu (Train - Validation - Test Split)
Phân phối ảnh để huấn luyện và đánh giá khách quan:
- **Tập Train (70%-80%):** Dùng để huấn luyện mô hình học trọng số.
- **Tập Validation (10%-20%):** Dùng để đánh giá độ chính xác chéo, tinh chỉnh siêu tham số và theo dõi xem mô hình có bị "học vẹt" (overfitting) hay không.
- **Tập Test (10%):** Tập dữ liệu hoàn toàn chưa từng thấy trong lúc train, dùng đo đạc độ chính xác thực tế cuối cùng trước khi đưa mô hình vào ứng dụng thực tế chạy thật.
