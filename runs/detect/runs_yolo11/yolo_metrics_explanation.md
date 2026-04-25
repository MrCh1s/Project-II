# Giải thích các thông số trong biểu đồ YOLO11 (thư mục `runs_yolo11`)

## 1. Metrics 

Trong Machine Learning và Computer Vision, **metrics** là các chỉ số dùng để đánh giá hiệu suất của mô hình bằng con số cụ thể.

Thay vì nhận xét cảm tính, metrics giúp trả lời chính xác:

* Mô hình phát hiện đúng bao nhiêu?
* Có bỏ sót không?
* Có nhận diện nhầm không?

---

## 2. Biểu đồ Recall - Confidence (`BoxR_curve.png`)

* **Trục X:** Confidence Score (Ngưỡng tự tin)
* **Trục Y:** Recall (Độ thu hồi)

### Công thức:

```
Recall = TP / (TP + FN)
```

### Ý nghĩa:

* Recall đo khả năng **không bỏ sót đối tượng**
* Ví dụ:

  * Có 100 biển số thật
  * Mô hình phát hiện được 80 → Recall = 80%

### Diễn giải:

* Confidence thấp → Recall cao (ít bỏ sót, nhưng nhiều nhiễu)
* Confidence cao → Recall giảm (bỏ sót nhiều hơn)

---

## 3. Biểu đồ Precision - Confidence (`BoxP_curve.png`)

* **Trục X:** Confidence Score
* **Trục Y:** Precision (Độ chính xác)

### Công thức:

```
Precision = TP / (TP + FP)
```

### Ý nghĩa:

* Precision đo **độ tin cậy của dự đoán**
* Confidence càng cao → Precision càng cao

---

## 4. Biểu đồ F1 - Confidence (`BoxF1_curve.png`)

* **Trục Y:** F1-score

### Công thức:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### Ý nghĩa:

* Kết hợp giữa Precision và Recall
* Đỉnh của đường cong → **ngưỡng Confidence tối ưu nhất**

---

## 5. Biểu đồ Precision - Recall (`BoxPR_curve.png`)

* **Trục X:** Recall
* **Trục Y:** Precision

### Ý nghĩa:

* Đánh giá tổng thể mô hình
* Đường cong càng sát góc trên phải → mô hình càng tốt

### mAP:

* Diện tích dưới đường cong = **mAP (mean Average Precision)**

---

## 6. Các thông số Loss (trong `results.png` / `results.csv`)

### `train/box_loss` và `val/box_loss`

* Sai số về vị trí bounding box
* Càng thấp → box càng chính xác

---

### `train/cls_loss` và `val/cls_loss`

* Sai số phân loại
* Càng thấp → nhận diện class càng đúng

---

### `train/dfl_loss` và `val/dfl_loss`

* Distribution Focal Loss
* Giúp cải thiện độ chính xác biên bounding box
* Càng thấp → box càng “mịn” và chuẩn

---

## 7. Các chỉ số mAP theo IoU

### IoU (Intersection over Union)

* Đo mức độ chồng lấp giữa:

  * Box dự đoán
  * Box thực tế

---

### `metrics/mAP50(B)`

* IoU ≥ 0.5
* Dễ đạt → đánh giá cơ bản

---

### `metrics/mAP50-95(B)`

* Trung bình mAP từ IoU 0.5 → 0.95
* Khắt khe hơn
* Đánh giá **chất lượng box toàn diện**

---

##  Tổng kết nhanh

| Chỉ số    | Ý nghĩa chính               |
| --------- | --------------------------- |
| Recall    | Không bỏ sót                |
| Precision | Không đoán sai              |
| F1        | Cân bằng Precision & Recall |
| PR Curve  | Tổng thể mô hình            |
| Box Loss  | Sai vị trí box              |
| Cls Loss  | Sai phân loại               |
| DFL Loss  | Độ mịn box                  |
| mAP50     | Đánh giá cơ bản             |
| mAP50-95  | Đánh giá nâng cao           |

---
