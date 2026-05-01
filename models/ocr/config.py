import os

# Paths
PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
YOLO_WEIGHTS   = os.path.join(PROJECT_ROOT, "runs/detect/runs_yolo11/plate_detection/weights/best.pt")
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, "data/yolo_dataset/images/test")
GROUND_TRUTH_CSV = os.path.join(PROJECT_ROOT, "data/vietnam-car-license-plate/Bike/GreenParking/location.csv")
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "models/ocr/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV_PADDLE = os.path.join(OUTPUT_DIR, "results_paddle_ocr.csv")
OUTPUT_CSV_EASY   = os.path.join(OUTPUT_DIR, "results_easy_ocr.csv")
METRICS_CSV       = os.path.join(OUTPUT_DIR, "metrics.csv")

# YOLO settings
YOLO_CONF_THRESHOLD  = 0.25
YOLO_IOU_THRESHOLD   = 0.45
BBOX_PADDING         = 5   

# OCR settings
PADDLE_LANG       = "en"
PADDLE_USE_ANGLE  = False      

EASYOCR_LANGUAGES = ["en", "vi"]
# EASYOCR_LANGUAGES = "en"
EASYOCR_GPU       = False
EASYOCR_BATCH_SIZE = 16

PROVINCE_CODES = {
    '11','12','14','15','16','17','18','19', 
    '20','21','22','23','24','25','26','27','28','29',
    '30','31','32','33','34','35','36','37','38','39',
    '40','41','43','47','48','49',
    '50','51','52','53','54','55','56','57','58','59',
    '60','61','62','63','64','65','66','67','68','69',
    '70','71','72','73','74','75','76','77','78','79',
    '81','82','83','84','85','86','88','89',
    '90','92','93','94','95','97','98','99',
}
