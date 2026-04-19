import os

# Paths
PROJECT_ROOT   = "/Users/binhminh/Project-II"
YOLO_WEIGHTS   = os.path.join(PROJECT_ROOT, "runs/detect/runs_yolo11/plate_detection/weights/best.pt")
TEST_IMAGE_DIR = os.path.join(PROJECT_ROOT, "yolo_dataset/images/test")
GROUND_TRUTH_CSV = os.path.join(
    PROJECT_ROOT,
    "data/vietnam-car-license-plate/Bike/GreenParking/location.csv"
)
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "OCR/output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_CSV_PADDLE = os.path.join(OUTPUT_DIR, "results_paddleocr.csv")
OUTPUT_CSV_EASY   = os.path.join(OUTPUT_DIR, "results_easyocr.csv")
METRICS_CSV       = os.path.join(OUTPUT_DIR, "metrics.csv")

# YOLO settings
YOLO_CONF_THRESHOLD  = 0.25
YOLO_IOU_THRESHOLD   = 0.45
BBOX_PADDING         = 5   

# OCR settings
PADDLE_LANG       = "en"
PADDLE_USE_ANGLE  = False      

EASYOCR_LANGUAGES = ["en", "vi"]
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

FLIP_MAP = {
    '9':'6', '6':'9',
    'L':'1', 'I':'1',
    'A':'V', 'V':'A',
    'E':'3', '3':'E',
    'S':'5', '5':'S',
    '7':'2', '2':'7',
    'J':'C', 'C':'J',
    'Z':'2',
}

CHAR_TO_NUM = {
    'E':'6', 'G':'6', 'S':'5', 'B':'8',
    'O':'0', 'D':'0', 'Z':'2', 'L':'1',
    'I':'1', 'T':'7', 'W':'7', 'A':'4',
}

NUM_TO_CHAR = {
    '6':'G', '5':'S', '8':'B', '0':'D',
    '2':'Z', '1':'L', '7':'T', '4':'A', '3':'E',
}
