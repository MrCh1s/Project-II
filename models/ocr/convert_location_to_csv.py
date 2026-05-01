"""
Định dạng: filename, full_plate, province, series, number
Ví dụ:
    1 0000_00532_b.jpg 1 145 73 72 62 59P1 66480
    -> filename = 0000_00532_b.jpg
    -> province = 59, series = P1, number = 66480
    -> full_plate = 59 P1 66480
"""
import re
import csv
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_TXT = os.path.join(
    PROJECT_ROOT,
    "data/vietnam-car-license-plate/Bike/GreenParking/location.txt"
)
DST_CSV = os.path.join(
    PROJECT_ROOT,
    "data/vietnam-car-license-plate/Bike/GreenParking/location.csv"
)

PROVINCE_CODES = {
    '11','12','13','14','15','16','17','18','19',
    '21','22','23','24','25','26','27','28','29',
    '30','31','32','33','34','35','36','37','38','39',
    '40','41','42','43','44','45','46','47','48','49',
    '50','51','52','53','54','55','56','57','58','59',
    '60','61','62','63','64','65','66','67','68','69',
    '70','71','72','73','74','75','76','77','78','79',
    '80','81','82','83','84','85','86','87','88','89',
    '90','91','92','93','94','95','96','97','98','99',
}

def clean_token(token: str) -> str:
    return re.sub(r'[^A-Za-z0-9]', '', token).strip()

# Tách plate_tokens thành (full_plate, province, series, number).
def parse_plate(tokens: list[str]) -> tuple[str, str, str, str]:
    if not tokens:
        return '', '', '', ''

    # Clean tất cả tokens
    cleaned = [clean_token(t) for t in tokens if clean_token(t)]
    if not cleaned:
        return '', '', '', ''

    province = ''
    series   = ''
    number   = ''

    first = cleaned[0]

    # Province 2 chữ số
    if len(first) >= 2 and first[:2].isdigit() and first[:2] in PROVINCE_CODES:
        province = first[:2]
        rest = first[2:]
        if rest:
            cleaned[0] = rest
        else:
            cleaned.pop(0)

    # Series = token đầu còn lại
    if cleaned:
        series = cleaned[0].upper()

    # Number = token còn lại, nối liền 
    if len(cleaned) > 1:
        number = ''.join(cleaned[1:])
    elif len(cleaned) == 1 and cleaned[0].isdigit():
        number = cleaned[0]
        series = ''

    parts = [p for p in [province, series, number] if p]
    full_plate = ' '.join(parts)

    return full_plate, province, series, number

def run():
    if not os.path.exists(SRC_TXT):
        print(f"[ERROR] Source file not found: {SRC_TXT}")
        sys.exit(1)

    rows    = []
    skipped = 0

    with open(SRC_TXT, 'r', encoding='utf-8') as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 8:
                skipped += 1
                continue

            filename   = parts[0]          
            plate_tokens = parts[6:]       

            full_plate, province, series, number = parse_plate(plate_tokens)

            rows.append({
                'filename':    filename,
                'full_plate':  full_plate,
                'province':    province,
                'series':      series,
                'number':      number,
            })

    os.makedirs(os.path.dirname(DST_CSV), exist_ok=True)
    with open(DST_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['filename', 'full_plate', 'province', 'series', 'number']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Parsed {len(rows)} rows")
    print(f"  Skipped {skipped} lines (format mismatch)")
    print(f"  CSV saved → {DST_CSV}")

    print("\n── Sample 10 rows ──")
    for r in rows[:10]:
        print(f"  {r['filename']}  →  {r['full_plate']}  "
              f"(province={r['province']}, series={r['series']}, number={r['number']})")

if __name__ == '__main__':
    run()
