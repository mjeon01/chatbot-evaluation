"""
Step 1: PDF 전처리
PDF → 페이지별 텍스트 추출 → refined_context.json
        
        python src/step1_preprocess_pdf.py

- 텍스트 레이어가 있는 페이지: PyMuPDF 직접 추출
- 이미지 전용 페이지: EasyOCR 폴백 (한국어+영어)
"""

import fitz  # PyMuPDF
import json
import sys
import numpy as np
from pathlib import Path

import easyocr


# ─────────────────────────────────────────
#  OCR 싱글턴
# ─────────────────────────────────────────
_ocr_reader = None

def get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is None:
        print("  EasyOCR 모델 로드 중 (첫 실행 시 다운로드)...")
        _ocr_reader = easyocr.Reader(["ko", "en"], gpu=True)
        print("  EasyOCR 로드 완료!")
    return _ocr_reader


def ocr_page(page: fitz.Page) -> str:
    """이미지 페이지를 EasyOCR로 텍스트 추출"""
    pix = page.get_pixmap(dpi=200)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:      # RGBA → RGB
        img = img[:, :, :3]
    results = get_ocr_reader().readtext(img)
    return " ".join(text for _, text, conf in results if conf > 0.3)


# ─────────────────────────────────────────
#  PDF 처리
# ─────────────────────────────────────────
def preprocess_pdf(pdf_path: str) -> list:
    print("\n" + "="*60)
    print(f"  처리 중: {Path(pdf_path).name}")
    print("-"*60)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  총 {total_pages} 페이지 감지됨. 추출 시작...\n")

    structured_data = []
    ocr_count = 0

    for i, page in enumerate(doc):
        page_num = i + 1
        text     = page.get_text("text").strip()
        method   = "텍스트"

        if not text:
            text   = ocr_page(page)
            method = "OCR"
            if text:
                ocr_count += 1

        if text:
            structured_data.append({
                "source": Path(pdf_path).name,
                "page":   page_num,
                "content": text,
            })
            if page_num % 5 == 0 or page_num == total_pages:
                preview = text[:60].replace("\n", " ")
                print(f"  [{page_num:>3}/{total_pages}] {method} 추출 | \"{preview}...\"")
        else:
            print(f"  [{page_num:>3}/{total_pages}] 추출 실패 (텍스트·OCR 모두 빈 페이지)")

    skipped = total_pages - len(structured_data)
    print(f"\n  추출된 페이지: {len(structured_data)}개 (텍스트: {len(structured_data)-ocr_count}개, OCR: {ocr_count}개) / 스킵: {skipped}개")
    return structured_data


# ─────────────────────────────────────────
#  실행 진입점
# ─────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR    = "./data"
    OUTPUT_JSON = "./output/context/refined_context.json"

    pdf_files = sorted(Path(DATA_DIR).glob("*.pdf"))
    if not pdf_files:
        print(f"오류: '{DATA_DIR}'에 PDF 파일이 없습니다.")
        sys.exit(1)

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)

    print(f"\n총 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
    all_data = []
    for pdf_path in pdf_files:
        all_data.extend(preprocess_pdf(str(pdf_path)))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=4)

    print("\n" + "="*60)
    print(f"  전체 완료! 총 {len(all_data)}개 페이지 → {OUTPUT_JSON}")
    print("="*60 + "\n")
