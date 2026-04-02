"""
Step 1: PDF 전처리
PDF → 페이지별 텍스트 추출 → refined_context.json
"""

import fitz  # PyMuPDF
import json
import sys
from pathlib import Path


def preprocess_pdf(pdf_path: str) -> list:
    print("\n" + "="*60)
    print(f"  처리 중: {Path(pdf_path).name}")
    print("-"*60)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  총 {total_pages} 페이지 감지됨. 추출 시작...\n")

    structured_data = []
    skipped = 0

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        page_num = i + 1

        if text:
            structured_data.append({
                "source": Path(pdf_path).name,
                "page": page_num,
                "content": text
            })
            if page_num % 5 == 0 or page_num == total_pages:
                preview = text[:60].replace("\n", " ")
                print(f"  [{page_num:>3}/{total_pages}] 추출 성공 | 미리보기: \"{preview}...\"")
        else:
            skipped += 1
            print(f"  [{page_num:>3}/{total_pages}] 빈 페이지 스킵")

    print(f"\n  추출된 페이지: {len(structured_data)}개 / 스킵: {skipped}개")
    return structured_data


if __name__ == "__main__":
    DATA_DIR = "./data"
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
