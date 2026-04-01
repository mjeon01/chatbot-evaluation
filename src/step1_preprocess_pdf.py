"""
Step 1: PDF 전처리
PDF → 페이지별 텍스트 추출 → refined_context.json
"""

import fitz  # PyMuPDF
import json
import sys
from pathlib import Path


def preprocess_pdf(pdf_path: str, output_json: str) -> None:
    print("\n" + "="*60)
    print("  📄 Step 1: PDF 전처리 시작")
    print("="*60)
    print(f"  입력 파일: {pdf_path}")
    print(f"  출력 파일: {output_json}")
    print("-"*60)

    if not Path(pdf_path).exists():
        print(f"  ❌ 오류: '{pdf_path}' 파일을 찾을 수 없습니다.")
        sys.exit(1)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"  📖 총 {total_pages} 페이지 감지됨. 추출 시작...\n")

    structured_data = []
    skipped = 0

    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        page_num = i + 1

        if text:
            structured_data.append({
                "page": page_num,
                "content": text
            })
            # 진행 상황 출력 (5페이지마다 또는 마지막 페이지)
            if page_num % 5 == 0 or page_num == total_pages:
                preview = text[:60].replace("\n", " ")
                print(f"  ✅ [{page_num:>3}/{total_pages}] 추출 성공 | 미리보기: \"{preview}...\"")
        else:
            skipped += 1
            print(f"  ⚠️  [{page_num:>3}/{total_pages}] 빈 페이지 스킵")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(structured_data, f, ensure_ascii=False, indent=4)

    print("\n" + "-"*60)
    print(f"  ✅ 전처리 완료!")
    print(f"  📊 추출된 페이지: {len(structured_data)}개")
    print(f"  ⚠️  스킵된 페이지: {skipped}개")
    print(f"  💾 저장 위치: {output_json}")
    print("="*60 + "\n")


if __name__ == "__main__":
    # 사용 예시
    PDF_PATH = "document.pdf"
    OUTPUT_JSON = "refined_context.json"

    preprocess_pdf(PDF_PATH, OUTPUT_JSON)
