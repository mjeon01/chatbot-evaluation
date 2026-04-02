"""
Step 3: QA 데이터셋 후처리 및 최종 출력
qa_dataset_raw.json → qa_dataset_final.json

- 중복 제거
- 품질 필터링
- 최종 통계 출력
- 언어별 분리 저장
"""

import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path


#  필터링 규칙

# 언어별 최소 글자 수 기준
# 한국어: 형태소 압축률 높아 짧아도 의미 있음
# 영어/인도네시아어: 단어 단위라 글자 수가 더 필요
# 베트남어/우즈벡어: 중간 수준
MIN_QUESTION_LENGTH = {
    "ko": 10,
    "en": 20,
    "id": 20,
    "vi": 15,
    "uz": 15,
}

MIN_ANSWER_LENGTH = {
    "ko": 15,
    "en": 30,
    "id": 30,
    "vi": 25,
    "uz": 20,
}

def rule_based_filter(item: dict) -> tuple[bool, str]:
    """규칙 기반 품질 필터 (언어별 기준 적용)"""
    lang = item.get("language", "ko")
    q    = item.get("question", "").strip()
    a    = item.get("answer",   "").strip()

    min_q = MIN_QUESTION_LENGTH.get(lang, 10)
    min_a = MIN_ANSWER_LENGTH.get(lang, 15)

    if len(q) < min_q:
        return False, f"질문이 너무 짧음 ({len(q)}자 < {min_q}자)"
    if len(a) < min_a:
        return False, f"답변이 너무 짧음 ({len(a)}자 < {min_a}자)"
    if len(q) > 500:
        return False, "질문이 너무 김 (> 500자)"
    if a.lower() in q.lower():
        return False, "질문에 답변 포함됨"
    if not item.get("ref_pages") and not item.get("is_not_answerable"):
        return False, "참조 페이지 없음"

    return True, "통과"


def deduplicate(items: list) -> tuple[list, int]:
    """토픽 키 기반 중복 제거 (언어+난이도 내에서)"""
    seen    = defaultdict(set) 
    result  = []
    removed = 0

    for item in items:
        key       = (item["language"], item["difficulty"])
        topic_key = item.get("topic_key", "").lower().strip()

        if topic_key and topic_key in seen[key]:
            removed += 1
            continue

        seen[key].add(topic_key)
        result.append(item)

    return result, removed


#  통계 출력
def print_statistics(items: list, title: str = "통계") -> None:
    print(f"\n  {'─'*50}")
    print(f"  📊 {title}")
    print(f"  {'─'*50}")
    print(f"  총 QA 쌍: {len(items)}개\n")

    # 언어별
    by_lang = defaultdict(list)
    for item in items:
        by_lang[item["language"]].append(item)

    diff_order = ["EASY", "MIDDLE", "HARD", "NOT_ANSWERABLE"]

    print("  [언어별]")
    for lang, lang_items in sorted(by_lang.items()):
        by_diff  = defaultdict(int)
        by_model = defaultdict(int)
        na_count = 0
        for it in lang_items:
            by_diff[it["difficulty"]] += 1
            by_model[it["model"].split("/")[-1]] += 1
            if it.get("is_not_answerable"):
                na_count += 1

        diff_str  = " / ".join(
            f"{d}:{by_diff[d]}" for d in diff_order if by_diff[d]
        )
        model_str = ", ".join(f"{m}:{c}" for m, c in sorted(by_model.items()))
        na_tag    = f"  (NA:{na_count}개)" if na_count else ""
        print(f"  ✅ {lang.upper()}: {len(lang_items):>3}개  |  난이도: {diff_str}{na_tag}")
        print(f"     모델: {model_str}")

    diff_order = ["EASY", "MIDDLE", "HARD", "NOT_ANSWERABLE"]
    print(f"\n  [난이도별]")
    by_diff = defaultdict(int)
    for item in items:
        by_diff[item["difficulty"]] += 1
    for diff in diff_order:
        count = by_diff.get(diff, 0)
        if count:
            bar = "█" * (count // 5)
            tag = " ← 환각탐지" if diff == "NOT_ANSWERABLE" else ""
            print(f"  {diff:<16}: {count:>3}개  {bar}{tag}")

    print(f"\n  [모델별]")
    by_model = defaultdict(int)
    for item in items:
        by_model[item["model"]] += 1
    for model, count in sorted(by_model.items()):
        print(f"  {model:<25}: {count:>3}개")

    print(f"  {'─'*50}")

#  메인
def postprocess(input_path: str = "qa_dataset_raw.json") -> None:
    print("\n" + "="*60)
    print(" Step 3: QA 데이터셋 후처리 시작")
    print("="*60)
    print(f"  입력: {input_path}")
    print(f"  시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 로드
    if not Path(input_path).exists():
        print(f"'{input_path}' 없음. step2_generate_qa.py 먼저 실행하세요.")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw_items = json.load(f)

    print(f"\n 원본 데이터 로드: {len(raw_items)}개")
    print_statistics(raw_items, "원본 통계")

    # 규칙 기반 필터
    print("\n 규칙 기반 필터링 중...")
    passed   = []
    filtered = []

    for item in raw_items:
        ok, reason = rule_based_filter(item)
        if ok:
            passed.append(item)
        else:
            filtered.append({"id": item.get("id"), "reason": reason})
            print(f" [{item.get('id','?')}] 필터 제거: {reason}")

    print(f"\n  필터링 결과: {len(passed)}개 통과 / {len(filtered)}개 제거")

    # 검증 실패 제거
    before_valid = len(passed)
    passed = [it for it in passed if it.get("is_valid", True)]
    invalid_removed = before_valid - len(passed)
    if invalid_removed:
        print(f" LLM 검증 실패 제거: {invalid_removed}개")

    # 중복 제거
    passed, dup_removed = deduplicate(passed)
    print(f" 중복 제거: {dup_removed}개")

    print_statistics(passed, "필터링 후 통계")

    # ID 재부여
    for i, item in enumerate(passed):
        item["id"] = f"qa_{i+1:04d}"

    # 최종 저장
    final_output = "qa_dataset_final.json"
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(passed, f, ensure_ascii=False, indent=2)
    print(f"\n 통합 저장: {final_output}")

    # 언어별 분리 저장
    by_lang = defaultdict(list)
    for item in passed:
        by_lang[item["language"]].append(item)

    print("\n 언어별 분리 저장:")
    for lang, items in sorted(by_lang.items()):
        lang_path = f"qa_{lang}_final.json"
        with open(lang_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"     {lang.upper()}: {lang_path}  ({len(items)}개)")

    # 필터링 로그 저장
    filter_log_path = "qa_filter_log.json"
    with open(filter_log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp":       datetime.now().isoformat(),
            "original_count":  len(raw_items),
            "final_count":     len(passed),
            "rule_filtered":   filtered,
            "invalid_removed": invalid_removed,
            "dup_removed":     dup_removed,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n  📋 필터링 로그: {filter_log_path}")

    print("\n" + "="*60)
    print(" 후처리 완료!")
    print(f"  원본: {len(raw_items)}개  →  최종: {len(passed)}개")
    print(f"  완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    postprocess("qa_dataset_raw.json")
