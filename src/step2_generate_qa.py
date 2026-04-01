"""
Step 2: 다국어 QA 합성 데이터 생성
refined_context.json → 언어별 50개 × 4언어 = 200개 QA → qa_dataset_raw.json

- 번역: facebook/nllb (HuggingFace, 무료)
- 생성: Ollama 로컬 모델 (qwen2.5:32b, llama3.1 등, 무료)
- 난이도: EASY / MIDDLE / HARD
- 각 언어별로 독립적인 질문 생성 (중복 방지)
"""

import json
import time
import random
import requests
from datetime import datetime
from transformers import pipeline


# ──────────────────────────────────────────────
#  설정
# ──────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"

# 언어별 설정
LANGUAGES = {
    "ko": {"name": "Korean",      "nllb_code": "kor_Hang"},
    "en": {"name": "English",     "nllb_code": "eng_Latn"},
    "id": {"name": "Indonesian",  "nllb_code": "ind_Latn"},
    "vi": {"name": "Vietnamese",  "nllb_code": "vie_Latn"},
    "uz": {"name": "Uzbek",       "nllb_code": "uzn_Latn"},
}

# 난이도별 설정
DIFFICULTIES = {
    "EASY":   {"count": 17, "pages_needed": 1,    "desc": "Simple fact from a single page"},
    "MIDDLE": {"count": 17, "pages_needed": "2-3", "desc": "Connect facts across 2-3 pages"},
    "HARD":   {"count": 16, "pages_needed": "3+",  "desc": "Complex reasoning from 3+ sections"},
}
# EASY 17 + MIDDLE 17 + HARD 16 = 50개

# 언어별 사용할 모델 (다양하게 설정)
LANG_MODEL_MAP = {
    "ko": ["qwen2.5:32b", "exaone3.5:32b"],
    "en": ["llama3.1:8b",  "qwen2.5:32b"],
    "id": ["qwen2.5:32b",  "llama3.1:8b"],
    "vi": ["qwen2.5:32b",  "llama3.1:8b"],
    "uz": ["qwen2.5:32b",  "llama3.1:8b"],
}

QA_PER_LANGUAGE = 50
SYSTEM_PROMPT   = "You are a professional QA dataset creator. Output ONLY valid JSON. No markdown."


# ──────────────────────────────────────────────
#  번역 모듈 (Facebook NLLB, 무료)
# ──────────────────────────────────────────────

_translator_cache = {}

def get_translator(src_lang: str, tgt_lang: str):
    """NLLB 번역 파이프라인 (캐시)"""
    key = f"{src_lang}-{tgt_lang}"
    if key not in _translator_cache:
        print(f"  🔄 NLLB 번역 모델 로드 중: {src_lang} → {tgt_lang}")
        _translator_cache[key] = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=512,
            device=-1,  # CPU; GPU 있으면 device=0
        )
    return _translator_cache[key]

def translate_text(text: str, src_lang_code: str, tgt_lang_code: str) -> str:
    """NLLB로 텍스트 번역"""
    try:
        translator = get_translator(src_lang_code, tgt_lang_code)
        result = translator(text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        print(f"  ⚠️  번역 실패: {e}")
        return text  # 실패 시 원문 반환


# ──────────────────────────────────────────────
#  컨텍스트 선택 모듈
# ──────────────────────────────────────────────

def select_context_pages(pages: list, difficulty: str) -> tuple[str, list]:
    """
    난이도에 따라 페이지 선택 후 컨텍스트 반환
    Returns: (context_text, ref_page_numbers)
    """
    if difficulty == "EASY":
        page = random.choice(pages)
        return page["content"], [page["page"]]

    elif difficulty == "MIDDLE":
        n = random.randint(2, 3)
        selected = random.sample(pages, min(n, len(pages)))
        selected.sort(key=lambda x: x["page"])
        context = "\n\n".join(
            f"[Page {p['page']}]\n{p['content']}" for p in selected
        )
        return context, [p["page"] for p in selected]

    else:  # HARD
        n = random.randint(3, min(5, len(pages)))
        selected = random.sample(pages, n)
        selected.sort(key=lambda x: x["page"])
        context = "\n\n".join(
            f"[Page {p['page']}]\n{p['content']}" for p in selected
        )
        return context, [p["page"] for p in selected]


# ──────────────────────────────────────────────
#  QA 생성 모듈 (Ollama)
# ──────────────────────────────────────────────

USER_PROMPT_TEMPLATE = """[Context]
{context}

[Instruction]
Generate ONE unique {difficulty} question and answer in {language}.
- EASY: A simple factual question answerable from a single page.
- MIDDLE: A question that connects information from 2-3 pages.
- HARD: A complex question requiring deep reasoning from multiple sections.

[Constraints]
- The question AND answer MUST be written 100% in {language}.
- Do NOT translate or repeat from these already-used topics: {history}
- The question must be DIFFERENT in topic and style from the history above.
- Answer should be detailed and reference the context.

[Output Format - JSON only]
{{
    "question": "...",
    "answer": "...",
    "ref_pages": [{ref_pages}],
    "topic_key": "short_2-3_word_keyword_in_english"
}}"""

def call_ollama(model: str, system: str, user: str, temperature: float = 0.8) -> dict | None:
    """Ollama API 호출"""
    payload = {
        "model":   model,
        "system":  system,
        "prompt":  user,
        "format":  "json",
        "stream":  False,
        "options": {"temperature": temperature, "num_predict": 1024},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        return json.loads(raw)
    except json.JSONDecodeError:
        print("  ⚠️  JSON 파싱 실패")
        return None
    except requests.exceptions.ConnectionError:
        print("  ❌ Ollama 연결 실패. 'ollama serve' 실행 여부 확인하세요.")
        return None
    except Exception as e:
        print(f"  ⚠️  Ollama 오류: {e}")
        return None

def generate_single_qa(
    lang_code: str,
    difficulty: str,
    context: str,
    ref_pages: list,
    history: list,
    model: str,
) -> dict | None:
    """단일 QA 쌍 생성"""
    lang_name   = LANGUAGES[lang_code]["name"]
    history_str = ", ".join(history[-10:]) if history else "none"
    ref_str     = ", ".join(str(p) for p in ref_pages)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        context    = context[:3000],  # 토큰 제한
        difficulty = difficulty,
        language   = lang_name,
        history    = history_str,
        ref_pages  = ref_str,
    )

    result = call_ollama(model, SYSTEM_PROMPT, user_prompt)
    if not result:
        return None

    # 필수 필드 검증
    if not all(k in result for k in ("question", "answer", "topic_key")):
        print("  ⚠️  필수 필드 누락")
        return None

    if len(result["question"].strip()) < 10:
        print("  ⚠️  질문이 너무 짧음")
        return None

    return result


# ──────────────────────────────────────────────
#  검증 모듈
# ──────────────────────────────────────────────

VALIDATE_PROMPT = """Check if this QA pair is accurate and answerable from the context.
Question: {question}
Answer: {answer}
Context: {context}

Respond ONLY in JSON: {{"is_valid": true or false, "reason": "brief reason"}}"""

def validate_qa(qa_item: dict, context: str, validator_model: str = "llama3.1:8b") -> dict:
    """다른 모델로 QA 검증"""
    prompt = VALIDATE_PROMPT.format(
        question = qa_item["question"],
        answer   = qa_item["answer"],
        context  = context[:2000],
    )
    result = call_ollama(validator_model, "You are a QA validator. Output only JSON.", prompt, temperature=0.0)
    if result and "is_valid" in result:
        return result
    return {"is_valid": True, "reason": "validation skipped"}


# ──────────────────────────────────────────────
#  메인 생성 루프
# ──────────────────────────────────────────────

def print_progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    pct    = current / total * 100
    return f"[{bar}] {pct:.0f}% ({current}/{total})"

def generate_language_dataset(
    lang_code: str,
    pages: list,
    output_path: str,
) -> list:
    """
    단일 언어에 대해 50개 QA 생성
    실시간으로 터미널에 진행 상황 출력
    """
    lang_name  = LANGUAGES[lang_code]["name"]
    models     = LANG_MODEL_MAP[lang_code]
    results    = []
    history    = []  # 중복 방지용 토픽 기록

    print("\n" + "="*60)
    print(f"  🌐 언어: {lang_name} ({lang_code.upper()})  |  목표: {QA_PER_LANGUAGE}개")
    print(f"  🤖 사용 모델: {' / '.join(models)}")
    print("="*60)

    # 난이도별 생성
    for diff, cfg in DIFFICULTIES.items():
        target_count = cfg["count"]
        print(f"\n  📊 난이도: {diff} (목표 {target_count}개)")
        print(f"     └─ {cfg['desc']}")
        print()

        success_count = 0
        attempt       = 0
        max_attempts  = target_count * 3  # 최대 재시도

        while success_count < target_count and attempt < max_attempts:
            attempt += 1
            # 모델 번갈아 사용
            model = models[attempt % len(models)]

            # 컨텍스트 선택
            context, ref_pages = select_context_pages(pages, diff)

            print(f"  [{diff}] 시도 {attempt:>2} | 모델: {model:<20} | 참조 페이지: {ref_pages}")

            # QA 생성
            t0  = time.time()
            qa  = generate_single_qa(lang_code, diff, context, ref_pages, history, model)
            elapsed = time.time() - t0

            if qa is None:
                print(f"           └─ ❌ 생성 실패 ({elapsed:.1f}s)")
                continue

            # 검증 (validator 모델 사용)
            validator = "llama3.1:8b" if model != "llama3.1:8b" else "qwen2.5:32b"
            validation = validate_qa(qa, context, validator)

            if not validation["is_valid"]:
                print(f"           └─ ⚠️  검증 실패: {validation['reason']}")
                continue

            # 저장
            topic_key = qa.get("topic_key", f"topic_{success_count}")
            history.append(topic_key)

            qa_record = {
                "id":          f"{lang_code}_{diff.lower()}_{success_count+1:03d}",
                "language":    lang_code,
                "lang_name":   lang_name,
                "difficulty":  diff,
                "question":    qa["question"],
                "answer":      qa["answer"],
                "ref_pages":   ref_pages,
                "topic_key":   topic_key,
                "model":       model,
                "validator":   validator,
                "is_valid":    validation["is_valid"],
                "valid_reason":validation["reason"],
                "elapsed_sec": round(elapsed, 2),
                "created_at":  datetime.now().isoformat(),
            }
            results.append(qa_record)
            success_count += 1

            progress = print_progress_bar(success_count, target_count)
            print(f"           └─ ✅ 성공 ({elapsed:.1f}s) | {progress}")
            print(f"              질문 미리보기: \"{qa['question'][:60]}...\"")

            # 중간 저장 (5개마다)
            if success_count % 5 == 0:
                _save_intermediate(results, output_path, lang_code)

        # 난이도 완료 요약
        print(f"\n  ✅ {diff} 완료: {success_count}/{target_count}개 생성")

    # 최종 저장
    _save_intermediate(results, output_path, lang_code)

    print(f"\n  🎉 {lang_name} 데이터셋 완료!")
    print(f"     총 {len(results)}개 생성 | 저장: {output_path}")
    print("="*60)

    return results


def _save_intermediate(results: list, output_path: str, lang_code: str) -> None:
    """중간 결과 저장"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  💾 중간 저장: {output_path} ({len(results)}개)")


# ──────────────────────────────────────────────
#  실행 진입점
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  🚀 Step 2: 다국어 QA 합성 데이터 생성 파이프라인")
    print("="*60)
    print(f"  대상 언어: {', '.join(LANGUAGES.keys())}")
    print(f"  언어당 QA: {QA_PER_LANGUAGE}개")
    print(f"  총 목표:   {QA_PER_LANGUAGE * len(LANGUAGES)}개  ({len(LANGUAGES)}개 언어 × {QA_PER_LANGUAGE}개)")
    print(f"  시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 컨텍스트 로드
    try:
        with open("refined_context.json", "r", encoding="utf-8") as f:
            pages = json.load(f)
        print(f"\n  📖 컨텍스트 로드 완료: {len(pages)} 페이지")
    except FileNotFoundError:
        print("  ❌ 'refined_context.json' 없음. step1_preprocess_pdf.py 먼저 실행하세요.")
        return

    all_results = {}
    total_generated = 0

    for lang_code in LANGUAGES.keys():
        output_path = f"qa_{lang_code}_raw.json"
        lang_results = generate_language_dataset(lang_code, pages, output_path)
        all_results[lang_code] = lang_results
        total_generated += len(lang_results)

    # 전체 통합 저장
    combined_output = "qa_dataset_raw.json"
    flat_results = [item for items in all_results.values() for item in items]
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(flat_results, f, ensure_ascii=False, indent=2)

    # 최종 통계 출력
    print("\n" + "="*60)
    print("  🏁 전체 생성 완료!")
    print("="*60)
    print(f"  📊 언어별 통계:")
    for lang_code, items in all_results.items():
        diff_counts = {}
        for item in items:
            diff_counts[item["difficulty"]] = diff_counts.get(item["difficulty"], 0) + 1
        counts_str = " | ".join(f"{d}:{c}" for d, c in diff_counts.items())
        print(f"     {lang_code.upper()}: {len(items)}개  ({counts_str})")
    print(f"\n  총 생성: {total_generated}개")
    print(f"  통합 저장: {combined_output}")
    print(f"  완료 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
