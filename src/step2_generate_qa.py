"""
Step 2: 다국어 QA 합성 데이터 생성
refined_context.json → 언어별 50개 × 5언어 = 250개 QA → qa_dataset_raw.json

- 번역: facebook/nllb (HuggingFace)
- 생성: Ollama 로컬 모델 (qwen2.5:7b 등)
- 난이도: EASY / MIDDLE / HARD
- 페르소나: LLM이 자동 생성 (국가/TOPIK 수준/상황)
"""

import json
import time
import random
import requests
from datetime import datetime
from transformers import pipeline

#  설정
OLLAMA_URL = "http://localhost:11434/api/generate"

LANGUAGES = {
    "ko": {"name": "Korean",      "nllb_code": "kor_Hang"},
    "en": {"name": "English",     "nllb_code": "eng_Latn"},
    "id": {"name": "Indonesian",  "nllb_code": "ind_Latn"},
    "vi": {"name": "Vietnamese",  "nllb_code": "vie_Latn"},
    "uz": {"name": "Uzbek",       "nllb_code": "uzn_Latn"},
}

DIFFICULTIES = {
    "EASY":   {"count": 17, "pages_needed": 1,    "desc": "Simple fact from a single page"},
    "MIDDLE": {"count": 17, "pages_needed": "2-3", "desc": "Connect facts across 2-3 pages"},
    "HARD":   {"count": 16, "pages_needed": "3+",  "desc": "Complex reasoning from 3+ sections"},
}

LANG_MODEL_MAP = {'''
    "ko": ["qwen2.5:32b", "exaone3.5:32b"],
    "en": ["llama3.1:8b",  "qwen2.5:32b"],
    "id": ["qwen2.5:32b",  "llama3.1:8b"],
    "vi": ["qwen2.5:32b",  "llama3.1:8b"],
    "uz": ["qwen2.5:32b",  "llama3.1:8b"]'''

    "ko": ["qwen2.5:7b"],
    "en": ["qwen2.5:7b"],
    "id": ["qwen2.5:7b"],
    "vi": ["qwen2.5:7b"],
    "uz": ["qwen2.5:7b"],
}

QA_PER_LANGUAGE = 50

SYSTEM_PROMPT = """System Role: You are a multilingual assessment data generator specialized in university administration and campus life.
Design scenarios where students from South Korea, USA, Indonesia, Vietnam, and Uzbekistan inquire about
university systems (course registration, scholarships, visas, dormitories).

Constraints:
1. Contextual Reality: Use realistic situations based on university academic handbooks or official notices.
2. Multi-turn Logic: Formulate the question as if it is part of an ongoing conversation. Use pronouns or references
   that require understanding the provided [Context]. (e.g., Where should I submit those documents?)
3. Language Consistency: Ensure the question and answer are naturally phrased in the target language,
   respecting cultural nuances (e.g., honorifics in Korean).
4. Strict Format: Output ONLY a single valid JSON object. Do not include any conversational filler,
   introductory text, or markdown code blocks."""


USER_PROMPT_TEMPLATE = """[Context]
{context}

[Instruction]
First, create a realistic student persona from one of these countries: Vietnam, Uzbekistan, Indonesia, USA, or Korea.
Then generate ONE unique {difficulty} question and answer IN {language} that reflects that persona language level and situation.

Persona guidelines:
- TOPIK 1~2: Simple words, short sentences, basic grammar only.
- TOPIK 3~4: Some academic terms, minor errors allowed.
- TOPIK 5~6 / Native: Full fluency, academic vocabulary.

Difficulty:
- EASY: Single fact from one page.
- MIDDLE: Connects 2-3 pages.
- HARD: Complex reasoning from 3+ sections.

[Constraints]
- Output language: 100% {language}
- Avoid these already-used topics: {history}
- Answer must be detailed and grounded in the context.

[Output - JSON only]
{{
    "question": "...",
    "answer": "...",
    "ref_pages": [{ref_pages}],
    "topic_key": "short_2-3_word_keyword_in_english",
    "persona": {{
        "country": "...",
        "topik_level": "...",
        "situation": "..."
    }}
}}"""

VALIDATE_PROMPT = """Check if this QA pair is accurate and answerable from the context.
Question: {question}
Answer: {answer}
Context: {context}

Respond ONLY in JSON: {{"is_valid": true or false, "reason": "brief reason"}}"""


#  번역 모듈 (Facebook NLLB)
_translator_cache = {}

def get_translator(src_lang, tgt_lang):
    key = f"{src_lang}-{tgt_lang}"
    if key not in _translator_cache:
        print(f"  NLLB 번역 모델 로드 중: {src_lang} -> {tgt_lang}")
        _translator_cache[key] = pipeline(
            "translation",
            model="facebook/nllb-200-distilled-600M",
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            max_length=512,
            device=-1,
        )
    return _translator_cache[key]

def translate_text(text, src_lang_code, tgt_lang_code):
    try:
        translator = get_translator(src_lang_code, tgt_lang_code)
        result = translator(text, max_length=512)
        return result[0]["translation_text"]
    except Exception as e:
        print(f"  번역 실패: {e}")
        return text


#  컨텍스트 선택
def select_context_pages(pages, difficulty):
    def ref(p):
        return {"source": p.get("source", "unknown"), "page": p["page"]}

    if difficulty == "EASY":
        page = random.choice(pages)
        return page["content"], [ref(page)]
    elif difficulty == "MIDDLE":
        n = random.randint(2, 3)
        selected = random.sample(pages, min(n, len(pages)))
        selected.sort(key=lambda x: x["page"])
        context = "\n\n".join(f"[Page {p['page']}]\n{p['content']}" for p in selected)
        return context, [ref(p) for p in selected]
    else:  # HARD
        n = random.randint(3, min(5, len(pages)))
        selected = random.sample(pages, n)
        selected.sort(key=lambda x: x["page"])
        context = "\n\n".join(f"[Page {p['page']}]\n{p['content']}" for p in selected)
        return context, [ref(p) for p in selected]


#  Ollama 호출
def call_ollama(model, system, user, temperature=0.8):
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
        print("  JSON 파싱 실패")
        return None
    except requests.exceptions.ConnectionError:
        print("  Ollama 연결 실패. 'ollama serve' 실행 여부 확인하세요.")
        return None
    except Exception as e:
        print(f"  Ollama 오류: {e}")
        return None


#  QA 생성
def generate_single_qa(lang_code, difficulty, context, ref_pages, history, model):
    lang_name   = LANGUAGES[lang_code]["name"]
    history_str = ", ".join(history[-10:]) if history else "none"
    ref_str     = ", ".join(str(p) for p in ref_pages)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        context    = context[:3000],
        difficulty = difficulty,
        language   = lang_name,
        history    = history_str,
        ref_pages  = ref_str,
    )

    result = call_ollama(model, SYSTEM_PROMPT, user_prompt)
    if not result:
        return None

    if not all(k in result for k in ("question", "answer", "topic_key")):
        print("  필수 필드 누락")
        return None

    if len(result["question"].strip()) < 10:
        print("  질문이 너무 짧음")
        return None

    # persona 필드 없으면 기본값
    if "persona" not in result or not isinstance(result["persona"], dict):
        result["persona"] = {"country": "unknown", "topik_level": "unknown", "situation": "unknown"}

    return result


#  검증
def validate_qa(qa_item, context, validator_model):
    prompt = VALIDATE_PROMPT.format(
        question = qa_item["question"],
        answer   = qa_item["answer"],
        context  = context[:2000],
    )
    result = call_ollama(validator_model, "You are a QA validator. Output only JSON.", prompt, temperature=0.0)
    if result and "is_valid" in result:
        return result
    return {"is_valid": True, "reason": "validation skipped"}


#  메인 생성 루프
def print_progress_bar(current, total, width=30):
    filled = int(width * current / total)
    bar    = "X" * filled + "." * (width - filled)
    pct    = current / total * 100
    return f"[{bar}] {pct:.0f}% ({current}/{total})"

def generate_language_dataset(lang_code, pages, output_path):
    lang_name = LANGUAGES[lang_code]["name"]
    models    = LANG_MODEL_MAP[lang_code]
    results   = []
    history   = []

    print("\n" + "="*60)
    print(f"  언어: {lang_name} ({lang_code.upper()})  |  목표: {QA_PER_LANGUAGE}개")
    print(f"  사용 모델: {' / '.join(models)}")
    print("="*60)

    for diff, cfg in DIFFICULTIES.items():
        target_count = cfg["count"]
        print(f"\n  [난이도: {diff}] 목표 {target_count}개")
        print(f"     └─ {cfg['desc']}\n")

        success_count = 0
        attempt       = 0
        max_attempts  = target_count * 3

        while success_count < target_count and attempt < max_attempts:
            attempt += 1
            model = models[attempt % len(models)]

            context, ref_pages = select_context_pages(pages, diff)
            print(f"  [{diff}] 시도 {attempt:>2} | 모델: {model:<20} | 참조 페이지: {ref_pages}")

            t0      = time.time()
            qa      = generate_single_qa(lang_code, diff, context, ref_pages, history, model)
            elapsed = time.time() - t0

            if qa is None:
                print(f"           └─ 생성 실패 ({elapsed:.1f}s)")
                continue

            # 생성 모델 != 검증 모델 (모델이 1개면 동일 모델 사용)
            validator  = models[(attempt + 1) % len(models)] if len(models) > 1 else model
            validation = validate_qa(qa, context, validator)

            if not validation["is_valid"]:
                print(f"           └─ 검증 실패: {validation['reason']}")
                continue

            # 페르소나 요약 출력
            persona     = qa.get("persona", {})
            persona_str = f"{persona.get('country','?')} / {persona.get('topik_level','?')} / {persona.get('situation','?')[:30]}"

            topic_key = qa.get("topic_key", f"topic_{success_count}")
            history.append(topic_key)

            qa_record = {
                "id":           f"{lang_code}_{diff.lower()}_{success_count+1:03d}",
                "language":     lang_code,
                "lang_name":    lang_name,
                "difficulty":   diff,
                "question":     qa["question"],
                "answer":       qa["answer"],
                "ref_pages":    ref_pages,
                "topic_key":    topic_key,
                "persona":      persona,
                "model":        model,
                "validator":    validator,
                "is_valid":     validation["is_valid"],
                "valid_reason": validation["reason"],
                "elapsed_sec":  round(elapsed, 2),
                "created_at":   datetime.now().isoformat(),
            }
            results.append(qa_record)
            success_count += 1

            progress = print_progress_bar(success_count, target_count)
            print(f"           └─ 성공 ({elapsed:.1f}s) | {progress}")
            print(f"              페르소나: {persona_str}")
            print(f"              질문 미리보기: \"{qa['question'][:60]}...\"")

            if success_count % 5 == 0:
                _save_intermediate(results, output_path)

        print(f"\n  {diff} 완료: {success_count}/{target_count}개 생성")

    _save_intermediate(results, output_path)
    print(f"\n  {lang_name} 완료! 총 {len(results)}개 | 저장: {output_path}")
    print("="*60)
    return results


def _save_intermediate(results, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  중간 저장: {output_path} ({len(results)}개)")


#  실행 진입점
def main():
    print("\n" + "="*60)
    print("  Step 2: 다국어 QA 합성 데이터 생성 파이프라인")
    print("="*60)
    print(f"  대상 언어: {', '.join(LANGUAGES.keys())}")
    print(f"  언어당 QA: {QA_PER_LANGUAGE}개")
    print(f"  총 목표:   {QA_PER_LANGUAGE * len(LANGUAGES)}개  ({len(LANGUAGES)}개 언어 x {QA_PER_LANGUAGE}개)")
    print(f"  시작 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    context_path = "./output/context/refined_context.json"
    try:
        with open(context_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
        print(f"\n  컨텍스트 로드 완료: {len(pages)} 페이지")
    except FileNotFoundError:
        print(f"  '{context_path}' 없음. step1_preprocess_pdf.py 먼저 실행하세요.")
        return

    from pathlib import Path
    Path("./output/qa_raw").mkdir(parents=True, exist_ok=True)

    all_results     = {}
    total_generated = 0

    for lang_code in LANGUAGES.keys():
        output_path  = f"./output/qa_raw/qa_{lang_code}_raw.json"
        lang_results = generate_language_dataset(lang_code, pages, output_path)
        all_results[lang_code]  = lang_results
        total_generated        += len(lang_results)

    combined_output = "./output/qa_raw/qa_dataset_raw.json"
    flat_results    = [item for items in all_results.values() for item in items]
    with open(combined_output, "w", encoding="utf-8") as f:
        json.dump(flat_results, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print("  전체 생성 완료!")
    print("="*60)
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