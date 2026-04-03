"""
Step 2: 다국어 QA 합성 데이터 생성

워크플로우:  python src/step2_generate_qa.py
  --stage ko          : 한국어 QA 생성 → 검수 파일 저장
  --stage en          : 검수된 한국어 → 영어 번역
  --stage multilingual: 영어 → ID / VI / UZ 번역
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import argparse
import json
import time
import random
from datetime import datetime
from pathlib import Path

from vllm import LLM, SamplingParams

MODEL_PATH = "/home/bufsgpu/Hugging-Face/models/Qwen3.5-122B-A10B-FP8"

LANGUAGES = {
    "ko": {"name": "Korean",      "nllb_code": "kor_Hang"},
    "en": {"name": "English",     "nllb_code": "eng_Latn"},
    "id": {"name": "Indonesian",  "nllb_code": "ind_Latn"},
    "vi": {"name": "Vietnamese",  "nllb_code": "vie_Latn"},
    "uz": {"name": "Uzbek",       "nllb_code": "uzn_Latn"},
}

TRANSLATION_CHAIN = {
    "en": "ko",  # 한국어 → 영어
    "id": "en",  # 영어 → 인도네시아어
    "vi": "en",  # 영어 → 베트남어
    "uz": "en",  # 영어 → 우즈벡어
}

LANG_COUNTRY = {
    "en": "USA",
    "id": "Indonesia",
    "vi": "Vietnam",
    "uz": "Uzbekistan",
}

DIFFICULTIES = {
    "EASY": {
        "count":          30,
        "pages_needed":   1,
        "desc":           "단일 페이지 단순 사실 질문",
        "max_new_tokens": 256,
    },
    "MIDDLE": {
        "count":          30,
        "pages_needed":   "2-3",
        "desc":           "2-3페이지에 걸친 정보 연결 질문",
        "max_new_tokens": 768,
    },
    "HARD": {
        "count":          30,
        "pages_needed":   "3+",
        "desc":           "비교·대조·복합 추론 — 두 조건 이상 분석 필요",
        "max_new_tokens": 1536,  # 추론 체인 고려
    },
    "NOT_ANSWERABLE": {
        "count":          10,
        "pages_needed":   "any",
        "desc":           "환각 탐지용 — 문서에 없는 내용에 대한 그럴듯한 질문",
        "max_new_tokens": 512,
    },
}

QA_PER_LANGUAGE = sum(cfg["count"] for cfg in DIFFICULTIES.values()) 

_llm = None


def load_model() -> LLM:
    global _llm
    if _llm is None:
        print(f"모델 로드 중: {MODEL_PATH}")
        _llm = LLM(
            model=MODEL_PATH,
            tensor_parallel_size=2,
            dtype="auto",
            trust_remote_code=True,
            additional_config={"gdn_prefill_backend": "triton"},
        )
        print("모델 로드 완료!")
    return _llm


def call_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.8,
    max_new_tokens: int = 1024,
) -> dict | None:
    llm = load_model()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    tokenizer = llm.get_tokenizer()
    text      = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate([text], sampling_params)
    raw     = outputs[0].outputs[0].text

    try:
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(raw[start:end])
    except json.JSONDecodeError:
        pass

    print(f"  JSON 파싱 실패: {raw[:120]}")
    return None


from prompts import (
    SYSTEM_PROMPT,
    EASY_PROMPT_TEMPLATE,
    MIDDLE_PROMPT_TEMPLATE,
    HARD_PROMPT_TEMPLATE,
    NOT_ANSWERABLE_PROMPT_TEMPLATE,
    VALIDATE_PROMPT,
    KO_TO_EN_TEMPLATE,
    EN_TO_LANG_TEMPLATE,
)


def build_retrieved_chunks(selected_pages: list) -> list:
    return [
        {
            "source": page.get("source", "unknown"),
            "page": page["page"],
            "text": page.get("content", ""),
        }
        for page in selected_pages
    ]


# 참고 문서, 페이지
def select_context_pages(pages: list, difficulty: str) -> tuple[str, list, list]:
    def ref(p):
        return {"source": p.get("source", "unknown"), "page": p["page"]}

    if difficulty == "EASY":
        page = random.choice(pages)
        return page["content"], [ref(page)], build_retrieved_chunks([page])
    elif difficulty == "MIDDLE":
        n        = random.randint(2, 3)
        selected = random.sample(pages, min(n, len(pages)))
        selected.sort(key=lambda x: x["page"])
        context  = "\n\n".join(f"[Page {p['page']}]\n{p['content']}" for p in selected)
        return context, [ref(p) for p in selected], build_retrieved_chunks(selected)
    else:  # HARD, NOT_ANSWERABLE
        n        = random.randint(3, min(5, len(pages)))
        selected = random.sample(pages, n)
        selected.sort(key=lambda x: x["page"])
        context  = "\n\n".join(f"[Page {p['page']}]\n{p['content']}" for p in selected)
        return context, [ref(p) for p in selected], build_retrieved_chunks(selected)


#  QA 생성
def build_user_prompt(
    lang_code: str,
    difficulty: str,
    context: str,
    ref_pages: list,
    history: list,
) -> str:
    lang_name   = LANGUAGES[lang_code]["name"]
    history_str = ", ".join(history[-10:]) if history else "none"

    ref_str = ", ".join(
        f"{p['source']} p.{p['page']}" for p in ref_pages
    )

    if difficulty == "EASY":
        return EASY_PROMPT_TEMPLATE.format(
            context  = context[:3000],
            language = lang_name,
            history  = history_str,
        )
    elif difficulty == "MIDDLE":
        return MIDDLE_PROMPT_TEMPLATE.format(
            context    = context[:3000],
            difficulty = difficulty,
            language   = lang_name,
            history    = history_str,
            ref_pages  = ref_str,
        )
    elif difficulty == "HARD":
        return HARD_PROMPT_TEMPLATE.format(
            context   = context[:3000],
            language  = lang_name,
            history   = history_str,
            ref_pages = ref_str,
        )
    elif difficulty == "NOT_ANSWERABLE":
        return NOT_ANSWERABLE_PROMPT_TEMPLATE.format(
            context  = context[:3000],
            language = lang_name,
            history  = history_str,
        )
    raise ValueError(f"Unsupported difficulty: {difficulty}")


def generate_single_qa(
    lang_code: str,
    difficulty: str,
    context: str,
    ref_pages: list,
    history: list,
) -> dict | None:
    user_prompt     = build_user_prompt(lang_code, difficulty, context, ref_pages, history)
    max_new_tokens  = DIFFICULTIES[difficulty]["max_new_tokens"]
    result          = call_model(SYSTEM_PROMPT, user_prompt, max_new_tokens=max_new_tokens)
    if not result:
        return None

    if not all(k in result for k in ("question", "answer", "topic_key")):
        print("필수 필드 누락")
        return None

    if len(result["question"].strip()) < 10:
        print("질문이 너무 짧음")
        return None

    if "persona" not in result or not isinstance(result["persona"], dict):
        result["persona"] = {"country": "unknown", "topik_level": "unknown", "situation": "unknown"}

    if difficulty == "NOT_ANSWERABLE":
        result["ref_pages"]         = []
        result["is_not_answerable"] = True

    return result

# 검증
def validate_qa(qa_item: dict, context: str) -> dict:
    if qa_item.get("is_not_answerable"):
        return {"is_valid": True, "reason": "not_answerable — validation skipped"}

    prompt = VALIDATE_PROMPT.format(
        question = qa_item["question"],
        answer   = qa_item["answer"],
        context  = context[:2000],
    )
    result = call_model("You are a QA validator. Output only JSON.", prompt, temperature=0.0)
    if result and "is_valid" in result:
        return result
    return {"is_valid": True, "reason": "validation skipped"}


#  진행률 바
def print_progress_bar(current: int, total: int, width: int = 30) -> str:
    filled = int(width * current / total)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {current/total*100:.0f}% ({current}/{total})"


# 언어별 데이터셋 생성 (all 모드)

def generate_language_dataset(lang_code: str, pages: list, output_path: str) -> list:
    lang_name = LANGUAGES[lang_code]["name"]
    results   = []
    history   = []

    print("\n" + "="*60)
    print(f"  언어: {lang_name} ({lang_code.upper()})  |  목표: {QA_PER_LANGUAGE}개")
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
            context, ref_pages, retrieved_chunks = select_context_pages(pages, diff)
            ref_str = ", ".join(f"{p['source']} p.{p['page']}" for p in ref_pages)
            print(f"  [{diff}] 시도 {attempt:>2} | 참조: {ref_str}")

            t0      = time.time()
            qa      = generate_single_qa(lang_code, diff, context, ref_pages, history)
            elapsed = time.time() - t0

            if qa is None:
                print(f"           └─ 생성 실패 ({elapsed:.1f}s)")
                continue

            validation = validate_qa(qa, context)
            if not validation["is_valid"]:
                print(f"           └─ 검증 실패: {validation['reason']}")
                continue

            topic_key = qa.get("topic_key", f"topic_{success_count}")
            history.append(topic_key)
            persona   = qa.get("persona", {})

            qa_record = {
                "id":                f"{lang_code}_{diff.lower()}_{success_count+1:03d}",
                "language":          lang_code,
                "lang_name":         lang_name,
                "difficulty":        diff,
                "question":          qa["question"],
                "answer":            qa["answer"],
                "ref_pages":         qa.get("ref_pages", ref_pages),
                "retrieved_chunks":  retrieved_chunks,
                "topic_key":         topic_key,
                "is_not_answerable": qa.get("is_not_answerable", False),
                "reasoning_type":    qa.get("reasoning_type"),
                "persona":           persona,
                "model":             MODEL_PATH,
                "is_valid":          validation["is_valid"],
                "valid_reason":      validation["reason"],
            }
            results.append(qa_record)
            success_count += 1

            persona_str = (
                f"{persona.get('country','?')} / "
                f"{persona.get('topik_level','?')} / "
                f"{persona.get('situation','?')[:30]}"
            )
            print(f"           └─ 성공 ({elapsed:.1f}s) | {print_progress_bar(success_count, target_count)}")
            print(f"              페르소나: {persona_str}")
            print(f"              질문:     \"{qa['question'][:60]}...\"")

            if success_count % 5 == 0:
                _save(results, output_path)

        print(f"\n  {diff} 완료: {success_count}/{target_count}개")

    _save(results, output_path)
    print(f"\n  {lang_name} 완료! 총 {len(results)}개 | 저장: {output_path}")
    print("="*60)
    return results


#  Stage 1: 한국어 베이스 생성
def stage_generate_korean(pages: list) -> None:
    print("\n" + "="*60)
    print("  [Stage ko] 한국어 베이스 QA 생성")
    print("="*60)

    Path("./output/qa_review").mkdir(parents=True, exist_ok=True)
    output_path = "./output/qa_review/qa_ko_pending.json"
    results     = generate_language_dataset("ko", pages, output_path)

    print(f"\n  한국어 생성 완료: {len(results)}개")
    print(f"  검수 파일: {output_path}")
    print("  ─────────────────────────────────────────")
    print("  [다음 단계 — 인간 검수]")
    print('  각 항목에 "human_approved": true / false 표시 후:')
    print("  python step2_generate_qa.py --stage en")
    print("  ─────────────────────────────────────────")


#  Stage 2: 한국어 → 영어 번역
def stage_translate_to_english() -> None:
    print("\n" + "="*60)
    print("  [Stage en] 한국어 → 영어 번역")
    print("="*60)

    pending_path = "./output/qa_review/qa_ko_pending.json"
    if not Path(pending_path).exists():
        print(f"  오류: 검수 파일 없음 ({pending_path})")
        print("  먼저 --stage ko 를 실행하세요.")
        return

    with open(pending_path, encoding="utf-8") as f:
        ko_items = json.load(f)

    # human_approved가 명시적으로 False인 항목만 제외
    approved = [it for it in ko_items if it.get("human_approved") is not False]
    rejected = len(ko_items) - len(approved)
    print(f"  한국어 베이스: {len(ko_items)}개 (승인: {len(approved)}개, 제외: {rejected}개)")

    Path("./output/qa_raw").mkdir(parents=True, exist_ok=True)

    ko_raw_path = "./output/qa_raw/qa_ko_raw.json"
    with open(ko_raw_path, "w", encoding="utf-8") as f:
        json.dump(approved, f, ensure_ascii=False, indent=2)
    print(f"  한국어 저장: {ko_raw_path}")

    en_results  = []
    country     = LANG_COUNTRY["en"]
    output_path = "./output/qa_raw/qa_en_raw.json"

    print(f"\n  영어 번역 시작... (총 {len(approved)}개)\n")

    for i, ko_item in enumerate(approved):
        user_prompt = KO_TO_EN_TEMPLATE.format(
            ko_question = ko_item["question"],
            ko_answer   = ko_item["answer"],
            country     = country,
            topic_key   = ko_item["topic_key"],
        )

        t0      = time.time()
        result  = call_model(SYSTEM_PROMPT, user_prompt)
        elapsed = time.time() - t0

        ref_str = ", ".join(
            f"p.{p}" if isinstance(p, int) else p if isinstance(p, str) else f"{p['source']} p.{p['page']}"
            for p in ko_item.get("ref_pages", [])
        )

        if not result or "question" not in result or "answer" not in result:
            print(f"  [{i+1:>3}] {ko_item['id']} → 번역 실패 ({elapsed:.1f}s), 원본 유지")
            result = {
                "question":  ko_item["question"],
                "answer":    ko_item["answer"],
                "topic_key": ko_item["topic_key"],
                "persona":   {"country": country, "topik_level": "Native", "situation": "unknown"},
            }
        else:
            print(f"  [{i+1:>3}] {ko_item['id']} → 완료 ({elapsed:.1f}s) | 참조: {ref_str}")
            print(f"        질문: \"{result['question'][:60]}...\"")

        en_record = {
            "id":                f"en_{ko_item['difficulty'].lower()}_{i+1:03d}",
            "language":          "en",
            "lang_name":         "English",
            "difficulty":        ko_item["difficulty"],
            "question":          result["question"],
            "answer":            result["answer"],
            "ref_pages":         ko_item["ref_pages"],
            "retrieved_chunks":  ko_item.get("retrieved_chunks", []),
            "topic_key":         result.get("topic_key", ko_item["topic_key"]),
            "is_not_answerable": ko_item.get("is_not_answerable", False),
            "reasoning_type":    ko_item.get("reasoning_type"),
            "persona":           result.get("persona", {}),
            "model":             MODEL_PATH,
            "source_ko_id":      ko_item["id"],
            "is_valid":          True,
            "valid_reason":      "translated from Korean",
        }
        en_results.append(en_record)

        if (i + 1) % 5 == 0:
            _save(en_results, output_path)

    _save(en_results, output_path)
    print(f"\n  영어 번역 완료: {len(en_results)}개 | 저장: {output_path}")
    print("  ─────────────────────────────────────────")
    print("  [다음 단계]")
    print("  python step2_generate_qa.py --stage multilingual")
    print("  ─────────────────────────────────────────")


#  Stage 3: 영어 → ID / VI / UZ 번역
def stage_expand_multilingual() -> None:
    print("\n" + "="*60)
    print("  [Stage multilingual] 영어 → ID / VI / UZ 번역")
    print("="*60)

    en_path = "./output/qa_raw/qa_en_raw.json"
    if not Path(en_path).exists():
        print(f"  오류: 영어 파일 없음 ({en_path})")
        print("  먼저 --stage en 을 실행하세요.")
        return

    with open(en_path, encoding="utf-8") as f:
        en_items = json.load(f)

    print(f"  영어 베이스: {len(en_items)}개\n")

    Path("./output/qa_raw").mkdir(parents=True, exist_ok=True)
    all_results = {}

    for lang_code in ["id", "vi", "uz"]:
        lang_name   = LANGUAGES[lang_code]["name"]
        country     = LANG_COUNTRY[lang_code]
        output_path = f"./output/qa_raw/qa_{lang_code}_raw.json"
        results     = []

        print(f"\n  [{lang_name}] 번역 시작... (총 {len(en_items)}개)")

        for i, en_item in enumerate(en_items):
            user_prompt = EN_TO_LANG_TEMPLATE.format(
                en_question = en_item["question"],
                en_answer   = en_item["answer"],
                language    = lang_name,
                country     = country,
                topic_key   = en_item["topic_key"],
            )

            t0      = time.time()
            result  = call_model(SYSTEM_PROMPT, user_prompt)
            elapsed = time.time() - t0

            if not result or "question" not in result or "answer" not in result:
                print(f"  [{lang_code}] {en_item['id']} → 번역 실패 ({elapsed:.1f}s), 영어 원본 유지")
                result = {
                    "question":  en_item["question"],
                    "answer":    en_item["answer"],
                    "topic_key": en_item["topic_key"],
                }
            else:
                ref_str = ", ".join(
                    f"p.{p}" if isinstance(p, int) else p if isinstance(p, str) else f"{p['source']} p.{p['page']}"
                    for p in en_item.get("ref_pages", [])
                )
                print(f"  [{i+1:>3}] {en_item['id']} → 완료 ({elapsed:.1f}s) | 참조: {ref_str}")
                print(f"        질문: \"{result['question'][:60]}...\"")

            qa_record = {
                "id":                f"{lang_code}_{en_item['difficulty'].lower()}_{i+1:03d}",
                "language":          lang_code,
                "lang_name":         lang_name,
                "difficulty":        en_item["difficulty"],
                "question":          result["question"],
                "answer":            result["answer"],
                "ref_pages":         en_item["ref_pages"],
                "retrieved_chunks":  en_item.get("retrieved_chunks", []),
                "topic_key":         result.get("topic_key", en_item["topic_key"]),
                "is_not_answerable": en_item.get("is_not_answerable", False),
                "reasoning_type":    en_item.get("reasoning_type"),
                "persona":           {"country": country, "topik_level": "unknown", "situation": "unknown"},
                "model":             MODEL_PATH,
                "source_en_id":      en_item["id"],
                "source_ko_id":      en_item.get("source_ko_id"),
                "is_valid":          True,
                "valid_reason":      "translated from English",
            }
            results.append(qa_record)

            if (i + 1) % 5 == 0:
                _save(results, output_path)

        _save(results, output_path)
        all_results[lang_code] = results
        print(f"  {lang_name} 완료: {len(results)}개")

    # 전체 통합 저장
    ko_raw  = json.load(open("./output/qa_raw/qa_ko_raw.json", encoding="utf-8"))
    en_raw  = json.load(open("./output/qa_raw/qa_en_raw.json", encoding="utf-8"))
    flat    = ko_raw + en_raw + [item for items in all_results.values() for item in items]

    combined_path = "./output/qa_raw/qa_dataset_raw.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)

    print(f"\n  통합 저장: {combined_path} ({len(flat)}개)")
    print("  ─────────────────────────────────────────")
    print("  [다음 단계]")
    print("  python step3_postprocess.py")
    print("  ─────────────────────────────────────────")


#  공통 저장
def _save(results: list, output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  중간 저장: {output_path} ({len(results)}개)")


def main():
    parser = argparse.ArgumentParser(
        description="다국어 QA 데이터 생성 파이프라인",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--stage",
        choices=["ko", "en", "multilingual", "all"],
        default="all",
        help=(
            "ko          : 한국어 QA 생성 → 검수 파일 저장\n"
            "en          : 검수된 한국어 → 영어 번역\n"
            "multilingual: 영어 → ID / VI / UZ 번역\n"
            "all         : 전체 언어 직접 생성 (검수 단계 없음)"
        ),
    )
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Step 2: 다국어 QA 합성 데이터 생성 파이프라인")
    print("="*60)
    print(f"  모드  : --stage {args.stage}")
    print(f"  모델  : {MODEL_PATH}")
    print(f"  시작  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    context_path = "./output/context/refined_context.json"

    if args.stage in ("ko", "all"):
        try:
            with open(context_path, encoding="utf-8") as f:
                pages = json.load(f)
            print(f"\n  컨텍스트 로드: {len(pages)} 페이지")
        except FileNotFoundError:
            print(f"  오류: '{context_path}' 없음. step1 먼저 실행하세요.")
            return

    if args.stage == "ko":
        stage_generate_korean(pages)

    elif args.stage == "en":
        stage_translate_to_english()

    elif args.stage == "multilingual":
        stage_expand_multilingual()

    elif args.stage == "all":
        Path("./output/qa_raw").mkdir(parents=True, exist_ok=True)
        all_results     = {}
        total_generated = 0

        for lang_code in LANGUAGES.keys():
            output_path  = f"./output/qa_raw/qa_{lang_code}_raw.json"
            lang_results = generate_language_dataset(lang_code, pages, output_path)
            all_results[lang_code]  = lang_results
            total_generated        += len(lang_results)

        flat          = [item for items in all_results.values() for item in items]
        combined_path = "./output/qa_raw/qa_dataset_raw.json"
        with open(combined_path, "w", encoding="utf-8") as f:
            json.dump(flat, f, ensure_ascii=False, indent=2)

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
        print(f"  저장  : {combined_path}")
        print(f"  완료  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
