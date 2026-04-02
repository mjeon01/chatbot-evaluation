# 다국어 QA 평가 데이터셋 생성 파이프라인

> **목적**: 다국어 챗봇 평가를 위한 합성 QA 데이터셋 자동 생성  
> **입력**: PDF 문서  
> **출력**: 5개 언어 × 50개 = **총 250개 QA 쌍** (JSON)

---

## 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                 INPUT:  부산외국어대학교 관련 규정                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: PDF 전처리  [step1_preprocess_pdf.py]                   │
│                                                                 │
│  ./data/ 내 모든 PDF 자동 순회                                    │
│  PyMuPDF(fitz) → 페이지별 텍스트 추출                             │
│                                                                 │
│  출력: { "source": "파일명.pdf", "page": 1, "content": "..." }   │
│        → output/context/refined_context.json                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │  output/context/refined_context.json
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: 다국어 QA 생성  [step2_generate_qa.py]                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  언어별 루프 (KO / EN / ID / VI / UZ)                    │   │
│  │                                                         │   │
│  │  ┌─────────────────────────────────────────────────┐   │   │
│  │  │  난이도별 루프                                    │   │   │
│  │  │                                                 │   │   │
│  │  │  EASY   (17개) → 페이지 1개 선택                 │   │   │
│  │  │  MIDDLE (17개) → 페이지 2~3개 선택               │   │   │
│  │  │  HARD   (16개) → 페이지 3개+ 선택                │   │   │
│  │  │                    │                            │   │   │
│  │  │                    ▼                            │   │   │
│  │  │          Ollama 로컬 모델 호출                   │   │   │
│  │  │  ┌─────────────────────────────────────────┐   │   │   │
│  │  │  │  전 언어: qwen2.5:7b                     │   │   │   │
│  │  │  └─────────────────────────────────────────┘   │   │   │
│  │  │                    │                            │   │   │
│  │  │                    ▼                            │   │   │
│  │  │          LLM-as-Judge 검증                      │   │   │
│  │  │                    │                            │   │   │
│  │  │          ┌─────────┴──────────┐                │   │   │
│  │  │          │ 통과               │ 실패            │   │   │
│  │  │          ▼                   ▼                 │   │   │
│  │  │       저장               재시도 (최대 3회)       │   │   │
│  │  └─────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  번역 방향: 언어별 독립 생성 (직접 생성, 번역 아님)               │
│  번역 도구: facebook/nllb-200 (우즈벡어 등 저자원 언어 보조용)    │
│                                                                 │
│  출력: output/qa_raw/qa_ko_raw.json / qa_en_raw.json / ...     │
│        output/qa_raw/qa_dataset_raw.json (통합)                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │  qa_dataset_raw.json
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: 후처리 및 최종 출력  [step3_postprocess.py]             │
│                                                                 │
│  규칙 기반 필터링                                                │
│  ├─ 질문 길이 < 10자 → 제거                                      │
│  ├─ 답변 길이 < 15자 → 제거                                      │
│  ├─ 질문에 답변 포함 → 제거                                      │
│  └─ 참조 페이지 없음 → 제거                                      │
│                                                                 │
│  중복 제거 (언어 + 난이도 내 topic_key 기반)                      │
│                                                                 │
│  최종 저장                                                       │
│  ├─ qa_ko_final.json                                           │
│  ├─ qa_en_final.json                                           │
│  ├─ qa_id_final.json                                           │
│  ├─ qa_uz_final.json                                           │
│  ├─ qa_dataset_final.json (통합)                                │
│  └─ qa_filter_log.json (필터링 로그)                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## 데이터 스펙

### QA 쌍 구조 (JSON)

```json
{
  "id": "qa_0001",
  "language": "ko",
  "lang_name": "Korean",
  "difficulty": "EASY",
  "question": "졸업 이수 학점은 몇 학점인가요?",
  "answer": "졸업 이수 학점은 총 130학점이며...",
  "ref_pages": [{"source": "2026학년도1학기학사안내.pdf", "page": 3}],
  "topic_key": "graduation credits",
  "model": "qwen2.5:7b",
  "validator": "qwen2.5:7b",
  "is_valid": true,
  "valid_reason": "Answer is directly supported by context",
  "elapsed_sec": 4.2,
  "created_at": "2025-01-01T12:00:00"
}
```

### 난이도 정의

| 난이도 | 개수/언어 | 참조 페이지 | 설명 |
|--------|-----------|-------------|------|
| **EASY**   | 17개 | 1개    | 단일 페이지에서 직접 확인 가능한 사실 질문 |
| **MIDDLE** | 17개 | 2~3개  | 여러 페이지 정보를 연결해야 답변 가능 |
| **HARD**   | 16개 | 3개 이상 | 복합적 추론이 필요한 심화 질문 |

### 언어 및 모델 구성

| 언어 | 코드 | 생성 모델 | 검증 모델 | 번역 보조 |
|------|------|-----------|-----------|-----------|
| 한국어     | `ko` | qwen2.5:32b / exaone3.5:32b | llama3.1:8b | - |
| 영어       | `en` | llama3.1:8b / qwen2.5:32b  | qwen2.5:32b | - |
| 인도네시아어 | `id` | qwen2.5:32b / llama3.1:8b  | llama3.1:8b | NLLB |
| 베트남어   | `vi` | qwen2.5:32b / llama3.1:8b  | llama3.1:8b | NLLB |
| 우즈벡어   | `uz` | qwen2.5:32b / llama3.1:8b  | llama3.1:8b | NLLB |

---

## 실행 방법

```bash
# 환경 설치
pip install pymupdf transformers tqdm requests

# Ollama 모델 준비
ollama pull qwen2.5:32b
ollama pull llama3.1:8b
ollama serve  # 별도 터미널에서 실행

# 파이프라인 순서대로 실행
python step1_preprocess_pdf.py     # PDF → refined_context.json
python step2_generate_qa.py        # → qa_dataset_raw.json
python step3_postprocess.py        # → qa_dataset_final.json
```

---

## 중복 방지 전략

각 언어는 **독립적으로 별도 질문을 생성**합니다 (번역 금지).

- 내부적으로 `topic_key`를 기록하여 같은 언어 내 중복 방지
- 생성 프롬프트에 `history` 파라미터로 이미 사용한 토픽 전달
- 언어별로 다른 모델 조합을 사용해 생성 다양성 확보

```
KO: "졸업 요건은 어떻게 되나요?"     → topic_key: "graduation requirements"
EN: "What are the admission procedures?"  → topic_key: "admission process"
ID: "Berapa kredit yang harus diselesaikan?"  → 독립 생성
UZ: "Kursni yakunlash uchun qanday talablar bor?"  → 독립 생성
```

---

## 품질 관리

### 이중 검증 구조

```
[생성 모델]          [검증 모델]
qwen2.5:32b    →    llama3.1:8b   (서로 다른 모델로 교차 검증)
llama3.1:8b    →    qwen2.5:32b
```

### 필터링 단계

1. **생성 시**: JSON 파싱 실패 / 필수 필드 누락 → 즉시 재시도
2. **검증 시**: LLM-as-judge `is_valid: false` → 제거
3. **후처리 시**: 규칙 기반 필터 + 중복 제거

### 언어별 최소 길이 기준

언어마다 글자 수 체계가 달라 동일 기준 적용 시 역차별 발생 → 언어별로 다르게 설정

| 언어 | 질문 최소 | 답변 최소 | 이유 |
|------|-----------|-----------|------|
| 한국어 `ko` | 10자 | 15자 | 형태소 압축률 높아 짧아도 의미 있음 |
| 영어 `en`   | 20자 | 30자 | 단어 단위라 글자 수가 더 필요 |
| 인도네시아어 `id` | 20자 | 30자 | 영어와 유사한 구조 |
| 베트남어 `vi` | 15자 | 25자 | 중간 수준 |
| 우즈벡어 `uz` | 15자 | 20자 | 저자원 언어, 관대하게 적용 |

**예시 비교:**

```
# 한국어 (10자 기준)
"졸업 요건은?"       → 8자  → ❌ 제거
"졸업 이수 학점은?"  → 11자 → ✅ 통과

# 영어 (20자 기준)
"Who is he?"                            → 10자 → ❌ 제거
"What are the graduation requirements?" → 38자 → ✅ 통과

# 우즈벡어 (15자 기준, 관대 적용)
"Talablar nima?"    → 14자 → ❌ 제거
"Bitirish talablari?" → 20자 → ✅ 통과
```

---

## 출력 파일 목록

```
refined_context.json      # Step 1 출력: 페이지별 텍스트
qa_ko_raw.json            # Step 2 중간 출력
qa_en_raw.json
qa_id_raw.json
qa_vi_raw.json
qa_uz_raw.json
qa_dataset_raw.json       # Step 2 통합 출력
qa_ko_final.json          # Step 3 최종 출력 (언어별)
qa_en_final.json
qa_id_final.json
qa_vi_final.json
qa_uz_final.json
qa_dataset_final.json     # Step 3 최종 통합
qa_filter_log.json        # 필터링 로그
```
