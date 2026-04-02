# 다국어 QA 평가 데이터셋 생성 파이프라인

> **목적**: 다국어 챗봇 평가를 위한 합성 QA 데이터셋 자동 생성  
> **입력**: PDF 문서  
> **출력**: 5개 언어 × 50개 = **총 250개 QA 쌍** (JSON)

---

## 전체 아키텍처

```
PDF 파일 (data/)
     │
     ▼
[추출 단계] step1_preprocess_pdf.py
     │  텍스트 레이어 있음?
     ├─ Yes → PyMuPDF 직접 추출
     └─ No  → EasyOCR 폴백 (한국어 + 영어)
     │
     ▼
[생성 단계] step2_generate_qa.py
     │
     │  한국어 우선 워크플로우
     ├─ 1단계: 한국어 QA 생성
     │            ↓ 수동 검수
     ├─ 2단계: 한국어 → 영어 번역
     │            ↓
     └─ 3단계: 영어 → 인도네시아어 / 베트남어 / 우즈벡어 번역
     │
     │  난이도별 생성
     ├─ EASY           간결한 단일 사실 질문
     ├─ MIDDLE         상황 설명 포함, 복합 질문
     ├─ HARD           비교·대조·추론 필수
     └─ NOT_ANSWERABLE 문서에 없는 질문 (환각 탐지)
     │
     │  생성 후 LLM-as-Judge 검증
     │
     ▼
[후처리 단계] step3_postprocess.py
     │  규칙 기반 필터링 (언어별 최소 길이)
     │  중복 제거 (토픽 키 기반)
     │
     ▼
최종 QA 데이터셋
     ├─ 언어별 JSON (ko / en / id / vi / uz)
     └─ 통합 JSON + 필터링 로그
```

[업데이트 2026-04-02] 기존에는 단일 언어(한국어)만 직접 생성했으나, 현재는 한국어 우선 생성 후 검수 → 영어 번역 → 다국어 번역 순서로 진행한다. 추출 단계에서는 이미지 PDF 대응을 위해 EasyOCR 폴백이 추가됐으며, 생성 모델은 Ollama에서 vLLM(Qwen3.5-122B FP8)으로 교체됐다. NOT_ANSWERABLE 난이도가 새로 추가돼 챗봇 환각 탐지 평가가 가능해졌다.

---

## 데이터 스펙

### QA 쌍 구조 (JSON)

```json
{
  "id": "qa_0001",
  "language": "ko",
  "lang_name": "Korean",
  "difficulty": "EASY",
  "question": "2025년도 입학한 외국인 유학생인데요, 졸업하려면 총 몇 학점 채워야 하나요?",
  "answer": "졸업 이수 학점은 총 130학점이며...",
  "ref_pages": [{"source": "2026학년도1학기학사안내.pdf", "page": 3}],
  "topic_key": "graduation credits",
  "is_not_answerable": false,
  "reasoning_type": null,
  "persona": {
    "country": "Vietnam",
    "topik_level": "TOPIK 2급",
    "situation": "Anxious about graduation requirements"
  },
  "model": "/home/.../Qwen3.5-122B-A10B-FP8",
  "is_valid": true,
  "valid_reason": "Answer is directly supported by context"
}
```

### 난이도 정의

| 난이도 | 개수/언어 | 참조 페이지 | 설명 |
|--------|-----------|-------------|------|
| **EASY**           | 17개 | 1개      | 간결한 단일 사실 질문 |
| **MIDDLE**         | 18개 | 2~3개    | 상황 설명 포함, 여러 페이지 정보 연결 |
| **HARD**           | 10개 | 3개 이상 | 조건/상황 명시 필수, 비교·대조·추론 패턴 |
| **NOT_ANSWERABLE** |  5개 | 없음     | 1~2문장, 문서에 없는 그럴듯한 질문 (환각 탐지) |

### 모델 구성

| 역할 | 모델 | 비고 |
|------|------|------|
| QA 생성 / 번역 / 검증 | Qwen3.5-122B-A10B-FP8 | vLLM, H100 × 2 |

### 번역 체인

```
KO (생성) → EN (번역) → ID / VI / UZ (번역)
```

---

## 실행 방법

```bash
# Step 1: PDF 전처리
python src/step1_preprocess_pdf.py

# Step 2: QA 생성 (한국어 우선 워크플로우)
python src/step2_generate_qa.py --stage ko          # 한국어 생성 → 수동 검수
python src/step2_generate_qa.py --stage en          # 한국어 → 영어 번역
python src/step2_generate_qa.py --stage multilingual # 영어 → ID/VI/UZ 번역

# (또는 전체 직접 생성)
python src/step2_generate_qa.py --stage all

# Step 3: 후처리
python src/step3_postprocess.py
```

### 환경 설치

```bash
pip install pymupdf easyocr numpy vllm
```

---

## 중복 방지 전략

- 내부적으로 `topic_key`를 기록하여 같은 언어 내 중복 방지
- 생성 프롬프트에 `history` 파라미터로 이미 사용한 토픽 전달
- 페르소나를 LLM이 자동 생성하여 질문 스타일 다양화

---

## 품질 관리

### 필터링 단계

1. **생성 시**: JSON 파싱 실패 / 필수 필드 누락 → 즉시 재시도
2. **검증 시**: LLM-as-judge `is_valid: false` → 제거
3. **후처리 시**: 규칙 기반 필터 + 중복 제거

### 언어별 최소 길이 기준

| 언어 | 질문 최소 | 답변 최소 | 이유 |
|------|-----------|-----------|------|
| 한국어 `ko`      | 10자 | 15자 | 형태소 압축률 높아 짧아도 의미 있음 |
| 영어 `en`        | 20자 | 30자 | 단어 단위라 글자 수가 더 필요 |
| 인도네시아어 `id` | 20자 | 30자 | 영어와 유사한 구조 |
| 베트남어 `vi`    | 15자 | 25자 | 중간 수준 |
| 우즈벡어 `uz`    | 15자 | 20자 | 저자원 언어, 관대하게 적용 |

---

## 출력 파일 목록

```
data/                          # 입력 PDF 폴더
output/
├── context/
│   └── refined_context.json  # Step 1 출력: 페이지별 텍스트
├── qa_raw/                   # Step 2 중간 출력
│   ├── qa_ko_raw.json
│   ├── qa_en_raw.json
│   ├── qa_id_raw.json
│   ├── qa_vi_raw.json
│   ├── qa_uz_raw.json
│   └── qa_dataset_raw.json  # 통합
├── qa_review/                # 수동 검수용
│   └── qa_ko_pending.json
└── qa_final/                 # Step 3 최종 출력
    ├── qa_ko_final.json
    ├── qa_en_final.json
    ├── qa_id_final.json
    ├── qa_vi_final.json
    ├── qa_uz_final.json
    ├── qa_dataset_final.json  # 통합
    └── qa_filter_log.json     # 필터링 로그
```

