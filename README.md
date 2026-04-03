# 다국어 QA 평가 데이터셋 생성 파이프라인

> **목적**: 다국어 챗봇 평가를 위한 합성 QA 데이터셋 자동 생성  
> **입력**: PDF 문서  
> **출력**: 5개 언어 × 100개 = **총 500개 QA 쌍** (JSON)

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

[UPDATED 2026-04-03] "업데이트" 현재 파이프라인은 한국어 우선 생성 → 수동 검수 → 영어 번역 → 다국어 번역 순서로 동작한다. 난이도별 생성 수는 EASY 30개, MIDDLE 30개, HARD 30개, NOT_ANSWERABLE 10개로 조정되어 언어당 총 100개, 전체 500개 QA를 생성한다. 또한 생성 메타데이터에 `retrieved_chunks` 필드를 추가해 실제로 검색된 원문 청크를 함께 확인할 수 있도록 변경했다.

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
  "retrieved_chunks": [
    {
      "source": "2026학년도1학기학사안내.pdf",
      "page": 3,
      "text": "졸업 이수 학점은 총 130학점이며..."
    }
  ],
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
| **EASY**           | 30개 | 1개      | 간결한 단일 사실 질문 |
| **MIDDLE**         | 30개 | 2~3개    | 상황 설명 포함, 여러 페이지 정보 연결 |
| **HARD**           | 30개 | 3개 이상 | 조건/상황 명시 필수, 비교·대조·추론 패턴 |
| **NOT_ANSWERABLE** | 10개 | 없음     | 1~2문장, 문서에 없는 그럴듯한 질문 (환각 탐지) |

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

## 페르소나 기반 질문 스타일

각 QA는 LLM이 자동 생성한 **페르소나(국적, TOPIK 수준, 상황)**를 반영하여 질문이 작성됩니다.  
TOPIK 수준이 낮은 페르소나(1~3급)의 경우 **구어체, 비격식체, 단순한 문장 구조**가 의도적으로 사용됩니다.

| 국적 | TOPIK | 상황 | 질문 속 표현 |
|------|-------|------|-------------|
| 미국 | 3급 | 한국 도착 후 환전 방법 문의 | "저 이제 한국에 와서 **계정**을 만들려고 하는데..." |
| 베트남 | 4급 | 인터넷 뱅킹 보안카드 vs OTP 비교 | "어떤 게 더 **convenient**한가요?" |

> **참고**: 질문의 어색한 표현, 단어 혼용, 외래어 코드스위칭은 오타나 오류가 아니라, 해당 페르소나의 한국어 숙련도를 반영한 **의도된 설계**입니다. 실제 외국인 학생이 챗봇에 입력하는 질문 패턴을 재현합니다.

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
