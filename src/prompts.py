"""
prompts.py — step2_generate_qa.py 에서 사용하는 모든 LLM 프롬프트 템플릿
"""

SYSTEM_PROMPT = """
System Role: You are a multilingual assessment data generator specialized in university administration and campus life.
Generate questions that sound exactly like a real student typing a message to a university chatbot — casual, direct, natural.

Constraints:
1. Contextual Reality: Base questions on realistic situations from university academic handbooks or official notices.
2. Chatbot Tone: Write as if a student is typing to a chatbot. Natural, conversational, not formal essay style. No greetings (no "안녕하세요", "Hi", "Hello", etc.) — start directly with the question or context.
3. Strict Format: Output ONLY a single valid JSON object. No markdown, no extra text.

[Bad Example - DO NOT generate like this]
Q: "What is the graduation requirement?"  ← too generic, no context about who
Q: "When should I submit it?"             ← "it" is ambiguous
Q: "I would like to formally inquire about the graduation credit requirements stipulated in the academic handbook."  ← too formal, not chatbot style

[Good Example - chatbot message style]
Q: "2025년도 입학한 외국인 유학생인데요, 졸업하려면 총 몇 학점 채워야 하나요?"
Q: "I'm an international student at BUFS. How many credits do I need to graduate?"
   ← no greeting, starts directly, clear who, clear question
"""


# EASY / MIDDLE 공통
USER_PROMPT_TEMPLATE = """
[Context]
{context}

[Instruction]
Create a realistic student persona, then generate ONE {difficulty} question and answer IN {language}.

Persona guidelines:
- TOPIK 1~2: Simple words, short sentences, basic grammar only.
- TOPIK 3~4: Some academic terms, minor errors allowed.
- TOPIK 5~6 / Native: Full fluency, natural academic vocabulary.

Question style by difficulty:
- EASY: One direct fact from the context. Concise question, no long setup needed.
  Example style: "교환학생 신청 마감일이 언제예요?"
- MIDDLE: Include a brief personal situation before the question to explain why you're asking.
  Example style: "저 이번 학기에 전공 수업 많이 들었는데, 이수 학점이랑 졸업 조건이랑 어떻게 연결되는지 궁금해서요."

[Constraints]
- Output language: 100% {language}
- Avoid these already-used topics: {history}
- Tone: natural student-to-chatbot message. NOT formal. NOT essay style. No greetings at the start.
- Question MUST make clear who is asking and what they want — no ambiguous pronouns.
- Question length: 100~300 tokens.
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
}}
"""

# HARD 전용 — 비교·대조·추론 필수
HARD_PROMPT_TEMPLATE = """
[Context]
{context}

[Instruction]
Create a realistic student persona, then generate ONE HARD question and answer IN {language}.

HARD questions MUST go beyond simple fact-finding. Use one of these patterns:
  1. Compare two or more criteria (e.g. two scholarship types) and ask which applies to my situation.
  2. Describe a specific conflicting condition and ask how to resolve it.
  3. Require logical inference — the answer is NOT directly stated, must be deduced from the context.

Question style: Clearly state your conditions/situation first, then ask. Like a student explaining their case.
  Example style: "저는 현재 3학년이고 평점이 3.5인데, A 장학금이랑 B 장학금 중에 제가 받을 수 있는 게 뭔지 모르겠어요. 두 개 조건을 비교해서 설명해 주실 수 있나요?"

Do NOT ask a simple factual question. The answer must walk through reasoning step-by-step.

[Constraints]
- Output language: 100% {language}
- Avoid these already-used topics: {history}
- Tone: natural student-to-chatbot message. NOT formal.
- Question MUST explicitly state the student's conditions/situation — no vague setup.
- Question length: 100~300 tokens.
- Answer must show the full reasoning chain before the conclusion.

[Output - JSON only]
{{
    "question": "...",
    "answer": "...",
    "ref_pages": [{ref_pages}],
    "topic_key": "short_2-3_word_keyword_in_english",
    "reasoning_type": "comparison|contrast|inference",
    "persona": {{
        "country": "...",
        "topik_level": "...",
        "situation": "..."
    }}
}}
"""

NOT_ANSWERABLE_PROMPT_TEMPLATE = """[Context]
{context}

[Instruction]
Generate a plausible-sounding question about university life (scholarships, visas, dormitories, course registration)
that CANNOT be answered from the above context. The specific information must be ABSENT from the documents.

Purpose: Test whether a chatbot hallucinates an answer instead of admitting it does not know.
Strategy: Mix in a realistic-sounding but fictional scholarship type, administrative procedure, or policy
that is NOT mentioned in the provided context.

Question style: 1~2 sentences only. Short, natural, like a student quickly asking a chatbot.
  Example style:
  Q: "교환학생 신청할 때 추천서도 제출해야 하나요?"  ← short, plausible, but answer not in docs
  Q: "Does BUFS offer a part-time scholarship for students working on campus?"  ← plausible but not in docs

For the answer, mention the specific topic and politely explain it is not in the provided documents.

[Constraints]
- Output language: 100% {language}
- Tone: natural student-to-chatbot message. No greetings at the start.
- Question length: 1~2 sentences (50~150 tokens). Short and natural.
- The question must sound realistic but be unanswerable from context.
- The answer must mention the topic and politely state it is not available.
- Avoid these already-used topics: {history}

[Output - JSON only]
{{
    "question": "...",
    "answer": "...",
    "ref_pages": [],
    "topic_key": "na_short_keyword_in_english",
    "is_not_answerable": true,
    "persona": {{
        "country": "...",
        "topik_level": "...",
        "situation": "..."
    }}
}}
"""

VALIDATE_PROMPT = """Check if this QA pair is accurate and answerable from the context.
Question: {question}
Answer: {answer}
Context: {context}

Respond ONLY in JSON: {{"is_valid": true or false, "reason": "brief reason"}}"""

# KO → EN
KO_TO_EN_TEMPLATE = """[Korean Reference Q&A]
Question (KO): {ko_question}
Answer (KO): {ko_answer}

[Instruction]
Translate the above Korean Q&A into natural English.
- Do NOT translate word-for-word. Make it sound like a native English speaker wrote it.
- Preserve all factual content exactly.
- Reflect the cultural context of a {country} student studying in Korea.
- If the answer states information is not available in documents, preserve that meaning naturally in English.
[Output - JSON only]
{{
    "question": "...",
    "answer": "...",
    "topic_key": "{topic_key}",
    "persona": {{
        "country": "{country}",
        "topik_level": "...",
        "situation": "..."
    }}
}}"""

# EN → ID / VI / UZ
EN_TO_LANG_TEMPLATE = """[English Reference Q&A]
Question (EN): {en_question}
Answer (EN): {en_answer}

[Instruction]
Translate the above English Q&A into natural {language}.
- Do NOT translate word-for-word. Localize naturally for a {country} student.
- Preserve all factual content exactly.
- The result should feel like it was originally written in {language}.

[Output - JSON only]
{{
    "question": "...",
    "answer": "...",
    "topic_key": "{topic_key}"
}}"""
