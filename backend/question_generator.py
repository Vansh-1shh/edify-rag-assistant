"""
question_generator.py
Generates exam-style questions — format matches the original project exactly.

Key insight: st.markdown() renders \n\n as a paragraph break (visible line break),
but collapses a single \n into a space. So every element that must appear on its
own line needs to be separated by \n\n, not \n.

format_output() post-processes the raw LLM text with regex to enforce this.
"""

from google import genai
import re
import json
import os
from dotenv import load_dotenv
load_dotenv()  # reads .env and loads GEMINI_API_KEY into environment

_KEY = os.environ.get("GEMINI_API_KEY")
if not _KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=_KEY)
_MODEL = "gemini-3-flash-preview"


# ─── Post-processor (core fix) ────────────────────────────────────────────────

def format_output(text: str, q_type: str) -> str:
    """
    Re-formats raw LLM output so every element is separated by \n\n,
    which st.markdown() renders as a proper line break.
    """
    # Strip markdown fences if model wrapped output
    text = re.sub(r"```[a-z]*\n?", "", text).strip()

    if q_type == "MCQs":
        # Each option letter on its own paragraph
        text = re.sub(r"\s*([A-D]\))\s*", r"\n\n\1 ", text)
        # Answer: on its own paragraph
        text = re.sub(r"\s*(Answer\s*:)\s*", r"\n\nAnswer: ", text)
        # Each numbered question starts a new paragraph
        text = re.sub(r"\s*(\d+\.)\s*", r"\n\n\1 ", text)
        return text.strip()

    else:
        # Short Answer / Long Answer
        # Split on numbered question markers
        parts = re.split(r"\n?\d+\.\s+", text)
        formatted = []

        for i, part in enumerate(parts[1:], 1):
            part = part.strip()
            if not part:
                continue

            # Split on "Answer:" label (case-insensitive)
            answer_match = re.split(r"(?i)answer\s*:\s*", part, maxsplit=1)

            if len(answer_match) == 2:
                question_part = answer_match[0].strip()
                answer_part   = answer_match[1].strip()
                formatted.append(
                    f"**{i}.** {question_part}\n\n**Answer:** {answer_part}"
                )
            elif "?" in part:
                # Fallback: split on the question mark
                q, *rest = part.split("?", 1)
                answer_part = rest[0].strip() if rest else ""
                formatted.append(
                    f"**{i}.** {q.strip()}?\n\n**Answer:** {answer_part}"
                )
            else:
                formatted.append(f"**{i}.** {part}")

        return "\n\n---\n\n".join(formatted).strip()


# ─── Main generator ───────────────────────────────────────────────────────────

def generate_questions(vector_store, difficulty: str, q_type: str, num_q: int) -> str:
    docs = vector_store.similarity_search("important concepts definitions examples", k=6)
    context = "\n\n".join([doc.page_content for doc in docs])

    if q_type == "MCQs":
        format_instruction = f"""Generate exactly {num_q} MCQs at {difficulty} difficulty.

Use this EXACT format — follow it strictly:

1. Question text here?
A) Option one
B) Option two
C) Option three
D) Option four
Answer: B

2. Next question here?
A) Option one
B) Option two
C) Option three
D) Option four
Answer: A

Rules:
- Exactly 4 options per question labeled A) B) C) D)
- Each option on its own line
- Answer: line immediately after option D
- One blank line between questions
- Difficulty: {difficulty}"""

    elif q_type == "Short Answer":
        format_instruction = f"""Generate exactly {num_q} short answer questions at {difficulty} difficulty.

Use this EXACT format — follow it strictly:

1. Question text here?
Answer: Write a concise 2-3 sentence answer here.

2. Next question here?
Answer: Write a concise 2-3 sentence answer here.

Rules:
- Question on its own line
- Answer: on the very next line immediately after the question
- One blank line between questions
- Difficulty: {difficulty}"""

    else:  # Long Answer
        format_instruction = f"""Generate exactly {num_q} long answer questions at {difficulty} difficulty.

Use this EXACT format — follow it strictly:

1. Question text here?
Answer: Write a detailed 5-8 sentence answer here covering all key concepts, examples, and significance. Suitable for a 10-mark exam response.

2. Next question here?
Answer: Write a detailed 5-8 sentence answer here covering all key concepts, examples, and significance.

Rules:
- Question on its own line
- Answer: on the very next line immediately after the question
- One blank line between questions
- Difficulty: {difficulty}"""

    prompt = f"""You are an academic question generator.

{format_instruction}

Context:
{context}

Output:"""

    try:
        response = client.models.generate_content(model=_MODEL, contents=prompt)
        return format_output(response.text, q_type)
    except Exception as e:
        print("Gemini error:", e)
        return "Unable to generate questions right now. Please try again."


# ─── Key topics extractor ─────────────────────────────────────────────────────

def extract_key_topics(vector_store) -> list[str]:
    docs = vector_store.similarity_search("main topics overview introduction", k=8)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""From the following academic content, extract exactly 6 key topics or concepts.
Return ONLY a JSON array of strings like: ["Topic 1", "Topic 2", ...]
No extra text, no markdown fences.

Content:
{context}"""

    try:
        response = client.models.generate_content(model=_MODEL, contents=prompt)
        text = response.text.strip().replace("```json", "").replace("```", "")
        topics = json.loads(text)
        return topics[:6] if isinstance(topics, list) else []
    except Exception as e:
        print("Topic extraction error:", e)
        return []
