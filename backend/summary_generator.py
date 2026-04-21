"""
summary_generator.py
Generates structured summaries of the loaded document.
"""

from google import genai
import os
from dotenv import load_dotenv
load_dotenv()  # reads .env and loads GEMINI_API_KEY into environment

_KEY = os.environ.get("GEMINI_API_KEY")
if not _KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=_KEY)
_MODEL = "gemini-3-flash-preview"


def generate_summary(vector_store, summary_type: str) -> str:
    docs = vector_store.similarity_search("main topics overview key concepts", k=10)
    context = "\n\n".join([doc.page_content for doc in docs])

    if summary_type == "Short Summary":
        instruction = """Write a concise summary (5-7 sentences).
Cover only the most essential ideas. Use plain, clear language."""

    elif summary_type == "Detailed Summary":
        instruction = """Write a comprehensive, well-structured summary.
Use markdown headings (##) to organize major sections.
Cover all key concepts, definitions, and important points.
Aim for 300-500 words."""

    else:  # Bullet-Point Notes
        instruction = """Generate structured bullet-point study notes.
Organize by topic using markdown headings (## Topic).
Use bullet points (- ) for each fact or concept.
Include sub-bullets (  - ) for important details.
Make it exam-ready and scannable."""

    prompt = f"""You are an expert academic summarizer.

{instruction}

Content to summarize:
{context}

Output:"""

    try:
        response = client.models.generate_content(model=_MODEL, contents=prompt)
        return response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return "Unable to generate summary right now. Please try again."
