"""
rag_pipeline.py
RAG-based QA with source citation metadata returned alongside the answer.
"""

from google import genai
from dotenv import load_dotenv
load_dotenv()  # reads .env and loads GEMINI_API_KEY into environment

# ── Replace with your actual key or load from env ──────────────────────────
import os
_KEY = os.environ.get("GEMINI_API_KEY")
if not _KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
client = genai.Client(api_key=_KEY)
# ───────────────────────────────────────────────────────────────────────────

_MODEL = "gemini-3-flash-preview"


def generate_answer(query: str, vector_store):
    """
    Returns (answer: str, sources: list[dict])
    Each source dict has keys: 'page_content', 'source', 'page' (if PDF).
    """
    docs_with_scores = vector_store.similarity_search_with_score(query, k=4)

    relevant = [(doc, score) for doc, score in docs_with_scores if score < 1.5]
    if not relevant:
        return (
            "The answer could not be found in the uploaded document.\n\n"
            "Try rephrasing your question or ask something more relevant to the content.",
            []
        )

    context = "\n\n".join([doc.page_content for doc, _ in relevant])

    prompt = f"""You are an academic assistant. Answer the question ONLY using the provided context.
If the answer is not in the context, say: "The answer is not available in the document."
Be concise and accurate. Format with markdown where helpful.

Context:
{context}

Question: {query}

Answer:"""

    try:
        response = client.models.generate_content(model=_MODEL, contents=prompt)
        answer = response.text.strip()

        # Build source list for citation display
        sources = []
        for doc, score in relevant:
            entry = {
                "page_content": doc.page_content[:200] + "…",
                "source": doc.metadata.get("source", "Uploaded Document"),
                "page": doc.metadata.get("page", None),
                "score": round(float(score), 3)
            }
            sources.append(entry)

        return answer, sources

    except Exception as e:
        print("Gemini error:", e)
        return "Unable to generate a response right now. Please try again.", []
