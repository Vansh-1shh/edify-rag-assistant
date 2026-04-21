"""
web_processor.py
Fetches and chunks content from a URL, returning LangChain Document objects
identical in shape to what load_and_chunk_pdf() returns.
"""

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urlparse
import re


def _clean_text(text: str) -> str:
    """Remove extra whitespace and blank lines."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_url_content(url: str) -> str:
    """Download a URL and return its main readable text."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove navigation, ads, scripts, styles
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe", "svg"]):
        tag.decompose()

    # Prefer <article> or <main> if present
    main = soup.find("article") or soup.find("main") or soup.find("body")
    raw = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

    return _clean_text(raw)


def load_and_chunk_url(url: str):
    """
    Fetch a URL, extract readable text, chunk it, and return
    a list of LangChain Document objects — same interface as load_and_chunk_pdf().
    """
    text = fetch_url_content(url)

    domain = urlparse(url).netloc
    metadata = {"source": url, "domain": domain}

    doc = Document(page_content=text, metadata=metadata)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents([doc])
    print(f"URL '{url}' → {len(chunks)} chunks")
    return chunks
