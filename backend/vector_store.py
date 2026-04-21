"""
vector_store.py
Creates and persists FAISS vector stores.

FIX: indexes are now stored under faiss_indexes/<user_id>/<doc_name>/
so each user has a completely isolated document space.
Both PDF and URL sources are saved to disk (URLs were missing save_local before).
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

BASE_DIR = "faiss_indexes"


def _user_dir(user_id: int) -> str:
    """Return the root directory for a specific user's indexes."""
    path = os.path.join(BASE_DIR, str(user_id))
    os.makedirs(path, exist_ok=True)
    return path


def create_vector_store(chunks, doc_name: str, user_id: int):
    """Create a FAISS index from chunks and save it under the user's directory."""
    vector_store = FAISS.from_texts(
        [chunk.page_content for chunk in chunks],
        embedding=embeddings,
        metadatas=[chunk.metadata for chunk in chunks]
    )
    path = os.path.join(_user_dir(user_id), doc_name)
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)
    return vector_store


def load_vector_store(doc_name: str, user_id: int):
    """Load a saved FAISS index for a specific user."""
    path = os.path.join(_user_dir(user_id), doc_name)
    if os.path.exists(path):
        return FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None


def save_vector_store(vector_store, doc_name: str, user_id: int):
    """Persist an existing (possibly merged) vector store back to disk."""
    path = os.path.join(_user_dir(user_id), doc_name)
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)


def list_user_docs(user_id: int) -> list[str]:
    """Return sorted list of saved doc names for a user."""
    d = _user_dir(user_id)
    return sorted(
        name for name in os.listdir(d)
        if os.path.isdir(os.path.join(d, name))
    )


def delete_user_doc(doc_name: str, user_id: int):
    """Delete a user's saved index directory."""
    import shutil
    path = os.path.join(_user_dir(user_id), doc_name)
    if os.path.exists(path):
        shutil.rmtree(path)


def add_chunks_to_store(existing_store, new_chunks):
    """Merge new chunks into an existing vector store (multi-source sessions)."""
    texts = [c.page_content for c in new_chunks]
    metas = [c.metadata for c in new_chunks]
    existing_store.add_texts(texts, metadatas=metas)
    return existing_store
