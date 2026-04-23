"""
app.py — Edify: RAG-Powered Academic Assistant  v3

Fixes in this version:
  1. Per-user document isolation — FAISS indexes stored under faiss_indexes/<user_id>/
     New users start with zero documents. No data shared between accounts.
  2. Per-user granular history deletion — delete individual items, delete by type,
     or delete all. Available separately for Chat, Questions, and Summaries.
  3. Light / Dark mode toggle — persisted in session state, CSS variables swap.
  4. URL sources saved to disk — URL indexes now get save_local() just like PDFs,
     so they appear in Saved Documents and survive app restarts.
"""

import os, shutil, time
import html as html_module

import streamlit as st

from backend.pdf_processor      import load_and_chunk_pdf
from backend.web_processor      import load_and_chunk_url
from backend.vector_store       import (
    create_vector_store, load_vector_store, save_vector_store,
    add_chunks_to_store, list_user_docs, delete_user_doc,
)
from backend.rag_pipeline       import generate_answer
from backend.question_generator import generate_questions, extract_key_topics
from backend.summary_generator  import generate_summary
from backend.auth import (
    init_db, register_user, login_user,
    save_chat_message, load_chat_history,
    get_all_chat_docs, delete_chat_history, delete_all_chat_history,
    save_generation, load_generations,
    delete_generation, delete_generations_by_type, delete_all_generations,
    get_user_stats,
)

init_db()

st.set_page_config(
    page_title="Edify — Academic Assistant",
    page_icon="c:\\Users\\vansh\\Downloads\\file.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session defaults ──────────────────────────────────────────────────────────
_D = {
    "logged_in": False, "user_id": None, "username": None,
    "dark_mode": True,                      # NEW: theme toggle
    "vector_store": None, "current_doc": None, "sources_loaded": [],
    "messages": [], "key_topics": [],
    "question_count": 0, "confirm_delete": False,
    "last_questions": "", "last_summary": "",
    "last_uploaded_pdf": None,
    "last_added_urls": [],
}
for k, v in _D.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# THEME CSS  — swaps via dark_mode flag
# ═══════════════════════════════════════════════════════════════════════════════
def _inject_theme():
    dark = st.session_state.dark_mode
    if dark:
        t = """
        :root{
            --bg:#0D0F14; --surface:#141720; --surface2:#1C2030; --border:#252A3A;
            --accent:#6C8EF5; --text:#E2E8F0; --muted:#64748B; --r:10px;
            --bot-bubble:#1C2030; --bot-border:#252A3A;
            --input-bg:#1C2030; --input-border:#252A3A;
            --card:#141720; --card-hover-border:#6C8EF5;
        }"""
    else:
        t = """
        :root{
            --bg:#F4F6FB; --surface:#FFFFFF; --surface2:#EEF1F8; --border:#D1D9EE;
            --accent:#3B5BDB; --text:#1A1F36; --muted:#6B7A99; --r:10px;
            --bot-bubble:#FFFFFF; --bot-border:#D1D9EE;
            --input-bg:#FFFFFF; --input-border:#C5CEDF;
            --card:#FFFFFF; --card-hover-border:#3B5BDB;
        }"""

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600&family=JetBrains+Mono:wght@400;500&display=swap');
{t}

html,body,[data-testid="stAppViewContainer"]{{
    background:var(--bg)!important;font-family:'DM Sans',sans-serif;color:var(--text);}}
[data-testid="stSidebar"]{{
    background:var(--surface)!important;border-right:1px solid var(--border);}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(button span),
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div.stMarkdown,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]{{
    color:var(--text)!important;}}
[data-testid="stHeader"]{{background:transparent!important;}}
.block-container{{padding-top:1.5rem!important;}}
#MainMenu,footer,[data-testid="stDecoration"]{{display:none!important;}}

.brand{{font-family:'DM Serif Display',serif;font-size:2.2rem;
    background:linear-gradient(120deg,var(--accent),#A78BFA);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;line-height:1;}}
.brand-sub{{font-size:.8rem;color:var(--muted);letter-spacing:.5px;font-weight:300;}}
.sec-label{{font-size:.68rem;font-weight:600;letter-spacing:.12em;
    text-transform:uppercase;color:var(--muted);margin:18px 0 8px;}}
.tab-heading{{font-family:'DM Serif Display',serif;font-size:1.3rem;
    color:var(--text);margin-bottom:2px;}}
.tab-sub{{font-size:.82rem;color:var(--muted);margin-bottom:18px;}}

/* tabs */
[data-baseweb="tab-list"]{{background:transparent!important;
    border-bottom:1px solid var(--border)!important;gap:2px;padding:0 2px;}}
[data-baseweb="tab"]{{background:transparent!important;color:var(--muted)!important;
    font-family:'DM Sans',sans-serif!important;font-size:.88rem!important;
    font-weight:500!important;padding:10px 22px!important;
    border-radius:10px 10px 0 0!important;border:1px solid transparent!important;
    border-bottom:none!important;transition:all .18s ease!important;}}
[data-baseweb="tab"]:hover{{color:var(--text)!important;
    background:rgba(108,142,245,.07)!important;border-color:var(--border)!important;}}
[aria-selected="true"][data-baseweb="tab"]{{color:var(--accent)!important;
    background:var(--surface)!important;border-color:var(--border)!important;
    border-bottom:2px solid var(--accent)!important;
    box-shadow:0 -2px 12px rgba(108,142,245,.15)!important;}}
[data-baseweb="tab-panel"]{{background:transparent!important;padding-top:22px!important;}}

/* buttons — all variants, both themes */
.stButton>button,
.stButton>button *,
button[data-testid],
[data-testid="stSidebar"] .stButton>button,
[data-testid="stSidebar"] .stButton>button *{{
    background:var(--surface2)!important;
    border:1px solid var(--border)!important;
    color:var(--text)!important;
    border-radius:var(--r)!important;
    font-family:'DM Sans',sans-serif!important;
    font-weight:500!important;
    transition:all .15s!important;
    box-shadow:none!important;}}
.stButton>button:hover,
[data-testid="stSidebar"] .stButton>button:hover{{
    border-color:var(--accent)!important;
    color:var(--accent)!important;
    background:var(--surface2)!important;}}
.stButton>button:hover *,
[data-testid="stSidebar"] .stButton>button:hover *{{
    color:var(--accent)!important;}}
/* primary buttons */
.stButton>button[kind="primary"],
.stButton>button[kind="primary"] *{{
    background:linear-gradient(135deg,#3D5AF1,var(--accent))!important;
    border-color:transparent!important;
    color:#ffffff!important;
    box-shadow:0 2px 14px rgba(108,142,245,.3)!important;}}
.stButton>button[kind="primary"]:hover{{
    box-shadow:0 4px 20px rgba(108,142,245,.45)!important;
    transform:translateY(-1px)!important;}}
/* download button */
.stDownloadButton>button,
.stDownloadButton>button *{{
    background:var(--surface2)!important;
    border:1px solid var(--border)!important;
    color:var(--text)!important;
    border-radius:var(--r)!important;
    font-family:'DM Sans',sans-serif!important;
    font-weight:500!important;}}
.stDownloadButton>button:hover,
.stDownloadButton>button:hover *{{
    border-color:var(--accent)!important;
    color:var(--accent)!important;}}

/* inputs */
.stTextInput>div>div>input,.stTextArea textarea,.stSelectbox>div>div{{
    background:var(--input-bg)!important;border:1px solid var(--input-border)!important;
    border-radius:var(--r)!important;color:var(--text)!important;
    font-family:'DM Sans',sans-serif!important;}}
.stTextInput>div>div>input:focus,.stTextArea textarea:focus{{
    border-color:var(--accent)!important;
    box-shadow:0 0 0 2px rgba(108,142,245,.15)!important;}}

/* pills */
.source-pill{{display:inline-block;background:var(--surface2);
    border:1px solid var(--border);border-radius:20px;padding:3px 10px;
    font-size:.72rem;color:var(--accent);margin:2px;
    font-family:'JetBrains Mono',monospace;
    max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}}
.topic-chip{{display:inline-block;background:rgba(108,142,245,.12);
    border:1px solid rgba(108,142,245,.3);border-radius:6px;padding:4px 10px;
    font-size:.75rem;color:var(--accent);margin:3px;font-weight:500;}}

/* chat */
.chat-user{{display:flex;justify-content:flex-end;margin:10px 0 2px;}}
.chat-user-bubble{{background:linear-gradient(135deg,#3D5AF1,var(--accent));
    padding:10px 16px;border-radius:18px 18px 4px 18px;max-width:68%;color:#fff;
    font-size:.9rem;line-height:1.55;box-shadow:0 2px 12px rgba(108,142,245,.25);}}
.chat-bot{{display:flex;justify-content:flex-start;margin:2px 0 2px;}}
.chat-bot-bubble{{background:var(--bot-bubble);border:1px solid var(--bot-border);
    padding:12px 16px;border-radius:18px 18px 18px 4px;max-width:74%;
    color:var(--text);font-size:.9rem;line-height:1.6;}}
.chat-ts{{font-size:.65rem;color:var(--muted);margin:1px 4px 8px;}}
.cite-box{{margin:2px 0 10px 4px;padding:8px 12px;
    background:rgba(108,142,245,.06);border-left:2px solid var(--accent);
    border-radius:0 6px 6px 0;font-size:.73rem;color:var(--muted);
    font-family:'JetBrains Mono',monospace;line-height:1.5;}}
.cite-src{{color:var(--accent);font-weight:500;}}

/* metric card */
.metric-card{{background:var(--card);border:1px solid var(--border);
    border-radius:var(--r);padding:20px 24px;text-align:center;transition:border-color .2s;}}
.metric-card:hover{{border-color:var(--card-hover-border);}}
.metric-val{{font-family:'DM Serif Display',serif;font-size:2rem;color:var(--accent);}}
.metric-lbl{{font-size:.72rem;color:var(--muted);text-transform:uppercase;
    letter-spacing:.08em;margin-top:4px;}}

/* theme toggle button */
.theme-btn{{cursor:pointer;font-size:1.2rem;background:none;border:none;
    padding:4px 8px;border-radius:8px;transition:background .15s;}}
.theme-btn:hover{{background:var(--surface2);}}

hr{{border-color:var(--border)!important;}}
::-webkit-scrollbar{{width:6px;}}
::-webkit-scrollbar-track{{background:var(--bg);}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:3px;}}
[data-testid="stFileUploader"]{{
    background:var(--surface2)!important;
    border:1px dashed var(--border)!important;
    border-radius:var(--r)!important;}}
[data-testid="stFileUploader"] *{{
    color:var(--text)!important;
    background:transparent!important;}}
[data-testid="stFileUploaderDropzone"]{{
    background:var(--surface2)!important;}}
/* selectbox dropdown menu */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="menu"]{{
    background:var(--surface)!important;
    border:1px solid var(--border)!important;}}
[data-baseweb="menu"] li,
[data-baseweb="option"]{{
    background:var(--surface)!important;
    color:var(--text)!important;}}
[data-baseweb="option"]:hover{{
    background:var(--surface2)!important;}}
/* checkbox, radio */
.stCheckbox label span,
.stRadio label span{{color:var(--text)!important;}}
/* captions and small text */
.stCaption,.stCaption *{{color:var(--muted)!important;}}
/* info / success / warning / error boxes */
.stAlert{{background:var(--surface2)!important;border-color:var(--border)!important;}}
/* expander */
[data-testid="stExpander"]{{
    background:var(--surface)!important;
    border:1px solid var(--border)!important;
    border-radius:var(--r)!important;}}
[data-testid="stExpander"] summary{{
    background:var(--surface)!important;
    color:var(--text)!important;}}
</style>
""", unsafe_allow_html=True)

_inject_theme()


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
def _auth_screen():
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            '<div style="text-align:center;margin-bottom:28px;">'
            '<span class="brand">Edify</span><br>'
            '<span class="brand-sub">RAG-Powered Academic Assistant</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        t_in, t_reg = st.tabs(["  Sign In  ", "  Create Account  "])
        with t_in:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            u  = st.text_input("Username", key="li_u",  placeholder="your username")
            p  = st.text_input("Password", key="li_p",  placeholder="••••••••", type="password")
            if st.button("Sign In", use_container_width=True, type="primary", key="btn_li"):
                if u and p:
                    ok, user, msg = login_user(u, p)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.user_id   = user["id"]
                        st.session_state.username  = user["username"]
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.warning("Please enter username and password.")
        with t_reg:
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            nu  = st.text_input("Username",         key="reg_u",  placeholder="min. 3 characters")
            np  = st.text_input("Password",         key="reg_p",  placeholder="min. 6 characters", type="password")
            np2 = st.text_input("Confirm password", key="reg_p2", placeholder="repeat password",   type="password")
            if st.button("Create Account", use_container_width=True, type="primary", key="btn_reg"):
                if not (nu and np and np2):    st.warning("Please fill all fields.")
                elif np != np2:               st.error("Passwords do not match.")
                else:
                    ok, msg = register_user(nu, np)
                    if ok: st.success(msg + "  You can now sign in.")
                    else:  st.error(msg)

if not st.session_state.logged_in:
    _auth_screen()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
UID = st.session_state.user_id


def _reset_session(new_doc: str):
    st.session_state.current_doc    = new_doc
    st.session_state.messages       = []
    st.session_state.key_topics     = []
    st.session_state.sources_loaded = []
    st.session_state.last_questions = ""
    st.session_state.last_summary   = ""


def _render_msg(msg: dict):
    safe = html_module.escape(msg["content"])
    ts   = msg.get("timestamp", "")[:16].replace("T", " ")
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user"><div class="chat-user-bubble">{safe}</div></div>'
            f'<div style="text-align:right"><span class="chat-ts">{ts}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-bot"><div class="chat-bot-bubble">{safe}</div></div>'
            f'<span class="chat-ts">{ts}</span>',
            unsafe_allow_html=True,
        )
        sources = msg.get("sources", [])
        if sources:
            lines = []
            for s in sources:
                src  = s.get("source", "Document")
                pg   = s.get("page")
                pg_s = f" · p.{pg+1}" if pg is not None else ""
                snip = s.get("page_content", "")[:100].replace("\n", " ")
                lines.append(
                    f'<span class="cite-src">{html_module.escape(src)}{pg_s}</span>'
                    f'  —  {html_module.escape(snip)}'
                )
            st.markdown(
                f'<div class="cite-box">📎 Sources:<br>{"<br>".join(lines)}</div>',
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # ── Brand + user info + theme toggle ─────────────────────────────────────
    b_col, t_col = st.columns([4, 1])
    with b_col:
        st.markdown(
            '<div class="brand" style="font-size:1.5rem">Edify</div>'
            '<div class="brand-sub">Academic Assistant</div>',
            unsafe_allow_html=True,
        )
    with t_col:
        # Light/dark toggle  ← FIX 3
        icon = "light" if st.session_state.dark_mode else "Dark"
        if st.button(icon, key="theme_toggle", help="Toggle light/dark mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

    u_col, o_col = st.columns([3, 1])
    with u_col:
        st.markdown(
            f'<div style="font-size:.78rem;color:var(--accent);margin-top:2px;">'
            f'👤 {st.session_state.username}</div>',
            unsafe_allow_html=True,
        )
    with o_col:
        if st.button("Logout", key="logout", help="Sign out"):
            for k in list(_D.keys()):
                st.session_state[k] = _D[k]
            st.rerun()

    st.markdown("---")

    # ── Upload PDF ────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Upload PDF</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["pdf"], label_visibility="collapsed", key="uploader")

    if uploaded is not None:
        if st.session_state.last_uploaded_pdf != uploaded.name:
            st.session_state.last_uploaded_pdf = uploaded.name
            doc_key = uploaded.name.replace(".pdf", "").replace(" ", "_")

            with open("temp_upload.pdf", "wb") as f:
                f.write(uploaded.read())

            _reset_session(doc_key)
            st.session_state.last_added_urls = []

            # FIX 1: load from user-specific directory
            vs = load_vector_store(doc_key, UID)
            if vs is None:
                with st.spinner("Processing PDF…"):
                    chunks = load_and_chunk_pdf("temp_upload.pdf")
                    vs = create_vector_store(chunks, doc_key, UID)
                st.success("PDF processed & indexed!")
            else:
                st.info("Loaded existing index.")

            st.session_state.vector_store   = vs
            st.session_state.sources_loaded = [uploaded.name]
            st.session_state.messages = load_chat_history(UID, doc_key)
            st.rerun()

    # ── Add URL ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">Add Web URL</div>', unsafe_allow_html=True)
    url_val = st.text_input("", placeholder="https://...", key="url_field",
                            label_visibility="collapsed")
    if st.button("➕ Add URL", use_container_width=True):
        url = url_val.strip()
        if url:
            if url in st.session_state.last_added_urls:
                st.info("Already loaded this URL.")
            else:
                from urllib.parse import urlparse
                domain  = urlparse(url).netloc or "web_source"
                # FIX 4: stable key (no timestamp suffix) so it can be reloaded
                url_key = "url_" + domain.replace(".", "_").replace("-", "_")

                with st.spinner(f"Fetching {domain}…"):
                    try:
                        chunks = load_and_chunk_url(url)

                        if st.session_state.vector_store is None:
                            # No doc loaded yet — create fresh store and SAVE it
                            vs = create_vector_store(chunks, url_key, UID)
                            _reset_session(url_key)
                            st.session_state.vector_store = vs
                            st.session_state.messages = load_chat_history(UID, url_key)
                        else:
                            # Merge into existing store then re-save
                            st.session_state.vector_store = add_chunks_to_store(
                                st.session_state.vector_store, chunks
                            )
                            # FIX 4: persist the merged store so URLs appear in Saved Docs
                            save_vector_store(
                                st.session_state.vector_store,
                                st.session_state.current_doc,
                                UID,
                            )

                        st.session_state.last_added_urls.append(url)
                        if domain not in st.session_state.sources_loaded:
                            st.session_state.sources_loaded.append(domain)
                        st.success(f"Added {domain} ({len(chunks)} chunks)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

    # ── Active sources ────────────────────────────────────────────────────────
    if st.session_state.sources_loaded:
        st.markdown('<div class="sec-label">Active Sources</div>', unsafe_allow_html=True)
        st.markdown(
            "".join(
                f'<span class="source-pill">📄 {html_module.escape(s)}</span>'
                for s in st.session_state.sources_loaded
            ),
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Saved documents (per-user) ────────────────────────────────────────────
    st.markdown('<div class="sec-label">Saved Documents</div>', unsafe_allow_html=True)
    # FIX 1: list only THIS user's docs
    docs = list_user_docs(UID)

    if docs:
        disp    = [d.replace("_", " ") for d in docs]
        cur     = st.session_state.current_doc
        idx     = docs.index(cur) if cur in docs else 0
        sel     = st.selectbox("", disp, index=idx, label_visibility="collapsed", key="doc_sel")
        sel_key = sel.replace(" ", "_")

        if sel_key != st.session_state.current_doc:
            vs = load_vector_store(sel_key, UID)
            if vs:
                _reset_session(sel_key)
                st.session_state.vector_store = vs
                st.session_state.messages = load_chat_history(UID, sel_key)
                st.session_state.last_uploaded_pdf = None
            st.rerun()

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🗑 Delete", use_container_width=True):
                st.session_state.confirm_delete = True
        with c2:
            if st.button("↺ Refresh", use_container_width=True):
                st.rerun()

        if st.session_state.confirm_delete:
            st.warning(f"Delete **{sel}**?")
            y, n = st.columns(2)
            with y:
                if st.button("Yes", use_container_width=True):
                    # FIX 1: delete from user-specific directory
                    delete_user_doc(sel_key, UID)
                    for k in ["vector_store", "current_doc", "messages", "sources_loaded"]:
                        st.session_state[k] = _D[k]
                    st.session_state.last_uploaded_pdf = None
                    st.session_state.confirm_delete = False
                    st.rerun()
            with n:
                if st.button("No", use_container_width=True):
                    st.session_state.confirm_delete = False
    else:
        st.caption("No saved documents yet.")

    # ── Key Topics ─────────────────────────────────────────────────────────────
    if st.session_state.vector_store is not None:
        st.markdown("---")
        st.markdown('<div class="sec-label">Key Topics</div>', unsafe_allow_html=True)
        if not st.session_state.key_topics:
            if st.button("🔍 Extract Topics", use_container_width=True):
                with st.spinner("Analysing…"):
                    st.session_state.key_topics = extract_key_topics(st.session_state.vector_store)
                st.rerun()
        else:
            st.markdown(
                "".join(
                    f'<span class="topic-chip">{html_module.escape(t)}</span>'
                    for t in st.session_state.key_topics
                ),
                unsafe_allow_html=True,
            )
            if st.button("↺ Re-extract", use_container_width=True):
                st.session_state.key_topics = []
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HEADER
# ═══════════════════════════════════════════════════════════════════════════════
lh, rh = st.columns([3, 1])
with lh:
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:2px;">'
        '<span class="brand">Edify</span>'
        '<span class="brand-sub">Your AI-Powered Academic Study Assistant</span>'
        '</div>',
        unsafe_allow_html=True,
    )
with rh:
    if st.session_state.current_doc:
        st.markdown(
            f'<div style="text-align:right;padding-top:10px;">'
            f'<span class="source-pill" style="font-size:.8rem;">'
            f'📑 {st.session_state.current_doc.replace("_"," ")}</span></div>',
            unsafe_allow_html=True,
        )
st.markdown("---")

if st.session_state.vector_store is None:
    mode_txt = "dark" if st.session_state.dark_mode else "light"
    st.markdown(
        '<div style="text-align:center;padding:70px 20px;">'
        '<div style="font-size:3rem;margin-bottom:16px;"></div>'
        '<div style="font-family:\'DM Serif Display\',serif;font-size:1.6rem;'
        'color:var(--text);margin-bottom:10px;">No document loaded</div>'
        '<div style="color:var(--muted);font-size:.9rem;max-width:420px;margin:auto;">'
        'Upload a PDF or paste a URL in the sidebar to begin.</div></div>',
        unsafe_allow_html=True,
    )
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  Chat", "  Questions", "  Summary", "  History", "📊  Dashboard",
])


# ── TAB 1: CHAT ───────────────────────────────────────────────────────────────
with tab1:
    st.markdown(
        '<div class="tab-heading">Chat with your documents</div>'
        '<div class="tab-sub">Answers grounded in your content — with source citations.</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.messages:
        st.markdown(
            '<div style="text-align:center;padding:28px;color:var(--muted);">'
            '✦ Ask anything about your document…</div>',
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.messages:
            _render_msg(msg)

    st.markdown("&nbsp;", unsafe_allow_html=True)

    if st.session_state.messages:
        if st.button("🗑 Clear chat", key="clr"):
            delete_chat_history(UID, st.session_state.current_doc)
            st.session_state.messages = []
            st.rerun()

    user_input = st.chat_input("Ask a question about your document…")
    if user_input:
        from datetime import datetime
        ts = datetime.now().isoformat()
        umsg = {"role": "user", "content": user_input, "sources": [], "timestamp": ts}
        st.session_state.messages.append(umsg)
        save_chat_message(UID, st.session_state.current_doc, "user", user_input, [])

        with st.spinner("Thinking…"):
            answer, sources = generate_answer(user_input, st.session_state.vector_store)

        bmsg = {"role": "bot", "content": answer, "sources": sources,
                "timestamp": datetime.now().isoformat()}
        st.session_state.messages.append(bmsg)
        save_chat_message(UID, st.session_state.current_doc, "bot", answer, sources)
        st.rerun()


# ── TAB 2: QUESTIONS ──────────────────────────────────────────────────────────
with tab2:
    st.markdown(
        '<div class="tab-heading">Generate Practice Questions</div>'
        '<div class="tab-sub">Exam-style questions auto-generated from your document.</div>',
        unsafe_allow_html=True,
    )
    qc1, qc2, qc3 = st.columns(3)
    with qc1: q_type     = st.selectbox("Type",       ["MCQs", "Short Answer", "Long Answer"])
    with qc2: difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    with qc3: num_q      = st.selectbox("Count",      [3, 5, 8, 10])

    if st.button("⚡ Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating…"):
            qs = generate_questions(st.session_state.vector_store, difficulty, q_type, num_q)
        st.session_state.last_questions = qs
        st.session_state.question_count += num_q
        save_generation(UID, st.session_state.current_doc, "questions",
                        {"type": q_type, "difficulty": difficulty, "count": num_q}, qs)

    if st.session_state.last_questions:
        st.markdown("---")
        st.markdown(st.session_state.last_questions)
        st.download_button("⬇ Download .txt", data=st.session_state.last_questions,
                           file_name=f"questions_{difficulty}_{q_type}.txt", mime="text/plain")


# ── TAB 3: SUMMARY ────────────────────────────────────────────────────────────
with tab3:
    st.markdown(
        '<div class="tab-heading">Generate Smart Summary</div>'
        '<div class="tab-sub">Structured study notes in your preferred format.</div>',
        unsafe_allow_html=True,
    )
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        s_type = st.selectbox("Format", ["Short Summary", "Detailed Summary", "Bullet-Point Notes"])
    with sc2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        gen_s = st.button("⚡ Generate", type="primary", use_container_width=True)

    if gen_s:
        with st.spinner("Generating…"):
            sm = generate_summary(st.session_state.vector_store, s_type)
        st.session_state.last_summary = sm
        save_generation(UID, st.session_state.current_doc, "summary", {"format": s_type}, sm)

    if st.session_state.last_summary:
        st.markdown("---")
        st.markdown(st.session_state.last_summary)
        st.download_button("⬇ Download .txt", data=st.session_state.last_summary,
                           file_name=f"summary_{s_type.replace(' ','_')}.txt", mime="text/plain")


# ── TAB 4: HISTORY ────────────────────────────────────────────────────────────
with tab4:
    st.markdown(
        '<div class="tab-heading">Your Activity History</div>'
        '<div class="tab-sub">All past chats, questions, and summaries — yours only.</div>',
        unsafe_allow_html=True,
    )

    h_chat, h_qs, h_sum = st.tabs(["💬 Chat History", "❓ Questions", "📝 Summaries"])

    # ── Chat History ──────────────────────────────────────────────────────────
    with h_chat:
        chat_docs = get_all_chat_docs(UID)
        if not chat_docs:
            st.info("No chat history yet. Start chatting in the Chat tab.")
        else:
            # FIX 2: delete ALL chats button
            del_all_col, _ = st.columns([1, 3])
            with del_all_col:
                if st.button("🗑 Delete ALL chat history", key="del_all_chat",
                             help="Permanently delete every chat message in your account"):
                    delete_all_chat_history(UID)
                    st.success("All chat history deleted.")
                    st.rerun()

            st.markdown("---")
            chosen = st.selectbox("Select document", chat_docs,
                                  format_func=lambda x: x.replace("_", " "),
                                  key="hist_sel")
            hist = load_chat_history(UID, chosen)
            if hist:
                hdr, del_btn = st.columns([3, 1])
                with hdr:
                    st.caption(f"{len(hist)} messages — {chosen.replace('_', ' ')}")
                with del_btn:
                    # FIX 2: delete chat for this specific document
                    if st.button("🗑 Delete this doc's chat", key="del_doc_chat",
                                 use_container_width=True):
                        delete_chat_history(UID, chosen)
                        st.success("Deleted.")
                        st.rerun()
                for m in hist:
                    _render_msg(m)
            else:
                st.info("No messages for this document.")

    # ── Questions History ─────────────────────────────────────────────────────
    with h_qs:
        gens_q = load_generations(UID, "questions")
        if not gens_q:
            st.info("No question sets generated yet.")
        else:
            # FIX 2: delete ALL questions button
            da_col, _ = st.columns([1, 3])
            with da_col:
                if st.button("🗑 Delete ALL questions", key="del_all_qs",
                             help="Permanently delete all your generated question sets"):
                    delete_generations_by_type(UID, "questions")
                    st.success("All question history deleted.")
                    st.rerun()

            st.markdown("---")
            for g in gens_q:
                ts_s = g["timestamp"][:16].replace("T", " ")
                p    = g["params"]
                lbl  = f"❓ {p.get('type','')} · {p.get('difficulty','')} · {p.get('count','')}q"
                with st.expander(f"{lbl}  ·  {g['doc_name'].replace('_',' ')}  ·  {ts_s}"):
                    st.markdown(g["output"])
                    dl_col, del_col = st.columns([2, 1])
                    with dl_col:
                        st.download_button("⬇ Download", data=g["output"],
                                           file_name=f"questions_{g['id']}.txt",
                                           mime="text/plain", key=f"dl_q_{g['id']}")
                    with del_col:
                        # FIX 2: delete individual question set
                        if st.button("🗑 Delete", key=f"del_q_{g['id']}",
                                     use_container_width=True):
                            delete_generation(UID, g["id"])
                            st.success("Deleted.")
                            st.rerun()

    # ── Summaries History ─────────────────────────────────────────────────────
    with h_sum:
        gens_s = load_generations(UID, "summary")
        if not gens_s:
            st.info("No summaries generated yet.")
        else:
            # FIX 2: delete ALL summaries button
            ds_col, _ = st.columns([1, 3])
            with ds_col:
                if st.button("🗑 Delete ALL summaries", key="del_all_sums",
                             help="Permanently delete all your generated summaries"):
                    delete_generations_by_type(UID, "summary")
                    st.success("All summary history deleted.")
                    st.rerun()

            st.markdown("---")
            for g in gens_s:
                ts_s = g["timestamp"][:16].replace("T", " ")
                p    = g["params"]
                lbl  = f"📝 {p.get('format', 'Summary')}"
                with st.expander(f"{lbl}  ·  {g['doc_name'].replace('_',' ')}  ·  {ts_s}"):
                    st.markdown(g["output"])
                    dl_col, del_col = st.columns([2, 1])
                    with dl_col:
                        st.download_button("⬇ Download", data=g["output"],
                                           file_name=f"summary_{g['id']}.txt",
                                           mime="text/plain", key=f"dl_s_{g['id']}")
                    with del_col:
                        # FIX 2: delete individual summary
                        if st.button("🗑 Delete", key=f"del_s_{g['id']}",
                                     use_container_width=True):
                            delete_generation(UID, g["id"])
                            st.success("Deleted.")
                            st.rerun()


# ── TAB 5: DASHBOARD ──────────────────────────────────────────────────────────
with tab5:
    st.markdown(
        '<div class="tab-heading">Dashboard</div>'
        '<div class="tab-sub">Live stats for your account and current session.</div>',
        unsafe_allow_html=True,
    )

    stats = get_user_stats(UID)
    try:    cc = len(st.session_state.vector_store.index_to_docstore_id)
    except: cc = 0

    total_saved = len(list_user_docs(UID))

    for col, val, lbl in zip(
        st.columns(5),
        [stats["docs_used"], stats["chat_turns"], stats["question_sets"], stats["summaries"], cc],
        ["Docs Used", "Chat Turns", "Question Sets", "Summaries", "Chunks (Current)"]
    ):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-val">{val}</div>'
                f'<div class="metric-lbl">{lbl}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    da, db = st.columns(2)
    with da:
        st.markdown('<div class="sec-label">Current Session</div>', unsafe_allow_html=True)
        cd = st.session_state.current_doc
        st.markdown(f"**Document:** {cd.replace('_',' ') if cd else '—'}")
        st.markdown(f"**Active sources:** {len(st.session_state.sources_loaded)}")
        for s in st.session_state.sources_loaded:
            st.markdown(f"  - `{s}`")
    with db:
        st.markdown('<div class="sec-label">System Status</div>', unsafe_allow_html=True)
        if st.session_state.vector_store: st.success("✓ Vector store ready")
        else: st.error("✗ No vector store")
        st.markdown(f"**Your saved indexes:** {total_saved}")
        st.markdown(f"**Topics extracted:** {'Yes' if st.session_state.key_topics else 'No'}")
        st.markdown(f"**Theme:** {'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'}")

    st.markdown("---")
    # st.markdown('<div class="sec-label">Pipeline</div>', unsafe_allow_html=True)
    # st.code(
    #     "PDF / URL  →  Text Extraction  →  Chunking (500 tok / 50 overlap)\n"
    #     "          →  HuggingFace Embeddings (all-MiniLM-L6-v2, local)\n"
    #     "          →  FAISS Vector Store  (saved under faiss_indexes/<user_id>/)\n"
    #     "          →  Similarity Search  (top-k=4, distance < 1.5)\n"
    #     "          →  Gemini 2.0 Flash   →  Response + Source Citations\n"
    #     "          →  SQLite DB          →  Per-user Chat & Generation History",
    #     language="text",
    # )
