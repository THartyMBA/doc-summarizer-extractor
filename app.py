# doc_summarizer_extractor.py
"""
Document Summarization & Keyword Extraction Pipeline  📄🔑
────────────────────────────────────────────────────────────────────────────
Upload a PDF or DOCX, automatically:

1. Splits the document into ~500-word sections.  
2. Summarizes each section into 3 bullet points via a free OpenRouter LLM.  
3. Extracts the top 20 keywords across all sections using TF-IDF.  
4. Displays section summaries and a keyword table.  
5. Lets you download summaries and keywords as CSVs.

*Proof-of-concept only*—no production ingestion or compliance.
For enterprise NLP pipelines, [contact me](https://drtomharty.com/bio).
"""

import os, io, requests
import streamlit as st
import pandas as pd

import PyPDF2
import docx

from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────── OpenRouter summarizer ───────────────────────────
API_KEY = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""
MODEL   = "mistralai/mistral-7b-instruct:free"

def openrouter_summarize(text: str, temperature=0.3) -> str:
    """Summarize text via OpenRouter LLM into 3 bullet points."""
    if not API_KEY:
        raise RuntimeError("Please set OPENROUTER_API_KEY in secrets or env")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = (
        "Summarize the following text in **3 concise bullet points**:\n\n"
        f"{text}"
    )
    body = {
        "model": MODEL,
        "messages": [
            {"role":"system","content":"You are a helpful summarization assistant."},
            {"role":"user","content":prompt}
        ],
        "temperature": temperature,
    }
    r = requests.post(url, headers=headers, json=body, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ────────────────────────── Document text extraction ──────────────────────────
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PyPDF2.PdfReader(uploaded_file)
    text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text.append(page_text)
    return "\n".join(text)

def extract_text_from_docx(uploaded_file) -> str:
    doc = docx.Document(uploaded_file)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

# ───────────────────────────── Text splitting ────────────────────────────────
def split_into_chunks(text: str, words_per_chunk=500) -> list[str]:
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]

# ─────────────────────────────────── UI ───────────────────────────────────────
st.set_page_config(page_title="Doc Summarizer & Keywords", layout="wide")
st.title("📄 Document Summarization & Keyword Extraction")

st.info(
    "🔔 **Demo Notice**  \n"
    "This is a lightweight proof-of-concept. For production NLP pipelines, "
    "[contact me](https://drtomharty.com/bio).",
    icon="💡"
)

uploaded = st.file_uploader(
    "Upload a PDF or DOCX document", type=["pdf","docx"]
)
if not uploaded:
    st.stop()

# Extract raw text
file_ext = uploaded.name.lower().split(".")[-1]
if file_ext == "pdf":
    raw_text = extract_text_from_pdf(uploaded)
else:
    raw_text = extract_text_from_docx(uploaded)

if not raw_text:
    st.error("No extractable text found in the document.")
    st.stop()

# Split into chunks
chunks = split_into_chunks(raw_text, words_per_chunk=500)
st.subheader(f"Document split into {len(chunks)} sections")

# Summarize & extract keywords
if st.button("🚀 Summarize & Extract Keywords"):
    summaries = []
    with st.spinner("Summarizing sections…"):
        for i, chunk in enumerate(chunks, start=1):
            summary = openrouter_summarize(chunk)
            summaries.append((i, summary))

    # Keyword extraction across all chunks
    vect = TfidfVectorizer(stop_words="english", max_features=50)
    tfidf_matrix = vect.fit_transform(chunks)
    scores = tfidf_matrix.mean(axis=0).A1
    terms = vect.get_feature_names_out()
    kw_scores = sorted(zip(terms, scores), key=lambda x: x[1], reverse=True)[:20]
    kw_df = pd.DataFrame(kw_scores, columns=["keyword","score"])

    # Display summaries
    st.subheader("🔖 Section Summaries")
    for idx, text in summaries:
        st.markdown(f"**Section {idx}**")
        st.markdown(text)

    # Display keywords
    st.subheader("🔑 Top Keywords Across Document")
    st.dataframe(kw_df)

    # Download buttons
    sum_df = pd.DataFrame({
        "section": [s[0] for s in summaries],
        "summary": [s[1] for s in summaries]
    })
    st.download_button(
        "⬇️ Download summaries CSV",
        data=sum_df.to_csv(index=False).encode(),
        file_name="section_summaries.csv",
        mime="text/csv"
    )
    st.download_button(
        "⬇️ Download keywords CSV",
        data=kw_df.to_csv(index=False).encode(),
        file_name="top_keywords.csv",
        mime="text/csv"
    )
