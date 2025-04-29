# doc-summarizer-extractor

📄 Document Summarization & Keyword Extraction Pipeline
A Streamlit proof-of-concept that turns any PDF or DOCX into concise section summaries and extracts top keywords—no complex setup needed.

Demo only—no production ingestion, compliance checks, or scaling.
For enterprise NLP pipelines with robust ETL, versioning, and privacy controls, contact me.

🔍 What it does
Upload a PDF or DOCX.

Extract raw text from each page or paragraph.

Split into ~500-word chunks.

Summarize each chunk into 3 bullet points via a free OpenRouter LLM.

Extract top 20 keywords across the entire document using TF-IDF.

Display section summaries and keyword table.

Download both summaries and keyword lists as CSVs.

✨ Key Features
Language-agnostic: works on any PDF or Word document.

Chunked processing: keeps prompts under token limits.

LLM summarization: leverages OpenRouter’s free models.

Keyword extraction: TF-IDF highlights the most salient terms.

Single-file app: all logic in doc_summarizer_extractor.py.

Downloadable outputs: CSVs for summaries and keywords.

🔑 Secrets
Add your OpenRouter API key so the summarizer can call the LLM.

Streamlit Community Cloud
Deploy this repo → ⋯ → Edit secrets.

Add:

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
Local development
Create ~/.streamlit/secrets.toml:

toml
Copy
Edit
OPENROUTER_API_KEY = "sk-or-xxxxxxxxxxxxxxxx"
—or—

bash
Copy
Edit
export OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxx
🚀 Quick Start (Local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/doc-summarizer-extractor.git
cd doc-summarizer-extractor
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run doc_summarizer_extractor.py
Navigate to http://localhost:8501.

Upload your document.

Click Summarize & Extract Keywords.

Review and download your CSV outputs.

☁️ Deploy on Streamlit Cloud (Free)
Push this repo (public or private) to GitHub under THartyMBA.

Go to streamlit.io/cloud → New app → select your repo & branch → Deploy.

Add OPENROUTER_API_KEY in Secrets—no other config needed.

🛠️ Requirements
shell
Copy
Edit
streamlit>=1.32
requests
PyPDF2
python-docx
scikit-learn
pandas
🗂️ Repo Structure
vbnet
Copy
Edit
doc-summarizer-extractor/
├─ doc_summarizer_extractor.py   ← single-file Streamlit app  
├─ requirements.txt  
└─ README.md                      ← you’re reading it  
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid Python UIs

OpenRouter – free LLM gateway

PyPDF2 & python-docx – document parsing

scikit-learn – TF-IDF vectorization

Pandas – data handling

Summarize long docs and surface key terms in seconds! 🎉
