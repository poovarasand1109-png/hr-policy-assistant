# Zyro HR Policy Assistant

Streamlit RAG application using PyMuPDF, LangChain text splitting (1000 characters / 200 overlap), Hugging Face `sentence-transformers/all-MiniLM-L6-v2`, FAISS similarity search (top 5), and Gemini 2.5 Flash.

## Setup
1. Put approved HR policy PDFs in `data/`.
2. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`.
3. Add your Gemini key to `secrets.toml`. Never commit this file or share the key.
4. In a terminal in this folder, run:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Push the project to GitHub, open https://share.streamlit.io/, create an app from the repository, select `app.py`, and add this under App settings → Secrets:

```toml
GEMINI_API_KEY = "your-key"
```

Then deploy.

## Important
This is a standalone implementation based on the described HR RAG design, not a verified extraction of every cell in your Kaggle notebook. Compare its answers with your notebook before using it as the final demonstration. The first run downloads the embedding model and builds the FAISS index in memory; an app restart may rebuild it. Do not publish confidential HR policy PDFs in a public repository.
