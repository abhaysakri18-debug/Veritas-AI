# Veritas AI 🔬

**Evidence-first research intelligence using hybrid retrieval and LLMs.** A master's-level Generative AI project prototype with research investigations, claim verification, contradiction analysis, document-concept knowledge graphs, and inspectable citations.

## Features
- Upload and index PDF, DOCX, and TXT research documents.
- Persistent ChromaDB vector store using local MiniLM embeddings.
- Hybrid semantic + BM25 retrieval fused with reciprocal rank fusion (RRF).
- Research investigation, claim verification, contradiction checks, and citation-linked evidence.
- Knowledge graph linking retrieved documents to model/metric terms.
- Retrieval inspection and Markdown report export.
- Gemini API (default) with live available-model selector, or local Ollama generation.

## Windows / VS Code setup

```powershell
cd Veritas_AI
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

For **Gemini (default)**, edit `.env` and set `GEMINI_API_KEY`. The sidebar lists models accessible to your API key; choose one there. `GEMINI_MODEL` is only a preferred default. If it is unavailable, choose another model from the list. Model availability and API quotas can change. The key is never required in source code.

For **Ollama**, install Ollama and run `ollama pull gemma3:4b`, then select Ollama in the sidebar.

```powershell
python -m streamlit run app.py
```

The first indexing or search request downloads the `all-MiniLM-L6-v2` embedding model. Internet access is needed for initial download and for Gemini.

## How to demo
1. Upload at least two papers containing related but distinct experiments.
2. Index them in **Library**.
3. Ask for a cross-paper comparison in **Investigation**.
4. Check a claim in **Verification** and inspect evidence.
5. Compare seemingly conflicting findings in **Contradictions**.
6. Explore **Knowledge Graph** and **Evaluation**.

## Architecture

`Document upload → extraction → overlapping chunks → MiniLM embeddings → ChromaDB → hybrid BM25+dense retrieval → RRF → task-specific evidence prompt → Gemini/Ollama → cited report`

## Research scope and limitations
- This is a functional **multi-stage evidence-first prototype**, not a production multi-agent LangGraph system. Separate tabs implement research roles; there is no autonomous agent orchestration.
- Verification and contradiction judgments are LLM-generated hypotheses, **not validated fact-checks**. Human review is essential.
- Source citations refer to retrieved chunks and can be checked in the evidence explorer. They do not guarantee claims are correct.
- Knowledge graph edges indicate term co-occurrence in retrieved text, not scientific causal relationships.
- Evaluation page is a retrieval inspection workspace, not a benchmark with measured accuracy. Create manually labeled evaluation queries before reporting quantitative metrics.
- Scanned/image-only PDFs need OCR (not included).
- Avoid indexing sensitive data into third-party LLM prompts without authorization.
- Large collections use in-memory BM25 indexing at query time and may be slow; production use needs a dedicated keyword index.

## Tests

```powershell
python -m pytest -q
```

## Security
`.env`, virtual environments, and `storage/` are excluded from Git. Never commit secrets. Review external API privacy requirements before uploading private documents.

## Upgrade from an earlier Veritas AI version
Unzip this project into a new folder, or replace `app.py`, `backend/core.py`, `requirements.txt`, `.env.example`, and `README.md` in your existing folder. **Keep your existing `.env` and `storage/`** to preserve your key and indexed documents. Change any old `GEMINI_MODEL=gemini-2.0-flash` line in your existing `.env` to `GEMINI_MODEL=gemini-2.5-flash` (or another model shown in the app's available-model dropdown). Reinstall dependencies with `python -m pip install -r requirements.txt`, then restart Streamlit. Never upload `.env` to GitHub.
