# 📊 FinSight AI — Evidence-Grounded Financial Research Assistant

FinSight AI is an AI-powered financial research application that helps users explore annual reports and other financial PDF documents. It retrieves relevant evidence, generates answers using a large language model, displays source passages, and checks citations and factual support.

## 🚀 Features

- **PDF Research:** Ask questions about STL's Annual Report 2024–25.
- **Multi-PDF Upload:** Upload multiple financial reports and research their contents.
- **Hybrid Search:** Combines semantic search using FAISS and Sentence Transformers with keyword search using BM25.
- **Reciprocal Rank Fusion (RRF):** Combines semantic and keyword search rankings to improve evidence retrieval.
- **RAG-Based Answers:** Generates responses using retrieved document excerpts.
- **Source Citations:** Displays retrieved report passages with page numbers.
- **Citation Validation:** Checks whether cited source metadata matches retrieved evidence.
- **Factual Support Validation:** Uses an LLM to assess whether claims are supported by the retrieved excerpts.
- **Interactive Interface:** Built using Streamlit, with research history and validation results.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| Streamlit | Frontend and user interface |
| LangChain | LLM integration |
| Groq API | Language model inference |
| Sentence Transformers | Semantic embeddings |
| FAISS | Vector similarity search |
| BM25 | Keyword-based retrieval |
| Reciprocal Rank Fusion | Hybrid result ranking |
| PyMuPDF | PDF text extraction |
| python-dotenv | Environment variable management |

## 🏗️ Architecture

```text
User
 |
 v
Streamlit Frontend (app.py)
 |
 +--> Upload PDF reports
 |
 +--> Enter research question
          |
          v
    PDF Processing
    (pdf_processor.py)
          |
          v
    Document Indexing
    (document_search.py)
          |
          v
    Hybrid Retrieval
    FAISS + BM25 + RRF
          |
          v
    Relevant Evidence
          |
          v
    Groq LLM
    (rag_answer.py)
          |
          v
    Citation Validation
    + Factual Support Validation
          |
          v
    Answer + Sources + Status
          |
          v
    Streamlit Interface
```

The built-in STL annual-report workflow uses the hybrid-search module. Uploaded documents are processed and indexed separately.

## 📁 Project Structure

```text
FinSight-AI/
├── app.py
├── rag_answer.py
├── hybrid_search.py
├── document_search.py
├── pdf_processor.py
├── chunk_report.py
├── extract_all_pages.py
├── extract_pages_with_metadata.py
├── extract_pdf.py
├── search_report.py
├── semantic_search.py
├── test_groq.py
├── data/
│   └── stl_pages.json
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

The extraction and search scripts are development utilities. The Streamlit application uses the core modules to serve the main research workflow.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/FinSight-AI.git
cd FinSight-AI
```

Replace `YOUR-USERNAME` with your GitHub username and adjust the repository name if necessary.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get an API key from the Groq Console: https://console.groq.com/

Never commit `.env` or expose API keys in source code.

### 5. Prepare the data

Place the required annual-report data at the paths expected by the application. The built-in hybrid-search workflow currently expects:

```text
data/stl_pages.json
```

You can generate page-level extracted text and metadata using the provided extraction utility, after configuring any required OCR dependencies.

If you distribute the annual report, ensure you have permission to do so. Otherwise, provide instructions for users to supply their own PDF.

### 6. Run the application

```bash
streamlit run app.py
```

Streamlit will display the local application URL in the terminal.

## 🔎 How It Works

1. The user selects the built-in report or uploads PDF reports.
2. The application extracts readable text and preserves page metadata.
3. The document is split into smaller chunks.
4. Sentence Transformers converts chunks into embeddings.
5. FAISS retrieves semantically similar passages, while BM25 retrieves keyword matches.
6. Reciprocal Rank Fusion combines the rankings.
7. The Groq-hosted LLM generates an answer from the retrieved evidence.
8. The application checks citation consistency and assesses factual support.
9. The final answer, retrieved sources, and validation results appear in the interface.

## ⚠️ Limitations

- The quality of answers depends on the relevance and accuracy of retrieved evidence.
- Scanned PDFs may require OCR; the current uploaded-PDF processing path extracts embedded text and does not automatically OCR image-only pages.
- Citation validation does not, by itself, prove that a claim is factually supported.
- LLM-based factual validation is an additional check, not a guarantee of correctness.
- The application does not automatically provide live stock prices or independently verify current market data.
- The built-in report workflow requires the expected report data files to be available.

## 🔐 Security

- Keep API keys in environment variables.
- Never commit `.env`, API credentials, virtual environments, or sensitive documents.
- Review document licensing before publishing reports.
- Treat generated financial research as informational rather than personalized investment advice.

## 🎯 Project Objective

The goal of FinSight AI is to make financial-report analysis more accessible by combining document retrieval, language-model reasoning, source citations, and validation in a single research interface.

## 👨‍💻 Future Improvements

- Add support for scanned PDFs using OCR.
- Add automated tests for retrieval, citations, and answer validation.
- Separate frontend and backend modules into dedicated packages.
- Add persistent vector indexes for faster repeated searches.
- Add a FastAPI backend with a separate web frontend.
- Add authorized live financial-data integrations.

---

**Built with Python, Streamlit, FAISS, Sentence Transformers, BM25, LangChain, and Groq.**
