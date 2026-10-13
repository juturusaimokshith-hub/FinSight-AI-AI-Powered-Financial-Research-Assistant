
import hashlib
import streamlit as st

from rag_answer import (
    answer_question,
    answer_uploaded_documents,
)
from pdf_processor import extract_pdf_pages
from document_search import build_document_index


# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="FinSight AI",
    page_icon="📊",
    layout="wide",
)


# --------------------------------------------------
# 2. Initialize session state
# --------------------------------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_indexes" not in st.session_state:
    st.session_state.document_indexes = {}

if "document_errors" not in st.session_state:
    st.session_state.document_errors = {}


# --------------------------------------------------
# 3. Helper functions
# --------------------------------------------------

def get_document_key(uploaded_file):
    """Create a unique key from the filename and file contents."""
    file_bytes = uploaded_file.getvalue()
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    return f"{uploaded_file.name}_{content_hash}"


def process_uploaded_pdf(uploaded_file):
    """Extract and index an uploaded PDF only when necessary."""
    document_key = get_document_key(uploaded_file)

    if document_key in st.session_state.document_indexes:
        return document_key

    file_bytes = uploaded_file.getvalue()

    try:
        with st.spinner(f"Processing {uploaded_file.name}..."):
            pages = extract_pdf_pages(file_bytes)

            if not pages:
                raise ValueError(
                    "No extractable text was found. "
                    "The PDF may be scanned or image-only and may require OCR."
                )

            document_index = build_document_index(
                pages,
                document_name=uploaded_file.name,
            )

            # Ensure metadata is available for citations.
            document_index["document_name"] = uploaded_file.name

            st.session_state.document_indexes[
                document_key
            ] = document_index

            st.session_state.document_errors.pop(
                document_key, None
            )

        return document_key

    except Exception as error:
        st.session_state.document_errors[
            document_key
        ] = str(error)
        return None


def display_sources(sources):
    """Display retrieved sources without assuming optional keys exist."""
    st.markdown("### Retrieved Report Sources")

    if not sources:
        st.info("No sources were retrieved.")
        return

    for position, source in enumerate(sources, start=1):
        if not isinstance(source, dict):
            continue

        # Use a fallback when the retrieval function omits rank.
        rank = source.get("rank", position)

        source_name = source.get(
            "document_name",
            source.get(
                "filename",
                "STL Annual Report 2024–25",
            ),
        )

        page = source.get("pdf_page", "Unknown")
        chunk_id = source.get("chunk_id", "N/A")
        score = source.get("score")
        source_text = source.get("text", "No source text available.")

        title = (
            f"Source {rank} — {source_name} — "
            f"PDF page {page} (Chunk {chunk_id})"
        )

        with st.expander(title):
            if isinstance(score, (int, float)):
                st.write(f"**Retrieval score:** {score:.4f}")

            st.text(source_text)


def display_citation_validation(citation):
    """Display citation validation without assuming a specific status."""
    st.markdown("### Citation Validation")

    status = citation.get("status", "NOT_CHECKED")
    message = citation.get(
        "message",
        "No citation validation details are available.",
    )

    if status == "PASS":
        st.success(f"{status}: {message}")
    elif status in ("FAIL", "ERROR"):
        st.error(f"{status}: {message}")
    elif status in ("WARNING", "REVIEW"):
        st.warning(f"{status}: {message}")
    else:
        st.info(f"{status}: {message}")


def display_factual_validation(factual):
    """Display factual validation separately from citation validation."""
    st.markdown("### Factual Support Validation")

    status = factual.get("status", "NOT_CHECKED")
    message = factual.get(
        "message",
        "No factual validation details are available.",
    )

    if status == "PASS":
        st.success("Overall result: PASS")
    elif status == "FAIL":
        st.error("Overall result: FAIL")
    elif status in ("REVIEW", "WARNING"):
        st.warning(f"Overall result: {status}")
    else:
        st.info(f"Validation status: {status}")

    st.text(message)


# --------------------------------------------------
# 4. Main page
# --------------------------------------------------

st.title("📊 FinSight AI")
st.subheader("Evidence-Grounded Financial Research Assistant")

st.write(
    "Research STL's Annual Report 2024–25 or upload multiple "
    "PDF reports to ask questions across your documents."
)


# --------------------------------------------------
# 5. Sidebar and PDF uploads
# --------------------------------------------------

with st.sidebar:
    st.header("About FinSight AI")

    st.write("AI-powered financial report research.")

    st.markdown("**Technologies**")
    st.write(
        "Python · Streamlit · LangChain · Groq · FAISS · "
        "Sentence Transformers · BM25 · RRF"
    )

    st.divider()

    st.header("📄 Upload PDF Reports")

    uploaded_files = st.file_uploader(
        "Select one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload text-based financial reports for research.",
        key="report_uploader",
    )

    active_document_keys = []

    if uploaded_files:
        for uploaded_file in uploaded_files:
            document_key = process_uploaded_pdf(uploaded_file)

            if document_key is not None:
                active_document_keys.append(document_key)
                st.success(f"✓ {uploaded_file.name}")
            else:
                error_message = st.session_state.document_errors.get(
                    get_document_key(uploaded_file),
                    "Unknown processing error.",
                )
                st.error(
                    f"{uploaded_file.name}: {error_message}"
                )

    st.divider()

    research_mode = st.radio(
        "Choose research source",
        [
            "STL Annual Report 2024–25",
            "Uploaded PDF Reports",
        ],
        index=0,
    )

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    if st.button(
        "Clear Uploaded PDF Indexes",
        use_container_width=True,
    ):
        st.session_state.document_indexes = {}
        st.session_state.document_errors = {}
        st.rerun()


# --------------------------------------------------
# 6. Determine active research documents
# --------------------------------------------------

# Only the files currently selected in the uploader are searched.
# This prevents unrelated, previously uploaded PDFs from being used.
active_indexes = [
    st.session_state.document_indexes[key]
    for key in active_document_keys
    if key in st.session_state.document_indexes
]


if research_mode == "STL Annual Report 2024–25":
    st.info(
        "Researching the built-in STL Annual Report 2024–25."
    )
else:
    if active_indexes:
        st.info(
            f"{len(active_indexes)} uploaded PDF report(s) "
            "are ready for research."
        )
    else:
        st.warning(
            "Upload at least one successfully processed PDF "
            "using the sidebar before researching uploaded reports."
        )


# --------------------------------------------------
# 7. Question input
# --------------------------------------------------

question = st.text_input(
    "Enter your research question",
    placeholder=(
        "Example: Compare FY 2024–25 revenue for TCS and STL."
    ),
    key="question_input",
)


# --------------------------------------------------
# 8. Run research
# --------------------------------------------------

if st.button("Research", type="primary"):
    if not question.strip():
        st.warning("Please enter a research question.")

    elif (
        research_mode == "Uploaded PDF Reports"
        and not active_indexes
    ):
        st.warning(
            "Upload and process at least one PDF before "
            "researching uploaded documents."
        )

    else:
        try:
            with st.spinner(
                "Searching documents and validating the answer..."
            ):
                if research_mode == "STL Annual Report 2024–25":
                    result = answer_question(question.strip())

                else:
                    result = answer_uploaded_documents(
                        question.strip(),
                        active_indexes,
                    )

            # Protect the UI if a function returns an unexpected object.
            if not isinstance(result, dict):
                raise TypeError(
                    "The answer function did not return a dictionary."
                )

            result.setdefault(
                "answer",
                "No answer was generated.",
            )
            result.setdefault("sources", [])
            result.setdefault(
                "citation_validation",
                {
                    "status": "NOT_CHECKED",
                    "message": "Citation validation was not returned.",
                },
            )
            result.setdefault(
                "factual_validation",
                {
                    "status": "NOT_CHECKED",
                    "message": "Factual validation was not returned.",
                },
            )

            st.session_state.chat_history.append({
                "question": question.strip(),
                "research_mode": research_mode,
                "result": result,
            })

            st.rerun()

        except Exception as error:
            st.error(f"Research failed: {error}")
            st.caption(
                "Check the VS Code terminal for the complete traceback."
            )


# --------------------------------------------------
# 9. Display research history
# --------------------------------------------------

st.markdown("---")
st.header("Research History")

if not st.session_state.chat_history:
    st.info(
        "Your questions and research results will appear here."
    )

else:
    history = list(reversed(st.session_state.chat_history))

    for position, item in enumerate(history, start=1):
        result = item.get("result", {})

        with st.expander(
            f"Q{len(history) - position + 1}: "
            f"{item.get('question', 'Unknown question')}",
            expanded=(position == 1),
        ):
            st.caption(
                "Research source: "
                + item.get(
                    "research_mode",
                    "STL Annual Report 2024–25",
                )
            )

            st.markdown("### Research Answer")
            st.markdown(
                result.get("answer", "No answer was generated.")
            )

            citation = result.get(
                "citation_validation",
                {
                    "status": "NOT_CHECKED",
                    "message": "No citation validation result.",
                },
            )
            display_citation_validation(citation)

            factual = result.get(
                "factual_validation",
                {
                    "status": "NOT_CHECKED",
                    "message": "No factual validation result.",
                },
            )
            display_factual_validation(factual)

            overall_status = result.get(
                "overall_status",
                "NOT_CHECKED",
            )
            st.markdown(
                f"**Combined validation status:** {overall_status}"
            )

            display_sources(result.get("sources", []))
