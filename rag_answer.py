
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from hybrid_search import hybrid_search
from document_search import search_document

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=api_key,
)


def normalize_filename(filename):
    """Normalize filenames before comparing them."""
    name = Path(str(filename)).name.strip().strip("'\"`")
    name = re.sub(r"\s+", "", name)
    return name.casefold()


def validate_answer(question, answer, context):
    """Ask the LLM to assess claims against retrieved evidence."""
    prompt = f"""
You are a strict fact-checker for a financial research assistant.

Check every important claim against the supplied excerpts.
Check figures, dates, units, comparisons, and citations.
Do not assume a claim is supported just because a page was retrieved.
If evidence is insufficient, mark the claim UNCERTAIN.
Do not invent evidence.

For each important claim, state:
CLAIM, STATUS (SUPPORTED/UNSUPPORTED/UNCERTAIN), and REASON.

Finish with exactly one:
OVERALL: PASS
OVERALL: REVIEW
OVERALL: FAIL

PASS means all claims are supported.
REVIEW means at least one claim is uncertain and none are unsupported.
FAIL means at least one claim is unsupported.

QUESTION:
{question}

REPORT EXCERPTS:
{context}

ANSWER TO CHECK:
{answer}
"""
    response = llm.invoke(prompt)
    return str(response.content).strip()


def factual_validation_result(question, answer, context):
    try:
        message = validate_answer(question, answer, context)
        match = re.search(
            r"OVERALL\s*:\s*(PASS|REVIEW|FAIL)",
            message,
            flags=re.IGNORECASE,
        )
        status = match.group(1).upper() if match else "REVIEW"
        return {"status": status, "message": message}
    except Exception as error:
        return {
            "status": "ERROR",
            "message": f"Factual validation failed: {error}",
        }


def validate_uploaded_citations(answer, sources):
    """
    Check cited source numbers, page numbers and filenames against
    the retrieved sources. This is a consistency check, not proof
    that the cited passage supports the claim.
    """
    if not sources:
        return {
            "status": "NOT_CHECKED",
            "message": "No retrieved sources are available.",
        }

    # Match [Source 1, PDF page 70, filename.pdf]
    pattern = re.compile(
        r"\[\s*Source\s+(\d+)\s*,\s*"
        r"PDF\s+page\s+(\d+)\s*,\s*([^\]]+?)\s*\]",
        re.IGNORECASE,
    )

    matches = list(pattern.finditer(answer))

    if not matches:
        return {
            "status": "WARNING",
            "message": (
                "No citations in the expected format were found. "
                "Expected: [Source 1, PDF page 70, filename.pdf]."
            ),
        }

    problems = []

    for match in matches:
        source_number = int(match.group(1))
        cited_page = int(match.group(2))
        cited_filename = match.group(3).strip()

        if source_number < 1 or source_number > len(sources):
            problems.append(
                f"Source {source_number} does not exist in retrieved sources."
            )
            continue

        source = sources[source_number - 1]
        expected_page = source.get("pdf_page")
        expected_filename = source.get(
            "document_name",
            source.get("filename", ""),
        )

        if str(cited_page) != str(expected_page):
            problems.append(
                f"Source {source_number}: cited page {cited_page}, "
                f"but retrieved page is {expected_page}."
            )

        if expected_filename and (
            normalize_filename(cited_filename)
            != normalize_filename(expected_filename)
        ):
            problems.append(
                f"Source {source_number}: filename mismatch. "
                f"Cited '{cited_filename}', expected '{expected_filename}'."
            )

    if problems:
        return {
            "status": "FAIL",
            "message": "Citation validation failed:\n- "
            + "\n- ".join(problems),
        }

    return {
        "status": "PASS",
        "message": (
            "Cited source numbers, page numbers and filenames match "
            "the retrieved source metadata. Claim-level support still "
            "requires factual review."
        ),
    }


def empty_result(message):
    return {
        "answer": message,
        "sources": [],
        "citation_validation": {
            "status": "NOT_CHECKED",
            "message": "No sources were available to check.",
        },
        "factual_validation": {
            "status": "NOT_CHECKED",
            "message": "No evidence was available for factual validation.",
        },
        "overall_status": "NOT_CHECKED",
    }


def answer_question(question):
    """Answer questions using the built-in STL annual report."""
    results = hybrid_search(
        question,
        top_k=6,
        candidate_k=20,
    )

    if not results:
        return empty_result(
            "No relevant STL report excerpts were retrieved. "
            "Try rephrasing your question."
        )

    sources = []

    for rank, item in enumerate(results, start=1):
        sources.append({
            "rank": rank,
            "chunk_id": item.get("chunk_id", "N/A"),
            "pdf_page": item.get("pdf_page", "Unknown"),
            "score": item.get("score", 0.0),
            "text": item.get("text", ""),
            "document_name": "STL Annual Report 2024–25",
        })

    context_parts = []

    for source in sources:
        context_parts.append(
            f"[Source {source['rank']}: "
            f"{source['document_name']}, "
            f"PDF page {source['pdf_page']}]\n"
            f"{source['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are FinSight AI, a financial-report research assistant.

Answer using ONLY the supplied excerpts.
Do not invent facts, figures, dates, or explanations.
If evidence is insufficient, say so.
Distinguish report facts from calculations.
Put a citation immediately after every factual claim.

Use this citation format:
[STL Annual Report 2024-25, PDF p. 19]

QUESTION:
{question}

REPORT EXCERPTS:
{context}

ANSWER:
"""

    try:
        answer = str(llm.invoke(prompt).content).strip()
    except Exception as error:
        return empty_result(f"Answer generation failed: {error}")

    page_pattern = re.compile(
        r"PDF\s*(?:page|pages|p\.?|pp\.?)\s*"
        r"(\d+(?:\s*[-–,]\s*\d+)*)",
        re.IGNORECASE,
    )

    cited_pages = []
    for match in page_pattern.findall(answer):
        cited_pages.extend(re.findall(r"\d+", match))

    retrieved_pages = {str(s["pdf_page"]) for s in sources}
    invalid_pages = [
        page for page in cited_pages if page not in retrieved_pages
    ]

    if not cited_pages:
        citation_validation = {
            "status": "WARNING",
            "message": "No PDF page citations were detected.",
        }
    elif invalid_pages:
        citation_validation = {
            "status": "WARNING",
            "message": (
                "Cited pages were not among the retrieved pages: "
                + ", ".join(sorted(set(invalid_pages)))
            ),
        }
    else:
        citation_validation = {
            "status": "PASS",
            "message": (
                "Cited page numbers are present among retrieved sources. "
                "This does not prove claim-level support."
            ),
        }

    factual_validation = factual_validation_result(
        question, answer, context
    )

    overall_status = combine_validation_status(
        citation_validation, factual_validation
    )

    return {
        "answer": answer,
        "sources": sources,
        "citation_validation": citation_validation,
        "factual_validation": factual_validation,
        "overall_status": overall_status,
    }


def normalize_document_indexes(document_indexes):
    """Accept either a list of indexes or a single index dictionary."""
    if isinstance(document_indexes, dict):
        document_indexes = [document_indexes]

    normalized = []

    for item in document_indexes or []:
        if not isinstance(item, dict):
            continue

        # Support session-state entries containing a nested index.
        index = item.get("document_index", item)

        if (
            isinstance(index, dict)
            and all(key in index for key in ("chunks", "index", "bm25"))
        ):
            normalized.append(index)

    return normalized


def answer_uploaded_documents(question, document_indexes):
    """Answer a question using one or more uploaded PDF indexes."""
    indexes = normalize_document_indexes(document_indexes)

    if not indexes:
        return empty_result(
            "No indexed PDF reports are available. Upload a PDF and "
            "wait for indexing to finish."
        )

    all_results = []

    for document_index in indexes:
        try:
            results = search_document(
                question,
                document_index,
                top_k=4,
                candidate_k=15,
            )
            all_results.extend(results)
        except Exception as error:
            print(f"Document search warning: {error}")

    if not all_results:
        return empty_result(
            "I couldn't find relevant information in the uploaded reports."
        )

    all_results.sort(
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )

    # Avoid duplicate excerpts from the same document and page.
    selected = []
    seen = set()

    for item in all_results:
        key = (
            normalize_filename(item.get("document_name", "")),
            str(item.get("pdf_page", "")),
            re.sub(r"\s+", " ", item.get("text", "")).strip().casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) == 6:
            break

    sources = []

    for rank, item in enumerate(selected, start=1):
        sources.append({
            "rank": rank,
            "chunk_id": item.get("chunk_id", "N/A"),
            "pdf_page": item.get("pdf_page", "Unknown"),
            "score": float(item.get("score", 0.0)),
            "text": item.get("text", ""),
            "document_name": item.get(
                "document_name",
                item.get("filename", "Unknown PDF"),
            ),
        })

    context = "\n\n".join(
        f"[Source {source['rank']}: {source['document_name']}, "
        f"PDF page {source['pdf_page']}]\n{source['text']}"
        for source in sources
    )

    prompt = f"""
You are FinSight AI, a financial-report research assistant.

Answer the question using ONLY the supplied document excerpts.
Do not invent facts or use outside knowledge to fill gaps.
If evidence is insufficient, say so clearly.
Distinguish reported figures from your own arithmetic.
Put citations immediately after factual claims.

Use exactly this citation format:
[Source 1, PDF page 70, annual-report.pdf]

Use the exact source number, filename and PDF page from the labels.
For comparisons, ensure the figures use comparable reporting periods,
units and accounting definitions. If they do not, explain the limitation.

QUESTION:
{question}

DOCUMENT EXCERPTS:
{context}

ANSWER:
"""

    try:
        answer = str(llm.invoke(prompt).content).strip()
    except Exception as error:
        return empty_result(f"Answer generation failed: {error}")

    citation_validation = validate_uploaded_citations(answer, sources)
    factual_validation = factual_validation_result(
        question, answer, context
    )

    overall_status = combine_validation_status(
        citation_validation, factual_validation
    )

    return {
        "answer": answer,
        "sources": sources,
        "citation_validation": citation_validation,
        "factual_validation": factual_validation,
        "overall_status": overall_status,
    }


def combine_validation_status(citation_validation, factual_validation):
    """Combine results without treating a validation error as a pass."""
    citation_status = citation_validation.get("status", "NOT_CHECKED")
    factual_status = factual_validation.get("status", "NOT_CHECKED")

    if "FAIL" in (citation_status, factual_status):
        return "FAIL"

    if "ERROR" in (citation_status, factual_status):
        return "REVIEW"

    if factual_status != "PASS" or citation_status not in ("PASS",):
        return "REVIEW"

    return "PASS"


if __name__ == "__main__":
    question = input("Ask a question about STL: ")
    result = answer_question(question)

    print("\n--- Answer ---\n")
    print(result["answer"])

    print("\n--- Citation Validation ---")
    print(result["citation_validation"]["message"])

    print("\n--- Factual Validation ---")
    print(result["factual_validation"]["message"])

    print("\n--- Retrieved Sources ---")
    for index, source in enumerate(result["sources"], start=1):
        print(
            f"Source {source.get('rank', index)} | "
            f"PDF page {source.get('pdf_page', 'Unknown')} | "
            f"Chunk {source.get('chunk_id', 'N/A')} | "
            f"Score {source.get('score', 0.0):.4f} | "
            f"{source.get('document_name', 'STL Annual Report')}"
        )
        print(source.get("text", ""))
        print("-" * 60)
