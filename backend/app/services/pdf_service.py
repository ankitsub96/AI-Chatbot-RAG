import os

os.environ["UNSTRUCTURED_SKIP_TORCH"] = "1"

from unstructured.partition.pdf import partition_pdf

# from docling.document_converter import DocumentConverter
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.base_models import InputFormat
# from docling.document_converter import PdfFormatOption
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging

from app.config.settings import TOP_K, CHUNK_SIZE
from app.utils.helpers import timer

logging.getLogger("pdfminer").setLevel(logging.ERROR)
# os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("docling").setLevel(logging.ERROR)


# =========================
# CHUNKING CONFIG
# =========================

CHUNK_SIZE = 5000
CHUNK_OVERLAP = 800
# merge multiple pages before chunking
PAGE_WINDOW_SIZE = 2

# =========================================================
# SPLITTER
# =========================================================

"""
IMPORTANT:

Do NOT split aggressively by sentences.

Sentence splitting creates:
- tiny chunks
- weak semantic continuity
- too many embeddings
- poor retrieval for long-form content

This splitter prioritizes:
1. paragraphs
2. new lines
3. spaces

Works well for:
- books
- documentation
- reports
- PDFs
- conversational text
"""
text_splitter = RecursiveCharacterTextSplitter(
    # separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
    # separators=["\n\n", "\n", "\t", " ", ""],
    separators=[
        "\n\n",
        "\n",
        "\t",
    ],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

table_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " | ", " ", ""],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

code_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ";", "{", "}", " ", ""],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=50,
)

markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["## ", "# ", "\n\n", "\n", ". ", " ", ""],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

# =========================================================
# PAGE GROUPING
# =========================================================


@timer
def group_pages(
    pages: list[dict],
    window_size: int = PAGE_WINDOW_SIZE,
):
    """
    Merge nearby pages together before chunking.

    WHY:
    Important context often spans multiple pages.

    Benefits:
    - better semantic continuity
    - fewer chunks
    - faster embeddings
    - smaller FAISS indexes
    - better retrieval quality
    """

    grouped_pages = []

    for start in range(0, len(pages), window_size):

        batch = pages[start : start + window_size]

        combined_text = "\n\n".join(page["text"] for page in batch).strip()
        combined_text = clean_text(combined_text)
        grouped_pages.append(
            {
                "page": batch[0]["page"],
                "text": combined_text,
            }
        )

    print(f"\nGrouped {len(pages)} pages into " f"{len(grouped_pages)} page groups")

    return grouped_pages


# =========================================================
# CHUNKING
# =========================================================


@timer
def chunk_pages(grouped_pages: list[dict]):
    """
    Split grouped pages into semantic chunks.
    """

    print(f"\nChunking {len(grouped_pages)} grouped pages...")

    documents = []

    for page_group in grouped_pages:

        chunks = text_splitter.split_text(page_group["text"])

        for chunk in chunks:
            chunk = clean_text(chunk)
            documents.append(
                {
                    "page": page_group["page"],
                    "text": chunk,
                }
            )

    print(f"Total chunks created: {len(documents)}")

    return documents


# =========================================================
# COMPLETE DOCUMENT PROCESSING PIPELINE
# =========================================================


@timer
def process_document_pages(pages: list[dict]):
    """
    Full document chunking pipeline.

    FLOW:

    Extracted Pages
        ↓
    Group Nearby Pages
        ↓
    Recursive Chunking
        ↓
    Final Chunks
    """

    print("\n" + "=" * 80)
    print("DOCUMENT CHUNKING PIPELINE")
    print("=" * 80)

    grouped_pages = group_pages(pages)

    documents = chunk_pages(grouped_pages)

    print("\nChunking pipeline complete")

    print(
        {
            "original_pages": len(pages),
            "grouped_pages": len(grouped_pages),
            "final_chunks": len(documents),
        }
    )

    print("=" * 80)

    return documents


def clean_text(text: str) -> str:
    if not text:
        return ""

    return text.replace("\x00", "").replace("\u0000", "").strip()  # ❗ critical fix


# =========================
# PDF EXTRACTION
# =========================


@timer
def extract_pdf_text(pdf_path: str) -> list[dict]:

    print("\nExtracting PDF with Unstructured (fast mode)...")

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="fast",
        include_page_breaks=True,
    )
    print("elements:", len(elements))

    extracted = []
    current_page = 1
    last_section = ""

    for el in elements:

        el_type = type(el).__name__

        if el_type == "PageBreak":
            current_page += 1
            continue

        text = el.text.strip() if el.text else ""

        if not text:
            continue

        if el_type in ("Title", "Header"):
            last_section = text

        extracted.append(
            {
                "page": current_page,
                "type": el_type,
                "section": last_section,
                "text": text,
            }
        )

    print(f"Extracted {len(extracted)} elements across {current_page} pages")

    return extracted


# =========================
# PDF EXTRACTION
# same output format as unstructured version:
# [{"page": int, "type": str, "section": str, "text": str}]
# =========================

# =========================================================
# DOCLING TYPE → UNSTRUCTURED TYPE MAPPING
# keeps chunk_text routing identical to unstructured version
# =========================================================

# DOCLING_TYPE_MAP = {
#     "section_header": "Title",
#     "page_header": "Header",
#     "page_footer": "Footer",
#     "table": "Table",
#     "code": "CodeSnippet",
#     "list_item": "ListItem",
#     "text": "NarrativeText",
#     "paragraph": "NarrativeText",
#     "caption": "NarrativeText",
#     "footnote": "NarrativeText",
#     "formula": "NarrativeText",
#     "picture": None,  # skip images
#     "page_break": None,  # skip page breaks
# }

# one converter instance — reused across calls

# _pipeline_options = PdfPipelineOptions()
# _pipeline_options.do_ocr = False  # no OCR
# _pipeline_options.do_table_structure = False  # no ML table detection

# _converter = DocumentConverter(
#     format_options={
#         InputFormat.PDF: PdfFormatOption(
#             pipeline_options=_pipeline_options,
#         )
#     }
# )


# def extract_pdf_text(pdf_path: str) -> list[dict]:

#     print("\nExtracting PDF with Docling...")

#     result = _converter.convert(pdf_path)

#     extracted = []
#     last_section = ""

#     for item, level in result.document.iterate_items():

#         # get docling label
#         label = getattr(item, "label", None)
#         if label is None:
#             continue

#         label_str = label.value if hasattr(label, "value") else str(label)

#         # map to unstructured-compatible type
#         el_type = DOCLING_TYPE_MAP.get(label_str)

#         # skip images, page breaks, footers, unknowns
#         if el_type is None or el_type == "Footer":
#             continue

#         # get text
#         text = item.text.strip() if hasattr(item, "text") and item.text else ""

#         if not text:
#             continue

#         # get page number
#         page = 1
#         if hasattr(item, "prov") and item.prov:
#             page = item.prov[0].page_no

#         # track last section title
#         if el_type in ("Title", "Header"):
#             last_section = text

#         extracted.append(
#             {
#                 "page": page,
#                 "type": el_type,
#                 "section": last_section,
#                 "text": text,
#             }
#         )

#     print(f"Extracted {len(extracted)} elements")

#     return extracted
