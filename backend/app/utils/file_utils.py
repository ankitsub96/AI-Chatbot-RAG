import os
import hashlib
import json


def get_pdf_name(filename: str):
    return os.path.splitext(filename)[0]


def get_index_path(filename: str):
    pdf_name = get_pdf_name(filename)
    return f"app/vector_store/indexes/{pdf_name}.index"


def get_metadata_path(filename: str):
    pdf_name = get_pdf_name(filename)
    return f"app/vector_store/metadata/{pdf_name}.json"


def calculate_checksum(file_bytes: bytes) -> str:
    """Stable SHA256 checksum for deduplication."""
    return hashlib.sha256(file_bytes).hexdigest()


# backward compat alias — remove once all callers are migrated
generate_file_hash = calculate_checksum


def load_documents(metadata_path: str):
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_file(path: str, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_file(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_text_file(path: str, content: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_bm25_path(filename: str) -> str:
    return os.path.join(
        "app/vector_store/bm25",
        f"{os.path.splitext(filename)[0]}.pkl",
    )
