"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

# pyright: reportMissingImports=false

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
PAGEINDEX_WORKSPACE = Path(__file__).parent.parent / "data" / ".pageindex_workspace"


def _get_pageindex_client():
    """
    Tạo PageIndex client dùng chung cho upload và truy vấn.
    """
    from pageindex import PageIndexClient

    # PageIndex Python SDK hiện tại chỉ nhận api_key.
    # Workspace local của bài lab vẫn được dùng riêng để lưu manifest.
    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_indexed_docs_manifest() -> dict[str, str]:
    """
    Lưu các file đã được index để tránh re-index lặp lại trên mỗi lần chạy.
    """
    manifest_path = PAGEINDEX_WORKSPACE / "indexed_documents.json"
    if not manifest_path.exists():
        return {}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    # Tương thích manifest cũ dạng list: chưa có doc_id nên cần upload lại.
    if isinstance(payload, list):
        return {}
    return {str(path): str(doc_id) for path, doc_id in payload.items()}


def _save_indexed_docs_manifest(indexed_files: dict[str, str]) -> None:
    manifest_path = PAGEINDEX_WORKSPACE / "indexed_documents.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(indexed_files, f, ensure_ascii=False, indent=2)


def _extract_json_payload(text: str):
    """
    Lấy JSON object/array đầu tiên từ output của model.

    PageIndex chat có thể trả về cả lời dẫn và JSON, nên cần tách phần JSON ra
    trước khi parse.
    """
    fenced_match = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidate = fenced_match.group(1).strip() if fenced_match else text.strip()

    start_candidates = [idx for idx in (candidate.find("{"), candidate.find("[")) if idx != -1]
    if start_candidates:
        start = min(start_candidates)
        end = max(candidate.rfind("}"), candidate.rfind("]"))
        if end >= start:
            candidate = candidate[start:end + 1].strip()

    return json.loads(candidate)


def _coerce_result(item: dict, score: float, doc_name: str) -> dict:
    content = item.get("content") or item.get("relevant_content") or item.get("text") or ""
    metadata = {
        "source": doc_name,
    }

    if "page" in item:
        metadata["page"] = item.get("page")
    if "section_title" in item:
        metadata["section_title"] = item.get("section_title")
    if "node_id" in item:
        metadata["node_id"] = item.get("node_id")
    if "title" in item:
        metadata["title"] = item.get("title")

    return {
        "content": content,
        "score": round(score, 4),
        "metadata": metadata,
        "source": "pageindex",
    }


def _search_single_document(client, doc_id: str, query: str, top_k: int, doc_rank: int) -> list[dict]:
    """
    Chạy một truy vấn trên từng document đã index và parse response thành các
    đoạn nội dung liên quan.
    """
    doc_info = client.get_document(doc_id)
    if isinstance(doc_info, str):
        try:
            doc_info = json.loads(doc_info)
        except json.JSONDecodeError:
            doc_info = {}

    if doc_info.get("status") not in {None, "completed"}:
        return []

    doc_name = doc_info.get("name") or doc_info.get("doc_name") or doc_id

    retrieval_prompt = f"""
Your job is to retrieve the raw relevant content from the document based on the user's query.

Query: {query}

Return in JSON format:
```json
[
  {{
    "page": <number>,
    "section_title": "<section title if available>",
    "content": "<raw text>"
  }},
  ...
]
```

Rules:
- Return only the JSON array.
- Use short raw excerpts from the document.
- Prefer the most relevant sections first.
""".strip()

    try:
        streamed_chunks = []
        for chunk in client.chat_completions(
            messages=[{"role": "user", "content": retrieval_prompt}],
            doc_id=doc_id,
            stream=True,
        ):
            streamed_chunks.append(chunk)

        response_text = "".join(streamed_chunks).strip()
        if not response_text:
            return []

        parsed = _extract_json_payload(response_text)
        if isinstance(parsed, dict):
            parsed = parsed.get("results") or parsed.get("items") or parsed.get("nodes") or []
        if not isinstance(parsed, list):
            return []

        results = []
        for item_rank, item in enumerate(parsed[:top_k], start=1):
            if not isinstance(item, dict):
                continue
            score = 1.0 / (doc_rank + item_rank)
            result = _coerce_result(item, score, doc_name)
            if result["content"]:
                results.append(result)
        return results

    except Exception as exc:
        print(f"  ⚠ PageIndex query failed for {doc_name}: {exc}")
        return []


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        return {}

    client = _get_pageindex_client()
    PAGEINDEX_WORKSPACE.mkdir(parents=True, exist_ok=True)

    indexed_files = _load_indexed_docs_manifest()
    uploaded: dict[str, str] = {}

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        md_path = str(md_file.resolve())
        if md_path in indexed_files:
            continue

        try:
            response = client.submit_document(md_path)
            doc_id = response.get("doc_id") if isinstance(response, dict) else None
            if not doc_id:
                raise RuntimeError(f"PageIndex không trả về doc_id: {response}")
            uploaded[md_path] = doc_id
            indexed_files[md_path] = doc_id
            print(f"  ✓ Indexed: {md_file.relative_to(STANDARDIZED_DIR)} -> {doc_id}")
        except Exception as exc:
            print(f"  ✗ Failed to index {md_file.name}: {exc}")

    if uploaded:
        _save_indexed_docs_manifest(indexed_files)

    return uploaded


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        return []

    client = _get_pageindex_client()
    PAGEINDEX_WORKSPACE.mkdir(parents=True, exist_ok=True)

    # Auto-index nếu chưa có dữ liệu local. Điều này giúp task chạy được ngay
    # sau khi cài dependencies mà không cần thao tác tay.
    indexed_files = _load_indexed_docs_manifest()
    if not indexed_files:
        upload_documents()
        indexed_files = _load_indexed_docs_manifest()

    if not indexed_files:
        return []

    all_results: list[dict] = []
    for doc_rank, doc_id in enumerate(indexed_files.values(), start=1):
        doc_results = _search_single_document(client, doc_id, query, top_k, doc_rank)
        all_results.extend(doc_results)

    # Deduplicate theo nội dung để giảm lặp giữa các document/section tương tự.
    deduped: list[dict] = []
    seen_contents: set[str] = set()
    for item in sorted(all_results, key=lambda row: row["score"], reverse=True):
        content_key = item["content"].strip()
        if not content_key or content_key in seen_contents:
            continue
        seen_contents.add(content_key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    return deduped


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
