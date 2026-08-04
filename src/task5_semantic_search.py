"""
Task 5 — Semantic Search Module (Dense Retrieval + HyDE).

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4

Hai chế độ retrieval:
    1. semantic_search()      — Standard dense retrieval (cosine similarity)
    2. semantic_search_hyde() — HyDE: Hypothetical Document Embedding (bonus +5đ)

============================================================
HyDE hoạt động thế nào:
============================================================
Vấn đề của dense retrieval thông thường:
    Query ngắn ("hoàn tiền shopee") có phân phối ngữ nghĩa RẤT KHÁC
    với document dài ("Quy trình hoàn tiền: Người mua cần gửi yêu cầu...").
    Vector của query và document nằm ở vùng khác nhau trong embedding space
    → cosine similarity thấp dù nội dung liên quan.

Giải pháp HyDE (Gao et al., 2022):
    1. Dùng LLM sinh ra một "hypothetical document" — đoạn văn GIẢ ĐỊNH
       như thể đã là câu trả lời cho query
    2. Embed hypothetical document thay vì query gốc
    3. Vector của hypothetical document gần hơn với document thật trong corpus
       vì cùng phân phối ngôn ngữ (đều là văn bản dài, cùng domain)
    4. Query ChromaDB bằng vector đó → precision cao hơn

Ví dụ:
    Query gốc:          "hoàn tiền shopee"
    Hypothetical doc:   "Theo chính sách của Shopee, người mua có thể yêu cầu
                         hoàn tiền trong vòng 15 ngày kể từ ngày nhận hàng.
                         Quy trình hoàn tiền bao gồm: gửi đơn yêu cầu, ..."
    → Vector của hypothetical doc gần với các chunk policy thật hơn

Trade-off:
    - Tốt hơn với query ngắn, mơ hồ
    - Chậm hơn (thêm 1 lần gọi LLM)
    - Nếu LLM hallucinate quá xa → vector nhiễu → kém hơn standard search
    → Dùng HyDE khi query ngắn (<5 từ) hoặc khi standard search trả về score thấp
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Import configuration từ Task 4
import sys
sys.path.append(str(Path(__file__).parent))
from task4_chunking_indexing import (
    EMBEDDING_MODEL,
    COLLECTION_NAME,
    CHROMA_DIR,
    EMBEDDING_DIM,
)

# ------------------------------------------------------------------ #
# Cấu hình LLM cho HyDE                                               #
# ------------------------------------------------------------------ #
# Đọc từ .env — cùng thứ tự ưu tiên với Task 10:
#   OPENROUTER_API_KEY (ưu tiên, có model free)
#   OPENAI_API_KEY     (fallback)
#   GEMINI_API_KEY     (fallback)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Model dùng để sinh hypothetical document (nhẹ, đủ dùng)
HYDE_LLM_MODEL = os.getenv("HYDE_LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

load_dotenv()


# ------------------------------------------------------------------ #
# Helpers — Collection & Embedding Model                              #
# ------------------------------------------------------------------ #

def get_collection():
    """Lấy ChromaDB collection đã được tạo trong Task 4."""
    import chromadb

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"ChromaDB directory not found at {CHROMA_DIR}. "
            "Please run task4_chunking_indexing.py first to create the index."
        )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection


def _embed_text(text: str) -> list[float]:
    """Embed một đoạn text thành vector float list.

    Strategy:
      - If `OPENAI_API_KEY` is set, use OpenAI embeddings (same as Task 4).
      - Otherwise, fall back to a deterministic local embedding (bag-of-words
        hashing) with dimension `EMBEDDING_DIM` so tests can run offline.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
            emb = response.data[0].embedding
            return emb
        except Exception as e:
            print(f"[embed] OpenAI embedding failed: {e}. Falling back to local embed.")

    # Fallback: deterministic local embedding (hash-based) using EMBEDDING_DIM
    try:
        dim = int(EMBEDDING_DIM)
    except Exception:
        dim = 1536
    vec = [0.0] * dim
    for token in str(text).split():
        idx = abs(hash(token)) % dim
        vec[idx] += 1.0
    # L2-normalize
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def _query_chroma(vector: list[float], top_k: int) -> list[dict]:
    """
    Query ChromaDB với vector đã cho, trả về danh sách kết quả đã format.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    collection = get_collection()
    results = collection.query(
        query_embeddings=[vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and len(results["documents"][0]) > 0:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB hnsw:cosine trả về cosine distance ∈ [0, 2]
            # → similarity = max(0, 1 - distance)  ∈ [0, 1]
            score = max(0.0, 1.0 - float(dist))
            output.append({
                "content": doc,
                "score": round(score, 4),
                "metadata": meta,
            })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


# ------------------------------------------------------------------ #
# HyDE — Hypothetical Document Generation                             #
# ------------------------------------------------------------------ #

def _generate_hypothetical_document(query: str) -> str:
    """
    Dùng LLM sinh ra một đoạn văn GIẢ ĐỊNH như thể đã là câu trả lời
    cho query — dùng làm embedding input trong HyDE.

    Thứ tự ưu tiên provider:
        1. OpenRouter (có model free, ưu tiên dùng cho lab)
        2. OpenAI
        3. Gemini

    Nếu không có API key nào → raise ValueError để caller fallback về
    standard search.

    Args:
        query: Câu truy vấn gốc của người dùng

    Returns:
        Hypothetical document string (100-200 từ)
    """
    system_prompt = (
        "Bạn là trợ lý hỗ trợ khách hàng thương mại điện tử Shopee Vietnam. "
        "Hãy viết một đoạn văn ngắn (100-150 từ) bằng tiếng Việt, "
        "giống như một đoạn trích từ tài liệu chính sách hoặc hướng dẫn, "
        "trả lời trực tiếp câu hỏi sau. "
        "QUAN TRỌNG: Chỉ trả về đoạn văn, không thêm giải thích hay tiêu đề."
    )
    user_prompt = f"Câu hỏi: {query}"

    # ---- Provider 1: OpenRouter ----
    if OPENROUTER_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )
            response = client.chat.completions.create(
                model=HYDE_LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.3,  # Thấp → ít hallucinate, bám sát domain
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[HyDE] OpenRouter failed: {e}. Trying next provider...")

    # ---- Provider 2: OpenAI ----
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=300,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[HyDE] OpenAI failed: {e}. Trying next provider...")

    # ---- Provider 3: Gemini ----
    if GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config={"max_output_tokens": 300, "temperature": 0.3},
            )
            return response.text.strip()
        except Exception as e:
            print(f"[HyDE] Gemini failed: {e}.")

    raise ValueError(
        "HyDE requires an LLM API key. "
        "Set OPENROUTER_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in .env"
    )


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa chuẩn sử dụng cosine similarity (dense retrieval).

    Pipeline:
        query string
            → embed bằng BAAI/bge-m3
            → query ChromaDB (hnsw:cosine)
            → chuyển đổi distance → similarity (1 - dist)
            → sort descending
            → top_k results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity ∈ [0, 1]
            'metadata': dict     # source, type, chunk_index
        }
        Sorted by score descending.
    """
    query_vector = _embed_text(query)
    return _query_chroma(query_vector, top_k)


def semantic_search_hyde(
    query: str,
    top_k: int = 10,
    fallback_on_error: bool = True,
) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng HyDE (Hypothetical Document Embedding).

    Tham khảo: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without Relevance Labels"
    https://arxiv.org/abs/2212.10496

    Pipeline:
        query string
            → LLM sinh hypothetical document (100-150 từ tiếng Việt)
            → embed hypothetical document bằng BAAI/bge-m3
            → query ChromaDB (hnsw:cosine) với vector của hypothetical doc
            → chuyển đổi distance → similarity
            → sort descending
            → top_k results

    Khi nào HyDE tốt hơn standard search:
        - Query ngắn, keyword-style: "hoàn tiền", "giao hàng trễ"
        - Query mơ hồ thiếu context: "làm sao để được hoàn?"
        - Query không khớp ngôn ngữ corpus (query tiếng Anh, corpus tiếng Việt)

    Khi nào standard search tốt hơn:
        - Query dài, đã rõ nghĩa: "Tôi muốn biết quy định về trả hàng trong 15 ngày"
        - LLM bị rate limit hoặc không có API key

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa
        fallback_on_error: Nếu True, tự động fallback về standard search
                           khi LLM gọi thất bại (mất kết nối, hết quota)

    Returns:
        List of {
            'content': str,
            'score': float,      # Cosine similarity ∈ [0, 1]
            'metadata': dict,
            'hyde_used': bool    # True nếu dùng HyDE, False nếu fallback
        }
        Sorted by score descending.
    """
    try:
        # Bước 1: Sinh hypothetical document bằng LLM
        hypothetical_doc = _generate_hypothetical_document(query)
        print(f"[HyDE] Hypothetical doc ({len(hypothetical_doc)} chars): "
              f"{hypothetical_doc[:80]}...")

        # Bước 2: Embed hypothetical document (không phải query gốc)
        hyde_vector = _embed_text(hypothetical_doc)

        # Bước 3: Query ChromaDB với vector hypothetical doc
        results = _query_chroma(hyde_vector, top_k)

        # Đánh dấu kết quả được trả về bằng HyDE
        for r in results:
            r["hyde_used"] = True
        return results

    except Exception as e:
        if fallback_on_error:
            print(f"[HyDE] Failed ({e}). Falling back to standard semantic search.")
            results = semantic_search(query, top_k)
            for r in results:
                r["hyde_used"] = False
            return results
        raise


def semantic_search_with_hyde_fusion(
    query: str,
    top_k: int = 10,
    hyde_weight: float = 0.6,
) -> list[dict]:
    """
    Kết hợp standard search + HyDE bằng score fusion (nâng cao).

    Lý do fusion tốt hơn dùng một mình:
        - Standard search tốt khi query đã rõ nghĩa
        - HyDE tốt khi query mơ hồ / ngắn
        - Fusion lấy điểm mạnh của cả hai, giảm thiểu trường hợp xấu

    Cách fusion:
        final_score = hyde_weight × hyde_score + (1 - hyde_weight) × standard_score

    Args:
        query:       Câu truy vấn
        top_k:       Số kết quả trả về
        hyde_weight: Trọng số cho HyDE score (0.0 → chỉ standard, 1.0 → chỉ HyDE)
                     Mặc định 0.6 vì HyDE thường tốt hơn với corpus tiếng Việt

    Returns:
        List of {'content', 'score', 'metadata', 'hyde_score', 'standard_score'}
        Sorted by fused score descending.
    """
    # Chạy song song (trong Python đơn luồng vẫn sequential nhưng code rõ ý định)
    standard_results = semantic_search(query, top_k=top_k * 2)

    try:
        hyde_results = semantic_search_hyde(query, top_k=top_k * 2, fallback_on_error=False)
    except Exception:
        print("[HyDE Fusion] HyDE unavailable, returning standard results only.")
        return standard_results[:top_k]

    # Build lookup dict: content → score (dedup bằng content)
    standard_map: dict[str, float] = {r["content"]: r["score"] for r in standard_results}
    hyde_map: dict[str, float] = {r["content"]: r["score"] for r in hyde_results}
    meta_map: dict[str, dict] = {r["content"]: r["metadata"] for r in standard_results}
    meta_map.update({r["content"]: r["metadata"] for r in hyde_results})

    # Merge tất cả unique content
    all_contents = set(standard_map.keys()) | set(hyde_map.keys())

    fused = []
    for content in all_contents:
        s_score = standard_map.get(content, 0.0)
        h_score = hyde_map.get(content, 0.0)
        fused_score = hyde_weight * h_score + (1 - hyde_weight) * s_score
        fused.append({
            "content": content,
            "score": round(fused_score, 4),
            "metadata": meta_map[content],
            "hyde_score": round(h_score, 4),
            "standard_score": round(s_score, 4),
        })

    fused.sort(key=lambda x: x["score"], reverse=True)
    return fused[:top_k]


# ------------------------------------------------------------------ #
# Test / __main__                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    test_queries = [
        "quy định trả hàng hoàn tiền shopee",
        "phương thức thanh toán",
        "chính sách bảo mật dữ liệu cá nhân",
    ]

    # ---- Standard Semantic Search ----
    print("=" * 60)
    print("Standard Semantic Search (cosine similarity)")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = semantic_search(query, top_k=3)
        if not results:
            print("  Không tìm thấy kết quả.")
        else:
            for i, r in enumerate(results, 1):
                source = r["metadata"].get("source", "unknown")
                print(f"  [{i}] score={r['score']:.4f}  src={source}")
                print(f"       {r['content'][:100]}...")

    # ---- HyDE Semantic Search ----
    print("\n" + "=" * 60)
    print("HyDE Semantic Search (Hypothetical Document Embedding)")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = semantic_search_hyde(query, top_k=3, fallback_on_error=True)
        if not results:
            print("  Không tìm thấy kết quả.")
        else:
            hyde_used = results[0].get("hyde_used", False)
            mode = "HyDE" if hyde_used else "standard (fallback)"
            print(f"  Mode: {mode}")
            for i, r in enumerate(results, 1):
                source = r["metadata"].get("source", "unknown")
                print(f"  [{i}] score={r['score']:.4f}  src={source}")
                print(f"       {r['content'][:100]}...")
