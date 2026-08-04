"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement — khuyến nghị vì không cần API key

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.

Lưu ý quan trọng về RRF (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164 (k=60),
bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng điểm RRF để
quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""

import os
import math
from typing import Optional


# =============================================================================
# Helpers
# =============================================================================

def _cosine_sim(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Tính cosine similarity giữa hai vector.

    cos(a, b) = (a · b) / (|a| × |b|)

    Trả về 0.0 nếu một trong hai vector là zero vector.
    Không dùng numpy để giữ dependency tối thiểu — hàm này chỉ
    được gọi trong rerank_mmr() với số lượng vector nhỏ (top_k).
    """
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# =============================================================================
# Method 1 — Cross-Encoder Reranker
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model qua Jina Reranker API.

    Cross-encoder khác bi-encoder (dense retrieval) ở chỗ:
        - Bi-encoder: embed query và document RIÊNG BIỆT → cosine similarity
          (nhanh, dùng được với ChromaDB)
        - Cross-encoder: nhận (query, document) cùng lúc → 1 scalar score
          (chính xác hơn nhưng không scale được cho full corpus — dùng để rerank)

    Model: jina-reranker-v2-base-multilingual
        - Hỗ trợ tiếng Việt tốt (multilingual)
        - API free tier: 1M tokens/tháng

    Fallback: Nếu không có JINA_API_KEY, tự rank lại dựa trên score gốc
    (giữ nguyên thứ tự, trả về top_k).

    Args:
        query:      Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k:      Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import requests

    jina_api_key = os.getenv("JINA_API_KEY", "")

    if not jina_api_key:
        # Fallback: trả về top_k theo score gốc (không gọi API)
        print("[cross_encoder] JINA_API_KEY không tìm thấy — dùng score gốc.")
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:top_k]

    documents = [c["content"] for c in candidates]

    try:
        response = requests.post(
            "https://api.jina.ai/v1/rerank",
            headers={
                "Authorization": f"Bearer {jina_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "documents": documents,
                "top_n": top_k,
            },
            timeout=15,
        )
        response.raise_for_status()
        reranked = response.json()["results"]

        return [
            {**candidates[r["index"]], "score": round(r["relevance_score"], 4)}
            for r in reranked
        ]

    except Exception as e:
        print(f"[cross_encoder] Jina API lỗi: {e} — fallback về score gốc.")
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:top_k]


# =============================================================================
# Method 2 — MMR (Maximal Marginal Relevance)
# =============================================================================

def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Công thức:
        MMR(d) = λ × sim(query, d)  -  (1-λ) × max_{s ∈ Selected} sim(d, s)

    Cơ chế:
        - Vòng lặp greedy: mỗi bước chọn document có MMR score cao nhất
          trong danh sách còn lại.
        - Thành phần thứ nhất (λ × relevance): ưu tiên doc liên quan đến query.
        - Thành phần thứ hai ((1-λ) × max_sim_to_selected): "phạt" doc quá
          giống với những doc đã chọn → tăng diversity.
        - λ = 1.0 → chỉ quan tâm relevance (giống top-k thuần)
          λ = 0.0 → chỉ quan tâm diversity (chọn doc khác nhau nhất)
          λ = 0.7 (mặc định) → thiên về relevance nhưng vẫn giảm trùng lặp

    Yêu cầu: candidates phải có trường "embedding" (vector float list).
    Nếu không có → fallback về score gốc.

    Args:
        query_embedding: Vector embedding của query (cùng dimension với candidates)
        candidates:      List of {'content': str, 'score': float,
                                  'embedding': list[float], 'metadata': dict}
        top_k:           Số lượng kết quả
        lambda_param:    Trade-off relevance (1.0) ↔ diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR, sorted by selection order
        (document được chọn đầu tiên = relevant nhất & diverse nhất).
    """
    if not candidates:
        return []

    # Kiểm tra candidates có embedding không
    if "embedding" not in candidates[0]:
        print("[MMR] candidates thiếu 'embedding' — fallback về score gốc.")
        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        return sorted_candidates[:top_k]

    selected_indices: list[int] = []
    remaining_indices: list[int] = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining_indices:
            # Thành phần 1: relevance giữa candidate và query
            relevance = _cosine_sim(query_embedding, candidates[idx]["embedding"])

            # Thành phần 2: max similarity với các doc đã chọn
            max_sim_to_selected = 0.0
            for sel_idx in selected_indices:
                sim = _cosine_sim(
                    candidates[idx]["embedding"],
                    candidates[sel_idx]["embedding"],
                )
                max_sim_to_selected = max(max_sim_to_selected, sim)

            # MMR score
            mmr_score = (
                lambda_param * relevance
                - (1 - lambda_param) * max_sim_to_selected
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

    # Gán MMR score vào kết quả (dùng thứ tự lựa chọn làm score tương đối)
    results = []
    for rank, idx in enumerate(selected_indices):
        item = candidates[idx].copy()
        # Score giảm dần theo thứ tự MMR selection (rank 0 = quan trọng nhất)
        item["score"] = round(1.0 - rank / max(len(selected_indices), 1), 4)
        results.append(item)

    return results


# =============================================================================
# Method 3 — RRF (Reciprocal Rank Fusion)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    Công thức:
        RRF(d) = Σ_r  1 / (k + rank_r(d))

    Cơ chế:
        - Mỗi document nhận điểm từ từng ranker theo công thức 1/(k+rank).
          rank bắt đầu từ 1 (document đầu tiên = rank 1).
        - k=60 (Cormack et al. 2009): làm mượt — tránh document ở rank 1
          của một ranker thống trị quá mức; khoảng cách điểm giữa rank 1
          và rank 2 bị thu hẹp: 1/61 vs 1/62 thay vì 1/1 vs 1/2.
        - Document xuất hiện ở cả semantic lẫn lexical list sẽ tích lũy
          điểm từ cả hai → tự nhiên được boost lên đầu.
        - Document chỉ xuất hiện ở 1 list → điểm thấp hơn dù rank cao
          trong list đó.

    Lưu ý quan trọng (Task 9):
        Điểm RRF KHÔNG phản ánh độ tương đồng thực. Top-1 luôn ≈ 1/(k+1)
        ≈ 0.016 bất kể query có liên quan hay không → đừng dùng làm
        score_threshold để quyết định fallback.

    Args:
        ranked_lists: List of ranked result lists — mỗi list từ 1 ranker
                      (ví dụ: [semantic_results, lexical_results])
        top_k:        Số lượng kết quả cuối cùng
        k:            Smoothing constant (default=60, Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}  # content → tổng RRF score
    content_map: dict[str, dict] = {}  # content → full dict (giữ metadata gốc)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ bản metadata đầu tiên khi document xuất hiện ở nhiều list
            if key not in content_map:
                content_map[key] = item

    # Sắp xếp theo RRF score giảm dần
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = round(score, 6)  # ghi đè bằng điểm RRF đã fuse
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "rrf",  # "cross_encoder" | "mmr" | "rrf"
    ranked_lists: Optional[list[list[dict]]] = None,
    query_embedding: Optional[list[float]] = None,
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query:           Câu truy vấn
        candidates:      Danh sách candidates từ retrieval
                         (dùng cho cross_encoder và rrf đơn list)
        top_k:           Số lượng kết quả sau rerank
        method:          Phương pháp: "cross_encoder" | "mmr" | "rrf"
        ranked_lists:    [method="rrf"] Truyền nhiều ranked lists để fuse
                         (ví dụ: [semantic_results, lexical_results]).
                         Nếu None → dùng candidates như 1 list duy nhất.
        query_embedding: [method="mmr"] Vector embedding của query.
                         Nếu None → fallback về score gốc trong rerank_mmr.

    Returns:
        List of top_k reranked candidates, sorted by score descending.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        # Nếu không có query_embedding, dùng score gốc làm proxy relevance
        emb = query_embedding if query_embedding is not None else []
        return rerank_mmr(emb, candidates, top_k)

    elif method == "rrf":
        # Nếu không truyền ranked_lists, wrap candidates thành 1 list.
        # Trong Task 9 nên truyền [semantic_results, lexical_results].
        lists = ranked_lists if ranked_lists is not None else [candidates]
        return rerank_rrf(lists, top_k=top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}")


# =============================================================================
# __main__ — smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Test rerank_rrf (2 ranked lists)")
    print("=" * 60)

    semantic_results = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.85, "metadata": {"source": "returns.md"}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.72, "metadata": {"source": "payment.md"}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.55, "metadata": {"source": "seller.md"}},
    ]
    lexical_results = [
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.9, "metadata": {"source": "seller.md"}},
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.6, "metadata": {"source": "returns.md"}},
        {"content": "Chính sách bảo mật thông tin cá nhân người dùng", "score": 0.4, "metadata": {"source": "privacy.md"}},
    ]

    rrf_results = rerank_rrf([semantic_results, lexical_results], top_k=4)
    print("RRF fused results:")
    for r in rrf_results:
        print(f"  [{r['score']:.6f}] {r['content'][:60]}")

    print("\n" + "=" * 60)
    print("Test rerank() unified interface — method='rrf'")
    print("=" * 60)

    combined = semantic_results + [
        {"content": "Chính sách bảo mật thông tin cá nhân người dùng", "score": 0.4, "metadata": {"source": "privacy.md"}}
    ]
    results = rerank("chính sách trả hàng shopee", combined, top_k=3)
    for r in results:
        print(f"  [{r['score']:.6f}] {r['content'][:60]}")

    print("\n" + "=" * 60)
    print("Test rerank_cross_encoder (fallback khi không có API key)")
    print("=" * 60)

    ce_results = rerank("hoàn tiền shopee", semantic_results, top_k=2, method="cross_encoder")
    for r in ce_results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}")

    print("\n" + "=" * 60)
    print("Test rerank_mmr (không có embedding → fallback score gốc)")
    print("=" * 60)

    mmr_results = rerank("hoàn tiền", semantic_results, top_k=2, method="mmr")
    for r in mmr_results:
        print(f"  [{r['score']:.4f}] {r['content'][:60]}")
