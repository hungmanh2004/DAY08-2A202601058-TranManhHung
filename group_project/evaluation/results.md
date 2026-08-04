# RAG Evaluation Results

## Framework sử dụng

> **RAGAS v0.4.1** — chuẩn industry cho RAG evaluation.
> Đánh giá trên **15 câu hỏi** về chính sách Shopee Việt Nam với 4 metrics chính.

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ (A − B) |
|--------|:--------------------------:|:---------------------:|:---------:|
| Faithfulness | **0.853** | 0.844 | +0.009 ✅ |
| Answer Relevance | **0.703** | 0.694 | +0.008 ✅ |
| Context Recall | 0.867 | **0.900** | −0.033 ⚠️ |
| Context Precision | **0.962** | 0.949 | +0.013 ✅ |
| **Average** | **0.846** | 0.847 | −0.001 |

---

## A/B Comparison Analysis

**Config A — Hybrid Search + Reranking:**
> Pipeline đầy đủ: semantic search (dense) + BM25 (sparse) → RRF merge → reranking → generation.
> Top-k = 5 chunks đưa vào LLM.

**Config B — Dense-Only (không reranking):**
> Chỉ dùng semantic search (ChromaDB cosine similarity), bỏ qua BM25 và bước reranking.
> Top-k = 5 chunks đưa vào LLM.

**Kết luận:**

Config A và Config B cho kết quả **gần như tương đương** (trung bình 0.846 vs 0.847, chênh 0.001).

**Điểm đáng chú ý:**
- Config A **tốt hơn** ở Faithfulness (+0.009), Answer Relevance (+0.008), và Context Precision (+0.013) — reranking đưa đúng chunk lên trên, giúp LLM trích dẫn chính xác hơn.
- Config B **tốt hơn** ở Context Recall (+0.033) — do không qua reranking, BM25 giữ lại nhiều candidate hơn, một số evidence bị reranking lọc ra ngoài top-5 ở Config A.
- **Trade-off**: Reranking tăng Precision nhưng có thể giảm Recall nếu top-k không đủ lớn. Giải pháp: tăng `candidate_k` trong bước merge trước khi rerank.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Root Cause |
|---|----------|:------------:|:---------:|:------:|------------|
| 1 | Khi gửi yêu cầu trả hàng, người mua cần chuẩn bị gì để làm bằng chứng? | 1.000 | 0.841 | 0.500 | Context Recall thấp — retriever chỉ lấy được 1 trong 2 evidence cần thiết |
| 2 | Trong trường hợp nào thì người mua được trả hàng vì không còn nhu cầu? | 1.000 | 1.000 | 0.500 | Context Recall thấp — điều kiện Trả hàng COM nằm rải rác nhiều chunk |
| 3 | Trả hàng COM có áp dụng cho sản phẩm thuộc Shopee Mart không? | 0.800 | 0.976 | 1.000 | Faithfulness thấp — LLM thêm thông tin ngoài context (hallucination nhẹ) |

**Nhận xét chung:** Các câu hỏi về **Trả hàng COM** gặp vấn đề vì thông tin nằm rải rác ở nhiều điều khoản (4.1, 4.2, 4.4) — cần cải thiện chunking để giữ nguyên cấu trúc điều khoản.

---

## Recommendations

### Cải tiến 1: Tăng candidate_k trước khi rerank
**Action:** Trong `task9_retrieval_pipeline.py`, tăng `candidate_k = top_k * 4` (hiện tại `* 2`).
Reranker có nhiều candidate hơn để chọn → giảm risk bỏ sót evidence quan trọng.
**Expected impact:** Tăng Context Recall từ 0.867 lên ~0.900+, giữ nguyên Precision.

### Cải tiến 2: Sentence-aware Chunking
**Action:** Thay chunking theo ký tự bằng chunking theo ranh giới câu/điều khoản.
Mỗi chunk chứa 1 điều khoản hoàn chỉnh (ví dụ: toàn bộ Điều 4.2) thay vì bị cắt giữa chừng.
**Expected impact:** Tăng Context Recall cho câu hỏi về Trả hàng COM, Context Precision giữ nguyên hoặc tăng.

### Cải tiến 3: Prompt Self-Check (Anti-hallucination)
**Action:** Thêm bước LLM tự kiểm tra sau khi sinh câu trả lời:
*"Mỗi khẳng định trong câu trả lời có xuất hiện trong Context không? Nếu không, hãy xóa bỏ."*
**Expected impact:** Tăng Faithfulness từ 0.853 lên ~0.90+, giảm hallucination nhẹ ở câu hỏi phức tạp.