---
title: E-commerce Support RAG Chatbot
emoji: 🛒
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.35.0"
app_file: app.py
pinned: false
---

# Ngày 8 — RAG Pipeline v2

**Chương 2 | Ngày 8 trong 15**

> Dùng chung chủ đề "E-commerce Policy / Customer Support" với biến thể K4 của Ngày 7 (`K4_VARIANT.md`), để pipeline Ngày 7 → Ngày 8 nhất quán.

---

## Mục Tiêu

Xây dựng một RAG pipeline thực tế, end-to-end, từ thu thập dữ liệu chính sách thương mại điện tử và hỗ trợ khách hàng → xử lý → indexing → retrieval (hybrid + vectorless fallback) → generation có citation.

---

## Chủ Đề Dữ Liệu

**Chính sách thương mại điện tử** (thanh toán, đổi trả/hoàn tiền, quy định người bán, quyền riêng tư) + **Hướng dẫn hỗ trợ khách hàng** (theo dõi đơn hàng, bằng chứng hoàn tiền, thay đổi phương thức thanh toán)

Dữ liệu mẫu trong repo được crawl thật từ trang trung tâm trợ giúp công khai của **Shopee Vietnam** (help.shopee.vn) — xem chi tiết URL nguồn trong `src/task1_collect_legal_docs.py` và `src/task2_crawl_news.py`.

---

## Cấu Trúc Thư Mục

```
K4-Day08-RAG-Pipeline-Starter/
├── README.md
├── LAB_GUIDE.md           ← Hướng dẫn chi tiết & Codelab
├── checkpoint_timer.html  ← Dashboard đếm ngược Checkpoint & Phân vai
├── app.py                 ← Streamlit chatbot (bài nhóm)
├── data/
│   ├── landing/           ← Task 1 & 2: raw files (PDF, JSON)
│   └── standardized/      ← Task 3: converted markdown files
├── src/
│   ├── __init__.py
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   ├── task10_generation.py
│   └── supervisor.py      ← Pattern nâng cao: Supervisor + Workers song song
├── chroma_db/             ← Task 4: vector store đã index (sinh ra khi chạy, không tự viết tay)
├── tests/
│   └── test_individual.py ← Chấm điểm phần Task 1-10 (pytest)
├── group_project/
│   ├── README.md          ← Hướng dẫn bài tập nhóm
│   └── evaluation/        ← golden_dataset.json, eval_pipeline.py, results.md
├── requirements.txt
└── .env.example
```

---

## Nhiệm Vụ Chi Tiết

### Task 1 — Thu Thập Văn Bản Chính Sách Thương Mại Điện Tử

Tìm và tải về **tối thiểu 3 văn bản chính sách/quy định** dạng PDF/DOCX về chính sách thương mại điện tử. Lưu vào `data/landing/`.

**Gợi ý nguồn** (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
- Chính sách trả hàng và hoàn tiền (Returns & Refund Policy)
- Phương thức thanh toán (Payment Methods)
- Chính sách bảo mật (Privacy Policy)
- Quy định đăng bán sản phẩm cho người bán (Product Listing Regulations)

**Yêu cầu:**
- Lưu file gốc (PDF/DOCX) vào `data/landing/legal/`
- Đặt tên file rõ ràng: `returns-refund-policy-shopee.pdf`, `payment-methods-shopee.pdf`, ...

---

### Task 2 — Crawl Bài Viết/Thông Báo

Crawl **tối thiểu 5 bài viết** hướng dẫn hỗ trợ khách hàng (theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền, mua hàng xuyên biên giới).

**Thư viện khuyến nghị:** [Crawl4AI](https://github.com/unclecode/crawl4ai)

**Yêu cầu:**
- Lưu output vào `data/landing/news/`
- Mỗi bài báo lưu thành 1 file (JSON hoặc HTML)
- Ghi rõ metadata: URL gốc, ngày crawl, tiêu đề bài báo

**Code mẫu (Crawl4AI):**
```python
from crawl4ai import AsyncWebCrawler

async def crawl_article(url: str, output_dir: str):
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        # Lưu result.markdown vào file
        ...
```

---

### Task 3 — Convert Sang Markdown

Sử dụng [MarkItDown](https://github.com/microsoft/markitdown) của Microsoft để convert toàn bộ file trong `data/landing/` thành Markdown.

**Cài đặt:**
```bash
pip install markitdown
```

**Code mẫu:**
```python
from markitdown import MarkItDown

md = MarkItDown()

# Convert PDF
result = md.convert("data/landing/legal/returns-refund-policy-shopee.pdf")
print(result.text_content)

# Convert DOCX
result = md.convert("data/landing/legal/product-listing-regulations-shopee.docx")
```

**Lưu ý:** MarkItDown cần cài thêm extra `pip install "markitdown[pdf]"` để convert được file
PDF — nếu chỉ `pip install markitdown` sẽ báo lỗi `MissingDependencyException` khi convert PDF.

**Yêu cầu:**
- Output lưu vào `data/standardized/`
- Giữ nguyên cấu trúc thư mục con (`legal/`, `news/`)
- Mỗi file output có tên tương ứng: `returns-refund-policy-shopee.md`

---

### Task 4 — Chunking & Indexing

Chọn **một loại chunking strategy** và **một embedding model** để index toàn bộ markdown files vào vector store.

**Chunking — khuyến khích dùng [langchain-text-splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/):**
```bash
pip install langchain-text-splitters
```

Các loại splitter phù hợp:
- `RecursiveCharacterTextSplitter` (mặc định, an toàn)
- `MarkdownHeaderTextSplitter` (tốt cho file có heading rõ)
- `SemanticChunker` (nâng cao, dùng embedding để tách)

**Embedding model gợi ý:**
- `sentence-transformers/all-MiniLM-L6-v2` (nhẹ, nhanh)
- `BAAI/bge-m3` (multilingual, tốt cho tiếng Việt)
- OpenAI `text-embedding-3-small` (nếu có API key)

**Vector Store — sử dụng ChromaDB (Vector Store mặc định của bài lab):**
```bash
pip install chromadb
```
- ChromaDB lưu trữ vector embeddings (`BAAI/bge-m3`), metadata và thông tin phân đoạn local tại thư mục `chroma_db/`
- Hỗ trợ truy vấn tìm kiếm tương đồng Cosine (Cosine Similarity Search) phục vụ Dense Retrieval ở Task 5

**Yêu cầu:**
- Ghi rõ trong code: dùng chunking nào, chunk_size bao nhiêu, overlap bao nhiêu, vì sao
- Ghi rõ embedding model nào, dimension bao nhiêu
- Index thành công toàn bộ documents

---

### Task 5 — Semantic Search Module

Viết module thực hiện **semantic search** (dense retrieval) trên vector store.

**Yêu cầu:**
```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    ...
```

- Input: query string + top_k
- Output: danh sách chunks có score, sorted descending
- Phải hoạt động được với embedding model đã chọn ở Task 4

---

### Task 6 — Lexical Search Module

Viết module thực hiện **lexical search**. Mặc định sử dụng **BM25**.

```bash
pip install rank-bm25
```

**Code mẫu BM25:**
```python
from rank_bm25 import BM25Okapi

# Tokenize corpus
tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Search
tokenized_query = query.split()
scores = bm25.get_scores(tokenized_query)
```

**Yêu cầu:**
```python
def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
    """
    ...
```

**Bonus:** Nếu dùng phương pháp khác (TF-IDF, Elasticsearch, Weaviate BM25 built-in), hãy giải thích cơ chế hoạt động trong buổi demo → **+5 điểm bonus**.

---

### Task 7 — Reranking Module

Viết module **reranking** để chấm lại độ liên quan của kết quả retrieval.

**Lựa chọn (chọn 1):**

| Phương pháp | Thư viện / Model | Đặc điểm |
|-------------|-----------------|-----------|
| Cross-encoder reranker | `jinaai/jina-reranker-v2-base-multilingual` | Multilingual, tốt cho tiếng Việt |
| Cross-encoder reranker | `Qwen/Qwen3-Reranker-0.6B` | Nhẹ, hiệu quả |
| MMR (Maximal Marginal Relevance) | Tự implement | Giảm trùng lặp, tăng diversity |
| RRF (Reciprocal Rank Fusion) | Tự implement | Gộp kết quả từ nhiều ranker |

**Code mẫu (Jina Reranker via API):**
```python
import requests

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": "Bearer YOUR_API_KEY"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": documents,
            "top_n": top_k
        }
    )
    return response.json()["results"]
```

**Yêu cầu:**
```python
def rerank(query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score and re-order candidates based on relevance to query.
    """
    ...
```

---

### Task 8 — PageIndex Vectorless RAG

Đăng ký tài khoản tại [https://pageindex.ai/](https://pageindex.ai/), sau đó sử dụng [PageIndex SDK](https://github.com/VectifyAI/PageIndex) để tạo một **vectorless RAG pipeline**.

**Cài đặt:**
```bash
pip install pageindex
```

**Tham khảo:** [https://github.com/VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)

**Yêu cầu:**
- Upload tài liệu lên PageIndex
- Viết function query PageIndex và trả về kết quả
```python
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval using PageIndex.
    Fallback khi hybrid search không trả về kết quả phù hợp.
    """
    ...
```

---

### Task 9 — Retrieval Pipeline Hoàn Chỉnh

Kết hợp tất cả modules thành một **retrieval pipeline** thống nhất với logic fallback:

```
Query
  │
  ├─→ Semantic Search (Task 5)  ──┐
  │                                ├─→ Merge + Rerank (Task 7) → Results
  ├─→ Lexical Search (Task 6)  ──┘
  │
  └─→ Nếu hybrid search không có kết quả đủ tốt (score < threshold)
        └─→ Fallback: PageIndex Vectorless (Task 8)
```

**Yêu cầu:**
```python
def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    """
    1. Chạy semantic_search + lexical_search
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback PageIndex
    5. Return top_k results
    """
    ...
```

> ⚠️ **Bẫy thường gặp:** nếu dùng RRF để merge (`RRF(d) = Σ 1/(k+rank)`, k=60), điểm số kết quả
> sau khi fuse **chỉ phụ thuộc thứ hạng**, không phản ánh độ liên quan thực sự — top-1 luôn
> xấp xỉ `1/(k+1) ≈ 0.016` dù nội dung có liên quan hay không. Nếu so `score_threshold` với
> điểm RRF đã fuse, fallback gần như **không bao giờ trigger** được (kể cả với query hoàn toàn
> lạc đề). Hãy so `score_threshold` với **điểm cosine similarity gốc** từ `semantic_search`
> (Task 5, thang đo `[0,1]` có ý nghĩa) — tách riêng khỏi điểm dùng để sắp xếp kết quả cuối cùng.



---

### Task 10 — Generation Có Citation

Sắp xếp lại context chunks sau reranking để **tránh lost in the middle**, inject vào prompt, và yêu cầu LLM trả lời có **citation**.

**Document Reordering (tránh lost in the middle):**
```python
def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks theo pattern: quan trọng nhất ở đầu và cuối,
    ít quan trọng hơn ở giữa.
    Ví dụ: [1, 3, 5, 4, 2] thay vì [1, 2, 3, 4, 5]
    """
    ...
```

**Prompt template:**
```python
SYSTEM_PROMPT = """Answer the following question comprehensively.
For every statement of fact or claim, immediately insert a citation
in brackets linking to the specific source
(e.g., [Author/Platform Name, Year]).
If the information is not explicitly stated in the provided context
or knowledge base, state 'I cannot verify this information'
rather than guessing."""

def generate_with_citation(query: str, context_chunks: list[dict]) -> str:
    """
    1. Reorder chunks để tránh lost in the middle
    2. Format context với source metadata
    3. Inject vào prompt với SYSTEM_PROMPT
    4. Gọi LLM (OpenAI, Gemini, hoặc local model)
    5. Return answer có citation
    """
    ...
```

**Yêu cầu:**
- Chọn top_k và top_p phù hợp (giải thích lý do trong code comment)
- Output phải có citation dạng `[Nguồn, Năm]`
- Nếu không đủ evidence → trả về "I cannot verify this information"

---

## Bài Tập Nhóm

> **Sau khi cả nhóm hoàn thành Task 1-10**, cùng nhau xây dựng **1 trong 2 sản phẩm** sau:

---

### Yêu cầu 1: Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về chính sách thương mại điện tử và hỗ trợ khách hàng liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

### Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

#### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

#### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

#### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

#### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

#### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="EcommerceSupport_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

#### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

### Yêu Cầu Chung

1. **Tích hợp pipeline** Task 1-10 mà cả nhóm đã xây dựng
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (xem `group_project/README.md`)

---

### Kiến Trúc Hệ Thống

```mermaid
flowchart TD
    User([Người Dùng]) --> UI[Giao diện Chat\n(Streamlit / Chainlit)]
    
    subgraph "Retrieval Pipeline"
        UI --> Query[User Query]
        Query --> Semantic[Semantic Search\n(ChromaDB)]
        Query --> Lexical[Lexical Search\n(BM25)]
        
        Semantic --> Merge[Merge Results\n(Reciprocal Rank Fusion)]
        Lexical --> Merge
        
        Merge --> Rerank[Reranking\n(Cross-Encoder)]
        
        Rerank --> Check{Best Score\n< Threshold?}
        Check -- Có --> Fallback[Fallback: PageIndex\nVectorless Search]
        Check -- Không --> TopK[Top K Chunks]
        Fallback --> TopK
    end
    
    subgraph "Generation Pipeline"
        TopK --> Reorder[Reorder Context\n(Chống Lost-in-the-middle)]
        Reorder --> Prompt[Build Prompt\n(Context + Query)]
        Prompt --> LLM[LLM Generation\n(OpenRouter/OpenAI)]
    end
    
    LLM --> Response[Câu trả lời có Citation]
    Response --> UI
```

---

### Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
|Trần Mạnh Hùng | 2A202601058 |Role 1 |Hoàn thành |
|Trương Đan Vi | 2A202601178 |Role 3 |Hoàn thành |
|Lê Văn Tuệ |2A202601048 |Role 4 |Hoàn thành |
|Nguyễn Cảnh Hoàng | 2A202601588 |Role 5 |Hoàn thành |
|Hồ Trọng Hảo | 2A202601358 |Role 2 |Hoàn thành |

---

### Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy app
streamlit run app.py
# hoặc
chainlit run app.py
```

---

### Lưu ý

Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.

---

## Cài Đặt Môi Trường

```bash
pip install -r requirements.txt
```

Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
# Điền API keys vào .env
```

---

## Chấm Điểm

### Tổng Quan Phân Bổ Điểm

| Thành phần | Tỷ trọng | Mô tả |
|-----------|----------|-------|
| **Pipeline Kỹ Thuật (Task 1-10)** | **50%** | 10 tasks, cả nhóm cùng làm, chấm bằng automated tests + manual review |
| **Bài Nhóm** | **30%** | RAG Chatbot + Evaluation pipeline |
| **Bonus** | **20%** | Các tiêu chí nâng cao (xem bên dưới) |

---

### Pipeline Kỹ Thuật (Task 1-10) — 50 điểm (50%)

Chấm bằng automated test suite (`pytest tests/ -v`). Mỗi task có test riêng.

| Task | Nội dung | Điểm | Test |
|------|----------|------|------|
| 1 | Thu thập văn bản chính sách thương mại điện tử (≥3 files tồn tại trong `data/landing/legal/`) | 3 | `test_task1_*` |
| 2 | Crawl bài viết/thông báo (≥5 files tồn tại trong `data/landing/news/`) | 3 | `test_task2_*` |
| 3 | Convert markdown (files tồn tại trong `data/standardized/`) | 4 | `test_task3_*` |
| 4 | Chunking + Indexing (vector store có data) | 7 | `test_task4_*` |
| 5 | Semantic search trả về kết quả đúng format, sorted | 6 | `test_task5_*` |
| 6 | Lexical search (BM25) trả về kết quả đúng format | 6 | `test_task6_*` |
| 7 | Reranking hoạt động, output re-sorted | 6 | `test_task7_*` |
| 8 | PageIndex query trả về kết quả | 4 | `test_task8_*` |
| 9 | Retrieval pipeline + fallback logic hoạt động | 7 | `test_task9_*` |
| 10 | Generation có citation + reorder | 4 | `test_task10_*` |
| **Tổng** | | **50** | |

---

### Bài Nhóm — 30 điểm (30%)

| Tiêu chí | Điểm |
|----------|------|
| RAG Chatbot demo hoạt động được | 8 |
| Tích hợp pipeline Task 1-10 đã xây dựng | 4 |
| Kiến trúc rõ ràng + README | 3 |
| Chất lượng câu trả lời (có citation, đúng nội dung) | 3 |
| **Evaluation pipeline** (DeepEval / RAGAS / TruLens) | **12** |
| — Golden dataset ≥15 Q&A pairs | 3 |
| — Chạy eval với ≥4 metrics | 4 |
| — So sánh A/B ≥2 configs + phân tích | 3 |
| — Báo cáo kết quả có phân tích worst performers | 2 |

---

### Bonus — 20 điểm (20%)

| Tiêu chí | Điểm |
|----------|------|
| Giải thích cơ chế lexical search khác BM25 (trong demo) | 5 |
| Implement phương pháp hỗ trợ Semantic Search (HyDE, Query Expansion, ...) | 5 |
| Deploy chatbot online (Hugging Face Spaces / Render / ...) | 4 |
| Conversation memory (multi-turn chat) | 3 |
| UI/UX chất lượng (hiển thị source, score, highlight) | 3 |

---

### Chạy Test Chấm Điểm Pipeline Kỹ Thuật (Task 1-10)

```bash
# Chạy toàn bộ test suite
pytest tests/ -v

# Chạy từng task
pytest tests/test_individual.py::TestTask1 -v
pytest tests/test_individual.py::TestTask5 -v
```

---

## Hướng Dẫn Thời Gian

Theo đúng 7 Checkpoint trong `checkpoint_timer.html` (tổng 180 phút = 3 giờ):

| Checkpoint | Thời gian | Khoảng giờ | Hoạt động |
|------------|-----------|-------------|-----------|
| CP0 | 10 phút | 0:00–0:10 | Setup môi trường & khai báo API keys |
| CP1 | 25 phút | 0:10–0:35 | Task 1–3: Thu thập data + convert markdown |
| CP2 | 25 phút | 0:35–1:00 | Task 4–6: Chunking, indexing, search modules |
| CP3 | 20 phút | 1:00–1:20 | Task 7–8: Reranking + PageIndex fallback |
| CP4 | 25 phút | 1:20–1:45 | Task 9–10: Pipeline hoàn chỉnh + generation (mốc 50đ Task 1-10) |
| CP5 | 30 phút | 1:45–2:15 | Bài nhóm: Chatbot UI & đánh giá RAGAS |
| CP6 | 45 phút | 2:15–3:00 | Thuyết trình demo live & nộp bài |

---

## Tài Liệu Tham Khảo

- [Crawl4AI](https://github.com/unclecode/crawl4ai) — Web crawling library
- [MarkItDown](https://github.com/microsoft/markitdown) — Microsoft document converter
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — Chunking strategies
- [Weaviate](https://weaviate.io/developers/weaviate) — Vector database with hybrid search
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 implementation
- [PageIndex](https://github.com/VectifyAI/PageIndex) — Vectorless RAG
- [Jina Reranker](https://jina.ai/reranker/) — Cross-encoder reranking API
- Liu et al. (2023), *Lost in the Middle: How Language Models Use Long Contexts*
