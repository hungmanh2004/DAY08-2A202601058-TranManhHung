"""
Task 6 — Lexical Search Module (BM25 + TF-IDF).

Cài đặt:
    pip install rank-bm25 scikit-learn

Hai phương pháp được implement:
    1. BM25 (mặc định) — rank-bm25 / BM25Okapi
    2. TF-IDF (bonus)  — sklearn TfidfVectorizer + cosine similarity

============================================================
BM25 hoạt động thế nào (giải thích cho demo +5 điểm bonus):
============================================================
BM25 (Best Match 25) là mô hình probabilistic ranking, cải tiến từ TF-IDF:

    score(q, d) = Σ IDF(qi) × [ tf(qi,d) × (k1 + 1) ] / [ tf(qi,d) + k1 × (1 - b + b × |d|/avgdl) ]

Trong đó:
    - tf(qi, d)  : số lần từ qi xuất hiện trong document d
    - IDF(qi)    : log((N - df(qi) + 0.5) / (df(qi) + 0.5) + 1)
                   → từ hiếm (ít document chứa) có IDF cao → được coi là quan trọng hơn
    - k1 = 1.5   : term frequency saturation — sau một ngưỡng nào đó, tf tăng thêm
                   không làm tăng score nhiều (tránh document lặp từ nhiều lần gian lận)
    - b  = 0.75  : length normalization — document dài hơn mức trung bình (avgdl)
                   bị "phạt" nhẹ để không ưu tiên quá mức chỉ vì chứa nhiều từ
    - avgdl      : độ dài trung bình của document trong corpus

So sánh BM25 vs TF-IDF:
    - TF-IDF: score = TF × IDF, nhân thẳng — không có saturation, document dài luôn
              được lợi (điểm TF tích lũy theo độ dài)
    - BM25: có upper bound cho TF (k1 saturation) + normalize theo độ dài tài liệu
              → robust hơn với document có độ dài không đồng đều

Corpus của bài lab gồm nhiều file markdown tiếng Việt độ dài rất khác nhau
(policy dài vài trang, news ngắn 1-2 đoạn) → BM25 phù hợp hơn TF-IDF vì
length normalization giúp document ngắn có cơ hội ranking ngang document dài
khi chứa đúng từ khóa truy vấn.
"""

import re
from pathlib import Path

from rank_bm25 import BM25Okapi
import numpy as np

# ------------------------------------------------------------------ #
# Cấu hình                                                            #
# ------------------------------------------------------------------ #
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Chunk size khi load corpus trực tiếp từ markdown (để phù hợp với
# kích thước chunk đã index ở Task 4: CHUNK_SIZE=500 ký tự)
CORPUS_CHUNK_SIZE = 500
CORPUS_CHUNK_OVERLAP = 50


# ------------------------------------------------------------------ #
# Tokenizer đơn giản hỗ trợ tiếng Việt                               #
# ------------------------------------------------------------------ #

def _tokenize(text: str) -> list[str]:
    """
    Tokenize văn bản tiếng Việt / Anh đơn giản.

    Cách hoạt động:
        1. Lowercase toàn bộ
        2. Dùng regex giữ lại chữ cái (bao gồm Unicode / tiếng Việt có dấu),
           chữ số, dấu gạch dưới — loại bỏ ký tự đặc biệt
        3. Split theo khoảng trắng → danh sách token
        4. Lọc token rỗng và token 1 ký tự (thường là nhiễu)

    Không dùng thư viện nặng (underthesea / pyvi) để giữ dependency tối thiểu.
    Với corpus chính sách thương mại điện tử, word-level tokenize đủ hiệu quả
    cho BM25 vì các từ khoá quan trọng ("hoàn tiền", "thanh toán", "trả hàng")
    là đơn từ hoặc bigram đơn giản.
    """
    text = text.lower()
    # Giữ lại chữ cái Unicode (bao gồm tiếng Việt), chữ số, khoảng trắng
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    tokens = text.split()
    # Loại bỏ token quá ngắn (1 ký tự) — thường là ký tự đơn lẻ nhiễu
    return [t for t in tokens if len(t) > 1]


def _expand_query(query: str) -> str:
    """Mở rộng một số cụm tiếng Anh phổ biến sang tiếng Việt.

    Corpus của bài lab chủ yếu là tiếng Việt, trong khi người dùng/test có
    thể hỏi bằng tiếng Anh. BM25 là lexical search nên không tự hiểu đồng
    nghĩa; query expansion giúp giữ token gốc và thêm token tương đương.
    """
    expansions = {
        "payment methods": "phương thức thanh toán",
        "payment method": "phương thức thanh toán",
        "order tracking": "theo dõi đơn hàng",
        "tracking order": "theo dõi đơn hàng",
        "tracking guide": "hướng dẫn theo dõi đơn hàng",
        "order tracking guide": "hướng dẫn theo dõi đơn hàng",
        "refund": "hoàn tiền",
        "return": "trả hàng",
        "returns": "trả hàng",
        "evidence": "bằng chứng",
    }
    expanded = query
    query_lower = query.lower()
    for english, vietnamese in expansions.items():
        if english in query_lower:
            expanded += " " + vietnamese
    return expanded


# ------------------------------------------------------------------ #
# Load & chunk corpus từ data/standardized/                           #
# ------------------------------------------------------------------ #

def _load_corpus() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/ và chia thành chunks.

    Dùng chunking đơn giản theo ký tự (khớp với CORPUS_CHUNK_SIZE) để corpus
    BM25 có granularity tương đương với ChromaDB collection ở Task 4 — đảm bảo
    khi merge kết quả ở Task 9, hai retriever trả về cùng cấp độ đoạn văn.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str, 'chunk_index': int}}
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Standardized data directory not found: {STANDARDIZED_DIR}. "
            "Please run task3_convert_markdown.py first."
        )

    corpus = []
    md_files = list(STANDARDIZED_DIR.rglob("*.md"))

    if not md_files:
        raise ValueError(
            f"No markdown files found in {STANDARDIZED_DIR}. "
            "Please run task3_convert_markdown.py to generate standardized files."
        )

    for md_file in sorted(md_files):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"

        # Chunk theo ký tự với overlap
        start = 0
        chunk_index = 0
        while start < len(content):
            end = start + CORPUS_CHUNK_SIZE
            chunk_text = content[start:end].strip()
            if chunk_text:
                corpus.append({
                    "content": chunk_text,
                    "metadata": {
                        "source": md_file.name,
                        "type": doc_type,
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1
            start = end - CORPUS_CHUNK_OVERLAP  # overlap để không cắt đứt ngữ cảnh

    return corpus


# ------------------------------------------------------------------ #
# Lazy-loaded global state                                            #
# ------------------------------------------------------------------ #

_corpus: list[dict] | None = None
_bm25_index: BM25Okapi | None = None
_tfidf_vectorizer = None   # sklearn TfidfVectorizer
_tfidf_matrix = None       # sparse matrix (n_chunks × vocab)


def _get_corpus() -> list[dict]:
    """Trả về corpus, load từ disk nếu chưa có."""
    global _corpus
    if _corpus is None:
        _corpus = _load_corpus()
    return _corpus


# ------------------------------------------------------------------ #
# BM25 index                                                          #
# ------------------------------------------------------------------ #

def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi object đã được fit trên corpus
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


def _get_bm25_index() -> BM25Okapi:
    """Trả về BM25 index, build nếu chưa có (lazy init)."""
    global _bm25_index
    if _bm25_index is None:
        corpus = _get_corpus()
        _bm25_index = build_bm25_index(corpus)
    return _bm25_index


# ------------------------------------------------------------------ #
# TF-IDF index (bonus)                                               #
# ------------------------------------------------------------------ #

def _build_tfidf_index(corpus: list[dict]):
    """
    Xây dựng TF-IDF index bằng sklearn TfidfVectorizer.

    TF-IDF hoạt động thế nào:
        - TF(t, d)  = (số lần t xuất hiện trong d) / (tổng số từ trong d)  [term frequency]
        - IDF(t)    = log((1 + N) / (1 + df(t))) + 1                        [smoothed IDF]
        - score(t, d) = TF(t, d) × IDF(t)
        - Cosine similarity giữa query vector và document vector → final score

    Lợi ích khi dùng sklearn TfidfVectorizer:
        - Tích hợp sẵn sublinear TF (log(1+tf)) để giảm ảnh hưởng của TF cao
        - L2-normalize vector → cosine similarity đo góc, không bị ảnh hưởng độ dài
        - Xử lý tốt corpus tiếng Việt vì analyzer="word" + unicode_decode

    Returns:
        (vectorizer, matrix) — cần giữ cả hai để transform query
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    texts = [doc["content"] for doc in corpus]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"(?u)\b\w\w+\b",  # ≥2 ký tự, hỗ trợ Unicode
        sublinear_tf=True,                # log(1+tf) — giảm bias term frequency cao
        min_df=1,                         # giữ cả từ hiếm (corpus nhỏ)
        max_df=0.95,                      # bỏ từ xuất hiện trong >95% documents (stop-word tự động)
        ngram_range=(1, 2),               # unigram + bigram để bắt cụm từ ("hoàn tiền", "trả hàng")
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def _get_tfidf_index():
    """Trả về (vectorizer, matrix), build nếu chưa có (lazy init)."""
    global _tfidf_vectorizer, _tfidf_matrix
    if _tfidf_vectorizer is None:
        corpus = _get_corpus()
        _tfidf_vectorizer, _tfidf_matrix = _build_tfidf_index(corpus)
    return _tfidf_vectorizer, _tfidf_matrix


# ------------------------------------------------------------------ #
# Public API                                                          #
# ------------------------------------------------------------------ #

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25 (mặc định).

    Thuật toán:
        1. Tokenize query (lowercase + regex unicode)
        2. Tính BM25 score cho toàn bộ corpus
        3. Lấy top_k indices theo score giảm dần
        4. Lọc bỏ kết quả có score = 0 (không có từ nào khớp)
        5. Normalize score về [0, 1] bằng min-max để tương thích khi
           merge với semantic search ở Task 9

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,    # BM25 score đã normalize về [0, 1]
            'metadata': dict   # source, type, chunk_index
        }
        Sorted by score descending.
    """
    corpus = _get_corpus()
    bm25 = _get_bm25_index()

    tokenized_query = _tokenize(_expand_query(query))
    if not tokenized_query:
        return []

    raw_scores = bm25.get_scores(tokenized_query)

    # Lấy top_k indices (argsort ascending → đảo ngược → slice)
    top_indices = np.argsort(raw_scores)[::-1][:top_k]

    # Normalize score về [0, 1] để tương thích với cosine score của Task 5
    max_score = float(raw_scores[top_indices[0]]) if len(top_indices) > 0 else 1.0
    if max_score == 0:
        return []  # Không có kết quả liên quan

    results = []
    for idx in top_indices:
        raw = float(raw_scores[idx])
        if raw <= 0:
            break  # Phần còn lại score = 0, bỏ qua
        normalized = round(raw / max_score, 4)  # min-max với min=0
        results.append({
            "content": corpus[idx]["content"],
            "score": normalized,
            "metadata": corpus[idx]["metadata"],
        })

    return results


def lexical_search_tfidf(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng TF-IDF + cosine similarity (phương pháp bonus).

    Khác biệt so với BM25:
        - TF-IDF đơn giản hơn: không có term saturation, không normalize theo độ dài tài liệu
        - sklearn TfidfVectorizer dùng sublinear_tf (log(1+tf)) để giảm thiểu phần nào
          điểm yếu của TF thuần
        - Phù hợp khi corpus đồng nhất về độ dài; kém hơn BM25 khi corpus hỗn hợp
          (vd: policy dài 5000 từ vs news 200 từ) — đây là trường hợp của bài lab

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,    # Cosine similarity TF-IDF ∈ [0, 1]
            'metadata': dict
        }
        Sorted by score descending.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    corpus = _get_corpus()
    vectorizer, matrix = _get_tfidf_index()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            break
        results.append({
            "content": corpus[idx]["content"],
            "score": round(score, 4),
            "metadata": corpus[idx]["metadata"],
        })

    return results


# ------------------------------------------------------------------ #
# Test / __main__                                                      #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    test_queries = [
        "phương thức thanh toán shopee",
        "quy định trả hàng hoàn tiền",
        "chính sách bảo vệ người mua",
    ]

    print("=" * 60)
    print("BM25 Lexical Search")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = lexical_search(query, top_k=5)
        if not results:
            print("  Không tìm thấy kết quả.")
        else:
            for i, r in enumerate(results, 1):
                source = r["metadata"].get("source", "unknown")
                print(f"  [{i}] score={r['score']:.4f}  src={source}")
                print(f"       {r['content'][:100]}...")

    print("\n" + "=" * 60)
    print("TF-IDF Lexical Search (bonus)")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        results = lexical_search_tfidf(query, top_k=5)
        if not results:
            print("  Không tìm thấy kết quả.")
        else:
            for i, r in enumerate(results, 1):
                source = r["metadata"].get("source", "unknown")
                print(f"  [{i}] score={r['score']:.4f}  src={source}")
                print(f"       {r['content'][:100]}...")
