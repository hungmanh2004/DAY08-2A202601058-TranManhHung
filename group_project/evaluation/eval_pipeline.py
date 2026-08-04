"""
RAG Evaluation Pipeline — RAGAS implementation.

Yeu cau:
    1. Load golden_dataset.json (>=15 Q&A pairs)
    2. Chay RAG pipeline tren tung question
    3. Evaluate voi 4 metrics: faithfulness, answer_relevancy, context_recall, context_precision
    4. So sanh A/B: Config A (hybrid+rerank) vs Config B (dense-only)
    5. Export results ra results.md

Luu y rate limit neu dung model OpenRouter ":free": RAGAS goi LLM rat nhieu lan
(khong phai 1 lan/cau hoi ma nhieu lan/metric/cau hoi). Model free gioi han 50 req/ngay.
Neu bi rate limit giua chung, dat EVAL_SUBSET=5 de chay subset.
"""

import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"

# Gioi han subset de tranh rate limit (None = chay toan bo)
EVAL_SUBSET = None   # Vi du: 5 -> chi chay 5 cau dau


def load_golden_dataset() -> list[dict]:
    """Load golden dataset tu JSON file."""
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# =============================================================================
# LLM / Embeddings helpers cho RAGAS
# =============================================================================

def _build_ragas_llm():
    """
    Tao LLM wrapper cho RAGAS.
    Thu tu uu tien: OPENROUTER_API_KEY -> OPENAI_API_KEY -> GEMINI_API_KEY
    """
    from ragas.llms import LangchainLLMWrapper

    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if openrouter_key:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "openai/gpt-4o-mini"),
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
    elif openai_key:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("LLM_MODEL", "gpt-4o-mini").replace("openai/", "")
        llm = ChatOpenAI(model=model_name, api_key=openai_key, temperature=0)
    elif gemini_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", google_api_key=gemini_key, temperature=0
        )
    else:
        raise RuntimeError(
            "Can it nhat 1 trong: OPENROUTER_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY"
        )

    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Tao Embeddings wrapper cho RAGAS."""
    from ragas.embeddings import LangchainEmbeddingsWrapper

    openai_key = os.getenv("OPENAI_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if openai_key:
        from langchain_openai import OpenAIEmbeddings
        emb = OpenAIEmbeddings(api_key=openai_key)
    elif gemini_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        emb = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=gemini_key
        )
    else:
        # Fallback: sentence-transformers local (khong ton API call)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return LangchainEmbeddingsWrapper(emb)


# =============================================================================
# Thu thap RAG outputs
# =============================================================================

def collect_rag_outputs(
    golden_dataset: list[dict],
    use_reranking: bool = True,
    config_name: str = "Config A",
) -> dict:
    """
    Chay RAG pipeline tren tung cau hoi trong golden_dataset.

    Args:
        golden_dataset : List Q&A dicts tu golden_dataset.json
        use_reranking  : True = hybrid+rerank (A), False = dense-only (B)
        config_name    : Ten config de log

    Returns:
        Dict voi 4 list: question, answer, contexts, ground_truth
    """
    from src.task10_generation import (
        generate_with_citation,
        reorder_for_llm,
        format_context,
        SYSTEM_PROMPT,
        TEMPERATURE,
        TOP_P,
        LLM_MODEL,
    )
    from src.task9_retrieval_pipeline import retrieve

    eval_data: dict = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }

    total = len(golden_dataset)
    for i, item in enumerate(golden_dataset, 1):
        question = item["question"]
        ground_truth = item["expected_answer"]

        label = question[:60] + "..." if len(question) > 60 else question
        print(f"  [{config_name}] Q{i:02d}/{total}: {label}")

        try:
            if use_reranking:
                # Config A: pipeline day du (hybrid + reranking)
                result = generate_with_citation(question, top_k=5)
                answer = result["answer"]
                sources = result["sources"]
            else:
                # Config B: chi dense-only, khong reranking
                chunks = retrieve(question, top_k=5, use_reranking=False)
                reordered = reorder_for_llm(chunks)
                context_str = format_context(reordered)
                user_msg = (
                    f"Context tai lieu:\n{context_str}\n\n"
                    f"Cau hoi: {question}\n\n"
                    "Chi su dung Context. Sau moi thong tin thuc te, ghi citation "
                    "theo dang [ten nguon, nam]. Neu khong co bang chung truc tiep, hay noi: "
                    "'Toi khong the xac minh thong tin nay tu nguon hien co'."
                )

                from openai import OpenAI
                openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
                openai_key = os.getenv("OPENAI_API_KEY", "")
                if openrouter_key:
                    client = OpenAI(
                        api_key=openrouter_key,
                        base_url="https://openrouter.ai/api/v1",
                    )
                    model = os.getenv("LLM_MODEL", LLM_MODEL)
                else:
                    client = OpenAI(api_key=openai_key)
                    model = os.getenv("LLM_MODEL", "gpt-4o-mini").replace("openai/", "")

                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )
                answer = (resp.choices[0].message.content or "").strip()
                sources = chunks

            eval_data["question"].append(question)
            eval_data["answer"].append(answer)
            eval_data["contexts"].append([c["content"] for c in sources] or [""])
            eval_data["ground_truth"].append(ground_truth)

        except Exception as exc:
            print(f"    Warning Q{i}: {exc}")
            eval_data["question"].append(question)
            eval_data["answer"].append("ERROR")
            eval_data["contexts"].append([""])
            eval_data["ground_truth"].append(ground_truth)

        # Tranh rate limit: nghi 2s giua cac cau hoi
        if i < total:
            time.sleep(2)

    return eval_data


# =============================================================================
# RAGAS Evaluation
# =============================================================================

def evaluate_with_ragas(eval_data: dict) -> dict:
    """
    Chay RAGAS evaluation tren bo du lieu da collect.

    4 metrics:
        - faithfulness       : cau tra loi co bam dung context khong?
        - answer_relevancy   : cau tra loi co dung cau hoi khong?
        - context_recall     : retriever co lay du evidence khong?
        - context_precision  : trong context lay ve, bao nhieu % thuc su huu ich?

    Returns:
        Dict[metric_name -> float (0-1)] + per_question list
    """
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision,
    )
    from datasets import Dataset

    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    # Inject LLM/embeddings vao tung metric
    for metric in [faithfulness, answer_relevancy, context_recall, context_precision]:
        metric.llm = ragas_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = ragas_emb

    dataset = Dataset.from_dict(eval_data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    )

    scores_df = result.to_pandas()
    return {
        "faithfulness": float(scores_df["faithfulness"].mean()),
        "answer_relevancy": float(scores_df["answer_relevancy"].mean()),
        "context_recall": float(scores_df["context_recall"].mean()),
        "context_precision": float(scores_df["context_precision"].mean()),
        "per_question": scores_df.to_dict(orient="records"),
    }


# =============================================================================
# A/B Comparison
# =============================================================================

def compare_configs(golden_dataset: list[dict]) -> tuple[dict, dict]:
    """
    So sanh A/B giua 2 configs:
        Config A : hybrid search + reranking  (pipeline day du)
        Config B : dense-only, khong reranking
    """
    print("\n[Step 1/4] Collecting Config A outputs (hybrid + reranking)...")
    data_a = collect_rag_outputs(
        golden_dataset, use_reranking=True, config_name="Config A"
    )

    print("\n[Step 2/4] Collecting Config B outputs (dense-only)...")
    data_b = collect_rag_outputs(
        golden_dataset, use_reranking=False, config_name="Config B"
    )

    print("\n[Step 3/4] Running RAGAS on Config A...")
    scores_a = evaluate_with_ragas(data_a)

    print("\n[Step 4/4] Running RAGAS on Config B...")
    scores_b = evaluate_with_ragas(data_b)

    return scores_a, scores_b


# =============================================================================
# Export Results
# =============================================================================

def export_results(
    scores_a: dict,
    scores_b: dict,
    golden_dataset: list[dict],
) -> None:
    """Export evaluation results ra results.md."""

    METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]
    LABELS = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevance",
        "context_recall": "Context Recall",
        "context_precision": "Context Precision",
    }

    def fmt(v: float) -> str:
        return f"{v:.3f}" if v is not None else "N/A"

    avg_a = sum(scores_a[m] for m in METRICS) / len(METRICS)
    avg_b = sum(scores_b[m] for m in METRICS) / len(METRICS)

    # Worst performers (bottom 3 theo combo score)
    worst: list[dict] = []
    if "per_question" in scores_a:
        for idx, row in enumerate(scores_a["per_question"]):
            q_text = (
                golden_dataset[idx]["question"]
                if idx < len(golden_dataset)
                else f"Q{idx + 1}"
            )
            faith = row.get("faithfulness", 1.0) or 1.0
            rel = row.get("answer_relevancy", 1.0) or 1.0
            rec = row.get("context_recall", 1.0) or 1.0
            worst.append(
                {
                    "question": q_text,
                    "faithfulness": faith,
                    "answer_relevancy": rel,
                    "context_recall": rec,
                    "combo": (faith + rel + rec) / 3,
                }
            )
        worst.sort(key=lambda x: x["combo"])
        worst = worst[:3]

    # ── Tao noi dung markdown ──────────────────────────────────────────────
    d_avg = avg_a - avg_b
    sign_avg = "+" if d_avg >= 0 else ""

    rows = []
    for m in METRICS:
        a_val, b_val = scores_a[m], scores_b[m]
        d = a_val - b_val
        sign = "+" if d >= 0 else ""
        rows.append(
            f"| {LABELS[m]} | {fmt(a_val)} | {fmt(b_val)} | {sign}{fmt(d)} |"
        )
    rows.append(
        f"| **Average** | **{fmt(avg_a)}** | **{fmt(avg_b)}** |"
        f" **{sign_avg}{fmt(d_avg)}** |"
    )

    worst_rows = []
    for rank, w in enumerate(worst, 1):
        q_short = (
            w["question"][:55] + "..." if len(w["question"]) > 55 else w["question"]
        )
        if w["faithfulness"] < 0.5:
            cause = "LLM hallucination - cau tra loi vuot ngoai context"
        elif w["context_recall"] < 0.5:
            cause = "Retriever khong lay du evidence"
        elif w["answer_relevancy"] < 0.5:
            cause = "Cau tra loi lac de so voi cau hoi goc"
        else:
            cause = "Tong hop diem thap, can xem xet toan bo pipeline"
        worst_rows.append(
            f"| {rank} | {q_short} | {fmt(w['faithfulness'])} |"
            f" {fmt(w['answer_relevancy'])} | {fmt(w['context_recall'])} | {cause} |"
        )

    if avg_a >= avg_b:
        diff_pct = abs(avg_a - avg_b) * 100
        conclusion = (
            f"> Config A (hybrid + rerank) tot hon Config B trung binh **{diff_pct:.1f}%**. "
            "Reranking giup dua dung context len top, cai thien Faithfulness va Context Precision. "
            "Khuyen nghi su dung Config A cho production."
        )
    else:
        diff_pct = abs(avg_b - avg_a) * 100
        conclusion = (
            f"> Config B (dense-only) but ngo vuot troi Config A **{diff_pct:.1f}%**. "
            "Co the reranking dang reorder sai thu tu voi corpus nay — can dieu chinh RERANK_METHOD."
        )

    content = "\n".join(
        [
            "# RAG Evaluation Results",
            "",
            "## Framework su dung",
            "",
            "> **RAGAS** — chuan industry cho RAG eval.",
            "> 4 metrics: Faithfulness, Answer Relevancy, Context Recall, Context Precision.",
            "",
            "---",
            "",
            "## Overall Scores",
            "",
            "| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Delta (A - B) |",
            "|--------|:--------------------------:|:---------------------:|:-------------:|",
        ]
        + rows
        + [
            "",
            "---",
            "",
            "## A/B Comparison Analysis",
            "",
            "**Config A - Hybrid Search + Reranking:**",
            "> Pipeline day du: semantic search + BM25 -> RRF merge -> reranking -> generation.",
            "",
            "**Config B - Dense-Only (khong reranking):**",
            "> Chi dung semantic search, bo qua BM25 va reranking.",
            "",
            "**Ket luan:**",
            conclusion,
            "",
            "---",
            "",
            "## Worst Performers (Bottom 3)",
            "",
            "| # | Question | Faithfulness | Relevance | Recall | Root Cause |",
            "|---|----------|:------------:|:---------:|:------:|------------|",
        ]
        + worst_rows
        + [
            "",
            "---",
            "",
            "## Recommendations",
            "",
            "### Cai tien 1: Prompt Engineering",
            "**Action:** Them explicit instruction yeu cau LLM chi trich dan thong tin trong",
            "context; them self-check step.",
            "**Expected impact:** Tang Faithfulness +0.05 ~ +0.10",
            "",
            "### Cai tien 2: Chunking Strategy",
            "**Action:** Giam chunk_size xuong ~256 tokens voi overlap ~64 tokens.",
            "Chunks nho hon -> context precision cao hon (it noise hon moi chunk).",
            "**Expected impact:** Tang Context Precision +0.08 ~ +0.15",
            "",
            "### Cai tien 3: Query Expansion (HyDE / Multi-Query)",
            "**Action:** Truoc retrieval, dung LLM sinh 2-3 cau hoi tuong duong.",
            "Merge ket qua tu nhieu variants -> tang recall cho cau hoi phuc hop.",
            "**Expected impact:** Tang Context Recall +0.05 ~ +0.12",
        ]
    )

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\n  Results exported -> {RESULTS_PATH}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_recall",
        "context_precision",
    ]

    print("=" * 65)
    print("  RAGAS Evaluation Pipeline")
    print("  E-commerce Support RAG Chatbot — Shopee Policies")
    print("=" * 65)

    golden_dataset = load_golden_dataset()
    total = len(golden_dataset)
    print(f"\nLoaded {total} test cases from {GOLDEN_DATASET_PATH.name}")

    if EVAL_SUBSET and EVAL_SUBSET < total:
        print(
            f"  Warning: Running subset {EVAL_SUBSET}/{total} "
            "(set EVAL_SUBSET=None for full run)"
        )
        golden_dataset = golden_dataset[:EVAL_SUBSET]

    # Kiem tra dependencies
    missing_pkgs = []
    for pkg, install_name in [
        ("ragas", "ragas"),
        ("datasets", "datasets"),
        ("langchain_openai", "langchain-openai"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing_pkgs.append(install_name)

    if missing_pkgs:
        print(f"\n  Missing packages: {missing_pkgs}")
        print("  Run: pip install " + " ".join(missing_pkgs))
        raise SystemExit(1)

    scores_a, scores_b = compare_configs(golden_dataset)

    avg_a = sum(scores_a[m] for m in METRICS) / len(METRICS)
    avg_b = sum(scores_b[m] for m in METRICS) / len(METRICS)

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Metric':<22} {'Config A':>12} {'Config B':>12} {'Delta':>8}")
    print("-" * 65)
    for m in METRICS:
        d = scores_a[m] - scores_b[m]
        print(f"  {m:<22} {scores_a[m]:>12.3f} {scores_b[m]:>12.3f} {d:>+8.3f}")
    print("-" * 65)
    d_avg = avg_a - avg_b
    print(f"  {'Average':<22} {avg_a:>12.3f} {avg_b:>12.3f} {d_avg:>+8.3f}")

    export_results(scores_a, scores_b, golden_dataset)
    print("\n  Done! Xem ket qua tai: group_project/evaluation/results.md")
