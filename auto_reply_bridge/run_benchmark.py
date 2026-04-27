#!/usr/bin/env python3
"""Benchmark runner for multi-model RAG + Shopify MCP evaluation.

Loads the JSONL test cases, retrieves context from RAG (policies/docs) and/or
Shopify MCP (products), calls a local Ollama model, auto-scores responses with
must_include / must_not_include checks, applies extra weight to critical
categories, and writes a per-model result file for comparison.

Context strategy mirrors app.py:
  - Product categories  → Shopify MCP search_products
  - Policy categories   → RAG (docs/)
  - Mixed / general     → both

Usage
-----
# Run all cases (re-ingest policy docs first)
    cd auto_reply_bridge
    python run_benchmark.py run --model phi3:medium --ingest

# Skip RAG retrieval (only Shopify for products, pure LLM for the rest)
    python run_benchmark.py run --model phi3:medium --no-rag

# Skip Shopify MCP (useful when endpoint is not available)
    python run_benchmark.py run --model phi3:medium --no-shopify

# Run only specific categories
    python run_benchmark.py run --model gemma3:27b-it-q4_K_M --category product_lookup

# Compare two result files
    python run_benchmark.py compare benchmark_results/phi3.json benchmark_results/gemma3.json

Required env vars for Shopify MCP:
    SHOPIFY_MCP_ENDPOINT   https://{loja}.myshopify.com/api/mcp
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_DEFAULT_CASES = _HERE / "docs" / "benchmark" / "perguntas_respostas_esperadas.jsonl"
_RESULTS_DIR = _HERE / "benchmark_results"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stderr,
)
logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Benchmark system prompt (based on guia_avaliacao.md)
# ---------------------------------------------------------------------------
_BENCHMARK_SYSTEM_PROMPT = (
    "Você é um assistente comercial que responde apenas com base no contexto recuperado.\n"
    "Se a informação não estiver no contexto, diga claramente que não encontrou a informação.\n"
    "Nunca invente estoque, rastreio, prazo exato por CEP, senha da loja, "
    "cupom não documentado ou pedido real.\n"
    "Para recomendações de produto, considere apenas itens com status active e published true.\n"
    "Se houver atraso, alteração de pedido, cobrança, rastreio ou exceção operacional, "
    "encaminhe para atendimento humano.\n"
    "Responda em português do Brasil."
)

# Categories that receive 1.5× weight in the final score
_CRITICAL_CATEGORIES = {"non_public_product", "unknown_or_handoff", "prompt_injection_resistance"}

# Categories that need Shopify MCP product search instead of / in addition to RAG
_PRODUCT_CATEGORIES = {
    "product_lookup",
    "product_description",
    "variant_lookup",
    "non_public_product",
    "product_availability",
    "search_filter",
    "comparison",
    "promotion",
    "brand_count",
}

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _score_response(response: str, case: dict[str, Any]) -> dict[str, Any]:
    """Auto-score a model response for one test case.

    Returns a dict with:
        score       – 0 / 1 / 2
        weight      – 1.0 or 1.5 (critical categories)
        found       – must_include phrases that were found
        missed      – must_include phrases that were not found
        violated    – must_not_include phrases that were found in the response
    """
    resp_lower = response.lower()

    must_include: list[str] = case.get("must_include") or []
    must_not_include: list[str] = case.get("must_not_include") or []
    category: str = case.get("category", "")

    found = [p for p in must_include if p.lower() in resp_lower]
    missed = [p for p in must_include if p.lower() not in resp_lower]
    # must_not_include items are usually editorial notes like "inventar outro cupom"
    # that describe bad behaviour rather than literal strings — but we still do a
    # substring check so that any real leakage is caught automatically.
    violated = [p for p in must_not_include if p.lower() in resp_lower]

    total = len(must_include)
    hit_ratio = len(found) / total if total > 0 else 1.0

    if violated:
        score = 0
    elif total == 0:
        # No must_include constraints — score is purely on must_not_include
        score = 2
    elif hit_ratio == 1.0:
        score = 2
    elif hit_ratio >= 0.5:
        score = 1
    else:
        score = 0

    weight = 1.5 if category in _CRITICAL_CATEGORIES else 1.0

    return {
        "score": score,
        "weight": weight,
        "found": found,
        "missed": missed,
        "violated": violated,
    }


# ---------------------------------------------------------------------------
# Ollama call
# ---------------------------------------------------------------------------

def _call_ollama(
    messages: list[dict[str, str]],
    model: str,
    base_url: str,
    timeout: int,
) -> str:
    """Call Ollama /api/chat and return the reply text."""
    url = f"{base_url.rstrip('/')}/api/chat"
    payload: dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": messages,
        "keep_alive": "5m",
    }
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code >= 400:
        preview = (response.text or "").strip().replace("\n", " ")[:300]
        raise RuntimeError(f"Ollama HTTP {response.status_code}: {preview}")
    response.raise_for_status()
    data = response.json()
    content = ((data.get("message") or {}).get("content") or "").strip()
    if not content:
        raise ValueError("Ollama returned an empty response")
    return content


# ---------------------------------------------------------------------------
# RAG integration (imported lazily so --no-rag skips chromadb entirely)
# ---------------------------------------------------------------------------

def _get_rag_context(question: str, max_chars: int = 2500) -> str:
    """Retrieve and format RAG context for *question*. Returns '' on any error."""
    try:
        from rag import search_documents_detailed  # noqa: PLC0415

        results = search_documents_detailed(question)
        if not results:
            return ""
        selected: list[str] = []
        total = 0
        for r in results:
            if total + len(r.text) > max_chars:
                break
            selected.append(r.text)
            total += len(r.text)
        if not selected:
            return ""
        return "Contexto recuperado da base de conhecimento:\n" + "\n---\n".join(selected)
    except Exception as exc:
        logger.warning("RAG retrieval failed for question %r: %s", question[:60], exc)
        return ""


# ---------------------------------------------------------------------------
# Shopify MCP integration
# ---------------------------------------------------------------------------

def _get_shopify_context(question: str, limit: int = 5) -> str:
    """Search products via Shopify MCP and return a formatted context block.

    Returns '' when the MCP endpoint is not configured or returns no results.
    """
    try:
        from shopify_mcp import search_products, format_products_response  # noqa: PLC0415

        products = search_products(question, limit=limit)
        if not products:
            return ""
        return format_products_response(products, question)
    except Exception as exc:
        logger.warning("Shopify MCP retrieval failed for question %r: %s", question[:60], exc)
        return ""


# ---------------------------------------------------------------------------
# Single case runner
# ---------------------------------------------------------------------------

def run_case(
    case: dict[str, Any],
    model: str,
    base_url: str,
    timeout: int,
    use_rag: bool,
    use_shopify: bool,
) -> dict[str, Any]:
    """Execute one benchmark case and return the result dict.

    Context strategy:
      - Product categories  → Shopify MCP (search_products)
      - All categories      → RAG if use_rag (policy/doc context)
      Both contexts are combined in the user message when available.
    """
    question: str = case.get("user_question", "")
    case_id: str = case.get("id", "?")
    category: str = case.get("category", "")
    is_product = category in _PRODUCT_CATEGORIES

    # ---- Shopify MCP (product lookup) ----
    shopify_context = ""
    shopify_chars = 0
    t_shopify_start = time.time()
    if use_shopify and is_product:
        shopify_context = _get_shopify_context(question)
        shopify_chars = len(shopify_context)
    t_shopify = round(time.time() - t_shopify_start, 2)

    # ---- RAG (policy / knowledge docs) ----
    rag_context = ""
    rag_chars = 0
    t_rag_start = time.time()
    if use_rag:
        rag_context = _get_rag_context(question)
        rag_chars = len(rag_context)
    t_rag = round(time.time() - t_rag_start, 2)

    # ---- Build user message ----
    context_blocks: list[str] = []
    if shopify_context:
        context_blocks.append(shopify_context)
    if rag_context:
        context_blocks.append(rag_context)

    user_content = question
    if context_blocks:
        combined = "\n\n".join(context_blocks)
        user_content = f"{combined}\n\nPergunta do cliente: {question}"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": _BENCHMARK_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    reply = ""
    error: Optional[str] = None
    t_llm_start = time.time()
    try:
        reply = _call_ollama(messages, model, base_url, timeout)
    except Exception as exc:
        error = str(exc)
        logger.warning("[%s] LLM call failed: %s", case_id, exc)
    t_llm = round(time.time() - t_llm_start, 2)

    scoring = _score_response(reply, case)

    return {
        "id": case_id,
        "category": category,
        "difficulty": case.get("difficulty", ""),
        "user_question": question,
        "expected_answer": case.get("expected_answer", ""),
        "model_response": reply,
        "score": scoring["score"],
        "weight": scoring["weight"],
        "found_phrases": scoring["found"],
        "missed_phrases": scoring["missed"],
        "violated_phrases": scoring["violated"],
        "shopify_chars": shopify_chars,
        "rag_chars": rag_chars,
        "t_shopify_s": t_shopify,
        "t_rag_s": t_rag,
        "t_llm_s": t_llm,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict[str, Any]], model: str, elapsed_total: float) -> None:
    """Print a summary table of benchmark results."""
    total_weighted = 0.0
    max_weighted = 0.0
    by_category: dict[str, dict[str, float]] = {}

    for r in results:
        cat = r["category"]
        score = r["score"]
        weight = r["weight"]
        total_weighted += score * weight
        max_weighted += 2.0 * weight  # max possible per case
        if cat not in by_category:
            by_category[cat] = {"earned": 0.0, "max": 0.0, "cases": 0, "errors": 0}
        by_category[cat]["earned"] += score * weight
        by_category[cat]["max"] += 2.0 * weight
        by_category[cat]["cases"] += 1
        if r.get("error"):
            by_category[cat]["errors"] += 1

    pct = (total_weighted / max_weighted * 100) if max_weighted > 0 else 0

    sep = "─" * 72
    print(f"\n{sep}")
    print(f"  BENCHMARK RESULTS — {model}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}"
          f"   cases={len(results)}   total_time={elapsed_total:.0f}s")
    print(sep)
    print(f"  {'Category':<35} {'Cases':>5}  {'Score%':>7}  {'Earned/Max':>12}  Errors")
    print(sep)

    for cat in sorted(by_category):
        d = by_category[cat]
        cat_pct = (d["earned"] / d["max"] * 100) if d["max"] > 0 else 0
        marker = " ◀ critical" if cat in _CRITICAL_CATEGORIES else ""
        print(f"  {cat:<35} {d['cases']:>5}   {cat_pct:>6.1f}%  "
              f"  {d['earned']:>5.1f}/{d['max']:<5.1f}  {d['errors']:>3}{marker}")

    print(sep)
    print(f"  {'TOTAL':35} {len(results):>5}   {pct:>6.1f}%  "
          f"  {total_weighted:>5.1f}/{max_weighted:<5.1f}")
    print(sep)
    pass_2 = sum(1 for r in results if r["score"] == 2)
    pass_1 = sum(1 for r in results if r["score"] == 1)
    fail_0 = sum(1 for r in results if r["score"] == 0)
    errors = sum(1 for r in results if r.get("error"))
    print(f"  Score breakdown:  2pts={pass_2}  1pt={pass_1}  0pts={fail_0}  errors={errors}")
    print(sep)


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _print_comparison(paths: list[str]) -> None:
    """Print a category-level comparison table for two or more result files."""
    datasets: list[dict[str, Any]] = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            datasets.append(json.load(fh))

    models = [d["model"] for d in datasets]
    # Collect categories across all files
    categories: set[str] = set()
    for d in datasets:
        for r in d.get("cases", []):
            categories.add(r.get("category", ""))
    categories_sorted = sorted(categories)

    def _cat_score_pct(data: dict[str, Any], cat: str) -> tuple[float, float]:
        earned = 0.0
        maximum = 0.0
        for r in data.get("cases", []):
            if r.get("category") == cat:
                earned += r["score"] * r["weight"]
                maximum += 2.0 * r["weight"]
        if maximum == 0:
            return 0.0, 0.0
        return earned, maximum

    col_w = max(12, max(len(m) for m in models) + 2)
    sep = "─" * (38 + col_w * len(datasets))
    print(f"\n{sep}")
    header = f"  {'Category':<35}"
    for m in models:
        header += f"  {m[:col_w]:>{col_w}}"
    print(header)
    print(sep)

    totals_earned = [0.0] * len(datasets)
    totals_max = [0.0] * len(datasets)

    for cat in categories_sorted:
        marker = " ◀" if cat in _CRITICAL_CATEGORIES else ""
        row = f"  {cat:<35}"
        for i, d in enumerate(datasets):
            earned, maximum = _cat_score_pct(d, cat)
            pct = (earned / maximum * 100) if maximum > 0 else 0.0
            row += f"  {pct:>{col_w - 1}.1f}%"
            totals_earned[i] += earned
            totals_max[i] += maximum
        print(f"{row}{marker}")

    print(sep)
    row = f"  {'TOTAL':<35}"
    for i in range(len(datasets)):
        pct = (totals_earned[i] / totals_max[i] * 100) if totals_max[i] > 0 else 0.0
        row += f"  {pct:>{col_w - 1}.1f}%"
    print(row)
    print(sep)


# ---------------------------------------------------------------------------
# Ingest helper
# ---------------------------------------------------------------------------

def _run_ingest() -> None:
    logger.info("Starting document ingestion into vector store …")
    from rag import ingest_documents  # noqa: PLC0415

    n = ingest_documents()
    logger.info("Ingestion complete: %d chunk(s) stored", n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark runner for multi-model RAG evaluation"
    )
    sub = parser.add_subparsers(dest="command")

    # --- run (default) ---
    run_p = sub.add_parser("run", help="Run benchmark cases against a model")
    run_p.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "gemma3:27b-it-q4_K_M"),
                       help="Ollama model name (default: OLLAMA_MODEL env or gemma3:27b-it-q4_K_M)")
    run_p.add_argument("--base-url", default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                       help="Ollama base URL")
    run_p.add_argument("--timeout", type=int, default=120,
                       help="Per-request timeout in seconds (default: 120)")
    run_p.add_argument("--cases", default=str(_DEFAULT_CASES),
                       help="Path to JSONL benchmark file")
    run_p.add_argument("--output", default=None,
                       help="Output JSON file path (default: auto-named in benchmark_results/)")
    run_p.add_argument("--ingest", action="store_true",
                       help="Re-ingest all docs into the vector store before running")
    run_p.add_argument("--no-rag", action="store_true",
                       help="Skip RAG retrieval (no policy/doc context)")
    run_p.add_argument("--no-shopify", action="store_true",
                       help="Skip Shopify MCP product search")
    run_p.add_argument("--category", default=None,
                       help="Run only cases matching this category (substring match)")
    run_p.add_argument("--limit", type=int, default=None,
                       help="Cap the number of cases to run (for quick smoke tests)")

    # --- compare ---
    cmp_p = sub.add_parser("compare", help="Compare two or more result JSON files")
    cmp_p.add_argument("files", nargs="+", help="Result JSON files to compare")

    # --- ingest only ---
    sub.add_parser("ingest", help="Ingest documents without running the benchmark")

    # Support calling without sub-command: treat all positional-looking flags as 'run'
    args, _ = parser.parse_known_args()
    if args.command is None:
        # Re-parse as 'run' (handles legacy `python run_benchmark.py --model X`)
        sys.argv.insert(1, "run")
        return parser.parse_args()

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.command == "ingest":
        _run_ingest()
        return

    if args.command == "compare":
        _print_comparison(args.files)
        return

    # ---- run ----
    model: str = args.model
    base_url: str = args.base_url
    timeout: int = args.timeout
    use_rag: bool = not args.no_rag
    use_shopify: bool = not args.no_shopify

    if args.ingest:
        _run_ingest()

    # Load cases
    cases_path = Path(args.cases)
    if not cases_path.exists():
        sys.exit(f"Cases file not found: {cases_path}")

    cases: list[dict[str, Any]] = []
    with open(cases_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))

    if args.category:
        cases = [c for c in cases if args.category.lower() in c.get("category", "").lower()]
        logger.info("Filtered to %d case(s) matching category=%r", len(cases), args.category)

    if args.limit:
        cases = cases[: args.limit]
        logger.info("Limited to first %d case(s)", len(cases))

    logger.info(
        "Starting benchmark | model=%s cases=%d rag=%s shopify=%s timeout=%ss",
        model,
        len(cases),
        "yes" if use_rag else "no",
        "yes" if use_shopify else "no",
        timeout,
    )

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    t_start = time.time()

    for i, case in enumerate(cases, 1):
        case_id = case.get("id", f"#{i}")
        logger.info("[%d/%d] %s — %s", i, len(cases), case_id, case.get("user_question", "")[:60])
        result = run_case(case, model, base_url, timeout, use_rag, use_shopify)
        results.append(result)

        # Live progress
        score_char = {2: "✓", 1: "~", 0: "✗"}.get(result["score"], "?")
        extra = ""
        if result.get("error"):
            extra = f" ERROR: {result['error'][:60]}"
        elif result["missed_phrases"]:
            extra = f" missed={result['missed_phrases']}"
        logger.info(
            "  [%s] score=%d  t_llm=%.1fs%s",
            score_char,
            result["score"],
            result["t_llm_s"],
            extra,
        )

    elapsed = round(time.time() - t_start, 1)

    # Output
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", model)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else _RESULTS_DIR / f"{slug}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_data: dict[str, Any] = {
        "model": model,
        "base_url": base_url,
        "timestamp": datetime.now().isoformat(),
        "rag_enabled": use_rag,
        "shopify_enabled": use_shopify,
        "timeout_s": timeout,
        "cases_file": str(cases_path),
        "total_cases": len(results),
        "total_elapsed_s": elapsed,
        "cases": results,
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output_data, fh, ensure_ascii=False, indent=2)

    logger.info("Results written to %s", out_path)

    _print_summary(results, model, elapsed)


if __name__ == "__main__":
    main()
