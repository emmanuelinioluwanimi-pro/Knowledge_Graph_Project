"""
evaluate.py — Formal evaluation for Multimodal Hybrid GraphRAG (V2)
====================================================================

BATCH MODE: on each session, prompts you for how many NEW questions to test.
Automatically resumes from the persistent run folder — questions already
completed (all 3 modes) are skipped. Progress persists across sessions
via `results.jsonl` in a persistent folder (`evaluation_results/current/`).

Metrics computed (deterministic, no LLM judge):
    RETRIEVAL     Recall@5, Precision@5, MRR, nDCG@5
    ANSWER        Exact-fact match, Refusal accuracy
    SYSTEM        Mean latency, Mean answer length

Outputs:
    evaluation_results/current/
        results.jsonl   — appended after every question (persistent progress)
        summary.json    — rewritten after every session (metrics over all
                          questions tested so far)
        report.md       — rewritten after every session (human-readable)

Usage
-----
    import evaluate
    evaluate.run_evaluation()   # prompts for batch size

Or non-interactive:
    evaluate.run_evaluation(questions_this_session=3)   # test next 3 untested
    evaluate.run_evaluation(questions_this_session=999) # test all remaining
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_PIPELINE_DIR = Path("/content/KG-RAG-System-V2")
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

import pipeline  # noqa: E402


# =============================================================================
# CONFIG
# =============================================================================

QUESTIONS_PATH = _PIPELINE_DIR / "data" / "evaluation" / "questions.json"
RESULTS_ROOT   = _PIPELINE_DIR / "evaluation_results"
RUN_DIR        = RESULTS_ROOT / "current"   # single persistent folder

MODES = ("baseline", "kg_only", "hybrid")

REFUSAL_MARKERS = (
    "outside the scope", "cannot answer", "not covered",
    "no information", "unable to answer", "i don't have",
    "not in the", "outside of the scope",
)


# =============================================================================
# LENIENT FACT MATCHING
# =============================================================================

_STRIP_CHARS = re.compile(r"[\s\.,;:!?\"()\[\]{}·°\u00a0\u201c\u201d\u2018\u2019/]+")


def _normalize(s: str) -> str:
    if not s:
        return ""
    return _STRIP_CHARS.sub("", s.lower())


def _fact_matches(fact: str, answer: str) -> bool:
    nf = _normalize(fact)
    if not nf:
        return False
    return nf in _normalize(answer)


def _exact_fact_score(key_facts: List[str], answer: str) -> float:
    if not key_facts:
        return 0.0
    hits = sum(1 for f in key_facts if _fact_matches(f, answer))
    return hits / len(key_facts)


def _looks_like_refusal(answer: str) -> bool:
    if not answer:
        return True
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


# =============================================================================
# RETRIEVAL METRICS
# =============================================================================

def _recall_at_k(expected: List[str], retrieved: List[str], k: int = 5) -> float:
    if not expected:
        return 0.0
    top_k = set(retrieved[:k])
    return sum(1 for e in expected if e in top_k) / len(expected)


def _precision_at_k(expected: List[str], retrieved: List[str], k: int = 5) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    expected_set = set(expected)
    return sum(1 for c in top_k if c in expected_set) / len(top_k)


def _reciprocal_rank(expected: List[str], retrieved: List[str]) -> float:
    expected_set = set(expected)
    for i, c in enumerate(retrieved, start=1):
        if c in expected_set:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(expected: List[str], retrieved: List[str], k: int = 5) -> float:
    if not expected or not retrieved:
        return 0.0
    expected_set = set(expected)
    dcg = sum(1.0 / math.log2(i + 1)
              for i, c in enumerate(retrieved[:k], start=1)
              if c in expected_set)
    ideal_hits = min(len(expected_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# =============================================================================
# METRIC AGGREGATION
# =============================================================================

def _compute_metrics(per_question: List[Dict[str, Any]],
                     mode: str) -> Dict[str, Any]:
    subset = [r for r in per_question if r["mode"] == mode]
    if not subset:
        return {}

    n = len(subset)
    in_scope  = [r for r in subset if r["expected_type"] != "out_of_scope"]
    out_scope = [r for r in subset if r["expected_type"] == "out_of_scope"]

    fact_scores = [_exact_fact_score(r["key_facts"], r["answer"]) for r in in_scope]
    exact_fact_mean = statistics.mean(fact_scores) if fact_scores else 0.0

    correct_refusals = 0
    for r in subset:
        refused = _looks_like_refusal(r["answer"])
        should_refuse = (r["expected_type"] == "out_of_scope")
        if refused == should_refuse:
            correct_refusals += 1
    refusal_acc = correct_refusals / n

    recalls, precisions, mrrs, ndcgs = [], [], [], []
    for r in in_scope:
        expected = r["expected_chunk_ids"]
        retrieved = [s["chunk_id"] for s in r["sources"]]
        if not expected:
            continue
        recalls.append(_recall_at_k(expected, retrieved, 5))
        precisions.append(_precision_at_k(expected, retrieved, 5))
        mrrs.append(_reciprocal_rank(expected, retrieved))
        ndcgs.append(_ndcg_at_k(expected, retrieved, 5))

    recall_at_5    = statistics.mean(recalls)    if recalls    else 0.0
    precision_at_5 = statistics.mean(precisions) if precisions else 0.0
    mrr            = statistics.mean(mrrs)       if mrrs       else 0.0
    ndcg_at_5      = statistics.mean(ndcgs)      if ndcgs      else 0.0

    latencies = [r["latency_ms"] / 1000.0 for r in subset]
    lengths   = [len(r["answer"].split()) for r in subset if r["answer"]]

    return {
        "exact_fact_match":         exact_fact_mean,
        "refusal_accuracy":         refusal_acc,
        "recall_at_5":              recall_at_5,
        "precision_at_5":           precision_at_5,
        "mrr":                      mrr,
        "ndcg_at_5":                ndcg_at_5,
        "mean_latency_sec":         statistics.mean(latencies) if latencies else 0.0,
        "mean_answer_length_words": statistics.mean(lengths)   if lengths   else 0.0,
        "n_questions":              n,
        "n_in_scope":               len(in_scope),
        "n_out_of_scope":           len(out_scope),
    }


# =============================================================================
# REPORT
# =============================================================================

def _format_summary_table(summary: Dict[str, Dict[str, Any]]) -> str:
    baseline = summary.get("baseline", {})
    kg_only  = summary.get("kg_only",  {})
    hybrid   = summary.get("hybrid",   {})

    def _fmt_pct(v):
        return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "n/a"
    def _fmt_num(v, d=3):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) else "n/a"

    rows = [
        ("Exact-fact match",           _fmt_pct(baseline.get("exact_fact_match")),
                                       _fmt_pct(kg_only.get("exact_fact_match")),
                                       _fmt_pct(hybrid.get("exact_fact_match"))),
        ("Refusal accuracy",           _fmt_pct(baseline.get("refusal_accuracy")),
                                       _fmt_pct(kg_only.get("refusal_accuracy")),
                                       _fmt_pct(hybrid.get("refusal_accuracy"))),
        ("Recall@5",                   _fmt_pct(baseline.get("recall_at_5")),
                                       _fmt_pct(kg_only.get("recall_at_5")),
                                       _fmt_pct(hybrid.get("recall_at_5"))),
        ("Precision@5",                _fmt_pct(baseline.get("precision_at_5")),
                                       _fmt_pct(kg_only.get("precision_at_5")),
                                       _fmt_pct(hybrid.get("precision_at_5"))),
        ("MRR",                        _fmt_num(baseline.get("mrr")),
                                       _fmt_num(kg_only.get("mrr")),
                                       _fmt_num(hybrid.get("mrr"))),
        ("nDCG@5",                     _fmt_num(baseline.get("ndcg_at_5")),
                                       _fmt_num(kg_only.get("ndcg_at_5")),
                                       _fmt_num(hybrid.get("ndcg_at_5"))),
        ("Mean latency (sec)",         _fmt_num(baseline.get("mean_latency_sec"), 2),
                                       _fmt_num(kg_only.get("mean_latency_sec"),  2),
                                       _fmt_num(hybrid.get("mean_latency_sec"),   2)),
        ("Mean answer length (words)", _fmt_num(baseline.get("mean_answer_length_words"), 1),
                                       _fmt_num(kg_only.get("mean_answer_length_words"),  1),
                                       _fmt_num(hybrid.get("mean_answer_length_words"),   1)),
    ]

    lines = [
        f"{'Metric':<32}{'Baseline':>12}{'KG-only':>12}{'Hybrid':>12}",
        "-" * 68,
    ]
    for name, b, k, h in rows:
        lines.append(f"{name:<32}{b:>12}{k:>12}{h:>12}")
    return "\n".join(lines)


def _write_report(out_dir: Path,
                  summary: Dict[str, Dict[str, Any]],
                  per_question: List[Dict[str, Any]],
                  n_total: int) -> None:
    n_tested = summary.get("baseline", {}).get("n_questions", 0)
    lines: List[str] = []
    lines.append(f"# Evaluation Report — {out_dir.name}")
    lines.append("")
    lines.append(f"**Progress**: {n_tested} / {n_total} questions completed")
    lines.append(f"**Modes evaluated**: {', '.join(MODES)}")
    lines.append("**Evaluation method**: fully deterministic (no LLM-as-judge)")
    lines.append("")

    lines.append("## Results table")
    lines.append("")
    lines.append("```")
    lines.append(_format_summary_table(summary))
    lines.append("```")
    lines.append("")

    lines.append("## Per-question breakdown")
    lines.append("")

    from collections import defaultdict
    by_qid: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_question:
        by_qid[r["id"]].append(r)

    for qid in sorted(by_qid):
        entries = by_qid[qid]
        q0 = entries[0]
        lines.append(f"### {qid} — {q0['question']}")
        lines.append(f"*Manual: {q0['manual']}  ·  Type: {q0['expected_type']}  "
                     f"·  Difficulty: {q0['difficulty']}*")
        lines.append("")
        lines.append(f"**Expected**: {q0['expected_answer']}")
        if q0["key_facts"]:
            lines.append(f"**Key facts**: {q0['key_facts']}")
        lines.append("")
        for mode in MODES:
            entry = next((e for e in entries if e["mode"] == mode), None)
            if not entry:
                continue
            fact_score = _exact_fact_score(entry["key_facts"], entry["answer"])
            found = [f for f in entry["key_facts"] if _fact_matches(f, entry["answer"])]
            missing = [f for f in entry["key_facts"] if not _fact_matches(f, entry["answer"])]
            lines.append(f"- **{mode}** — exact-fact {fact_score:.0%}, "
                         f"latency {entry['latency_ms']/1000:.1f}s")
            if entry["key_facts"]:
                if found:
                    lines.append(f"    facts found: {found}")
                if missing:
                    lines.append(f"    facts missing: {missing}")
            ans = (entry["answer"] or "").replace("\n", " ")
            if len(ans) > 300:
                ans = ans[:300] + " ..."
            lines.append(f"    Answer: {ans}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))


# =============================================================================
# CHECKPOINT + PROGRESS
# =============================================================================

def _load_questions() -> List[Dict[str, Any]]:
    with open(QUESTIONS_PATH) as f:
        data = json.load(f)
    return data["questions"]


def _load_checkpoint(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _append_checkpoint(path: Path, entry: Dict[str, Any]) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def _completed_question_ids(checkpoint: List[Dict[str, Any]]) -> set:
    """A question is 'done' if all 3 modes are in checkpoint."""
    from collections import defaultdict
    modes_by_qid: Dict[str, set] = defaultdict(set)
    for r in checkpoint:
        modes_by_qid[r["id"]].add(r["mode"])
    return {qid for qid, modes in modes_by_qid.items()
            if set(MODES).issubset(modes)}


# =============================================================================
# MAIN
# =============================================================================

def run_evaluation(questions_this_session: Optional[int] = None) -> Path:
    """
    Batch-mode evaluation.

    questions_this_session:
        None  → prompt user interactively
        int   → run at most this many NEW questions this session

    Progress persists in evaluation_results/current/ across sessions.
    Run multiple times to complete the full question set incrementally.
    """
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = RUN_DIR / "results.jsonl"

    questions = _load_questions()
    n_total = len(questions)

    checkpoint = _load_checkpoint(checkpoint_path)
    done_ids = _completed_question_ids(checkpoint)
    untested = [q for q in questions if q["id"] not in done_ids]

    print(f"[eval] persistent run folder: {RUN_DIR}")
    print(f"[eval] {len(done_ids)}/{n_total} questions already completed")
    print(f"[eval] {len(untested)} questions remaining")

    if not untested:
        print("[eval] all questions already completed — nothing to do")
        print("[eval] regenerating summary + report from existing checkpoint...")
        summary = {mode: _compute_metrics(checkpoint, mode) for mode in MODES}
        (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        _write_report(RUN_DIR, summary, checkpoint, n_total)
        print(_format_summary_table(summary))
        return RUN_DIR

    # Decide batch size
    if questions_this_session is None:
        default_n = min(3, len(untested))
        try:
            raw = input(f"[eval] how many questions this session? "
                        f"(default {default_n}, max {len(untested)}): ").strip()
            questions_this_session = int(raw) if raw else default_n
        except (EOFError, ValueError):
            questions_this_session = default_n
    batch_ids = [q["id"] for q in untested[:max(0, questions_this_session)]]
    print(f"[eval] running {len(batch_ids)} question(s) this session: {batch_ids}")
    print(f"[eval] each question = 3 modes = {len(batch_ids) * 3} pipeline runs")

    per_question: List[Dict[str, Any]] = list(checkpoint)
    done_keys = {(r["id"], r["mode"]) for r in checkpoint}
    total_runs_this_session = len(batch_ids) * len(MODES)
    run_idx = 0
    t_wall_start = time.time()

    for q in questions:
        if q["id"] not in batch_ids:
            continue
        for mode in MODES:
            key = (q["id"], mode)
            if key in done_keys:
                continue
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs_this_session}] {q['id']}  mode={mode}")
            print(f"    Q: {q['question']}")

            t0 = time.time()
            try:
                result = pipeline.answer_query(q["question"], mode=mode)
            except Exception as e:
                print(f"    [pipeline error] {e}")
                result = {
                    "answer": "",
                    "sources": [],
                    "confidence": "error",
                    "top_score": 0.0,
                    "kg_triples": [],
                    "latency_ms": int((time.time() - t0) * 1000),
                }

            fact_score = _exact_fact_score(q["key_facts"], result["answer"])

            entry = {
                "id":                 q["id"],
                "question":           q["question"],
                "manual":             q["manual"],
                "expected_type":      q["expected_type"],
                "difficulty":         q["difficulty"],
                "expected_answer":    q["expected_answer"],
                "expected_chunk_ids": q["expected_chunk_ids"],
                "key_facts":          q["key_facts"],
                "mode":               mode,
                "answer":             result["answer"],
                "sources":            result["sources"],
                "kg_triples":         result["kg_triples"],
                "confidence":         result["confidence"],
                "top_score":          result["top_score"],
                "latency_ms":         result["latency_ms"],
            }
            per_question.append(entry)
            _append_checkpoint(checkpoint_path, entry)
            print(f"    exact-fact {fact_score:.0%}   ({result['latency_ms']/1000:.1f}s)")

    # Rebuild summary + report over EVERYTHING tested so far
    print("\n[eval] session complete. rebuilding summary + report over all completed questions...")
    summary = {mode: _compute_metrics(per_question, mode) for mode in MODES}
    (RUN_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    _write_report(RUN_DIR, summary, per_question, n_total)

    new_done_ids = _completed_question_ids(per_question)
    wall = time.time() - t_wall_start
    print(f"[eval] session wall time: {wall/60:.1f} min")
    print(f"[eval] total progress: {len(new_done_ids)}/{n_total} questions")
    print(f"[eval] outputs at: {RUN_DIR}")
    print()
    print(_format_summary_table(summary))
    return RUN_DIR


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="Number of questions to test this session (default: prompt)")
    args = ap.parse_args()
    run_evaluation(questions_this_session=args.n)
