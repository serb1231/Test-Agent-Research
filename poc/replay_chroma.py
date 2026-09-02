"""Phase 2 — replay (article §01, §05). Reads trace.jsonl + vectors.npy
produced by golden_run.py and feeds the identical operation sequence into a
fresh Chroma collection. No LLM calls, no new embeddings — every vector is
looked up by the integer index recorded during the golden run. $0 to run.

For every search op, diffs Chroma's answer against the golden answer that
was embedded inline in the trace:
  - pool recall@k'  : did Chroma's ANN candidate pool contain the same ids
                       the exact reference found?
  - composite agreement@k : after rescoring Chroma's own candidates with
                       the identical composite formula, does its top-k match
                       the golden top-k?

At this tiny scale (~90 memories) both should land near 1.0 — Chroma's
HNSW is effectively exact over a corpus this small. That's the correct
sanity-check result (mirrors the article's P3 gate: the harness is only
trustworthy once it reproduces the reference on data too small to diverge).
Drift only shows up at the scale the real study targets.
"""

import json
import shutil
import time
from datetime import datetime

import numpy as np

from chroma_store import ChromaStore
from config import VECTORS_PATH, TRACE_PATH, CHROMA_PATH, REPLAY_METRICS_PATH
from rescore import rescore


def parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def run():
    # Replay must be a pure function of the trace: re-running it against a
    # store that already holds the previous run's data silently duplicates
    # every id (no error) and corrupts recall — confirmed empirically (1.000
    # dropped to 0.631 on a stale collection). Always start from a clean slate.
    shutil.rmtree(CHROMA_PATH, ignore_errors=True)
    vectors = np.load(VECTORS_PATH)
    store = ChromaStore(path=CHROMA_PATH)

    with open(TRACE_PATH) as f:
        lines = f.readlines()

    print(f"Replaying {len(lines)} ops into Chroma ({vectors.shape[0]} vectors loaded)...")

    recall_pool_scores = []
    composite_agreement_scores = []
    n_search = 0
    n_insert = 0
    n_memories = 0  # running count of rows in the store as replay progresses

    # Every timed op gets its own record — seq, verb, subtype, latency, and
    # how many memories existed in the store at that moment — written to
    # REPLAY_METRICS_PATH so it can be plotted against operations elapsed
    # later (article §09's "seq_at_sample" framing), not just aggregated
    # into one blind p50/p95/p99 at the end.
    metrics_fh = open(REPLAY_METRICS_PATH, "w")
    # (op_verb, sub_kind) -> [latency_us, ...], for the end-of-run summary
    latencies_by = {}

    def record(seq, op_verb, sub_kind, latency_us, count_at_time):
        metrics_fh.write(json.dumps({
            "seq": seq, "op": op_verb, "kind": sub_kind,
            "latency_us": latency_us, "store_count": count_at_time,
        }) + "\n")
        latencies_by.setdefault((op_verb, sub_kind), []).append(latency_us)

    for line in lines:
        op = json.loads(line)
        op_verb = op["op"]

        if op_verb == "insert":
            vec = vectors[op["vec"]].tolist()
            meta = {
                "agent_id": op["agent"], "type": op["type"], "text": op["text"],
                "importance": op["importance"], "created_at": parse_dt(op["created_at"]),
                "last_accessed_at": parse_dt(op["created_at"]),
                "derived_from": op.get("derived_from", []),
            }
            t0 = time.perf_counter_ns()
            store.insert(op["id"], vec, meta)
            latency_us = (time.perf_counter_ns() - t0) // 1000
            n_memories += 1
            record(op["seq"], "insert", op.get("kind"), latency_us, n_memories)
            n_insert += 1

        elif op_verb == "search":
            qvec = vectors[op["qvec"]].tolist()
            now = parse_dt(op["sim_t"])
            t0 = time.perf_counter_ns()
            candidates = store.search(qvec, op["filter"], op["k_prime"], now)
            latency_us = (time.perf_counter_ns() - t0) // 1000
            record(op["seq"], "search", op.get("kind"), latency_us, n_memories)

            golden_pool_ids = {p["id"] for p in op["pool"]}
            returned_ids = {c["id"] for c in candidates}
            if golden_pool_ids:
                recall_pool_scores.append(len(returned_ids & golden_pool_ids) / len(golden_pool_ids))

            chroma_top_ids = {c["id"] for c in rescore(candidates, now, op["k"])}
            golden_composite_ids = set(op["composite_top"])
            if golden_composite_ids:
                composite_agreement_scores.append(
                    len(chroma_top_ids & golden_composite_ids) / len(golden_composite_ids)
                )
            n_search += 1

        elif op_verb == "touch":
            t0 = time.perf_counter_ns()
            store.touch(op["ids"], parse_dt(op["at"]))
            latency_us = (time.perf_counter_ns() - t0) // 1000
            record(op["seq"], "touch", op.get("kind"), latency_us, n_memories)

        elif op_verb == "update":
            store.update(op["id"], op["fields"])

        elif op_verb == "delete":
            store.delete(op["ids"])

    metrics_fh.close()

    print("\n=== Replay done ===")
    print("Chroma store stats:", store.stats())
    print(f"Ops replayed: inserts={n_insert}, searches={n_search}")
    if recall_pool_scores:
        print(f"Mean pool recall@k': {sum(recall_pool_scores)/len(recall_pool_scores):.3f}  (n={len(recall_pool_scores)})")
    if composite_agreement_scores:
        print(f"Mean composite agreement@k: {sum(composite_agreement_scores)/len(composite_agreement_scores):.3f}  (n={len(composite_agreement_scores)})")

    print("\n=== Latency (store-only, microseconds), broken down by op subtype ===")
    for (op_verb, sub_kind), samples in sorted(latencies_by.items()):
        arr = np.array(samples)
        p50, p95, p99 = np.percentile(arr, [50, 95, 99])
        label = f"{op_verb}:{sub_kind}" if sub_kind else op_verb
        print(f"{label:20s}  n={len(arr):4d}  p50={p50:8.1f}us  p95={p95:8.1f}us  p99={p99:8.1f}us  mean={arr.mean():8.1f}us")
    print(f"\nPer-op records written to {REPLAY_METRICS_PATH} ({sum(len(v) for v in latencies_by.values())} rows) for plotting against seq/store_count later.")


if __name__ == "__main__":
    run()
