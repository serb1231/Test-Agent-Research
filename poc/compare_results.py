"""Print the Qdrant and PostgreSQL runs side by side.

Run each benchmark first (poc/Qdrant/main.py, poc/PostgreSQL/main.py); each one
drops a JSON record in poc/results/. This only reads those files, so the two runs
need not happen on the same day -- though they should happen on the same machine
with the same MAX_THREADS, or the latencies are not comparable.

    python compare_results.py
"""

import json
import sys
from pathlib import Path

from benchmark_report import TIMING_SERIES

RESULTS_DIR = Path(__file__).resolve().parent / "results"
# (column heading, file). Qdrant first, PG second, so the ratio reads
# "Qdrant / PostgreSQL": below 1.0 means Qdrant was faster.
ENGINES = (("Qdrant", "qdrant.json"), ("PostgreSQL", "postgresql.json"))

QUALITY_KEYS = ("overlap", "precision_at_5", "ndcg")


def load():
    reports = []
    for heading, filename in ENGINES:
        path = RESULTS_DIR / filename
        if not path.exists():
            sys.exit(f"missing {path} -- run that engine's main.py first")
        reports.append((heading, json.loads(path.read_text())))
    return reports


def ratio(a, b):
    """a relative to b, guarding the division so a zero never kills the table."""
    if not b:
        return "n/a"
    return f"{a / b:.2f}x"


def row(label, left, right, fmt="{:.3f}", with_ratio=True):
    l_txt = fmt.format(left) if left is not None else "-"
    r_txt = fmt.format(right) if right is not None else "-"
    tail = ratio(left, right) if (with_ratio and left is not None and right is not None) else ""
    print(f"{label:<26}{l_txt:>14}{r_txt:>14}{tail:>10}")


def main():
    reports = load()
    (left_name, left), (right_name, right) = reports

    print()
    print(f"{'':<26}{left_name:>14}{right_name:>14}{'ratio':>10}")
    print("-" * 64)

    for heading, rep in reports:
        if rep["failed_agents"]:
            print(f"!! {heading}: agents crashed mid-run: {rep['failed_agents']}")
    if left["max_threads"] != right["max_threads"]:
        print(f"!! different MAX_THREADS ({left['max_threads']} vs "
              f"{right['max_threads']}); latencies are not comparable")
    if left["total_ops"] != right["total_ops"]:
        print(f"!! different op counts ({left['total_ops']} vs {right['total_ops']}); "
              f"the two runs did not replay the same trace")

    print("\nrun")
    row("agents", left["agents"], right["agents"], "{:.0f}", with_ratio=False)
    row("total ops", left["total_ops"], right["total_ops"], "{:.0f}", with_ratio=False)
    row("wall clock (s)", left["wall_clock_s"], right["wall_clock_s"], "{:.1f}")
    row("ops/s", left["ops_per_s"], right["ops_per_s"], "{:.0f}")

    # The ratio column is the useful part: a p99 ratio far from the p50 ratio is
    # the interesting result, because it means the two engines differ in tail
    # behaviour under concurrency rather than in raw speed.
    for stat in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
        print(f"\n{stat.replace('_ms', '')} latency (ms)")
        for label, _ in TIMING_SERIES:
            l = left["latency"].get(label)
            r = right["latency"].get(label)
            row(label, l and l[stat], r and r[stat])

    print("\nthroughput by op (ops/s)")
    for label, _ in TIMING_SERIES:
        l = left["latency"].get(label)
        r = right["latency"].get(label)
        row(label, l and l["throughput_per_s"], r and r["throughput_per_s"], "{:.0f}")

    lq, rq = left["quality"], right["quality"]
    if lq and rq:
        print("\nretrieval quality (micro-averaged over all searches)")
        row("searches", lq["searches"], rq["searches"], "{:.0f}", with_ratio=False)
        for key in QUALITY_KEYS:
            row(key, lq[key], rq[key], "{:.4f}")
        print("\nper-agent spread (min / median / max)")
        for key in QUALITY_KEYS:
            for heading, q in ((left_name, lq), (right_name, rq)):
                s = q[f"{key}_per_agent"]
                print(f"  {key:<16}{heading:<12}"
                      f"{s['min']:.4f} / {s['median']:.4f} / {s['max']:.4f}")
    print()


if __name__ == "__main__":
    main()
