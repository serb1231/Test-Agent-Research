"""Run-level aggregation shared by the PostgreSQL and Qdrant benchmarks.

Both Agent classes already collect the same four timing lists and the same
cumulative quality sums, so the aggregation lives here once. Keeping a single
implementation is the point: if the two engines summarised their numbers with
even slightly different definitions, the comparison would be meaningless.

Imported by each main.py, which sits one directory below this file.
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np

NS_PER_MS = 1_000_000

# (label, Agent attribute). The four phases are timed separately because they
# cost very different things: a vector write, an ANN read, a payload/row update,
# and the audit-log write that both engines pay on every search and touch.
TIMING_SERIES = (
    ("insertion", "insertion_times"),
    ("search", "search_times"),
    ("modification", "modification_times"),
    ("log", "log_times"),
)


def _latency_stats(samples_ns):
    """Percentiles over the pooled samples of every agent.

    Pooling the raw samples is the only correct way to do this. Averaging each
    agent's p95 would produce a number that is not a percentile of anything --
    the mean of 100 tail estimates is not the tail of the run.
    """
    if not samples_ns:
        return None
    arr = np.asarray(samples_ns, dtype=np.float64) / NS_PER_MS
    p50, p95, p99 = (float(x) for x in np.percentile(arr, (50, 95, 99)))
    return {
        "count": int(arr.size),
        "mean_ms": float(arr.mean()),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": float(arr.max()),
        "total_s": float(arr.sum() / 1000.0),
    }


def _quality_stats(agents):
    """Retrieval quality micro-averaged over every search in the run.

    Micro, not macro: agents do not all issue the same number of searches, so
    averaging their per-agent averages would weight a 40-search agent the same
    as a 900-search one. The per-agent spread is reported separately, which says
    more about consistency than a macro average would anyway.
    """
    searching = [a for a in agents if a.total_searches > 0]
    total = sum(a.total_searches for a in searching)
    if not total:
        return None

    out = {"searches": total, "agents_searching": len(searching)}
    for key, attr in (("overlap", "cum_overlap"),
                      ("precision_at_5", "cum_p_at_5"),
                      ("ndcg", "cum_ndcg")):
        out[key] = sum(getattr(a, attr) for a in searching) / total
        per_agent = sorted(getattr(a, attr) / a.total_searches for a in searching)
        out[f"{key}_per_agent"] = {
            "min": per_agent[0],
            "median": float(np.median(per_agent)),
            "max": per_agent[-1],
        }
    return out


def summarise(engine, agents, wall_clock_s, max_threads, failed_agents=()):
    """Collapse every agent's samples into one comparable record."""
    ops = {}
    for label, attr in TIMING_SERIES:
        stats = _latency_stats([t for a in agents for t in getattr(a, attr)])
        if stats is not None:
            # Throughput is over the whole run, so it reflects what the engine
            # actually delivered with MAX_THREADS agents competing, not what one
            # idle client could achieve.
            stats["throughput_per_s"] = stats["count"] / wall_clock_s if wall_clock_s else None
            ops[label] = stats

    total_ops = sum(s["count"] for s in ops.values())
    return {
        "engine": engine,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "agents": len(agents),
        "failed_agents": sorted(failed_agents),
        "max_threads": max_threads,
        "wall_clock_s": wall_clock_s,
        "total_ops": total_ops,
        "ops_per_s": total_ops / wall_clock_s if wall_clock_s else None,
        "latency": ops,
        "quality": _quality_stats(agents),
    }


def log_summary(report):
    engine = report["engine"]
    logging.info("=" * 68)
    logging.info(f"RUN SUMMARY -- {engine}")
    logging.info("=" * 68)
    logging.info(
        f"{report['agents']} agents over {report['max_threads']} threads, "
        f"{report['total_ops']} ops in {report['wall_clock_s']:.1f}s "
        f"({report['ops_per_s']:.0f} ops/s)"
    )
    if report["failed_agents"]:
        logging.warning(f"agents that crashed: {report['failed_agents']}")

    # Latencies are measured with max_threads agents hitting the engine at once,
    # so they are latencies under load. Worth remembering before reading a p99.
    logging.info(f"--- {engine} latency, pooled over all agents (ms) ---")
    logging.info(f"{'op':<14}{'n':>9}{'mean':>10}{'p50':>10}{'p95':>10}{'p99':>10}{'ops/s':>10}")
    for label, _ in TIMING_SERIES:
        s = report["latency"].get(label)
        if s:
            logging.info(
                f"{label:<14}{s['count']:>9}{s['mean_ms']:>10.3f}{s['p50_ms']:>10.3f}"
                f"{s['p95_ms']:>10.3f}{s['p99_ms']:>10.3f}{s['throughput_per_s']:>10.0f}"
            )

    q = report["quality"]
    if q:
        logging.info(f"--- {engine} retrieval quality over {q['searches']} searches ---")
        for key in ("overlap", "precision_at_5", "ndcg"):
            spread = q[f"{key}_per_agent"]
            logging.info(
                f"{key:<16}{q[key]:>8.4f}   "
                f"(per-agent min {spread['min']:.4f} / median {spread['median']:.4f} / max {spread['max']:.4f})"
            )
    logging.info("=" * 68)


def write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    logging.info(f"summary written to {path}")
    return path
