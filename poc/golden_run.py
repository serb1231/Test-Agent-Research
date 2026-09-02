"""Phase 1 — the golden run (article §01, §05). Runs the agent simulation
ONCE, driven by real LLM calls, against NumpyReference (exact, no
approximation). Every operation is logged to trace.jsonl; every vector
(memory content AND query vectors) is assigned a integer index and appended
to a shared array saved as vectors.npy. Search ops additionally record the
golden answer inline — the exact candidate pool and the composite-rescored
top-k — so replay never needs a second ground-truth pass.

This is the only script that costs money or calls an LLM. Replay (see
replay_chroma.py) reads the artifacts this produces and is free.
"""

import json
import re
from datetime import datetime, timedelta

import numpy as np

from config import (
    VECTORS_PATH, TRACE_PATH, K, K_PRIME, OBSERVATIONS_PER_TICK, TICKS,
    REACT_IMPORTANCE_THRESHOLD, REFLECTION_IMPORTANCE_THRESHOLD,
    REFLECT_QUESTIONS, REFLECT_K,
)
from llm import embed, chat, usage_report
from numpy_store import NumpyReference
from personas import PERSONAS, LOCATIONS
from rescore import rescore

SIM_START = datetime(2026, 3, 4, 9, 0, 0)
TICK_MINUTES = 15

trace_fh = None
_seq = 0
_vectors: list[list[float]] = []


def log(op: str, **fields):
    global _seq
    _seq += 1
    rec = {"seq": _seq, "op": op, **fields}
    trace_fh.write(json.dumps(rec, default=str) + "\n")
    return rec


def register_vector(vec: list[float]) -> int:
    idx = len(_vectors)
    _vectors.append(vec)
    return idx

#inserts the vectors into the log which acts as the ground truth
def do_insert(store, agent_id, mid, vec, vec_idx, kind, mtype, importance, created_at, text, state, derived_from=None):
    derived_from = derived_from or []
    store.insert(mid, vec, {
        "agent_id": agent_id, "type": mtype, "text": text,
        "importance": importance, "created_at": created_at,
        "last_accessed_at": created_at,
        "derived_from": derived_from,
    })
    log("insert", agent=agent_id, id=mid, kind=kind, type=mtype, importance=importance,
        created_at=created_at, derived_from=derived_from, text=text, vec=vec_idx)
    state[agent_id]["recent"].append({"id": mid, "text": text, "importance": importance})
    return mid


def do_search(store, agent_id, qvec, qvec_idx, filter_, k_prime, k, now, trigger, kind, state):
    """Run an exact search against the golden store, log the full candidate
    pool AND the composite-rescored top-k as ground truth, touch the
    returned rows, and return the rescored top-k.

    kind labels *why* this search happened ("react" or "reflect") — logged
    explicitly rather than inferred later from the trigger string, so
    replay/analysis can bucket metrics by op subtype without string-sniffing.
    """
    candidates = store.search(qvec, filter_, k_prime, now)
    top = rescore(candidates, now, k)
    ids = [c["id"] for c in top]
    if ids:
        store.touch(ids, at=now)
    log("search", agent=agent_id, kind=kind, trigger=trigger, filter=filter_, qvec=qvec_idx,
        k=k, k_prime=k_prime, sim_t=now,
        pool=[{"id": c["id"], "distance": c["distance"]} for c in candidates],
        composite_top=ids)
    log("touch", agent=agent_id, kind=kind, ids=ids, at=now)
    return top


#start with a few sample memories for each persona so the simulation has something to work with , add them to the store and to the state.recent list so they can be used for reflection and retrieval
def seed(store, state):
    print("\n=== Seeding personas (golden run) ===")
    for agent_id, persona in PERSONAS.items():
        texts = [t for _, t in persona["seed_memories"]]
        vecs = embed(texts)
        created = SIM_START - timedelta(days=1)
        for i, ((importance, text), vec) in enumerate(zip(persona["seed_memories"], vecs)):
            idx = register_vector(vec)
            mid = f"{agent_id}_seed_{i}"
            do_insert(store, agent_id, mid, vec, idx, "seed", 0, importance, created, text, state)
        print(f"  {persona['name']}: {len(texts)} seed memories")


def perceive(agent_id, tick_time, ambient, state):
    persona = PERSONAS[agent_id]
    other_id = "a2" if agent_id == "a1" else "a1"
    other_name = PERSONAS[other_id]["name"]
    recent_texts = "\n".join(f"- {m['text']}" for m in state[agent_id]["recent"][-4:])

    system = "You simulate a specific person's stream of consciousness. Be concrete and concise."
    user = f"""You are simulating {persona['name']}, {persona['occupation']}.
Traits: {persona['traits']}. Current concerns: {persona['concerns']}.
It is {tick_time.strftime('%H:%M')} at {LOCATIONS['ceramics_lab']}.
{other_name} is also present.
{"Ambient event just now: " + ambient if ambient else ""}

Their last few memories:
{recent_texts if recent_texts else "(none yet)"}

Emit exactly {OBSERVATIONS_PER_TICK} things {persona['name']} notices or does in the next
15 minutes, one per line, each rated 1-10 for importance (1 = mundane, 10 = life-changing).
Format strictly as: RATING | TEXT
No preamble, no numbering, just the {OBSERVATIONS_PER_TICK} lines."""

    raw = chat(system, user)
    obs = []
    for line in raw.strip().splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*(.+)", line)
        if m:
            obs.append((int(m.group(1)), m.group(2).strip()))
    return obs[:OBSERVATIONS_PER_TICK]


def reflect(store, agent_id, tick_time, state):
    persona = PERSONAS[agent_id]
    recent = state[agent_id]["recent"][-10:]
    recent_block = "\n".join(f"{i+1}. {m['text']}" for i, m in enumerate(recent))

    q_system = "You identify high-level questions a stream of memories raises."
    q_user = f"""Here are {persona['name']}'s most recent memories:
{recent_block}

What are the {REFLECT_QUESTIONS} most salient high-level questions we can ask about this?
One per line, no numbering, no preamble."""
    questions_raw = chat(q_system, q_user)
    questions = [l.strip() for l in questions_raw.strip().splitlines() if l.strip()][:REFLECT_QUESTIONS]
    print(f"    [reflect] {agent_id} questions: {questions}")

    retrieved_by_q = []
    qvecs = embed(questions)
    for q, qvec in zip(questions, qvecs):
        qidx = register_vector(qvec)
        top = do_search(store, agent_id, qvec, qidx, {"agent_id": agent_id}, K_PRIME, REFLECT_K,
                         tick_time, f"reflect_q:{q}", "reflect", state)
        retrieved_by_q.append((q, top))

    numbered, index_to_id, counter = [], {}, 1
    for q, mems in retrieved_by_q:
        for m in mems:
            numbered.append(f"{counter}. {m['text']}")
            index_to_id[counter] = m["id"]
            counter += 1
    numbered_block = "\n".join(numbered)

    #give the past 4 insights to the model so it can avoid repeating them in the new insights , 
    # this is a defensive measure to avoid the model just repeating the same insight over and over again
    past_insights = state[agent_id]["past_insights"][-4:]
    past_block = (
        "\n".join(f"- {t}" for t in past_insights)
        if past_insights else "(none yet — this is the first reflection)"
    )

    s_system = "You synthesize high-level insights from a person's memories, citing which memories support each insight."
    s_user = f"""{persona['name']}'s memories, numbered:
{numbered_block}

Insights already reached in earlier reflections (do not just restate these —
if the evidence still points the same way, find a more specific detail, a
new angle, or how the situation has changed instead):
{past_block}

What 2 high-level insights can you infer? One per line.
Rules:
- Output ONLY the insight text followed by its citation — no leading label,
  no word "insight" or "INSIGHT" anywhere in the line.
- Format strictly as: <insight text> (because of N, M)
  where N, M are memory numbers from the list above.
- Cite each memory number at most once per insight.
Example of the expected shape (do not reuse this content):
Teo is starting to trust Mira's judgment on ramp-rate settings (because of 2, 7)"""
    insights_raw = chat(s_system, s_user)

    inserted = []
    for line in [l.strip() for l in insights_raw.strip().splitlines() if l.strip()]:
        m = re.match(r"(.+?)\s*\(because of\s*([\d,\s]+)\)", line)
        if not m:
            continue
        text = m.group(1).strip()
        text = re.sub(r"^insight\s*:?\s*", "", text, flags=re.IGNORECASE)  # defensive: strip a leading label even if the model still emits one
        if not text:
            continue
        nums = [int(n.strip()) for n in m.group(2).split(",") if n.strip().isdigit()]
        derived_ids = list(dict.fromkeys(index_to_id[n] for n in nums if n in index_to_id))  # defensive: dedupe citations, preserve order
        if not derived_ids:
            continue  # drop insights whose citations don't resolve (article §04)
        vec = embed([text])[0]
        idx = register_vector(vec)
        mid = f"{agent_id}_refl_{tick_time.strftime('%H%M')}_{len(inserted)}"
        do_insert(store, agent_id, mid, vec, idx, "reflection", 1, 7, tick_time, text, state, derived_ids)
        state[agent_id]["past_insights"].append(text)
        inserted.append(mid)
        print(f"    [reflect] insight: \"{text}\" <- {derived_ids}")
    return inserted


def run():
    global trace_fh
    store = NumpyReference()
    trace_fh = open(TRACE_PATH, "w")

    state = {aid: {"recent": [], "cum_importance": 0, "past_insights": []} for aid in PERSONAS}
    seed(store, state)

    ambient_events = {0: "Batch 7 came out of the kiln cracked along the rim."}

    print(f"\n=== Running {TICKS} ticks (golden run) ===")
    tick_time = SIM_START
    for t in range(TICKS):
        print(f"\n--- Tick {t} · {tick_time.strftime('%H:%M')} ---")
        ambient = ambient_events.get(t, "")
        for agent_id in PERSONAS:
            obs = perceive(agent_id, tick_time, ambient, state)
            if not obs:
                print(f"  {agent_id}: (perceive returned nothing parseable)")
                continue
            texts = [text for _, text in obs]
            vecs = embed(texts)
            for i, ((importance, text), vec) in enumerate(zip(obs, vecs)):
                idx = register_vector(vec)
                mid = f"{agent_id}_t{t}_{i}"
                do_insert(store, agent_id, mid, vec, idx, "observation", 0, importance, tick_time, text, state)
                state[agent_id]["cum_importance"] += importance
                print(f"  {agent_id} [{importance}] {text}")

                #This is crucial , an important obs occured , we give a visibility boost for the 5 closest memories(K) to change their last accessed_at to now so they will be more likely to be retrieved when queries related to this vector occur , they get a boost as relavency will be improved . 
                # This will give them an edge to other similarly important memories .
                if importance >= REACT_IMPORTANCE_THRESHOLD:
                    top = do_search(store, agent_id, vec, idx, {"agent_id": agent_id}, K_PRIME, K,
                                     tick_time, text, "react", state)
                    if top:
                        print(f"    [react] retrieved (top: \"{top[0]['text'][:60]}...\")")

            if state[agent_id]["cum_importance"] >= REFLECTION_IMPORTANCE_THRESHOLD:
                reflect(store, agent_id, tick_time, state)
                state[agent_id]["cum_importance"] = 0

        #each tick is 15 minutes of sim time, so advance the clock by that much
        tick_time += timedelta(minutes=TICK_MINUTES)

    trace_fh.close()
    np.save(VECTORS_PATH, np.array(_vectors, dtype=np.float32))

    print("\n=== Golden run done ===")
    print("Reference store stats:", store.stats())
    print("LLM usage:", usage_report())
    print(f"Trace written to {TRACE_PATH} ({_seq} ops)")
    print(f"Vectors written to {VECTORS_PATH} ({len(_vectors)} vectors)")


if __name__ == "__main__":
    run()
