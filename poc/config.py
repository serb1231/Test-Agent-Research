"""Constants pinned once, shared by every module. Mirrors the run plan's
'pre-registration' principle (§08) at PoC scale: nothing is a magic number
buried in sim.py.
"""

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536

CHAT_MODEL = "gpt-4o-mini"

# Composite score = ALPHA*recency + BETA*importance + GAMMA*relevance
# Same weights Park et al. (Generative Agents) use.
ALPHA = 1.0
BETA = 1.0
GAMMA = 1.0
RECENCY_DECAY = 0.99  # per sim-hour

K = 5          # final results returned to the agent
K_PRIME = 20   # candidates pulled from the store before rescoring (small: only ~40-60 memories exist total in this PoC)

REFLECTION_IMPORTANCE_THRESHOLD = 60  # cumulative importance since last reflection. Park et al. use 150, calibrated for a full simulated day (~448 ticks); at 20 (our original value) reflection fired almost every single one of our 6 ticks, leaving too little new material between cycles and producing near-duplicate insights (observed empirically — see findings.md). 60 gives roughly one reflection every ~2 ticks at our per-tick importance sum (~28-30/agent), enough separation to matter at this tiny scale while still firing 2-3x per agent so the mechanism is actually exercised.
REACT_IMPORTANCE_THRESHOLD = 7  # an observation at/above this triggers a react-search

TICKS = 6
OBSERVATIONS_PER_TICK = 4
REFLECT_QUESTIONS = 2
REFLECT_K = 5  # candidates kept per reflection question, after rescoring

CHROMA_PATH = "./chroma_db_replay"
TRACE_PATH = "./trace.jsonl"
VECTORS_PATH = "./vectors.npy"
REPLAY_METRICS_PATH = "./replay_metrics.jsonl"
