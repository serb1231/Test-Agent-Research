"""Thin OpenAI wrapper. All embedding and chat calls funnel through here so
we have one place tracking token usage/cost — the article flags this as
worth doing (§06 golden-run cost table) even at full scale; more so here
where we want to know exactly what a 2-agent PoC costs before scaling up.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from config import EMBED_MODEL, CHAT_MODEL

load_dotenv()

_client: OpenAI | None = None

# Per-1M-token prices as of writing; only used for a rough running total.
_PRICES = {
    "text-embedding-3-small": {"input": 0.02},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

_usage = {"embed_tokens": 0, "chat_input_tokens": 0, "chat_output_tokens": 0, "calls": 0}


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY not set (expected in poc/.env)")
        _client = OpenAI(api_key=key)
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    client = _get_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    _usage["embed_tokens"] += resp.usage.total_tokens
    _usage["calls"] += 1
    return [d.embedding for d in resp.data]


def chat(system: str, user: str, temperature: float = 0.8) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    _usage["chat_input_tokens"] += resp.usage.prompt_tokens
    _usage["chat_output_tokens"] += resp.usage.completion_tokens
    _usage["calls"] += 1
    return resp.choices[0].message.content


def usage_report() -> dict:
    cost = (
        _usage["embed_tokens"] / 1_000_000 * _PRICES["text-embedding-3-small"]["input"]
        + _usage["chat_input_tokens"] / 1_000_000 * _PRICES["gpt-4o-mini"]["input"]
        + _usage["chat_output_tokens"] / 1_000_000 * _PRICES["gpt-4o-mini"]["output"]
    )
    return {**_usage, "estimated_cost_usd": round(cost, 5)}
