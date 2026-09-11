"""Tools the agent can call.

A LangChain "tool" is just a function wrapped with @tool. The decorator reads the
function name, the type hints, and the docstring and turns them into a JSON schema
that the model sees. In production that schema is how Gemini decides *which* tool
to call and *what arguments* to pass. So the docstring is not a comment — it is
prompt text the model reads. Write it like an instruction.

For Phase 1 the tools read seeded JSON from ../data. Later phases point the same
tool signatures at real backends (SQLite, an API) without the graph changing.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

# Resolve ../data relative to this file so it works regardless of the cwd.
_DATA = Path(__file__).resolve().parent.parent / "data"


def _load(name: str):
    return json.loads((_DATA / name).read_text())


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of a customer order by its numeric order ID.

    Use this whenever the user asks where an order is, whether it shipped, or
    when it will arrive. Returns a JSON string; `found` is false if the order
    number is not in the system.
    """
    orders = _load("orders.json")
    record = orders.get(str(order_id))
    if record is None:
        return json.dumps({"found": False, "order_id": order_id})
    return json.dumps({"found": True, "order_id": order_id, **record})


@tool
def search_catalog(query: str, max_price: float | None = None) -> str:
    """Search the product catalog by free-text query, optionally capping the price.

    Use this for product questions ("do you have X", "is Y in stock under $Z").
    `query` is the shopper's description; `max_price` filters out anything above
    that price. Returns a JSON string with a `results` list, best matches first.
    """
    products = _load("products.json")

    # Naive keyword relevance: score each product by how many query tokens it
    # matches, then return the highest scorers. A real system would use the
    # RAG retriever from Phase 2 (embeddings + vector search) here instead.
    tokens = [t for t in query.lower().replace("$", " ").split() if len(t) > 2]

    scored = []
    for product in products:
        haystack = " ".join(
            [product["name"], product.get("description", ""), *product.get("attrs", [])]
        ).lower()
        score = sum(1 for t in tokens if t in haystack)
        if score == 0:
            continue
        if max_price is not None and product["price"] > max_price:
            continue
        scored.append((score, product))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    results = [product for _, product in scored[:5]]
    return json.dumps({"query": query, "max_price": max_price, "results": results})
