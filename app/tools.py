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

# Cosine floor below which a retrieved passage is treated as "the corpus probably
# does not cover this".
#
# MEASURED, AND THE MEASUREMENT IS THE POINT. Over 9 in-scope and 7 out-of-scope
# questions, in-scope top hits scored 0.188-0.569 and out-of-scope ones scored
# 0.000-0.283. Those ranges OVERLAP, so no single threshold separates them, and
# any number chosen here is a trade-off rather than a fix. The failure has a
# specific cause: short chunks score high on almost anything, because cosine
# normalises away length -- every out-of-scope question containing the word
# "policy" retrieved the same three-line "If the size is wrong" section at ~0.25.
#
# So this floor is an honest cheap pre-filter for the obviously-irrelevant tail,
# NOT a relevance oracle. Deciding whether a passage actually ANSWERS the question
# needs to read both of them together -- which is exactly the grounding gate node
# in Phase 2b. Keeping the threshold low avoids false refusals and leaves the real
# judgement to the component that can make it.
_RELEVANCE_FLOOR = 0.12


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


@tool
def check_inventory(sku: str, location: str | None = None) -> str:
    """Check how many units of a specific SKU are in stock, optionally at one warehouse.

    Use this after search_catalog when the shopper asks whether an item is
    actually available, not just whether it exists. `sku` is the exact product
    code (e.g. "TR-BL-10"). Returns a JSON string with per-location quantities and
    a `total`; `found` is false if the SKU is unknown.
    """
    inventory = _load("inventory.json")
    record = inventory.get(sku.upper())
    if record is None:
        return json.dumps({"found": False, "sku": sku})

    if location is not None:
        quantity = record.get(location, 0)
        return json.dumps({
            "found": True, "sku": sku.upper(), "location": location,
            "quantity": quantity, "in_stock": quantity > 0,
        })

    total = sum(record.values())
    return json.dumps({
        "found": True, "sku": sku.upper(), "by_location": record,
        "total": total, "in_stock": total > 0,
    })


@tool
def get_policy(topic: str) -> str:
    """Look up store policy on returns, refunds, exchanges, shipping, warranty, or sizing.

    Use this for any question about the RULES of the store rather than about a
    specific order or product — "can I return worn shoes", "how long is the
    warranty", "do you ship internationally". Pass the shopper's question as
    `topic`. Returns a JSON string whose `passages` are verbatim policy excerpts,
    each with a `citation`. Answer only from these passages and cite them; if
    `confident` is false, say the policy does not clearly cover the question
    instead of guessing.
    """
    # Imported here rather than at module scope so the order/catalog tools stay
    # importable without the RAG dependencies (faiss, the splitters) installed.
    from app.rag import citation_for, search

    hits = search(topic, k=3)
    passages = [
        {
            "citation": citation_for(doc),
            "text": doc.page_content.strip(),
            "score": round(float(score), 3),
        }
        for doc, score in hits
    ]

    # A retriever ALWAYS returns its k nearest neighbours, even when the corpus
    # has nothing relevant -- "nearest" is not "relevant". Passing that judgement
    # to the model as an explicit flag is what turns a silent failure into an
    # honest "I don't know". Phase 2b's grounding gate will enforce it rather
    # than merely suggesting it.
    best = passages[0]["score"] if passages else 0.0
    return json.dumps({
        "topic": topic,
        "confident": best >= _RELEVANCE_FLOOR,
        "passages": passages,
    })
