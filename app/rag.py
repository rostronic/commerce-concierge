"""RAG: turn the policy corpus into something the agent can retrieve from.

WHY THIS FILE EXISTS
--------------------
The Phase 1 tools answer from *structured* data — an order id maps to a record, a
SKU maps to a quantity. Policy questions are not like that. "Can I return shoes
I've worn once?" has no lookup key; the answer lives in a paragraph of prose. You
cannot write a `get_return_answer(question)` function, because the space of
questions is open.

Retrieval-Augmented Generation is the standard answer: instead of teaching the
model the policy, you *find the relevant paragraph at question time* and hand it
to the model as context. Three steps, and that is genuinely all RAG is:

    1. CHUNK    split the corpus into passages small enough to be one idea
    2. EMBED    turn each passage into a vector whose geometry encodes meaning
    3. RETRIEVE embed the question the same way, return the nearest passages

The important architectural claim: retrieval is a *tool the agent chooses to
call*, not a step bolted in front of every request. That is why `get_policy` in
tools.py is a @tool — the model decides "this is a policy question, retrieve"
versus "this is an order question, call the order tool." Deciding which source
of truth to consult IS the agentic behaviour worth demonstrating.

THE TWO-EMBEDDINGS DESIGN (same trick as the two models)
--------------------------------------------------------
Phase 1.5 made the *reasoner* swappable: StubChatModel offline, Gemini live, and
the graph never changed. This file does the identical thing one layer down for
the *retriever*. `HashingEmbeddings` is a deterministic, dependency-light,
zero-key stand-in; `GoogleGenerativeAIEmbeddings` is the real thing. FAISS does
not know or care which it was handed — a vector store consumes vectors, full
stop. Keeping the offline path working is what keeps `--demo` free to run, fully
reproducible, and safe to hand to a stranger.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

_POLICIES = Path(__file__).resolve().parent.parent / "data" / "policies"

# Chunk size is a real tuning knob, not a magic number. Too large and the model
# gets a wall of mostly-irrelevant prose (and grounding gets hard to check); too
# small and a chunk loses the context that makes it meaningful ("$6.95" with no
# sentence around it is useless). ~600 chars is about a paragraph.
_CHUNK_CHARS = 600
_CHUNK_OVERLAP = 80

# Hash buckets. Wide, because collisions are what actually degrades ranking here
# (see HashingEmbeddings) and the corpus is tiny, so the memory cost is nothing.
_HASH_DIM = 16384

_STOPWORDS = frozenset("""
a an and are as at be but by can can't cant do does for from get got has have how i i'm
if in is it its me my of on or our so than that the their them then there these they this
to too was we were what when where which who why will with would you your
""".split())


# --- 1. EMBEDDINGS ---------------------------------------------------------
# LangChain's Embeddings interface is two methods. That is the whole contract,
# and it is why swapping providers is a one-line change: anything implementing
# these two is a legal embedding model as far as FAISS is concerned.
#
#     embed_documents(list[str]) -> list[vector]   # indexing time, batched
#     embed_query(str)           -> vector         # query time, one at a time
#
# (They are separate because some real models embed a *query* and a *passage*
# asymmetrically — the same text can get different vectors depending on role.)
class HashingEmbeddings(Embeddings):
    """A deterministic offline embedding — the 'stub model' of the RAG layer.

    Three classical ideas stacked, and each one earns its place:

    HASHING TRICK. Hash every word to a fixed bucket and accumulate weight there.
    That yields a fixed-width vector without ever building or storing a
    vocabulary, so the index has no "unseen word" failure mode.

    TF-IDF WEIGHTING. Term frequency alone ranks badly here, and the corpus shows
    exactly why: "return" appears in most of returns.md, so it carries almost no
    information about *which* rule you want, while "worn" appears in one chunk and
    pins the answer precisely. Inverse document frequency — log(N / docs
    containing the term) — scales each word by how rare it is, so discriminating
    words dominate common ones. Term frequency itself is sublinear, 1 + log(count),
    so ten mentions beat one without counting ten times as much.

PICK THE DIMENSION BY MEASURING IT. Hashing's one real cost is collisions:
    two unrelated words landing in the same bucket become indistinguishable. This
    is not theoretical — at 768 buckets the corpus's 246 terms produced 36
    colliding terms and a refund question retrieved "Order processing" as its top
    hit. Widening to 16384 left 2 collisions and fixed it. The lesson generalises:
    when a retriever ranks badly, check the representation before you start
    rewriting the prompt.

    CRUDE STEMMING. "sole" must match "soles" and "ship" must match "shipping".
    A bag of words has no idea those are related, so we strip a few common
    suffixes. It is a hack, and it is the kind of hack real embeddings make
    unnecessary — which is precisely the point of having both paths.

    Vectors are L2-normalised at the end, so a dot product between two of them is
    exactly their cosine similarity.

    BE HONEST ABOUT WHAT THIS IS. Even with IDF these vectors encode *lexical
    overlap*, not meaning: "worn shoes" and "used footwear" still land in
    unrelated buckets, so this retriever cannot do the synonym matching that is
    the whole selling point of real embeddings. It exists so the offline demo runs
    with zero keys and returns byte-identical results every time — exactly what
    you want underneath the Phase 3 eval harness. Run --live to see what real
    semantics buy you.
    """

    def __init__(self, dim: int = _HASH_DIM) -> None:
        self.dim = dim
        # Fitted from the corpus in embed_documents, then reused by embed_query.
        # This is why the interface has two methods and not one: indexing sees the
        # whole corpus and can learn from it, querying sees one string.
        self._idf: dict[str, float] = {}

    @staticmethod
    def _stem(token: str) -> str:
        for suffix in ("ing", "ies", "ed", "es", "s"):
            if len(token) > len(suffix) + 2 and token.endswith(suffix):
                return token[: -len(suffix)]
        return token

    def _tokens(self, text: str) -> list[str]:
        return [self._stem(t) for t in re.findall(r"[a-z0-9$.']+", text.lower())
                if t not in _STOPWORDS and len(t) > 1]

    def _bucket(self, token: str) -> int:
        # NOT Python's built-in hash(): it is randomly salted per process for
        # strings, so an index built in one run would not match a query in the
        # next. A cryptographic digest is stable across processes and machines,
        # which is the property "deterministic" actually requires.
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _vector(self, tokens: list[str]) -> list[float]:
        vec = [0.0] * self.dim
        for token, count in Counter(tokens).items():
            tf = 1.0 + math.log(count)
            # Unfitted (embed_query before embed_documents) falls back to 1.0,
            # which degrades to plain TF rather than crashing.
            vec[self._bucket(token)] += tf * self._idf.get(token, 1.0)

        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:                       # a query of pure stopwords
            return vec
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        tokenised = [self._tokens(t) for t in texts]

        # Fit IDF on the corpus. The +1s are smoothing: they keep a term that
        # appears in every chunk from getting weight exactly 0 (which would
        # silently delete it) and keep the log from ever going negative.
        n_docs = len(tokenised)
        doc_freq = Counter(token for tokens in tokenised for token in set(tokens))
        self._idf = {t: math.log((n_docs + 1) / (df + 1)) + 1.0 for t, df in doc_freq.items()}

        return [self._vector(tokens) for tokens in tokenised]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(self._tokens(text))


def live_embeddings(model: str = "models/gemini-embedding-001") -> Embeddings:
    """The real thing: Gemini embeddings over the same free AI Studio key.

    Imported lazily so the offline path never needs the package installed.
    gemini-embedding-001 is the GA model (3072 dimensions) and is what the
    --live flag wires in; the graph, the store and the tool are unchanged.
    """
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model)


# --- 2. CHUNKING -----------------------------------------------------------
def load_policy_chunks() -> list[Document]:
    """Read data/policies/*.md and split it into retrievable passages.

    Two splitters, in sequence, and the order matters:

    MarkdownHeaderTextSplitter first, because our corpus already has human-authored
    semantic boundaries — the `##` sections. Splitting on those means a chunk is
    "the return-shipping rule", never half of one rule plus half of the next. It
    also *lifts the headers into metadata*, which is what lets the tool cite
    "returns.md > Return shipping cost" instead of an opaque chunk number. Free
    citations are worth a lot; a grounded answer nobody can verify is still a
    trust problem.

    RecursiveCharacterTextSplitter second, as a safety net for any section that is
    longer than _CHUNK_CHARS. "Recursive" = it tries paragraph breaks first, then
    lines, then words, so it cuts at the least damaging boundary available rather
    than blindly at character 600.
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "doc_title"), ("##", "section")],
        strip_headers=False,     # keep the heading text IN the chunk: it is
                                 # strong retrieval signal, not decoration
    )
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_CHARS,
        chunk_overlap=_CHUNK_OVERLAP,   # overlap so a rule split across a seam
                                        # survives intact in one of the halves
    )

    chunks: list[Document] = []
    for path in sorted(_POLICIES.glob("*.md")):
        for doc in header_splitter.split_text(path.read_text()):
            doc.metadata["source"] = path.name
            chunks.extend(size_splitter.split_documents([doc]))
    return chunks


def citation_for(doc: Document) -> str:
    """Render a chunk's provenance, e.g. 'returns.md > Worn shoes — the 30-day trial'."""
    section = doc.metadata.get("section")
    source = doc.metadata.get("source", "policy")
    return f"{source} > {section}" if section else source


# --- 3. THE VECTOR STORE ---------------------------------------------------
# The retriever is process-global and built lazily, for a specific reason: the
# @tool functions in tools.py are plain module-level functions with a schema the
# model sees, so they cannot take a retriever as an argument — a tool's signature
# belongs to the model, not to us. Configuring the module and letting the tool
# read it keeps the tool's public schema clean while still making the embedding
# backend swappable at startup.
_configured: Optional[Embeddings] = None


def configure(embeddings: Optional[Embeddings] = None) -> None:
    """Choose the embedding backend before the first retrieval. Called by the CLI."""
    global _configured
    _configured = embeddings
    _build_store.cache_clear()


@lru_cache(maxsize=1)
def _build_store() -> FAISS:
    chunks = load_policy_chunks()
    return FAISS.from_documents(
        chunks,
        _configured or HashingEmbeddings(),
        # FAISS's default index measures straight-line (L2) distance. Normalising
        # every vector to unit length first makes L2 rank identically to cosine
        # similarity, which is what you actually want for text — it compares
        # *direction* (what the passage is about) and ignores magnitude (how long
        # it is). Without this a wordy chunk would beat a precise short one.
        normalize_L2=True,
    )


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Return the k nearest policy chunks as (document, similarity in 0..1)."""
    hits = _build_store().similarity_search_with_score(query, k=k)

    # FAISS hands back a DISTANCE (lower is better); a score you show a human or
    # threshold on should be a SIMILARITY (higher is better).
    #
    # MIND THE SQUARE. faiss.IndexFlatL2 returns the SQUARED L2 distance, not the
    # distance -- a documented detail that is very easy to miss. For unit vectors
    # d^2 = 2 - 2*cos, so the conversion is cos = 1 - d_squared/2. Getting this
    # wrong does not break the ranking (it is monotonic either way), which is
    # exactly what makes it dangerous: the ordering looks right while every score
    # you print or threshold on is garbage. Phase 3 will threshold on this number,
    # so it has to be the real cosine.
    return [(doc, max(0.0, 1.0 - distance / 2.0)) for doc, distance in hits]
