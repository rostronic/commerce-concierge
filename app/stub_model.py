"""A deterministic, offline stand-in for a real chat model (e.g. Gemini).

WHY THIS EXISTS
---------------
LangGraph orchestrates *around* a model; it does not care which model. This class
lets you learn the LangGraph mechanics — the ReAct loop, tool routing, and
checkpointed state — with zero API keys and zero cost.

A real model decides "call a tool" vs. "write the final answer" by *reasoning*.
This stub makes the same decision with a few keyword rules, so its behaviour is
100% reproducible. Crucially, it implements the same interface a real chat model
does: `.bind_tools([...])` and returning an `AIMessage` whose `.tool_calls` the
graph can route on.

GOING LIVE
----------
Swapping to production is a one-line change: construct a real model and pass it
to build_graph(model=...). The CLI does this behind its --live flag:

    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

That uses an AI Studio API key from GOOGLE_API_KEY (free tier, no GCP project).
Leave vertexai at its default of False: vertexai=True routes through Vertex AI,
which bills a GCP project — avoid it unless you deliberately want that path.

The graph, the state, the tools, and the CLI do not change. That decoupling is
the entire point of building on LangGraph.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

_ORDER_ID_RE = re.compile(r"\b(\d{4,})\b")
# A price is a number attached to "$" or to a cap word like "under"/"less than".
# (A bare number like the "10" in "size 10" must NOT be read as a price.)
_PRICE_RE = re.compile(r"(?:\$|under|less than|below|max)\s*\$?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)

_ORDER_WORDS = ("order", "where", "status", "track", "shipped", "deliver", "arrive")
_PRODUCT_WORDS = (
    "do you have", "in stock", "under", "less than", "price", "cost", "shoe",
    "trail", "runner", "sneaker", "boot", "size", "color", "colour", "blue",
    "red", "black", "white", "brown", "catalog", "buy", "looking for", "$",
)


class StubChatModel(BaseChatModel):
    """Rule-based chat model that mimics tool-calling. Not for production."""

    # A real model tracks which tools it was given via .bind_tools(); we store
    # their names so the stub only emits calls for tools that actually exist.
    bound_tool_names: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "stub-commerce-model"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "StubChatModel":
        names = [getattr(t, "name", getattr(t, "__name__", str(t))) for t in tools]
        # model_copy is the pydantic-v2 way to return a configured clone.
        return self.model_copy(update={"bound_tool_names": names})

    # --- BaseChatModel requires _generate; everything else is our own logic ---
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._decide(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    # ------------------------------------------------------------------ logic
    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        last = messages[-1]

        # OBSERVE phase: a tool just ran (the last message is its result).
        # Turn that result into a final natural-language answer -> loop ends.
        if isinstance(last, ToolMessage):
            return self._final_answer(messages)

        # REASON phase: read the newest human turn and decide whether to call a
        # tool. This is exactly the decision a real model makes when it emits
        # (or doesn't emit) tool_calls.
        human_text = self._last_human_text(messages)
        text = human_text.lower()

        order_match = _ORDER_ID_RE.search(human_text)
        if order_match and any(w in text for w in _ORDER_WORDS) and "get_order_status" in self.bound_tool_names:
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_order_status",
                    "args": {"order_id": order_match.group(1)},
                    "id": "call_order_1",
                    "type": "tool_call",
                }],
            )

        if any(w in text for w in _PRODUCT_WORDS) and "search_catalog" in self.bound_tool_names:
            max_price = None
            if any(w in text for w in ("under", "less than", "$")):
                price_match = _PRICE_RE.search(human_text)
                if price_match:
                    max_price = float(price_match.group(1))
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_catalog",
                    "args": {"query": human_text, "max_price": max_price},
                    "id": "call_catalog_1",
                    "type": "tool_call",
                }],
            )

        # Nothing to look up -> answer directly, no tool call, loop ends.
        return AIMessage(content=(
            "Hi! I'm the Commerce Concierge. I can check an order (give me the "
            "order number) or search our catalog (tell me what you're shopping for)."
        ))

    @staticmethod
    def _last_human_text(messages: list[BaseMessage]) -> str:
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m.content if isinstance(m.content, str) else str(m.content)
        return ""

    @staticmethod
    def _final_answer(messages: list[BaseMessage]) -> AIMessage:
        latest = [m for m in messages if isinstance(m, ToolMessage)][-1]
        try:
            payload = json.loads(latest.content)
        except (json.JSONDecodeError, TypeError):
            return AIMessage(content=f"Result: {latest.content}")

        if latest.name == "get_order_status":
            if payload.get("found"):
                return AIMessage(content=(
                    f"Order {payload['order_id']} is **{payload['status']}**. "
                    f"{payload.get('detail', '')}"
                ).strip())
            return AIMessage(content=f"I couldn't find order {payload.get('order_id', '?')} in our system.")

        if latest.name == "search_catalog":
            results = payload.get("results", [])
            if not results:
                return AIMessage(content="I didn't find any catalog items matching that. Want to broaden the search?")
            lines = [f"- {r['name']} (SKU {r['sku']}) — ${r['price']:g}" for r in results]
            return AIMessage(content="Here's what I found:\n" + "\n".join(lines))

        return AIMessage(content=f"Result: {latest.content}")
