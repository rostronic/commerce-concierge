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
    model = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

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

_SKU_RE = re.compile(r"\b([A-Z]{2}-[A-Z]{2}-\d{2})\b")

_ORDER_WORDS = ("order", "where", "status", "track", "shipped", "deliver", "arrive")
_PRODUCT_WORDS = (
    "do you have", "in stock", "under", "less than", "price", "cost", "shoe",
    "trail", "runner", "sneaker", "boot", "size", "color", "colour", "blue",
    "red", "black", "white", "brown", "catalog", "buy", "looking for", "$",
)
# Deliberately specific. "ship" alone would fire on "has my order shipped?",
# which is an ORDER question, not a policy one -- a crude router has to be
# narrow to avoid being wrong. A real model reads the tool docstrings and makes
# this distinction from meaning, which is exactly the work the stub cannot do.
_POLICY_WORDS = (
    "return", "refund", "exchange", "warranty", "policy", "defect", "guarantee",
    "run big", "run small", "run large", "true to size", "half size", "fit",
    "do you ship", "ship to", "shipping cost", "free shipping", "shipping free",
    "never arrived", "lost package", "worn", "final sale",
)
_STOCK_WORDS = ("in stock", "available", "how many", "left", "stock")

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
        """Pick the next action: call a tool, or write the final answer.

        Phase 2 turned this from a one-shot router into a genuine multi-hop loop.
        Before, one human turn meant at most one tool call. But a real question
        like "can I return order 10432?" needs TWO lookups -- the order and the
        policy -- and the honest way to model that is to re-ask "is anything
        still missing?" after every tool result, rather than answering from the
        first thing that came back.

        That is the whole ReAct idea, and note WHERE it lives: the graph already
        loops tools -> agent unconditionally, so multi-hop needed no graph change
        at all. Whether to stop is the reasoner's judgement, and the reasoner is
        this method.
        """
        human_text = self._last_human_text(messages)
        already_run = self._tools_run_this_turn(messages)

        pending = self._next_tool_call(human_text, already_run, messages)
        if pending is not None:
            return pending

        # Nothing left to look up. If tools ran, summarise them; otherwise this
        # was small talk and no tool was ever warranted.
        if already_run:
            return self._final_answer(messages)

        return AIMessage(content=(
            "Hi! I'm the Commerce Concierge. I can check an order, search the "
            "catalog, check stock on a SKU, or explain our return, shipping, "
            "warranty and sizing policies."
        ))

    def _next_tool_call(
        self, human_text: str, already_run: set[str], messages: list[BaseMessage]
    ) -> Optional[AIMessage]:
        """The router. Returns a tool-call message, or None when nothing is missing.

        `already_run` is what keeps this terminating: each tool fires at most once
        per user turn, so the agent <-> tools loop always drains. An agent that can
        re-request a tool it already called is an agent that can spin forever --
        LangGraph's recursion_limit would eventually stop it, but running out of
        budget is not the same as knowing when you are done.
        """
        text = human_text.lower()
        # Classify ONCE, up front. Computing this inside the product branch was a
        # real bug: "can I return these shoes?" contains "shoe", so after
        # get_policy answered it the router fell through and also ran
        # search_catalog, stapling irrelevant product listings onto a policy
        # answer. A tool that fires when it has nothing to contribute is not a
        # harmless extra -- it is latency, cost, and noise in the model's context.
        is_policy = any(w in text for w in _POLICY_WORDS)

        # ORDER -- most specific signal (a 4+ digit id), so it is checked first.
        order_match = _ORDER_ID_RE.search(human_text)
        if (order_match and any(w in text for w in _ORDER_WORDS)
                and self._can_call("get_order_status", already_run)):
            return self._tool_call("get_order_status", {"order_id": order_match.group(1)})

        # POLICY -- before the product branch, because "can I return these shoes"
        # contains "shoe" and would otherwise be misrouted to the catalog.
        if is_policy and self._can_call("get_policy", already_run):
            return self._tool_call("get_policy", {"topic": human_text})

        # INVENTORY, case 1: the shopper named an exact SKU.
        sku_match = _SKU_RE.search(human_text.upper())
        if sku_match and self._can_call("check_inventory", already_run):
            return self._tool_call("check_inventory", {"sku": sku_match.group(1), "location": None})

        # PRODUCT -- find candidates in the catalog. Suppressed for policy
        # questions (see is_policy above).
        if (not is_policy and any(w in text for w in _PRODUCT_WORDS)
                and self._can_call("search_catalog", already_run)):
            max_price = None
            if any(w in text for w in ("under", "less than", "$")):
                price_match = _PRICE_RE.search(human_text)
                if price_match:
                    max_price = float(price_match.group(1))
            return self._tool_call("search_catalog", {"query": human_text, "max_price": max_price})

        # INVENTORY, case 2: the interesting hop. The shopper asked whether
        # something is "in stock" without knowing a SKU, so the catalog had to run
        # first -- and now we read the SKU OUT OF THAT TOOL'S RESULT to build the
        # next call. Chaining where one tool's output becomes the next tool's
        # input is the thing that makes this an agent and not a lookup table.
        if (any(w in text for w in _STOCK_WORDS)
                and "search_catalog" in already_run
                and self._can_call("check_inventory", already_run)):
            sku = self._first_sku_from_catalog(messages)
            if sku:
                return self._tool_call("check_inventory", {"sku": sku, "location": None})

        return None

    def _can_call(self, name: str, already_run: set[str]) -> bool:
        return name in self.bound_tool_names and name not in already_run

    @staticmethod
    def _tool_call(name: str, args: dict) -> AIMessage:
        return AIMessage(content="", tool_calls=[{
            "name": name, "args": args, "id": f"call_{name}", "type": "tool_call",
        }])

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _this_turn(messages: list[BaseMessage]) -> list[BaseMessage]:
        """Messages since the latest human turn.

        The checkpointer means `messages` is the WHOLE conversation, not just this
        exchange. Scoping to the current turn is what stops a get_policy call made
        three questions ago from being mistaken for one already answered here.
        """
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], HumanMessage):
                return messages[i + 1:]
        return list(messages)

    def _tools_run_this_turn(self, messages: list[BaseMessage]) -> set[str]:
        return {m.name for m in self._this_turn(messages) if isinstance(m, ToolMessage)}

    def _first_sku_from_catalog(self, messages: list[BaseMessage]) -> Optional[str]:
        for m in self._this_turn(messages):
            if isinstance(m, ToolMessage) and m.name == "search_catalog":
                try:
                    results = json.loads(m.content).get("results", [])
                except (json.JSONDecodeError, TypeError):
                    return None
                if results:
                    return results[0]["sku"]
        return None

    @staticmethod
    def _last_human_text(messages: list[BaseMessage]) -> str:
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                return m.content if isinstance(m.content, str) else str(m.content)
        return ""

    # ---------------------------------------------------------------- answer
    def _final_answer(self, messages: list[BaseMessage]) -> AIMessage:
        """Compose one answer out of EVERY tool that ran this turn.

        Phase 1 rendered only the most recent tool result, which was fine when a
        turn meant one call. Once a turn can chain two tools, reading just the
        last one silently drops half the answer -- so this walks the whole turn.
        """
        tool_messages = [m for m in self._this_turn(messages) if isinstance(m, ToolMessage)]

        sections = [rendered for m in tool_messages if (rendered := self._render(m))]
        if not sections:
            return AIMessage(content="I wasn't able to look that up.")
        return AIMessage(content="\n\n".join(sections))

    @staticmethod
    def _render(message: ToolMessage) -> str:
        try:
            payload = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            return f"Result: {message.content}"

        if message.name == "get_order_status":
            if payload.get("found"):
                return (f"Order {payload['order_id']} is **{payload['status']}**. "
                        f"{payload.get('detail', '')}").strip()
            return f"I couldn't find order {payload.get('order_id', '?')} in our system."

        if message.name == "search_catalog":
            results = payload.get("results", [])
            if not results:
                return "I didn't find any catalog items matching that. Want to broaden the search?"
            lines = [f"- {r['name']} (SKU {r['sku']}) — ${r['price']:g}" for r in results]
            return "Here's what I found:\n" + "\n".join(lines)

        if message.name == "check_inventory":
            if not payload.get("found"):
                return f"I don't have a product with SKU {payload.get('sku', '?')}."
            if not payload.get("in_stock"):
                return f"{payload['sku']} is currently **out of stock** in every warehouse."
            if "quantity" in payload:
                return f"{payload['sku']}: {payload['quantity']} in stock at {payload['location']}."
            spread = ", ".join(f"{loc} {qty}" for loc, qty in payload["by_location"].items())
            return f"{payload['sku']}: **{payload['total']} in stock** ({spread})."

        if message.name == "get_policy":
            # Honouring `confident` is the point. The retriever always returns its
            # nearest neighbours, so without this check an off-topic question gets
            # confidently answered from whatever passage happened to be closest.
            if not payload.get("confident"):
                return ("I can't find anything in our policies that covers that. "
                        "I'd rather say so than guess.")
            best = payload["passages"][0]
            return f"{best['text']}\n\n_(source: {best['citation']})_"

        return f"Result: {message.content}"
