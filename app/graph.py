"""The LangGraph graph — a hand-rolled ReAct loop.

This is the core lesson. A ReAct agent is a loop:

    reason -> (maybe) call a tool -> observe the result -> reason again -> ... -> answer

LangGraph expresses that loop as a *graph* of nodes and edges over a shared
*state*. Reading top to bottom:

  STATE   what flows between nodes (here: the running list of messages).
  NODES   units of work. "agent" calls the model; "tools" runs any tool the
          model asked for.
  EDGES   wiring. A normal edge always goes A->B. A *conditional* edge picks the
          next node at runtime based on the state.
  COMPILE turns the definition into a runnable, and attaches a checkpointer so
          state survives across turns (keyed by a thread_id you pass at call time).

The prebuilt helpers used here (ToolNode, tools_condition) are the exact same
building blocks langchain.agents.create_agent uses under the hood — we just wire
them by hand so the control flow is visible instead of hidden.
"""

from __future__ import annotations

from typing import Annotated, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

from app.stub_model import StubChatModel
from app.tools import get_order_status, search_catalog

TOOLS = [get_order_status, search_catalog]

SYSTEM_PROMPT = SystemMessage(content=(
    "You are Commerce Concierge, a helpful assistant for an online shoe store. "
    "Use get_order_status for order questions and search_catalog for product "
    "questions. Only state facts you got from a tool result; if you cannot "
    "confirm something, say so plainly rather than guessing."
))


# STATE ---------------------------------------------------------------------
# The graph's memory. `add_messages` is a *reducer*: instead of overwriting the
# list when a node returns {"messages": [...]}, it APPENDS (and de-dupes by id).
# That append-only behaviour is what makes the message history accumulate as the
# loop runs. This one line is doing a lot of work.
class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(
    model: Optional[BaseChatModel] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
):
    """Assemble and compile the ReAct graph.

    Pass a real model (e.g. ChatGoogleGenerativeAI) to go live; defaults to the
    offline StubChatModel. Pass a SqliteSaver to persist across process restarts;
    defaults to in-memory MemorySaver.
    """
    llm = (model or StubChatModel()).bind_tools(TOOLS)

    # NODE 1 — the "agent". Calls the model on the whole conversation and returns
    # its reply. Because of the add_messages reducer, returning {"messages":[reply]}
    # appends the reply rather than replacing the history.
    def agent(state: State) -> dict:
        # Prepend the system prompt every call. A real model reads it to decide
        # how to behave; the stub ignores it. Keeping it here means "going live"
        # needs no other change.
        reply = llm.invoke([SYSTEM_PROMPT] + state["messages"])
        return {"messages": [reply]}

    builder = StateGraph(State)
    builder.add_node("agent", agent)
    # NODE 2 — the prebuilt ToolNode. It reads the last AIMessage's tool_calls,
    # runs the matching tool(s), and appends a ToolMessage per call.
    builder.add_node("tools", ToolNode(TOOLS))

    # EDGES -----------------------------------------------------------------
    builder.add_edge(START, "agent")           # every run starts at the agent
    # Conditional edge: after the agent speaks, `tools_condition` inspects the
    # last message. If it contains tool_calls -> go to "tools". Otherwise the
    # agent gave a final answer -> go to END. THIS is the ReAct branch.
    builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")         # after tools run, loop back to reason

    # COMPILE — attach a checkpointer so multi-turn memory works. At call time
    # you pass config={"configurable": {"thread_id": "..."}} to pick the thread.
    return builder.compile(checkpointer=checkpointer or MemorySaver())
