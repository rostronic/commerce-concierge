"""Run the agent from the terminal and watch the ReAct loop happen.

    python -m app.cli            # interactive chat, offline stub model (Ctrl-C to quit)
    python -m app.cli --demo     # run a scripted set of queries and print the trace
    python -m app.cli --demo --live  # SAME graph, driven by a live Gemini model
    python -m app.cli --graph    # print the graph as a Mermaid diagram and exit

The key teaching move is graph.stream(..., stream_mode="updates"): instead of
just getting the final answer, we get one chunk per node execution, so you can
SEE agent -> tools -> agent as it runs.
"""

from __future__ import annotations

import argparse

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.graph import build_graph

# Each query is here to exercise a DIFFERENT path through the graph. A demo that
# only shows the happy path is a demo that hides its own failure modes.
_DEMO_QUERIES = [
    "Where is my order 10432?",                                  # 1 tool: structured lookup
    "Do you have trail runners in blue, size 10, under $120?",   # 1 tool: catalog search
    "Can I return shoes I've worn once?",                        # 1 tool: RAG over policies
    "I want to return order 10432 — what's the process and will I get a full refund?",
                                                                 # 2 tools: order + policy
    "Is the Summit hiking boot in stock?",                       # 2 tools, CHAINED:
                                                                 # catalog -> SKU -> inventory
    "What's your policy on renting shoes for a wedding?",        # KNOWN FAILURE, kept
                                                                 # deliberately: the corpus
                                                                 # says nothing about rentals,
                                                                 # but a short chunk scores
                                                                 # high enough to look
                                                                 # confident. This is the
                                                                 # case Phase 2b's grounding
                                                                 # gate exists to catch.
]

# Pinned, not "latest". DESIGN.md §5 flagged that gemini-2.5-flash shuts down on
# 2026-10-20; the id below was picked by listing the models this key can actually
# see and smoke-testing tool-calling on each. A `-latest` alias would silently
# change the reasoner underneath the Phase 3 eval harness, which makes any
# regression it reports impossible to attribute. Pin, then bump deliberately.
MODEL_ID = "gemini-3.8-flash"
EMBEDDING_ID = "models/gemini-embedding-001"


def _text(content) -> str:
    """Flatten a message's content to plain text.

    A STACK-CURRENCY DETAIL worth knowing. `AIMessage.content` is not always a
    string. Older models (and our stub) return one, but current Gemini models
    return a LIST OF CONTENT BLOCKS -- typed dicts like {"type": "text", ...},
    alongside blocks carrying reasoning signatures and other metadata. Printing
    `msg.content` directly therefore works perfectly offline and dumps raw dicts
    and multi-kilobyte signature blobs the moment you pass --live.

    This is the same class of bug as the deprecated class names in DESIGN.md §5:
    code written against last year's shape still runs, just wrongly. Anything that
    renders model output has to handle both shapes.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def _run(graph, text: str, config: dict) -> None:
    """Stream one user turn through the graph, printing each node's output."""
    print(f"\n\033[1m👤 {text}\033[0m")
    for chunk in graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
        # chunk looks like {"agent": {"messages": [...]}} or {"gate": {"gate_status": ...}}.
        for node, update in chunk.items():
            # NOT EVERY UPDATE CARRIES MESSAGES. A node's update is whatever keys that
            # node chose to write, and the Phase 2b gate writes only gate_attempts /
            # gate_status when a draft passes -- it has nothing to add to the
            # conversation. Indexing update["messages"] crashed on the first passing
            # answer. Read the update with .get(), never assume its shape.
            for msg in update.get("messages", []):
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"  🧠 [{node}] decides to call: {call['name']}({call['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"  🔧 [{node}] {msg.name} returned: {_text(msg.content)}")
                elif isinstance(msg, AIMessage) and node == "agent":
                    # A DRAFT, not an answer: the gate has not seen it yet.
                    print(f"  💬 [{node}] drafts: {_text(msg.content)}")
            if node == "gate":
                line = f"  🚦 [{node}] {update.get('gate_status')}"
                if update.get("gate_status") == "retry":  # the reasons it bounced
                    line += f" — {_text(update['messages'][0].content)}"
                print(line)
    # A refusal REPLACES the draft in state (see the reducer trick in app/gate.py),
    # so the draft printed above is not what the shopper got. Say what was.
    state = graph.get_state(config).values
    if state.get("gate_status") == "refused":
        print(f"  🛑 sent instead: {_text(state['messages'][-1].content)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Commerce Concierge")
    parser.add_argument("--demo", action="store_true", help="run scripted demo queries")
    parser.add_argument("--graph", action="store_true", help="print the graph as Mermaid and exit")
    parser.add_argument("--live", action="store_true",
                        help="use live Gemini instead of the offline stub model "
                             "(needs GOOGLE_API_KEY in .env; free AI Studio key, NOT Vertex)")
    args = parser.parse_args()

    # THE ONE CHANGE THAT GOES LIVE. The graph, state, tools and CLI below are
    # identical either way — build_graph() already accepts any BaseChatModel, so
    # swapping the reasoner is just constructing a different model here.
    # Imports are lazy so the offline stub path needs none of these packages.
    model = None
    if args.live:
        from dotenv import load_dotenv
        from langchain_google_genai import ChatGoogleGenerativeAI

        from app import rag

        load_dotenv()  # read GOOGLE_API_KEY from .env (never committed)
        # AI Studio free-tier key: vertexai defaults to False, so NO GCP billing.
        model = ChatGoogleGenerativeAI(model=MODEL_ID, temperature=0)

        # --live now swaps TWO models, because Phase 2 added a second one. The
        # reasoner and the retriever are independent choices -- you can run a real
        # LLM over hash embeddings, or the stub over Gemini embeddings, and the
        # graph is indifferent. Being able to vary them separately is how you find
        # out whether a bad answer came from bad reasoning or bad retrieval.
        rag.configure(rag.live_embeddings(EMBEDDING_ID))

    graph = build_graph(model=model)

    if args.graph:
        print(graph.get_graph().draw_mermaid())
        return

    # A thread_id names the conversation. Reuse it across turns and the
    # checkpointer replays the accumulated history — that's the "memory".
    config = {"configurable": {"thread_id": "cli-session-1"}}

    if args.demo:
        for query in _DEMO_QUERIES:
            _run(graph, query, config)
        return

    flavour = f"live: {MODEL_ID}" if args.live else "offline stub model"
    print(f"Commerce Concierge ({flavour}). Type a question, or Ctrl-C to quit.")
    while True:
        try:
            query = input("\n👤 ")
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if query.strip():
            _run(graph, query, config)


if __name__ == "__main__":
    main()
