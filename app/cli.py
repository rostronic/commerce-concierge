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

_DEMO_QUERIES = [
    "Where is my order 10432?",                                 # order tool -> answer
    "Do you have trail runners in blue, size 10, under $120?",  # catalog tool -> answer
    "What's the status of order 99999?",                        # tool -> "not found"
    "thanks!",                                                  # no tool -> direct answer
]


def _run(graph, text: str, config: dict) -> None:
    """Stream one user turn through the graph, printing each node's output."""
    print(f"\n\033[1m👤 {text}\033[0m")
    for chunk in graph.stream({"messages": [HumanMessage(text)]}, config, stream_mode="updates"):
        # chunk looks like {"agent": {"messages": [...]}} or {"tools": {...}}.
        for node, update in chunk.items():
            for msg in update["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for call in msg.tool_calls:
                        print(f"  🧠 [{node}] decides to call: {call['name']}({call['args']})")
                elif isinstance(msg, ToolMessage):
                    print(f"  🔧 [{node}] {msg.name} returned: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"  💬 [{node}] answers: {msg.content}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Commerce Concierge (Phase 1)")
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

        load_dotenv()  # read GOOGLE_API_KEY from .env (never committed)
        # AI Studio free-tier key: vertexai defaults to False, so NO GCP billing.
        model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

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

    print("Commerce Concierge (stub model). Type a question, or Ctrl-C to quit.")
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
