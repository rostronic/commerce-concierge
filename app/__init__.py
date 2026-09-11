"""Commerce Concierge — a LangGraph learning POC.

Phase 1: a hand-rolled ReAct agent (agent node <-> tools node) driven by a
deterministic *stub* model so the graph mechanics can be learned with no API
keys and no cloud cost. Later phases swap the stub for Vertex AI Gemini and add
RAG, a grounding gate, an eval harness, and a Cloud Run deploy (see DESIGN.md).
"""
