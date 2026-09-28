"""The grounding gate — the node that refuses to let an ungrounded answer out.

WHY THIS NODE EXISTS
--------------------
Phase 2a ended on a measurement, not a hunch: across 9 in-scope and 7 out-of-scope
questions, retrieval scores were 0.188-0.569 and 0.000-0.283 respectively. Those
ranges OVERLAP, so no similarity threshold separates "the corpus covers this" from
"the corpus does not". A similarity score answers *what is nearest*; a guardrail
needs *does this answer the question*. Different predicates.

TWO PREDICATES, NOT ONE
-----------------------
DESIGN.md originally specified this node as "verify the answer's claims trace to
tool output". Check that specification against the bug it was written to fix:

    Q: "What's your policy on renting shoes for a wedding?"
    A: "Size exchanges ship free in both directions..."  (verbatim from sizing.md)

Every claim traces perfectly to a retrieved document. A pure trace-check PASSES
this bug. The conflation is between:

    FAITHFULNESS      is the answer entailed by the evidence?   catches invention
    ANSWER-RELEVANCE  does the evidence address the question?   catches the above

A real gate needs both, and they are not equally hard.

WHAT IS SOUNDLY DECIDABLE WITHOUT A MODEL
-----------------------------------------
The tempting shortcut is a lexical relevance heuristic -- refuse when the question's
words are missing from the evidence. That was MEASURED here and it does not work
either (docs/lessons, "coverage overlaps too"): evidence-coverage ran 0.000-1.000
in-scope against 0.000-0.600 out-of-scope, overlapping for a structural reason --

    "My sole came apart after a month"    IN-scope,  coverage 0.20
    "renting shoes for a wedding?"        OUT-scope, coverage 0.40

An in-scope question phrased in synonyms is indistinguishable, to a bag of words,
from an out-of-scope question about something else. A gate built on that heuristic
would refuse a legitimate warranty claim. That is worse than no gate at all.

So this module draws the line at what is *sound*:

    StubJudge   deterministic, offline, zero keys. Checks only what can be decided
                WITHOUT understanding meaning -- invented figures, and assertion
                through an explicit found=false / confident=false flag. It makes no
                claim to judge relevance, and the demo's rentals case still gets
                through it. That is a documented limit, not an oversight.

    LLMJudge    reads question and evidence together and answers the predicate a
                lexical check cannot. This is what catches the rentals case.

That split is the lesson. A guardrail that is honest about which half of its job
needs a model is worth more than one that pretends a threshold covers both.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

# How many times the gate will bounce a draft back to the agent before it gives
# up and substitutes an honest refusal. Bounded because an agent that can be
# asked to try again forever is an agent that can burn a budget forever;
# LangGraph's recursion_limit would eventually stop it, but running out of fuel
# is not the same as deciding you are done.
MAX_ATTEMPTS = 1

# What the gate substitutes when the draft cannot be made grounded.
REFUSAL = (
    "I can't confirm that from our order records or policy documents, so I'd "
    "rather say so than guess. If you can give me an order number, or ask about "
    "returns, shipping, warranty or sizing, I can look that up properly."
)

# Figures worth checking: currency amounts, bare numbers, percentages, and SKUs.
# These are the claims that are verbatim-checkable -- a number is either in the
# evidence or it is not, and deciding that needs no understanding of meaning.
_FIGURE_RE = re.compile(r"\$?\d+(?:\.\d+)?%?|\b[A-Z]{2}-[A-Z]{2}-\d{2}\b")

# Phrases that count as the answer actually abstaining. Deliberately generous:
# a false NEGATIVE here (missing an abstention that was phrased unusually) costs
# one needless retry, while a false POSITIVE lets an unflagged assertion out.
_ABSTENTION_RE = re.compile(
    r"can'?t (?:find|confirm|verify)|cannot (?:find|confirm|verify)|couldn'?t find|"
    r"unable to|don'?t have|do not have|didn'?t find|not (?:able to )?(?:find|confirm)|"
    r"no (?:record|information)|doesn'?t (?:cover|say)|does not (?:cover|say)|"
    r"rather say so|not clearly cover",
    re.IGNORECASE,
)


# The gate's retry feedback goes back to the agent as a HumanMessage (Gemini
# rejects a system turn mid-conversation). That creates a trap worth naming: every
# component here scopes "this turn" by scanning back to the last HumanMessage, so
# an unmarked feedback message would look like a NEW USER QUESTION -- resetting the
# turn boundary, hiding the tool results from the gate on the retry pass (empty
# evidence => an automatic pass), and convincing the stub reasoner that no tools had
# run yet, so it would run them all again.
#
# Marking the message keeps it inert: it is visible to the model as an instruction,
# and invisible to every "where does this turn start" calculation.
GATE_FEEDBACK_KEY = "gate_feedback"


def is_gate_feedback(message: BaseMessage) -> bool:
    return bool(getattr(message, "additional_kwargs", {}).get(GATE_FEEDBACK_KEY))


def _feedback_message(text: str) -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs={GATE_FEEDBACK_KEY: True})


def turn_start_index(messages: list[BaseMessage]) -> int:
    """Index just after the human message that began the current turn.

    Skips gate-feedback messages (see GATE_FEEDBACK_KEY) so a retry does not move
    the turn boundary out from under the caller.
    """
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, HumanMessage) and not is_gate_feedback(message):
            return i + 1
    return 0


@dataclass
class Verdict:
    """The gate's decision, plus WHY -- the reasons become the retry instruction.

    A gate that only returns a boolean can bounce a draft back but cannot tell the
    agent what to fix, so the agent redraws the same answer and the loop burns its
    budget. Carrying the reasons is what makes the retry edge worth having.
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)

    def as_feedback(self) -> str:
        return (
            "GROUNDING CHECK FAILED. Do not repeat the previous answer. "
            + " ".join(self.reasons)
            + " Rewrite using ONLY facts present in the tool results above, or say "
            "plainly that you cannot confirm it."
        )


class Judge(Protocol):
    """The judge interface -- one method, so the backend is swappable.

    Exactly the shape the reasoner (StubChatModel / ChatGoogleGenerativeAI) and the
    retriever (HashingEmbeddings / gemini-embedding-001) already have. Phase 2b adds
    a THIRD model to this project, and the value of varying them independently is
    diagnostic: when an answer is bad you can ask whether the reasoning, the
    retrieval, or the judging was at fault, one at a time.
    """

    def check(self, question: str, draft: str, evidence: list[ToolMessage]) -> Verdict: ...


# --- helpers shared by both judges ----------------------------------------
def _evidence_text(evidence: list[ToolMessage]) -> str:
    return "\n".join(_content_text(m.content) for m in evidence)


def _content_text(content: Any) -> str:
    """Flatten message content to text (see cli.py `_text` for why this is needed)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str) or isinstance(block, dict)
        )
    return str(content)


def _flagged_tools(evidence: list[ToolMessage]) -> list[str]:
    """Tools that explicitly reported they could not answer.

    `found: false` and `confident: false` are the tools telling the truth about
    their own limits. The failure this catches is an answer that sails past them --
    which is precisely what a model does when the retrieved passage reads plausibly.
    """
    flagged = []
    for message in evidence:
        try:
            payload = json.loads(_content_text(message.content))
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("found") is False or payload.get("confident") is False:
            flagged.append(message.name or "a tool")
    return flagged


class StubJudge:
    """Deterministic, offline, sound-only. The 'stub model' of the gate layer.

    It checks TWO things, both decidable from strings alone:

    1. UNSUPPORTED FIGURES. Every number, price, percentage and SKU in the draft
       must appear in the tool output. This is the cheap half of faithfulness and
       it catches the expensive half of hallucination: a model that turns a 60-day
       window into 45 days, or invents a restocking fee, is wrong in a way a
       shopper acts on. Prose can be paraphrased; a figure cannot.

    2. ASSERTION THROUGH AN ABSTENTION FLAG. If any tool returned found=false or
       confident=false, the draft has to acknowledge it. The stub reasoner already
       honours those flags voluntarily -- this makes it ENFORCED, which matters
       because a live model has no such rule, only a suggestion in a prompt.

    WHAT IT DELIBERATELY DOES NOT DO: judge whether the evidence is relevant to the
    question. That predicate was measured to be undecidable lexically (see the
    module docstring), so attempting it here would mean refusing real questions.
    The rentals demo query therefore still passes this judge. Run --live for a
    judge that can catch it.
    """

    name = "stub"

    def check(self, question: str, draft: str, evidence: list[ToolMessage]) -> Verdict:
        if not evidence:
            # No tools ran: nothing was claimed from a source, so there is nothing
            # to be unfaithful to. Small talk should not be gated.
            return Verdict(ok=True)

        reasons: list[str] = []
        haystack = _evidence_text(evidence)

        # 1. figures ---------------------------------------------------------
        # Compare on digits only, so "$165" in the draft matches "price": 165.0
        # in the tool's JSON and "**30 days**" matches "30".
        supported = set(_FIGURE_RE.findall(haystack))
        supported_digits = {f.lstrip("$").rstrip("%") for f in supported}
        unsupported = []
        for figure in _FIGURE_RE.findall(draft):
            digits = figure.lstrip("$").rstrip("%")
            if digits in supported_digits:
                continue
            # A truncation like 165 vs 165.0 is still supported.
            if any(h.startswith(digits) or digits.startswith(h) for h in supported_digits):
                continue
            unsupported.append(figure)
        if unsupported:
            reasons.append(
                "These figures appear in your answer but in no tool result: "
                + ", ".join(sorted(set(unsupported)))
                + "."
            )

        # 2. abstention flags ------------------------------------------------
        flagged = _flagged_tools(evidence)
        if flagged and not _ABSTENTION_RE.search(draft):
            reasons.append(
                f"{', '.join(sorted(set(flagged)))} reported it could not answer "
                "(found=false or confident=false), but the answer asserts anyway."
            )

        return Verdict(ok=not reasons, reasons=reasons)


class LLMJudge:
    """The judge that can actually read. Answers the predicate a string check cannot.

    Note what is being asked of the model here, because it is NOT "answer the
    question" -- it is a narrow, checkable classification over text it is handed:
    does this evidence address this question, and is this draft supported by it.
    A small, cheap model is appropriate; the difficulty is in the retrieval and the
    reasoning, not in the judging.

    LLM-as-judge has known biases (position, verbosity, self-preference) -- which is
    why Phase 3 calibrates it against a handful of human labels instead of trusting
    the score. Two mitigations are already in place: the judge is asked for strict
    JSON rather than prose (so the verdict cannot hide in hedging), and it falls
    back to the sound StubJudge if the model errors or returns something unparseable
    -- a guardrail that fails OPEN on a network blip is not a guardrail.
    """

    name = "llm"

    _PROMPT = (
        "You are a strict grounding checker for a shoe store's support assistant. "
        "You are NOT answering the question.\n\n"
        "QUESTION:\n{question}\n\nEVIDENCE (tool results):\n{evidence}\n\n"
        "DRAFT ANSWER:\n{draft}\n\n"
        "Decide two things:\n"
        "1. relevant: does the EVIDENCE actually address what the QUESTION asks "
        "about? If the question is about a topic the evidence never covers (e.g. "
        "asking about rentals when the evidence is about size exchanges), this is "
        "false -- even if both concern shoes.\n"
        "2. supported: is every factual claim in the DRAFT present in the EVIDENCE?\n\n"
        'Reply with ONLY strict JSON: {{"relevant": bool, "supported": bool, '
        '"reason": "<one short sentence>"}}'
    )

    def __init__(self, model: BaseChatModel, fallback: Optional[Judge] = None) -> None:
        # No .bind_tools() here: the judge must not be able to call tools. Its only
        # job is to read what it was given and return a verdict.
        self.model = model
        self.fallback = fallback or StubJudge()

    def check(self, question: str, draft: str, evidence: list[ToolMessage]) -> Verdict:
        if not evidence:
            return Verdict(ok=True)

        prompt = self._PROMPT.format(
            question=question, evidence=_evidence_text(evidence), draft=draft
        )
        try:
            raw = _content_text(self.model.invoke(prompt).content)
            payload = json.loads(_strip_fence(raw))
        except Exception as exc:  # network, quota, or unparseable JSON
            # FAIL CLOSED onto the sound checks rather than waving the draft through.
            verdict = self.fallback.check(question, draft, evidence)
            verdict.reasons.append(f"(judge unavailable: {type(exc).__name__})")
            return verdict

        reason = str(payload.get("reason", "")).strip()
        reasons = []
        if payload.get("relevant") is False:
            reasons.append(
                f"The retrieved evidence does not address what was asked. {reason}"
            )
        if payload.get("supported") is False:
            reasons.append(f"The answer states something the evidence does not. {reason}")
        return Verdict(ok=not reasons, reasons=reasons)


def _strip_fence(text: str) -> str:
    """Models wrap JSON in ```json fences no matter how firmly you ask them not to."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return text.strip()


# --- the node itself -------------------------------------------------------
def last_human_text(messages: list[BaseMessage]) -> str:
    """The question that started this turn -- never the gate's own feedback."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not is_gate_feedback(message):
            return _content_text(message.content)
    return ""


def tool_messages_this_turn(messages: list[BaseMessage]) -> list[ToolMessage]:
    """Tool results since the latest real human message.

    Scoped to the turn for the same reason the stub reasoner scopes its own view:
    the checkpointer means `messages` is the WHOLE conversation, and evidence from
    three questions ago is not evidence for this one.
    """
    return [m for m in messages[turn_start_index(messages):] if isinstance(m, ToolMessage)]


def make_gate_node(judge: Judge, max_attempts: int = MAX_ATTEMPTS):
    """Build the gate node. Returns a function with the standard node signature."""

    def gate(state: dict) -> dict:
        messages = state["messages"]
        draft = messages[-1]
        question = last_human_text(messages)
        evidence = tool_messages_this_turn(messages)

        verdict = judge.check(question, _content_text(draft.content), evidence)
        attempts = state.get("gate_attempts", 0)

        if verdict.ok:
            # Reset the counter so the NEXT user turn starts fresh. The gate is the
            # only writer of this key, and it runs exactly once per completed turn,
            # which is what makes resetting here correct rather than lucky.
            return {"gate_attempts": 0, "gate_status": "pass"}

        if attempts < max_attempts:
            # Bounce it back with the REASONS attached, as a marked HumanMessage
            # (see GATE_FEEDBACK_KEY for why it is a human turn, and why marking it
            # is load-bearing rather than cosmetic).
            return {
                "messages": [_feedback_message(verdict.as_feedback())],
                "gate_attempts": attempts + 1,
                "gate_status": "retry",
            }

        # Out of attempts. REPLACE the draft instead of appending a correction
        # after it -- an answer followed by a retraction still shows the shopper
        # the wrong answer.
        #
        # THE REDUCER TRICK, and it is the LangGraph lesson of this phase:
        # `add_messages` is append-only UNLESS the message you return carries an
        # ID THAT ALREADY EXISTS IN STATE, in which case it OVERWRITES that entry.
        # So returning a message with the draft's own id edits history in place.
        return {
            "messages": [AIMessage(content=REFUSAL, id=draft.id)],
            "gate_attempts": 0,
            "gate_status": "refused",
        }

    return gate


def gate_router(state: dict) -> str:
    """Where to go after the gate: back to the agent to revise, or out to the user."""
    return "agent" if state.get("gate_status") == "retry" else "__end__"
