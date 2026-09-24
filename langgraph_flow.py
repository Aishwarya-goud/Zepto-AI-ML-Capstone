
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

# -----------------------------
# MOCK MODE (Masai requirement)
# -----------------------------
MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"

# Keywords required in acceptance criteria
POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours"
]


class GraphState(TypedDict):
    question: str
    intent: str
    context: str
    answer: str


# -----------------------------
# Intent Classification
# -----------------------------
def classify_intent(state: GraphState):
    question = state["question"].lower()

    if any(word in question for word in POLICY_KEYWORDS):
        state["intent"] = "policy_question"
    else:
        state["intent"] = "general_question"

    return state


# -----------------------------
# Retrieve + Mock Answer
# -----------------------------
def retrieve_and_answer(state: GraphState):

    # Context will be replaced from FAISS inside main.py
    context = state.get("context", "")

    if MOCK_LLM:
        state["answer"] = (
            "Based on the retrieved context: " +
            context[:250]
        )
    else:
        state["answer"] = "Real Gemini answer here."

    return state


# -----------------------------
# Direct Answer
# -----------------------------
def direct_answer(state: GraphState):

    if MOCK_LLM:
        state["answer"] = (
            "This is a canned response for a general question."
        )
    else:
        state["answer"] = "Real Gemini general answer."

    return state


# -----------------------------
# Router
# -----------------------------
def router(state: GraphState):
    return state["intent"]


# -----------------------------
# Build LangGraph
# -----------------------------
builder = StateGraph(GraphState)

builder.add_node("classify_intent", classify_intent)
builder.add_node("retrieve_and_answer", retrieve_and_answer)
builder.add_node("direct_answer", direct_answer)

builder.set_entry_point("classify_intent")

builder.add_conditional_edges(
    "classify_intent",
    router,
    {
        "policy_question": "retrieve_and_answer",
        "general_question": "direct_answer",
    },
)

builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph = builder.compile()