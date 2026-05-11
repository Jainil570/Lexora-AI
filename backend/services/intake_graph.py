"""LangGraph-based conversational intake workflow.

Implements a StateGraph with three core nodes:
  1. analyze_state_node  — determines what data is still needed
  2. ask_user_node       — uses the LLM to formulate the next question
  3. generate_document_node — calls the existing document generators

The graph uses conditional edges to route between asking more questions
and triggering document generation.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from models.agent_state import (
    AgentState,
    ChatMessage,
    DOCUMENT_FIELD_REGISTRY,
    DOCUMENT_TYPE_LABELS,
    SUPPORTED_DOCUMENT_TYPES,
    create_initial_state,
)
from models.legal_state import (
    NDALegalState, FounderAgreementLegalState,
    ESOPLegalState, VendorContractLegalState, EmploymentAgreementLegalState,
)
from services.llm_client import get_llm_provider
from services.document_generator import (
    generate_nda, generate_founder_agreement,
    generate_esop, generate_vendor_contract, generate_employment_agreement,
)
from services.file_exporter import generate_docx, generate_pdf
from utils.logger import log_event

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


# ── Helper utilities ─────────────────────────────────────────────────────────

def _load_system_prompt() -> str:
    """Load the intake orchestrator system prompt."""
    path = PROMPTS_DIR / "intake_orchestrator.txt"
    return path.read_text(encoding="utf-8")


def _get_missing_fields(doc_type: str, collected: Dict[str, Any]) -> List[str]:
    """Return list of required fields not yet collected for the document type."""
    registry = DOCUMENT_FIELD_REGISTRY.get(doc_type)
    if not registry:
        return []
    return [f for f in registry["required"] if f not in collected or collected[f] is None]


def _build_user_context(state: AgentState) -> str:
    """Build the user-context message that includes conversation history,
    collected fields, and missing fields for the LLM."""
    doc_type = state.get("document_type")
    collected = state.get("collected_fields", {})
    missing = state.get("missing_fields", [])
    history = state.get("conversation_history", [])

    parts = []

    # Current document type
    if doc_type:
        label = DOCUMENT_TYPE_LABELS.get(doc_type, doc_type)
        parts.append(f"DOCUMENT TYPE: {label} ({doc_type})")
    else:
        parts.append("DOCUMENT TYPE: Not yet determined")
        parts.append(f"AVAILABLE TYPES: {', '.join(SUPPORTED_DOCUMENT_TYPES)}")

    # Already collected fields
    if collected:
        parts.append("\nFIELDS ALREADY COLLECTED:")
        for k, v in collected.items():
            parts.append(f"  - {k}: {v}")

    # Missing fields
    if missing and doc_type:
        registry = DOCUMENT_FIELD_REGISTRY.get(doc_type, {})
        field_descs = registry.get("field_descriptions", {})
        parts.append("\nFIELDS STILL NEEDED (required):")
        for f in missing:
            desc = field_descs.get(f, "")
            parts.append(f"  - {f}: {desc}")

    # Conversation history (last 10 messages for context window efficiency)
    recent = history[-10:] if len(history) > 10 else history
    if recent:
        parts.append("\nCONVERSATION HISTORY:")
        for msg in recent:
            parts.append(f"  [{msg['role'].upper()}]: {msg['content']}")

    # Latest user message (the most recent one)
    if recent and recent[-1]["role"] == "user":
        parts.append(f"\nUSER'S LATEST MESSAGE: {recent[-1]['content']}")

    return "\n".join(parts)


def _parse_llm_response(raw: str) -> Dict[str, Any]:
    """Parse the JSON response from the orchestrator LLM.
    Handles cases where the LLM wraps JSON in markdown code fences."""
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse LLM JSON response: {text[:200]}")
        # Fallback: treat the whole response as the assistant message
        return {
            "assistant_message": raw.strip(),
            "extracted_fields": {},
            "fields_complete": False,
        }


# ── Graph Nodes ──────────────────────────────────────────────────────────────

async def analyze_state_node(state: AgentState) -> AgentState:
    """Determine what data is still missing and update the state accordingly.

    This node is the router: it inspects collected fields and decides if we
    should keep asking questions or proceed to document generation.
    """
    doc_type = state.get("document_type")

    # If no document type yet, we need to keep asking
    if not doc_type:
        return {**state, "status": "awaiting_document_type", "missing_fields": []}

    # Calculate missing required fields
    collected = state.get("collected_fields", {})
    missing = _get_missing_fields(doc_type, collected)

    if missing:
        return {**state, "status": "collecting_fields", "missing_fields": missing}
    else:
        return {**state, "status": "ready_to_generate", "missing_fields": []}


async def ask_user_node(state: AgentState) -> AgentState:
    """Use the LLM to formulate the next conversational question.

    This node sends the current state context to the LLM and parses
    the structured JSON response to extract any new field values and
    the assistant's reply message.
    """
    system_prompt = _load_system_prompt()
    user_context = _build_user_context(state)

    provider = get_llm_provider()
    raw_response = await provider.generate(
        system=system_prompt,
        user=user_context,
        temperature=0.3,
    )

    parsed = _parse_llm_response(raw_response)

    # Extract new fields from the LLM's analysis of the user's message
    extracted = parsed.get("extracted_fields", {})
    fields_complete = parsed.get("fields_complete", False)
    assistant_message = parsed.get("assistant_message", "I'm sorry, could you rephrase that?")

    # Merge extracted fields into collected
    collected = {**state.get("collected_fields", {})}

    # Handle document_type extraction
    if "document_type" in extracted:
        detected_type = extracted.pop("document_type")
        if detected_type in SUPPORTED_DOCUMENT_TYPES:
            state = {**state, "document_type": detected_type}

    # Merge remaining extracted fields
    for key, value in extracted.items():
        if value is not None and value != "":
            collected[key] = value

    # Add assistant message to history
    history = list(state.get("conversation_history", []))
    history.append(ChatMessage(role="assistant", content=assistant_message))

    new_state = {
        **state,
        "collected_fields": collected,
        "conversation_history": history,
    }

    # If LLM says all fields are complete, mark as ready
    if fields_complete:
        new_state["status"] = "ready_to_generate"
        new_state["missing_fields"] = []
    else:
        # Recalculate missing fields
        doc_type = new_state.get("document_type")
        if doc_type:
            new_state["missing_fields"] = _get_missing_fields(doc_type, collected)
            new_state["status"] = "collecting_fields"
        else:
            new_state["status"] = "awaiting_document_type"

    return new_state


async def generate_document_node(state: AgentState) -> AgentState:
    """Generate the final legal document using the existing document generators.

    This node bridges the conversational state into the existing Pydantic
    LegalState models and calls the appropriate generation function.
    """
    doc_type = state.get("document_type")
    collected = state.get("collected_fields", {})
    session_id = state.get("session_id", "unknown")

    await log_event(
        event_type="chat_generation_started",
        session_id=session_id,
        data={"document_type": doc_type, "collected_fields": collected},
    )

    try:
        generated_text = await _dispatch_generation(doc_type, collected)
    except Exception as e:
        logger.error(f"Document generation failed: {e}")
        await log_event(
            event_type="chat_generation_failed",
            session_id=session_id,
            error=str(e),
        )
        history = list(state.get("conversation_history", []))
        history.append(ChatMessage(
            role="assistant",
            content=f"I'm sorry, document generation encountered an error: {str(e)}. Please try again.",
        ))
        return {
            **state,
            "status": "error",
            "error": str(e),
            "conversation_history": history,
        }

    # Generate export files
    try:
        docx_path = generate_docx(generated_text, session_id, doc_type)
        pdf_path = generate_pdf(generated_text, session_id, doc_type)
    except Exception as e:
        logger.error(f"File export failed: {e}")
        docx_path = ""
        pdf_path = ""

    session_short = session_id[:8]
    docx_url = f"/api/documents/download/{doc_type}_{session_short}.docx"
    pdf_url = f"/api/documents/download/{doc_type}_{session_short}.pdf"

    await log_event(
        event_type="chat_generation_completed",
        session_id=session_id,
        data={
            "document_type": doc_type,
            "text_length": len(generated_text),
            "docx_path": docx_path,
            "pdf_path": pdf_path,
        },
    )

    # Add completion message to history
    label = DOCUMENT_TYPE_LABELS.get(doc_type, doc_type)
    history = list(state.get("conversation_history", []))
    history.append(ChatMessage(
        role="assistant",
        content=f"Your {label} has been generated successfully! You can preview it below and download it as PDF or DOCX.",
    ))

    return {
        **state,
        "status": "completed",
        "generated_text": generated_text,
        "docx_download_url": docx_url,
        "pdf_download_url": pdf_url,
        "conversation_history": history,
    }


async def _dispatch_generation(doc_type: str, fields: Dict[str, Any]) -> str:
    """Route to the correct document generator based on document type."""
    generators = {
        "nda": (NDALegalState, generate_nda),
        "founder_agreement": (FounderAgreementLegalState, generate_founder_agreement),
        "esop": (ESOPLegalState, generate_esop),
        "vendor_contract": (VendorContractLegalState, generate_vendor_contract),
        "employment_agreement": (EmploymentAgreementLegalState, generate_employment_agreement),
    }

    if doc_type not in generators:
        raise ValueError(f"Unsupported document type: {doc_type}")

    StateClass, gen_fn = generators[doc_type]
    legal_state = StateClass(**fields)
    return await gen_fn(legal_state)


# ── Graph Construction ───────────────────────────────────────────────────────

def _should_generate(state: AgentState) -> str:
    """Conditional edge: decide whether to generate or keep asking."""
    if state.get("status") == "ready_to_generate":
        return "generate"
    return "ask"


def build_intake_graph() -> StateGraph:
    """Build and compile the LangGraph intake workflow.

    Graph flow:
        analyze_state → (conditional) → ask_user OR generate_document
        ask_user → END (returns to user for next message)
        generate_document → END (returns completed document)
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("analyze_state", analyze_state_node)
    graph.add_node("ask_user", ask_user_node)
    graph.add_node("generate_document", generate_document_node)

    # Set entry point
    graph.set_entry_point("analyze_state")

    # Conditional routing from analyze_state
    graph.add_conditional_edges(
        "analyze_state",
        _should_generate,
        {
            "ask": "ask_user",
            "generate": "generate_document",
        },
    )

    # Both ask_user and generate_document terminate the current graph run
    # (the user will send another message to continue the conversation)
    graph.add_edge("ask_user", END)
    graph.add_edge("generate_document", END)

    return graph


# Module-level compiled graph (reusable across requests)
_compiled_graph = None


def get_intake_graph():
    """Get or create the compiled intake graph (singleton)."""
    global _compiled_graph
    if _compiled_graph is None:
        graph = build_intake_graph()
        _compiled_graph = graph.compile()
    return _compiled_graph


async def run_intake_step(state: AgentState) -> AgentState:
    """Run one step of the intake graph with the given state.

    This is the main entry point called by the API router.
    Each call processes one user message through the graph and returns
    the updated state.
    """
    graph = get_intake_graph()
    result = await graph.ainvoke(state)
    return result
