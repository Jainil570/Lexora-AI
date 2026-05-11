"""Chat router — conversational intake endpoint powered by LangGraph.

Provides a stateful, multi-turn conversation API that replaces the need
for users to fill out large legal forms. State is managed in-memory per
session (will be backed by MongoDB in a future phase).
"""
import logging
from typing import Dict

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form

from models.agent_state import (
    AgentState,
    ChatMessage,
    DOCUMENT_FIELD_REGISTRY,
    DOCUMENT_TYPE_LABELS,
    create_initial_state,
)
from models.chat_models import (
    ChatRequest,
    ChatResponse,
    CollectedFieldsSnapshot,
    UpdateStateRequest,
)
from services.intake_graph import run_intake_step
from services.document_parser import parse_and_chunk
from services.llm_client import get_llm_provider
from utils.auth import get_current_user
from utils.logger import log_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ── In-memory session store ──────────────────────────────────────────────────
# Maps session_id → AgentState. In production this would be Redis or MongoDB.
_sessions: Dict[str, AgentState] = {}


def _build_collected_snapshot(state: AgentState) -> CollectedFieldsSnapshot:
    """Build a CollectedFieldsSnapshot from the current agent state."""
    doc_type = state.get("document_type")
    collected = state.get("collected_fields", {})
    missing = state.get("missing_fields", [])

    # Calculate progress
    total_required = 0
    filled = 0
    if doc_type and doc_type in DOCUMENT_FIELD_REGISTRY:
        required = DOCUMENT_FIELD_REGISTRY[doc_type]["required"]
        total_required = len(required)
        filled = sum(1 for f in required if f in collected and collected[f] is not None)

    progress = (filled / total_required * 100) if total_required > 0 else 0.0

    return CollectedFieldsSnapshot(
        document_type=doc_type,
        document_type_label=DOCUMENT_TYPE_LABELS.get(doc_type) if doc_type else None,
        fields=collected,
        missing_fields=missing,
        progress_percent=round(progress, 1),
    )


def _get_last_assistant_message(state: AgentState) -> str:
    """Get the most recent assistant message from conversation history."""
    history = state.get("conversation_history", [])
    for msg in reversed(history):
        if msg["role"] == "assistant":
            return msg["content"]
    return "Hello! I'm your Lexora Legal Copilot. What legal document would you like to create today?"


@router.post("/message", response_model=ChatResponse)
async def send_chat_message(
    request: ChatRequest,
    user_id: str = Depends(get_current_user),
):
    """Process a user message through the LangGraph intake workflow.

    - If `session_id` is omitted, a new conversation session is started.
    - If `session_id` is provided, the existing session state is loaded
      and the user's message is appended before running the next graph step.

    Returns the assistant's response and the current state snapshot.
    """
    # Load or create session
    if request.session_id and request.session_id in _sessions:
        state = _sessions[request.session_id]
        session_id = request.session_id

        # Don't allow further messages on completed sessions
        if state.get("status") == "completed":
            return ChatResponse(
                session_id=session_id,
                status="completed",
                assistant_message="This document has already been generated. Start a new conversation to create another document.",
                collected_state=_build_collected_snapshot(state),
                preview_text=state.get("generated_text"),
                docx_download_url=state.get("docx_download_url"),
                pdf_download_url=state.get("pdf_download_url"),
            )
    else:
        state = create_initial_state()
        session_id = state["session_id"]

    # Append the user's message to conversation history
    history = list(state.get("conversation_history", []))
    history.append(ChatMessage(role="user", content=request.message))
    state = {**state, "conversation_history": history}

    # Log the incoming message
    await log_event(
        event_type="chat_message_received",
        session_id=session_id,
        data={
            "user_id": user_id,
            "message_length": len(request.message),
            "status": state.get("status"),
            "document_type": state.get("document_type"),
        },
    )

    # Run one step of the LangGraph workflow
    try:
        state = await run_intake_step(state)
    except Exception as e:
        logger.error(f"Intake graph error: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Conversational intake service encountered an error: {str(e)}",
        )

    # Persist updated state
    _sessions[session_id] = state

    # Build response
    assistant_msg = _get_last_assistant_message(state)
    status = state.get("status", "collecting_fields")

    response = ChatResponse(
        session_id=session_id,
        status=status,
        assistant_message=assistant_msg,
        collected_state=_build_collected_snapshot(state),
    )

    # Attach document data if generation is complete
    if status == "completed":
        response.preview_text = state.get("generated_text")
        response.docx_download_url = state.get("docx_download_url")
        response.pdf_download_url = state.get("pdf_download_url")

    await log_event(
        event_type="chat_response_sent",
        session_id=session_id,
        data={
            "user_id": user_id,
            "status": status,
            "document_type": state.get("document_type"),
            "progress": response.collected_state.progress_percent,
        },
    )

    return response


@router.get("/session/{session_id}")
async def get_session_state(
    session_id: str,
    user_id: str = Depends(get_current_user),
):
    """Retrieve the current state of a chat session (for page refreshes)."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = _sessions[session_id]

    # Return the conversation history and current state
    return {
        "session_id": session_id,
        "status": state.get("status"),
        "document_type": state.get("document_type"),
        "collected_state": _build_collected_snapshot(state).model_dump(),
        "conversation_history": [
            {"role": msg["role"], "content": msg["content"]}
            for msg in state.get("conversation_history", [])
        ],
        "preview_text": state.get("generated_text"),
        "docx_download_url": state.get("docx_download_url"),
        "pdf_download_url": state.get("pdf_download_url"),
    }


import json

@router.post("/update-state", response_model=ChatResponse)
async def update_session_state(
    request: UpdateStateRequest,
    user_id: str = Depends(get_current_user),
):
    """Manually update the session state with form fields and run the graph."""
    if request.session_id and request.session_id in _sessions:
        state = _sessions[request.session_id]
        session_id = request.session_id
    else:
        state = create_initial_state()
        session_id = state["session_id"]
    
    state = {**state, "document_type": request.document_type}
    collected = {**state.get("collected_fields", {})}
    for k, v in request.fields.items():
        if v is not None and str(v).strip() != "":
            collected[k] = v
            
    state["collected_fields"] = collected
    
    history = list(state.get("conversation_history", []))
    history.append(ChatMessage(role="user", content="[Manual form submission updated the data]"))
    state["conversation_history"] = history

    try:
        state = await run_intake_step(state)
    except Exception as e:
        logger.error(f"Intake graph error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

    _sessions[session_id] = state

    status = state.get("status", "collecting_fields")
    response = ChatResponse(
        session_id=session_id,
        status=status,
        assistant_message=_get_last_assistant_message(state),
        collected_state=_build_collected_snapshot(state),
    )
    if status == "completed":
        response.preview_text = state.get("generated_text")
        response.docx_download_url = state.get("docx_download_url")
        response.pdf_download_url = state.get("pdf_download_url")

    return response


@router.post("/upload", response_model=ChatResponse)
async def upload_document_for_extraction(
    session_id: str = Form(None),
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
):
    """Upload a document to extract fields automatically using PyMuPDF and LLM."""
    if session_id and session_id in _sessions:
        state = _sessions[session_id]
    else:
        state = create_initial_state()
        session_id = state["session_id"]
        
    doc_type = state.get("document_type")
    
    # 1. Parse Document
    try:
        file_bytes = await file.read()
        chunks = parse_and_chunk(file_bytes, file.filename)
        document_text = "\n\n".join(chunks[:3]) # Limit to 3 chunks to avoid massive context
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse document: {str(e)}")
        
    # 2. Extract using LLM
    provider = get_llm_provider()
    system_prompt = "You are a legal AI. Extract structured data from the following document text."
    
    # Build a specific extraction prompt
    extraction_prompt = f"Document Type: {doc_type or 'Unknown'}\n\n"
    if doc_type and doc_type in DOCUMENT_FIELD_REGISTRY:
        req_fields = DOCUMENT_FIELD_REGISTRY[doc_type]["required"]
        extraction_prompt += f"Required Fields to extract: {', '.join(req_fields)}\n\n"
    
    extraction_prompt += f"Document Text:\n{document_text}\n\n"
    extraction_prompt += "Respond ONLY with a valid JSON object containing the extracted fields. Do NOT invent information. If a field is not found, omit it."
    
    try:
        raw_response = await provider.generate(
            system=system_prompt,
            user=extraction_prompt,
            temperature=0.0
        )
        # Parse JSON
        text = raw_response.strip()
        if text.startswith("```json"): text = text[7:]
        elif text.startswith("```"): text = text[3:]
        if text.endswith("```"): text = text[:-3]
        extracted_fields = json.loads(text.strip())
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        extracted_fields = {}
        
    # 3. Update State
    collected = {**state.get("collected_fields", {})}
    for k, v in extracted_fields.items():
        if v is not None and str(v).strip() != "":
            collected[k] = v
            
    state["collected_fields"] = collected
    
    history = list(state.get("conversation_history", []))
    history.append(ChatMessage(role="user", content=f"[Uploaded document: {file.filename}]"))
    state["conversation_history"] = history
    
    try:
        state = await run_intake_step(state)
    except Exception as e:
        logger.error(f"Intake graph error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))

    _sessions[session_id] = state

    status = state.get("status", "collecting_fields")
    response = ChatResponse(
        session_id=session_id,
        status=status,
        assistant_message=_get_last_assistant_message(state),
        collected_state=_build_collected_snapshot(state),
    )
    if status == "completed":
        response.preview_text = state.get("generated_text")
        response.docx_download_url = state.get("docx_download_url")
        response.pdf_download_url = state.get("pdf_download_url")

    return response
