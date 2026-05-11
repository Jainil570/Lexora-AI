"""Pydantic request/response models for the conversational chat intake API."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessageResponse(BaseModel):
    """A single message in the conversation response."""
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    """Request body for sending a message in the conversational intake flow."""
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for an ongoing conversation. Omit or null to start a new session.",
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user's message text.",
    )


class UpdateStateRequest(BaseModel):
    """Request body for manually injecting extracted/form fields into the session state."""
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID. Omit to start a new session.",
    )
    document_type: str = Field(..., description="The type of document (e.g., nda, esop)")
    fields: Dict[str, Any] = Field(..., description="Dictionary of form fields to inject into the state")


class CollectedFieldsSnapshot(BaseModel):
    """Snapshot of all fields collected so far, for the frontend state panel."""
    document_type: Optional[str] = None
    document_type_label: Optional[str] = None
    fields: Dict[str, Any] = {}
    missing_fields: List[str] = []
    progress_percent: float = 0.0


class ChatResponse(BaseModel):
    """Response body for a conversational intake message."""
    session_id: str
    status: str = Field(
        ...,
        description="Current workflow status: awaiting_document_type, collecting_fields, ready_to_generate, generating, completed, error",
    )
    assistant_message: str = Field(
        ...,
        description="The AI assistant's response message.",
    )
    collected_state: CollectedFieldsSnapshot = Field(
        default_factory=CollectedFieldsSnapshot,
        description="Current state of collected fields for the sidebar visualization.",
    )
    # Only present when status == "completed"
    preview_text: Optional[str] = None
    docx_download_url: Optional[str] = None
    pdf_download_url: Optional[str] = None
