"""Unified Agent State for LangGraph conversational intake workflow.

This module defines the TypedDict-based state that flows through the
LangGraph StateGraph. It holds conversation history, collected form fields,
and metadata about the current document generation session.
"""
import uuid
from typing import Any, Dict, List, Literal, Optional, TypedDict


# ── Document type registry ───────────────────────────────────────────────────
# Maps each document type to its REQUIRED and OPTIONAL fields so the
# analyze_state_node can determine what's missing without hard-coding logic.

DOCUMENT_FIELD_REGISTRY: Dict[str, Dict[str, List[str]]] = {
    "nda": {
        "required": [
            "disclosing_party", "receiving_party", "nda_type",
            "purpose", "confidentiality_duration_years",
            "jurisdiction_state", "startup_stage",
        ],
        "optional": ["governing_law"],
        "field_descriptions": {
            "disclosing_party": "Name of the party sharing confidential information",
            "receiving_party": "Name of the party receiving confidential information",
            "nda_type": "Type of NDA — 'mutual' (both parties share info) or 'one_way' (only discloser shares)",
            "purpose": "Business purpose of the NDA (e.g., 'evaluating potential partnership for SaaS integration')",
            "confidentiality_duration_years": "How many years confidentiality obligations last (1-10)",
            "jurisdiction_state": "Indian state for legal jurisdiction (e.g., 'Maharashtra', 'Karnataka')",
            "startup_stage": "Current stage of the startup — 'idea', 'incorporated', 'seed', or 'series_a'",
            "governing_law": "Governing law (defaults to 'India')",
        },
    },
    "founder_agreement": {
        "required": [
            "company_name", "jurisdiction_state", "startup_stage",
            "founder_1_name", "founder_1_role", "founder_1_equity_percent",
            "founder_2_name", "founder_2_role", "founder_2_equity_percent",
        ],
        "optional": [
            "company_type", "vesting_schedule", "ip_assignment",
            "non_compete_duration_months", "governing_law",
        ],
        "field_descriptions": {
            "company_name": "Registered or proposed company name",
            "company_type": "Type of entity (defaults to 'Private Limited Company')",
            "jurisdiction_state": "Indian state for legal jurisdiction",
            "startup_stage": "Current stage — 'idea', 'incorporated', 'seed', or 'series_a'",
            "founder_1_name": "Full name of the first founder",
            "founder_1_role": "Role / designation of the first founder (e.g., 'CEO')",
            "founder_1_equity_percent": "Equity percentage for the first founder (0-100)",
            "founder_2_name": "Full name of the second founder",
            "founder_2_role": "Role / designation of the second founder (e.g., 'CTO')",
            "founder_2_equity_percent": "Equity percentage for the second founder (0-100)",
            "vesting_schedule": "Equity vesting schedule — '4yr_1yr_cliff', '3yr_monthly', '2yr_monthly', or 'custom'",
            "ip_assignment": "Whether IP is assigned to the company (true/false)",
            "non_compete_duration_months": "Non-compete duration in months (6-60, default 24)",
            "governing_law": "Governing law (defaults to 'India')",
        },
    },
    "esop": {
        "required": [
            "company_name", "jurisdiction_state", "startup_stage",
            "total_esop_pool_percent",
        ],
        "optional": [
            "vesting_type", "vesting_period_years", "cliff_period_months",
            "exercise_price_inr", "exercise_window_months", "governing_law",
        ],
        "field_descriptions": {
            "company_name": "Company name",
            "jurisdiction_state": "Indian state for legal jurisdiction",
            "startup_stage": "Current stage — 'idea', 'incorporated', 'seed', or 'series_a'",
            "total_esop_pool_percent": "Total ESOP pool as % of share capital (1-30)",
            "vesting_type": "Vesting type — 'cliff', 'graded', or 'milestone' (default 'graded')",
            "vesting_period_years": "Vesting period in years (1-6, default 4)",
            "cliff_period_months": "Cliff period in months (0-24, default 12)",
            "exercise_price_inr": "Exercise price per share in INR (default 1.0)",
            "exercise_window_months": "Exercise window in months post-vesting (6-120, default 36)",
            "governing_law": "Governing law (defaults to 'India')",
        },
    },
    "vendor_contract": {
        "required": [
            "company_name", "vendor_name", "service_description",
            "contract_value_inr", "jurisdiction_state", "startup_stage",
        ],
        "optional": [
            "contract_duration", "payment_terms_days", "sla_included",
            "ip_ownership", "governing_law",
        ],
        "field_descriptions": {
            "company_name": "Your company name (the client)",
            "vendor_name": "Vendor / service provider name",
            "service_description": "Description of the services being contracted (min 10 chars)",
            "contract_value_inr": "Total contract value in INR",
            "contract_duration": "Duration type — 'monthly', 'quarterly', 'annual', or 'fixed_term' (default 'annual')",
            "payment_terms_days": "Payment terms in days (7-90, default 30)",
            "jurisdiction_state": "Indian state for legal jurisdiction",
            "startup_stage": "Current stage — 'idea', 'incorporated', 'seed', or 'series_a'",
            "sla_included": "Whether SLA clauses should be included (true/false, default true)",
            "ip_ownership": "IP ownership model — 'client', 'vendor', or 'shared' (default 'client')",
            "governing_law": "Governing law (defaults to 'India')",
        },
    },
    "employment_agreement": {
        "required": [
            "company_name", "employee_name", "designation",
            "department", "ctc_annual_inr", "jurisdiction_state", "startup_stage",
        ],
        "optional": [
            "employment_type", "probation_period_months", "notice_period_days",
            "work_location", "non_compete_months", "governing_law",
        ],
        "field_descriptions": {
            "company_name": "Company name (the employer)",
            "employee_name": "Full name of the employee",
            "designation": "Job title / designation",
            "department": "Department name",
            "ctc_annual_inr": "Annual CTC (Cost to Company) in INR",
            "employment_type": "Employment type — 'full_time', 'part_time', 'contract', or 'probation' (default 'full_time')",
            "probation_period_months": "Probation period in months (0-12, default 3)",
            "notice_period_days": "Notice period in days (0-180, default 30)",
            "work_location": "Work location (e.g., 'office', 'remote', 'hybrid')",
            "non_compete_months": "Non-compete duration in months (0-24, default 12)",
            "jurisdiction_state": "Indian state for legal jurisdiction",
            "startup_stage": "Current stage — 'idea', 'incorporated', 'seed', or 'series_a'",
            "governing_law": "Governing law (defaults to 'India')",
        },
    },
}

# All supported document types
SUPPORTED_DOCUMENT_TYPES = list(DOCUMENT_FIELD_REGISTRY.keys())

# Human-friendly labels for document types
DOCUMENT_TYPE_LABELS = {
    "nda": "Non-Disclosure Agreement (NDA)",
    "founder_agreement": "Founder Agreement",
    "esop": "ESOP Policy",
    "vendor_contract": "Vendor / Service Contract",
    "employment_agreement": "Employment Agreement",
}


class ChatMessage(TypedDict):
    """A single message in the conversation."""
    role: Literal["user", "assistant", "system"]
    content: str


class AgentState(TypedDict, total=False):
    """Unified state that flows through the LangGraph intake workflow.

    Fields:
        session_id: Unique session identifier.
        document_type: Which document is being generated (None if not yet chosen).
        collected_fields: Key-value pairs of fields collected from the user so far.
        missing_fields: List of field names still needed.
        conversation_history: Full message history for context.
        status: Current workflow status.
        generated_text: The final generated document text (set after generation).
        error: Error message if something went wrong.
    """
    session_id: str
    document_type: Optional[str]
    collected_fields: Dict[str, Any]
    missing_fields: List[str]
    conversation_history: List[ChatMessage]
    status: Literal[
        "awaiting_document_type",   # Haven't determined document type yet
        "collecting_fields",        # Actively asking questions
        "ready_to_generate",        # All fields collected, about to generate
        "retrieving_clauses",       # RAG clause retrieval in progress
        "generating",               # Document generation in progress
        "completed",                # Document generated successfully
        "error",                    # An error occurred
    ]
    generated_text: Optional[str]
    docx_download_url: Optional[str]
    pdf_download_url: Optional[str]
    retrieved_clauses: Optional[str]
    retrieved_clause_ids: Optional[List[str]]
    error: Optional[str]


def create_initial_state(session_id: Optional[str] = None) -> AgentState:
    """Create a fresh AgentState for a new conversation."""
    return AgentState(
        session_id=session_id or str(uuid.uuid4()),
        document_type=None,
        collected_fields={},
        missing_fields=[],
        conversation_history=[],
        status="awaiting_document_type",
        generated_text=None,
        docx_download_url=None,
        pdf_download_url=None,
        retrieved_clauses=None,
        retrieved_clause_ids=None,
        error=None,
    )
