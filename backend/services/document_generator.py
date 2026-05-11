"""Document generator service — orchestrates legal state → prompt → LLM → text."""
import logging
from pathlib import Path

from models.legal_state import (
    NDALegalState, FounderAgreementLegalState,
    ESOPLegalState, VendorContractLegalState, EmploymentAgreementLegalState
)
from services.llm_client import get_llm_provider
from utils.logger import log_event

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = PROMPTS_DIR / filename
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _build_nda_user_prompt(state: NDALegalState, grounding_context: str = "") -> str:
    """Build the user prompt from the NDA legal state.

    If grounding_context is provided (from RAG retrieval), it is prepended
    to instruct the LLM to use retrieved clauses as the legal foundation.
    """
    nda_type_label = "Mutual" if state.nda_type.value == "mutual" else "One-Way"
    stage_label = state.startup_stage.value.replace("_", " ").title()

    base_prompt = f"""Generate a {nda_type_label} Non-Disclosure Agreement with the following details:

- Disclosing Party: {state.disclosing_party}
- Receiving Party: {state.receiving_party}
- NDA Type: {nda_type_label}
- Purpose: {state.purpose}
- Confidentiality Duration: {state.confidentiality_duration_years} year(s)
- Jurisdiction: {state.jurisdiction_state}, India
- Startup Stage: {stage_label}
- Governing Law: Laws of India, with courts in {state.jurisdiction_state} having exclusive jurisdiction"""

    if grounding_context:
        return f"""{grounding_context}

{base_prompt}

IMPORTANT: Use the REFERENCE LEGAL CLAUSES above as your primary legal foundation.
Adapt and customize each clause for the specific parties, jurisdiction, and purpose described.
Do NOT invent new legal structures — build from the retrieved reference clauses.
Ensure all clauses are adapted for Indian jurisdiction and the specified governing law.

Generate the complete NDA document now."""
    else:
        return base_prompt + "\n\nGenerate the complete NDA document now."


def _build_founder_agreement_user_prompt(state: FounderAgreementLegalState) -> str:
    """Build the user prompt from the Founder Agreement legal state."""
    stage_label = state.startup_stage.value.replace("_", " ").title()
    ip_clause = (
        "All intellectual property created by founders shall be fully assigned to the company."
        if state.ip_assignment
        else "IP assignment to be governed by individual IP agreements."
    )
    vesting_label = {
        "4yr_1yr_cliff": "4-year vesting with a 1-year cliff (25% vests at month 12, then monthly over 36 months)",
        "3yr_monthly": "3-year monthly vesting with no cliff",
        "2yr_monthly": "2-year monthly vesting with no cliff",
        "custom": "custom vesting schedule as determined by the Board",
    }.get(state.vesting_schedule.value, state.vesting_schedule.value)

    return f"""Generate a Founder Agreement for an Indian {state.company_type} with the following details:

COMPANY:
- Company Name: {state.company_name}
- Entity Type: {state.company_type}
- Jurisdiction: {state.jurisdiction_state}, India
- Startup Stage: {stage_label}

FOUNDERS:
- Founder 1: {state.founder_1_name}, Role: {state.founder_1_role}, Equity: {state.founder_1_equity_percent}%
- Founder 2: {state.founder_2_name}, Role: {state.founder_2_role}, Equity: {state.founder_2_equity_percent}%

KEY TERMS:
- Vesting Schedule: {vesting_label}
- IP Assignment: {ip_clause}
- Non-Compete Duration: {state.non_compete_duration_months} months post-departure
- Governing Law: Laws of India, with courts in {state.jurisdiction_state} having exclusive jurisdiction

Generate the complete Founder Agreement document now."""


async def generate_nda(state: NDALegalState, grounding_context: str = "") -> str:
    """Generate an NDA document from the validated legal state.

    Args:
        state: Validated NDA legal state.
        grounding_context: Optional RAG-retrieved clause context for grounding.
    """
    await log_event(
        event_type="generation_started",
        session_id=state.session_id,
        data={
            "document_type": "nda",
            "nda_type": state.nda_type.value,
            "startup_stage": state.startup_stage.value,
            "jurisdiction": state.jurisdiction_state,
            "has_grounding": bool(grounding_context),
        },
        prompt_id="nda_generation",
    )

    try:
        system_prompt = _load_prompt("nda_generation.txt")
        user_prompt = _build_nda_user_prompt(state, grounding_context)

        provider = get_llm_provider()
        generated_text = await provider.generate(
            system=system_prompt,
            user=user_prompt,
            temperature=0.15,
        )

        await log_event(
            event_type="generation_completed",
            session_id=state.session_id,
            data={
                "document_type": "nda",
                "text_length": len(generated_text),
                "used_rag_grounding": bool(grounding_context),
            },
            prompt_id="nda_generation",
        )

        return generated_text

    except Exception as e:
        await log_event(
            event_type="generation_failed",
            session_id=state.session_id,
            error=str(e),
            prompt_id="nda_generation",
        )
        raise


async def generate_founder_agreement(state: FounderAgreementLegalState) -> str:
    """Generate a Founder Agreement document from the validated legal state."""
    await log_event(
        event_type="generation_started",
        session_id=state.session_id,
        data={
            "document_type": "founder_agreement",
            "company_name": state.company_name,
            "startup_stage": state.startup_stage.value,
            "jurisdiction": state.jurisdiction_state,
        },
        prompt_id="founder_agreement_generation",
    )

    try:
        system_prompt = _load_prompt("founder_agreement_generation.txt")
        user_prompt = _build_founder_agreement_user_prompt(state)

        provider = get_llm_provider()
        generated_text = await provider.generate(
            system=system_prompt,
            user=user_prompt,
            temperature=0.15,
        )

        await log_event(
            event_type="generation_completed",
            session_id=state.session_id,
            data={"document_type": "founder_agreement", "text_length": len(generated_text)},
            prompt_id="founder_agreement_generation",
        )

        return generated_text

    except Exception as e:
        await log_event(
            event_type="generation_failed",
            session_id=state.session_id,
            error=str(e),
            prompt_id="founder_agreement_generation",
        )
        raise


def _build_esop_user_prompt(state: ESOPLegalState) -> str:
    """Build the user prompt for ESOP Policy generation."""
    stage_label = state.startup_stage.value.replace("_", " ").title()
    vesting_map = {"cliff": "cliff vesting (all options vest at end of period)",
                   "graded": "graded monthly vesting after cliff",
                   "milestone": "milestone-based vesting"}
    return f"""Generate an ESOP Policy for an Indian startup with the following details:

COMPANY: {state.company_name}
JURISDICTION: {state.jurisdiction_state}, India
STARTUP STAGE: {stage_label}

ESOP PLAN TERMS:
- Total ESOP Pool: {state.total_esop_pool_percent}% of total paid-up share capital
- Vesting Type: {vesting_map.get(state.vesting_type.value, state.vesting_type.value)}
- Vesting Period: {state.vesting_period_years} years
- Cliff Period: {state.cliff_period_months} months
- Exercise Price: INR {state.exercise_price_inr:.2f} per share
- Exercise Window: {state.exercise_window_months} months post-vesting
- Governing Law: Laws of India, with courts in {state.jurisdiction_state} having exclusive jurisdiction

Generate the complete ESOP Policy document now."""


async def generate_esop(state: ESOPLegalState) -> str:
    """Generate an ESOP Policy document from the validated legal state."""
    await log_event(event_type="generation_started", session_id=state.session_id,
                    data={"document_type": "esop", "company_name": state.company_name,
                          "startup_stage": state.startup_stage.value}, prompt_id="esop_generation")
    try:
        system_prompt = _load_prompt("esop_generation.txt")
        generated_text = await get_llm_provider().generate(
            system=system_prompt, user=_build_esop_user_prompt(state), temperature=0.15)
        await log_event(event_type="generation_completed", session_id=state.session_id,
                        data={"document_type": "esop", "text_length": len(generated_text)},
                        prompt_id="esop_generation")
        return generated_text
    except Exception as e:
        await log_event(event_type="generation_failed", session_id=state.session_id,
                        error=str(e), prompt_id="esop_generation")
        raise


def _build_vendor_contract_user_prompt(state: VendorContractLegalState) -> str:
    """Build the user prompt for Vendor Contract generation."""
    stage_label = state.startup_stage.value.replace("_", " ").title()
    ip_map = {"client": "all deliverables and work product owned by the client (company) upon payment",
              "vendor": "vendor retains IP ownership; client receives perpetual license",
              "shared": "joint ownership with equal usage rights"}
    duration_map = {"monthly": "month-to-month", "quarterly": "quarterly",
                    "annual": "one-year fixed term", "fixed_term": "fixed term"}
    return f"""Generate a Vendor/Service Agreement for an Indian startup with the following details:

PARTIES:
- Client Company: {state.company_name}
- Vendor/Service Provider: {state.vendor_name}

SERVICES: {state.service_description}

KEY TERMS:
- Contract Value: INR {state.contract_value_inr:,.2f}
- Contract Duration: {duration_map.get(state.contract_duration.value, state.contract_duration.value)}
- Payment Terms: Net {state.payment_terms_days} days from invoice date
- SLA Included: {"Yes" if state.sla_included else "No"}
- IP Ownership: {ip_map.get(state.ip_ownership, state.ip_ownership)}
- Startup Stage: {stage_label}
- Jurisdiction: {state.jurisdiction_state}, India

Generate the complete Vendor Contract document now."""


async def generate_vendor_contract(state: VendorContractLegalState) -> str:
    """Generate a Vendor Contract document from the validated legal state."""
    await log_event(event_type="generation_started", session_id=state.session_id,
                    data={"document_type": "vendor_contract", "company_name": state.company_name,
                          "vendor_name": state.vendor_name}, prompt_id="vendor_contract_generation")
    try:
        system_prompt = _load_prompt("vendor_contract_generation.txt")
        generated_text = await get_llm_provider().generate(
            system=system_prompt, user=_build_vendor_contract_user_prompt(state), temperature=0.15)
        await log_event(event_type="generation_completed", session_id=state.session_id,
                        data={"document_type": "vendor_contract", "text_length": len(generated_text)},
                        prompt_id="vendor_contract_generation")
        return generated_text
    except Exception as e:
        await log_event(event_type="generation_failed", session_id=state.session_id,
                        error=str(e), prompt_id="vendor_contract_generation")
        raise


def _build_employment_agreement_user_prompt(state: EmploymentAgreementLegalState) -> str:
    """Build the user prompt for Employment Agreement generation."""
    stage_label = state.startup_stage.value.replace("_", " ").title()
    emp_type_map = {"full_time": "Full-Time Permanent", "part_time": "Part-Time",
                    "contract": "Fixed-Term Contract", "probation": "Probationary"}
    return f"""Generate an Employment Agreement for an Indian startup with the following details:

COMPANY: {state.company_name}
EMPLOYEE: {state.employee_name}

EMPLOYMENT TERMS:
- Designation: {state.designation}
- Department: {state.department}
- Employment Type: {emp_type_map.get(state.employment_type.value, state.employment_type.value)}
- Annual CTC: INR {state.ctc_annual_inr:,.2f}
- Work Location: {state.work_location}
- Probation Period: {state.probation_period_months} months
- Notice Period: {state.notice_period_days} days
- Non-Compete Duration Post-Termination: {state.non_compete_months} months
- Startup Stage: {stage_label}
- Jurisdiction: {state.jurisdiction_state}, India
- Governing Law: Laws of India and the applicable Shops & Establishments Act of {state.jurisdiction_state}

Generate the complete Employment Agreement document now."""


async def generate_employment_agreement(state: EmploymentAgreementLegalState) -> str:
    """Generate an Employment Agreement document from the validated legal state."""
    await log_event(event_type="generation_started", session_id=state.session_id,
                    data={"document_type": "employment_agreement",
                          "company_name": state.company_name,
                          "employee_name": state.employee_name}, prompt_id="employment_agreement_generation")
    try:
        system_prompt = _load_prompt("employment_agreement_generation.txt")
        generated_text = await get_llm_provider().generate(
            system=system_prompt, user=_build_employment_agreement_user_prompt(state), temperature=0.15)
        await log_event(event_type="generation_completed", session_id=state.session_id,
                        data={"document_type": "employment_agreement", "text_length": len(generated_text)},
                        prompt_id="employment_agreement_generation")
        return generated_text
    except Exception as e:
        await log_event(event_type="generation_failed", session_id=state.session_id,
                        error=str(e), prompt_id="employment_agreement_generation")
        raise
