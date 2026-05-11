"""Documents router — handles document generation endpoints."""
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import FileResponse

from models.legal_state import (
    NDALegalState, FounderAgreementLegalState,
    ESOPLegalState, VendorContractLegalState, EmploymentAgreementLegalState
)
from models.requests import (
    NDAGenerateRequest, FounderAgreementGenerateRequest,
    ESOPGenerateRequest, VendorContractGenerateRequest, EmploymentAgreementGenerateRequest
)
from models.responses import DocumentGenerateResponse
from services.document_generator import (
    generate_nda, generate_founder_agreement,
    generate_esop, generate_vendor_contract, generate_employment_agreement
)
from services.file_exporter import generate_docx, generate_pdf
from utils.logger import log_event
from utils.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("/generate", response_model=DocumentGenerateResponse)
async def generate_document(
    request: NDAGenerateRequest,
    user_id: str = Depends(get_current_user)
):
    """Generate an NDA document from structured input."""
    # Build the validated legal state from request
    try:
        legal_state = NDALegalState(
            disclosing_party=request.disclosing_party,
            receiving_party=request.receiving_party,
            nda_type=request.nda_type,
            purpose=request.purpose,
            confidentiality_duration_years=request.confidentiality_duration_years,
            jurisdiction_state=request.jurisdiction_state,
            startup_stage=request.startup_stage,
            governing_law=request.governing_law,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    await log_event(
        event_type="document_request_received",
        session_id=legal_state.session_id,
        data={"document_type": "nda", "request": request.model_dump(), "user_id": user_id},
    )

    # Retrieve clauses for RAG grounding
    grounding_context = ""
    try:
        from services.clause_retriever import get_clause_retriever
        retriever = get_clause_retriever()
        result = await retriever.retrieve_for_nda(
            nda_type=legal_state.nda_type.value,
            startup_stage=legal_state.startup_stage.value,
            jurisdiction=legal_state.jurisdiction_state,
            purpose=legal_state.purpose,
            session_id=legal_state.session_id,
        )
        grounding_context = retriever.build_grounding_context(result.clauses)
        logger.info(
            f"RAG: Retrieved {result.total_retrieved} clauses for NDA generation "
            f"({len(result.clause_types_covered)} types)"
        )
    except Exception as e:
        logger.warning(f"Clause retrieval failed, proceeding without RAG: {e}")

    # Generate document text via LLM
    try:
        generated_text = await generate_nda(legal_state, grounding_context=grounding_context)
    except RuntimeError as e:
        logger.error(f"LLM generation failed: {e}")
        await log_event(
            event_type="generation_failed",
            session_id=legal_state.session_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=503,
            detail="Document generation service is temporarily unavailable. Please ensure Ollama is running and try again.",
        )

    # Generate DOCX and PDF files
    try:
        docx_path = generate_docx(generated_text, legal_state.session_id, "nda")
        pdf_path = generate_pdf(generated_text, legal_state.session_id, "nda")
    except Exception as e:
        logger.error(f"File export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate document files.",
        )

    await log_event(
        event_type="document_files_generated",
        session_id=legal_state.session_id,
        data={
            "docx_path": docx_path,
            "pdf_path": pdf_path,
            "user_id": user_id,
        },
    )

    # Build download URLs
    session_short = legal_state.session_id[:8]
    return DocumentGenerateResponse(
        session_id=legal_state.session_id,
        document_type="nda",
        preview_text=generated_text,
        docx_download_url=f"/api/documents/download/nda_{session_short}.docx",
        pdf_download_url=f"/api/documents/download/nda_{session_short}.pdf",
    )


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a generated document file."""
    outputs_dir = Path(__file__).parent.parent / "outputs"
    file_path = outputs_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine media type
    if filename.endswith(".docx"):
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif filename.endswith(".pdf"):
        media_type = "application/pdf"
    else:
        media_type = "application/octet-stream"

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type=media_type,
    )


@router.post("/generate-founder-agreement", response_model=DocumentGenerateResponse)
async def generate_founder_agreement_doc(
    request: FounderAgreementGenerateRequest,
    user_id: str = Depends(get_current_user)
):
    """Generate a Founder Agreement document from structured input."""
    try:
        legal_state = FounderAgreementLegalState(
            company_name=request.company_name,
            company_type=request.company_type,
            jurisdiction_state=request.jurisdiction_state,
            startup_stage=request.startup_stage,
            founder_1_name=request.founder_1_name,
            founder_1_role=request.founder_1_role,
            founder_1_equity_percent=request.founder_1_equity_percent,
            founder_2_name=request.founder_2_name,
            founder_2_role=request.founder_2_role,
            founder_2_equity_percent=request.founder_2_equity_percent,
            vesting_schedule=request.vesting_schedule,
            ip_assignment=request.ip_assignment,
            non_compete_duration_months=request.non_compete_duration_months,
            governing_law=request.governing_law,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    await log_event(
        event_type="document_request_received",
        session_id=legal_state.session_id,
        data={
            "document_type": "founder_agreement",
            "request": request.model_dump(),
            "user_id": user_id,
        },
    )

    try:
        generated_text = await generate_founder_agreement(legal_state)
    except RuntimeError as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Document generation service is temporarily unavailable. Please ensure Ollama is running and try again.",
        )

    try:
        docx_path = generate_docx(generated_text, legal_state.session_id, "founder_agreement")
        pdf_path = generate_pdf(generated_text, legal_state.session_id, "founder_agreement")
    except Exception as e:
        logger.error(f"File export failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate document files.")

    await log_event(
        event_type="document_files_generated",
        session_id=legal_state.session_id,
        data={"docx_path": docx_path, "pdf_path": pdf_path, "user_id": user_id},
    )

    session_short = legal_state.session_id[:8]
    return DocumentGenerateResponse(
        session_id=legal_state.session_id,
        document_type="founder_agreement",
        preview_text=generated_text,
        docx_download_url=f"/api/documents/download/founder_agreement_{session_short}.docx",
        pdf_download_url=f"/api/documents/download/founder_agreement_{session_short}.pdf",
    )

@router.get("/history")
async def get_document_history(
    request: Request,
    user_id: str = Depends(get_current_user)
):
    """Fetch the user's generated documents history from audit logs."""
    db = request.app.state.db
    if db is None:
        return []

    # Find all successful generation events for this user
    cursor = db["audit_logs"].find(
        {
            "event_type": "document_files_generated",
            "data.user_id": user_id
        }
    ).sort("timestamp", -1).limit(50)

    history = []
    async for doc in cursor:
        session_id = doc.get("session_id")
        timestamp = doc.get("timestamp")
        data = doc.get("data", {})
        docx_path = data.get("docx_path", "")
        pdf_path = data.get("pdf_path", "")
        
        # Get the original request to find parties and type
        req_doc = await db["audit_logs"].find_one({
            "event_type": "document_request_received",
            "session_id": session_id
        })
        
        doc_type = "NDA"
        parties = "Unknown"
        if req_doc and "data" in req_doc and "request" in req_doc["data"]:
            req_data = req_doc["data"]["request"]
            doc_type = req_doc["data"].get("document_type", "nda").replace("_", " ").upper()
            # NDA shows disclosing & receiving party; Founder Agreement shows company name
            if "disclosing_party" in req_data:
                parties = f"{req_data.get('disclosing_party', 'A')} & {req_data.get('receiving_party', 'B')}"
            elif "company_name" in req_data:
                parties = f"{req_data.get('company_name', 'Unknown')} ({req_data.get('founder_1_name', '?')} & {req_data.get('founder_2_name', '?')})"
            else:
                parties = "Unknown"

        session_short = session_id[:8] if session_id else ""
        history.append({
            "session_id": session_id,
            "filename": f"{doc_type}_V1_{session_short}.PDF",
            "document_type": doc_type,
            "parties": parties,
            "date": timestamp,
            "status": "COMPLETED",
            "risk": "LOW",
            "docx_url": f"/api/documents/download/{doc_type.lower()}_{session_short}.docx",
            "pdf_url": f"/api/documents/download/{doc_type.lower()}_{session_short}.pdf",
        })

    return history


def _make_endpoint(doc_type_key: str, LegalStateClass, generate_fn, request_fields_fn):
    """Factory to create a standardized generation endpoint."""
    async def endpoint(request, user_id: str = Depends(get_current_user)):
        try:
            legal_state = LegalStateClass(**request_fields_fn(request))
        except Exception as e:
            raise HTTPException(status_code=422, detail=str(e))

        await log_event(event_type="document_request_received", session_id=legal_state.session_id,
                        data={"document_type": doc_type_key, "request": request.model_dump(), "user_id": user_id})
        try:
            generated_text = await generate_fn(legal_state)
        except RuntimeError as e:
            raise HTTPException(status_code=503,
                detail="Document generation service is temporarily unavailable. Please ensure Ollama is running.")

        try:
            docx_path = generate_docx(generated_text, legal_state.session_id, doc_type_key)
            pdf_path = generate_pdf(generated_text, legal_state.session_id, doc_type_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Failed to generate document files.")

        await log_event(event_type="document_files_generated", session_id=legal_state.session_id,
                        data={"docx_path": docx_path, "pdf_path": pdf_path, "user_id": user_id})

        session_short = legal_state.session_id[:8]
        return DocumentGenerateResponse(
            session_id=legal_state.session_id,
            document_type=doc_type_key,
            preview_text=generated_text,
            docx_download_url=f"/api/documents/download/{doc_type_key}_{session_short}.docx",
            pdf_download_url=f"/api/documents/download/{doc_type_key}_{session_short}.pdf",
        )
    return endpoint


@router.post("/generate-esop", response_model=DocumentGenerateResponse)
async def generate_esop_doc(
    request: ESOPGenerateRequest,
    user_id: str = Depends(get_current_user)
):
    """Generate an ESOP Policy document."""
    try:
        legal_state = ESOPLegalState(**request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    await log_event(event_type="document_request_received", session_id=legal_state.session_id,
                    data={"document_type": "esop", "request": request.model_dump(), "user_id": user_id})
    try:
        generated_text = await generate_esop(legal_state)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="Document generation service unavailable.")
    try:
        docx_path = generate_docx(generated_text, legal_state.session_id, "esop")
        pdf_path = generate_pdf(generated_text, legal_state.session_id, "esop")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate document files.")
    await log_event(event_type="document_files_generated", session_id=legal_state.session_id,
                    data={"docx_path": docx_path, "pdf_path": pdf_path, "user_id": user_id})
    session_short = legal_state.session_id[:8]
    return DocumentGenerateResponse(
        session_id=legal_state.session_id, document_type="esop", preview_text=generated_text,
        docx_download_url=f"/api/documents/download/esop_{session_short}.docx",
        pdf_download_url=f"/api/documents/download/esop_{session_short}.pdf",
    )


@router.post("/generate-vendor-contract", response_model=DocumentGenerateResponse)
async def generate_vendor_contract_doc(
    request: VendorContractGenerateRequest,
    user_id: str = Depends(get_current_user)
):
    """Generate a Vendor Contract document."""
    try:
        legal_state = VendorContractLegalState(**request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    await log_event(event_type="document_request_received", session_id=legal_state.session_id,
                    data={"document_type": "vendor_contract", "request": request.model_dump(), "user_id": user_id})
    try:
        generated_text = await generate_vendor_contract(legal_state)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="Document generation service unavailable.")
    try:
        docx_path = generate_docx(generated_text, legal_state.session_id, "vendor_contract")
        pdf_path = generate_pdf(generated_text, legal_state.session_id, "vendor_contract")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate document files.")
    await log_event(event_type="document_files_generated", session_id=legal_state.session_id,
                    data={"docx_path": docx_path, "pdf_path": pdf_path, "user_id": user_id})
    session_short = legal_state.session_id[:8]
    return DocumentGenerateResponse(
        session_id=legal_state.session_id, document_type="vendor_contract", preview_text=generated_text,
        docx_download_url=f"/api/documents/download/vendor_contract_{session_short}.docx",
        pdf_download_url=f"/api/documents/download/vendor_contract_{session_short}.pdf",
    )


@router.post("/generate-employment-agreement", response_model=DocumentGenerateResponse)
async def generate_employment_agreement_doc(
    request: EmploymentAgreementGenerateRequest,
    user_id: str = Depends(get_current_user)
):
    """Generate an Employment Agreement document."""
    try:
        legal_state = EmploymentAgreementLegalState(**request.model_dump())
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))
    await log_event(event_type="document_request_received", session_id=legal_state.session_id,
                    data={"document_type": "employment_agreement", "request": request.model_dump(), "user_id": user_id})
    try:
        generated_text = await generate_employment_agreement(legal_state)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail="Document generation service unavailable.")
    try:
        docx_path = generate_docx(generated_text, legal_state.session_id, "employment_agreement")
        pdf_path = generate_pdf(generated_text, legal_state.session_id, "employment_agreement")
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to generate document files.")
    await log_event(event_type="document_files_generated", session_id=legal_state.session_id,
                    data={"docx_path": docx_path, "pdf_path": pdf_path, "user_id": user_id})
    session_short = legal_state.session_id[:8]
    return DocumentGenerateResponse(
        session_id=legal_state.session_id, document_type="employment_agreement", preview_text=generated_text,
        docx_download_url=f"/api/documents/download/employment_agreement_{session_short}.docx",
        pdf_download_url=f"/api/documents/download/employment_agreement_{session_short}.pdf",
    )
