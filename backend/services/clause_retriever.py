"""Clause retrieval pipeline for RAG-based document generation.

Provides metadata-aware clause retrieval with logging, and builds
structured grounding context for injection into generation prompts.

Usage:
    from services.clause_retriever import ClauseRetriever
    retriever = ClauseRetriever()
    result = await retriever.retrieve_for_nda(
        nda_type="mutual",
        startup_stage="seed",
        jurisdiction="Maharashtra",
        purpose="Protect trade secrets during investor discussions",
    )
    grounding = retriever.build_grounding_context(result.clauses)
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from services.vector_store import get_vector_store
from utils.logger import log_event

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedClause:
    """A single retrieved clause with metadata and score."""
    clause_id: str
    clause_type: str
    clause_name: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a clause retrieval operation."""
    clauses: List[RetrievedClause]
    query: str
    filters_used: Dict[str, Any]
    total_retrieved: int

    @property
    def clause_ids(self) -> List[str]:
        return [c.clause_id for c in self.clauses]

    @property
    def clause_types_covered(self) -> List[str]:
        return sorted(set(c.clause_type for c in self.clauses))


# ── Core clause types that an NDA should cover ───────────────────────────────
NDA_CORE_TYPES = [
    "definitions",
    "confidentiality",
    "exclusions",
    "permitted_disclosure",
    "term",
    "return_of_information",
    "remedies",
    "governing_law",
    "severability",
    "entire_agreement",
]

NDA_OPTIONAL_TYPES = [
    "compelled_disclosure",
    "indemnification",
    "injunctive_relief",
    "no_license",
    "attorneys_fees",
    "assignment",
    "recitals",
    "non_compete",
    "relationship_disclaimer",
]


class ClauseRetriever:
    """Metadata-aware clause retrieval with coverage guarantees.

    Retrieves clauses using:
    1. Semantic similarity (embedding-based)
    2. Metadata filters (document_type, startup_stage, etc.)
    3. Coverage-based supplementation (ensures core clause types are present)
    """

    def __init__(self):
        self._store = get_vector_store()

    async def retrieve_for_nda(
        self,
        nda_type: str = "mutual",
        startup_stage: str = "seed",
        jurisdiction: str = "",
        purpose: str = "",
        session_id: str = "unknown",
        top_k: int = 8,
    ) -> RetrievalResult:
        """Retrieve clauses relevant to an NDA generation request.

        Strategy:
        1. Build a semantic query from NDA parameters
        2. Apply metadata filters for document_type
        3. Retrieve top-K results
        4. Supplement with coverage-based retrieval for core clause types
        5. Log the retrieval

        Args:
            nda_type: "mutual" or "one_way"
            startup_stage: Normalized startup stage
            jurisdiction: Jurisdiction string
            purpose: Purpose/context for the NDA
            session_id: Session ID for logging
            top_k: Max initial results

        Returns:
            RetrievalResult with clauses and metadata
        """
        # Build semantic query
        query_parts = [
            f"Non-disclosure agreement for {nda_type} confidentiality",
            f"startup stage {startup_stage}",
        ]
        if purpose:
            query_parts.append(f"purpose: {purpose}")
        if jurisdiction:
            query_parts.append(f"jurisdiction: {jurisdiction}")

        query = ". ".join(query_parts)

        # Build metadata filters
        filters: Dict[str, Any] = {"document_type": "nda"}

        await log_event(
            event_type="clause_retrieval_started",
            session_id=session_id,
            data={
                "query": query,
                "filters": filters,
                "nda_type": nda_type,
                "startup_stage": startup_stage,
                "top_k": top_k,
            },
        )

        # Phase 1: Semantic retrieval
        raw_results = self._store.retrieve(
            query=query,
            filters=filters,
            top_k=top_k,
        )

        retrieved = self._parse_results(raw_results)

        # Phase 2: Coverage-based supplementation
        # Ensure we have at least one clause per core type
        covered_types = set(c.clause_type for c in retrieved)
        missing_core = [t for t in NDA_CORE_TYPES if t not in covered_types]

        if missing_core:
            logger.info(
                f"  Supplementing {len(missing_core)} missing core types: "
                f"{missing_core}"
            )
            for clause_type in missing_core:
                supplement = self._store.retrieve_by_type(
                    clause_type=clause_type,
                    top_k=1,
                )
                if supplement:
                    parsed = self._parse_results(supplement)
                    retrieved.extend(parsed)

        # Deduplicate by clause_id
        seen_ids = set()
        unique_retrieved = []
        for clause in retrieved:
            if clause.clause_id not in seen_ids:
                seen_ids.add(clause.clause_id)
                unique_retrieved.append(clause)

        # Sort by score (best first), then by clause type importance
        type_order = {t: i for i, t in enumerate(NDA_CORE_TYPES + NDA_OPTIONAL_TYPES)}
        unique_retrieved.sort(
            key=lambda c: (type_order.get(c.clause_type, 99), -c.score)
        )

        result = RetrievalResult(
            clauses=unique_retrieved,
            query=query,
            filters_used=filters,
            total_retrieved=len(unique_retrieved),
        )

        await log_event(
            event_type="clause_retrieval_completed",
            session_id=session_id,
            data={
                "clause_ids": result.clause_ids,
                "clause_types_covered": result.clause_types_covered,
                "total_retrieved": result.total_retrieved,
                "scores": [c.score for c in result.clauses],
            },
        )

        logger.info(
            f"  Retrieved {result.total_retrieved} clauses covering "
            f"{len(result.clause_types_covered)} types"
        )

        return result

    async def retrieve_by_clause_type(
        self,
        clause_type: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedClause]:
        """Retrieve clauses filtered by clause_type."""
        raw_results = self._store.retrieve_by_type(
            clause_type=clause_type,
            top_k=top_k,
            additional_filters=filters,
        )
        return self._parse_results(raw_results)

    def build_grounding_context(
        self,
        clauses: List[RetrievedClause],
    ) -> str:
        """Build structured grounding context from retrieved clauses.

        Organizes clauses by type for clean prompt injection.
        """
        if not clauses:
            return ""

        # Group by clause type
        by_type: Dict[str, List[RetrievedClause]] = {}
        for clause in clauses:
            by_type.setdefault(clause.clause_type, []).append(clause)

        sections: List[str] = []
        sections.append("=" * 60)
        sections.append("REFERENCE LEGAL CLAUSES")
        sections.append(
            "Use these as grounding — adapt to the specific parties and terms."
        )
        sections.append("=" * 60)

        for clause_type in NDA_CORE_TYPES + NDA_OPTIONAL_TYPES + ["miscellaneous"]:
            type_clauses = by_type.get(clause_type, [])
            if not type_clauses:
                continue

            type_label = clause_type.replace("_", " ").upper()
            sections.append(f"\n--- {type_label} ---")

            for clause in type_clauses:
                sections.append(f"[{clause.clause_id}] {clause.clause_name}")
                sections.append(f"{clause.text}")
                sections.append("")

        sections.append("=" * 60)
        sections.append(
            "Generate the NDA using the above reference clauses as your legal foundation."
        )
        sections.append(
            "Customize the language for the specific parties, jurisdiction, and purpose."
        )
        sections.append(
            "Do NOT invent new legal structures — build from the retrieved clauses."
        )
        sections.append("=" * 60)

        return "\n".join(sections)

    @staticmethod
    def _parse_results(
        raw_results: List[Dict[str, Any]],
    ) -> List[RetrievedClause]:
        """Convert raw vector store results to RetrievedClause objects."""
        parsed = []
        for r in raw_results:
            meta = r.get("metadata", {})
            parsed.append(
                RetrievedClause(
                    clause_id=r.get("clause_id", meta.get("clause_id", "")),
                    clause_type=meta.get("clause_type", ""),
                    clause_name=meta.get("clause_name", ""),
                    text=r.get("text", ""),
                    score=r.get("score", 0.0),
                    metadata=meta,
                )
            )
        return parsed


# ── Module-level singleton ───────────────────────────────────────────────────
_retriever_instance: Optional[ClauseRetriever] = None


def get_clause_retriever() -> ClauseRetriever:
    """Get or create the singleton ClauseRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = ClauseRetriever()
    return _retriever_instance
