"""Clause Library Normalization Pipeline.

Loads all raw clause JSON files, normalizes metadata, standardizes
clause types, startup stages, jurisdictions, deduplicates boilerplate,
and outputs a unified normalized clause library.

Usage:
    python scripts/normalize_clauses.py
"""
import json
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw_clauses"
OUTPUT_DIR = BASE_DIR / "data" / "normalized"
OUTPUT_FILE = OUTPUT_DIR / "normalized_clause_library.json"

# ── Canonical Clause Type Mapping ────────────────────────────────────────────
# Maps raw clause_type values → canonical normalized types
CLAUSE_TYPE_MAP: Dict[str, str] = {
    # Confidentiality & Non-Disclosure
    "confidentiality_obligation": "confidentiality",
    "non_disclosure": "confidentiality",
    "confidential_information_definition": "definitions",
    "trade_secret": "confidentiality",
    "trade_secret_acknowledgment": "confidentiality",

    # Definitions
    "definitions": "definitions",
    "definition": "definitions",
    "disclosure_scope": "definitions",

    # Governing Law
    "governing_law": "governing_law",
    "jurisdiction_venue": "governing_law",

    # Remedies
    "remedies": "remedies",
    "breach_remedies": "remedies",
    "breach_and_remedies": "remedies",
    "breach_cause_for_action": "remedies",

    # Return of Information
    "return_of_information": "return_of_information",
    "return_or_destruction": "return_of_information",
    "return_of_materials": "return_of_information",

    # Exclusions
    "exclusions": "exclusions",
    "exceptions": "exclusions",

    # Term / Duration
    "term": "term",
    "term_duration": "term",
    "term_and_duration": "term",
    "term_termination": "term",
    "confidentiality_period": "term",

    # Assignment
    "assignment": "assignment",
    "non_assignment": "assignment",

    # Severability
    "severability": "severability",

    # Attorney Fees
    "attorneys_fees": "attorneys_fees",
    "prevailing_party_fees": "attorneys_fees",
    "enforcement": "attorneys_fees",

    # Entire Agreement
    "entire_agreement": "entire_agreement",
    "amendments": "entire_agreement",

    # Compelled Disclosure
    "compelled_disclosure": "compelled_disclosure",

    # No License / No Rights
    "no_license": "no_license",
    "no_rights_granted": "no_license",
    "disclaimer_of_rights": "no_license",

    # Indemnification
    "indemnification": "indemnification",
    "indemnification_remedies": "indemnification",

    # Injunctive Relief
    "injunctive_relief": "injunctive_relief",

    # Permitted Disclosure / Use
    "permitted_disclosure": "permitted_disclosure",
    "permitted_use": "permitted_disclosure",
    "purpose_limitation": "permitted_disclosure",

    # Recitals / Parties
    "recitals": "recitals",
    "parties": "recitals",
    "representation": "recitals",
    "relationship_definition": "recitals",

    # Non-Compete
    "no_obligation_no_compete": "non_compete",

    # Relationship Disclaimer
    "relationship_disclaimer": "relationship_disclaimer",
    "relationship": "relationship_disclaimer",

    # Miscellaneous — catch-all for specialized types
    "counterparts": "miscellaneous",
    "waiver": "miscellaneous",
    "no_waiver": "miscellaneous",
    "binding_arrangement": "miscellaneous",
    "binding_and_survival": "miscellaneous",
    "survival": "miscellaneous",
    "successors_assigns": "miscellaneous",
    "property_rights": "miscellaneous",
    "derivative_materials": "miscellaneous",
    "warranty": "miscellaneous",
    "warranty_disclaimer": "miscellaneous",
    "disclaimer_warranty": "miscellaneous",
    "disclaimer_of_warranty": "miscellaneous",
    "meta_confidentiality": "miscellaneous",
    "non_disparagement": "miscellaneous",
    "non_circumvention": "miscellaneous",
    "breach_notification": "miscellaneous",
    "no_obligation": "miscellaneous",
    "no_contact": "miscellaneous",
    "access_control": "miscellaneous",
    "compliance_training": "miscellaneous",
    "inventions_prior": "miscellaneous",
    "inventions_assignment": "miscellaneous",
    "inventions_personal": "miscellaneous",
    "security_obligations": "miscellaneous",
    "company_obligations": "miscellaneous",
    "tester_obligations": "miscellaneous",
    "ip_rights": "miscellaneous",
    "limitation_of_liability": "miscellaneous",
    "post_non_acquisition_obligation": "miscellaneous",
    "representative_obligations": "miscellaneous",
    "duty_of_care": "miscellaneous",
    "disclaimer_of_employment": "miscellaneous",

    # Additional types discovered in raw data
    "notices": "miscellaneous",
    "disclosure_procedure": "permitted_disclosure",
    "ownership_and_remedies": "remedies",
    "independent_development": "exclusions",
    "breach_notice": "miscellaneous",
    "demand_to_cease": "remedies",
    "compliance_demand": "remedies",
    "litigation_threat": "remedies",
    "binding_effect": "miscellaneous",
    "agreement_type": "recitals",
    "confidentiality": "confidentiality",
    "integration": "entire_agreement",
    "academic_carveout": "exclusions",
    "location_confidentiality": "confidentiality",
    "notification": "miscellaneous",
    "acknowledgment": "recitals",
}

# ── Canonical Startup Stages ────────────────────────────────────────────────
STAGE_MAP: Dict[str, str] = {
    "idea": "idea",
    "seed": "seed",
    "early": "early",
    "series_a": "early",
    "growth": "growth",
    "late": "late",
    "exit": "exit",
    "scale": "growth",
    "all": "__all__",  # sentinel: expand to all stages
}

ALL_STAGES = ["idea", "seed", "early", "growth", "late", "exit"]

# ── Jurisdiction Normalization ───────────────────────────────────────────────
JURISDICTION_MAP: Dict[str, str] = {
    "usa_state_generic": "generic_common_law",
    "usa_federal": "generic_common_law",
    "usa_state_specific": "generic_common_law",
    "general": "adaptable",
}

# ── Risk Level Normalization ─────────────────────────────────────────────────
VALID_RISK_LEVELS = {"low", "medium", "high", "critical"}
RISK_MAP: Dict[str, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "critical": "high",  # fold critical into high
}

# ── Importance Normalization ─────────────────────────────────────────────────
IMPORTANCE_MAP: Dict[str, str] = {
    "low": "standard",
    "standard": "standard",
    "medium": "important",
    "important": "important",
    "high": "important",
    "critical": "critical",
}

# ── Boilerplate types eligible for deduplication ─────────────────────────────
BOILERPLATE_TYPES = {
    "severability",
    "entire_agreement",
    "governing_law",
    "attorneys_fees",
    "miscellaneous",  # counterparts, waiver, etc.
}

# ── Required schema fields ───────────────────────────────────────────────────
REQUIRED_FIELDS = {"id", "document_type", "clause_type", "text"}
OPTIONAL_FIELDS = {
    "source_document", "nda_subtype", "clause_name",
    "risk_level", "startup_stage", "jurisdiction",
    "tags", "importance",
}


# ── Loading ──────────────────────────────────────────────────────────────────

def load_raw_clauses() -> List[Dict[str, Any]]:
    """Load and merge all JSON clause files from the raw directory."""
    all_clauses: List[Dict[str, Any]] = []
    json_files = sorted(RAW_DIR.glob("*.json"))

    if not json_files:
        logger.error(f"No JSON files found in {RAW_DIR}")
        sys.exit(1)

    for fp in json_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                logger.info(f"  Loaded {len(data):>3} clauses from {fp.name}")
                all_clauses.extend(data)
            else:
                logger.warning(f"  Skipping {fp.name}: not a JSON array")
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"  Failed to parse {fp.name}: {e}")

    logger.info(f"Total raw clauses loaded: {len(all_clauses)}")
    return all_clauses


# ── Validation ───────────────────────────────────────────────────────────────

def validate_clause(clause: Dict[str, Any], index: int) -> bool:
    """Validate a single clause has required fields."""
    missing = REQUIRED_FIELDS - set(clause.keys())
    if missing:
        logger.warning(
            f"  Clause #{index} (id={clause.get('id', 'UNKNOWN')}): "
            f"missing required fields: {missing}"
        )
        return False

    if not clause.get("text", "").strip():
        logger.warning(f"  Clause #{index} (id={clause.get('id')}): empty text")
        return False

    return True


# ── Normalization ────────────────────────────────────────────────────────────

def normalize_clause_type(raw_type: str) -> str:
    """Map raw clause type to canonical type."""
    normalized = CLAUSE_TYPE_MAP.get(raw_type)
    if normalized is None:
        logger.warning(f"  Unknown clause_type '{raw_type}' → 'miscellaneous'")
        return "miscellaneous"
    return normalized


def normalize_startup_stages(raw_stages: Any) -> List[str]:
    """Normalize startup stage list."""
    if not raw_stages:
        return list(ALL_STAGES)  # default: all stages

    if isinstance(raw_stages, str):
        raw_stages = [raw_stages]

    normalized: Set[str] = set()
    for stage in raw_stages:
        stage = stage.strip().lower()
        mapped = STAGE_MAP.get(stage)
        if mapped == "__all__":
            return list(ALL_STAGES)
        elif mapped:
            normalized.add(mapped)
        else:
            logger.warning(f"  Unknown stage '{stage}' → skipped")

    return sorted(normalized, key=lambda s: ALL_STAGES.index(s)) if normalized else list(ALL_STAGES)


def normalize_jurisdiction(raw_jurisdiction: str) -> Tuple[str, bool]:
    """Normalize jurisdiction and detect adaptability.

    Returns (normalized_jurisdiction, is_adaptable).
    """
    raw = raw_jurisdiction.strip().lower() if raw_jurisdiction else "general"
    normalized = JURISDICTION_MAP.get(raw, "adaptable")
    return normalized, True  # all current clauses are adaptable


def normalize_risk_level(raw: str) -> str:
    """Normalize risk level."""
    return RISK_MAP.get(raw.strip().lower() if raw else "medium", "medium")


def normalize_importance(raw: str) -> str:
    """Normalize importance level."""
    return IMPORTANCE_MAP.get(raw.strip().lower() if raw else "standard", "standard")


def normalize_tags(raw_tags: Any) -> List[str]:
    """Normalize and deduplicate tags."""
    if not raw_tags:
        return []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]
    return sorted(set(tag.strip().lower() for tag in raw_tags if tag.strip()))


def detect_adaptable_text(text: str) -> bool:
    """Detect if clause text contains placeholder markers (jurisdiction-adaptable)."""
    placeholders = ["___", "[STATE]", "[DURATION]", "[insert", "[PROJECT NAME]"]
    return any(p in text for p in placeholders)


def normalize_single_clause(
    clause: Dict[str, Any], seq_id: int
) -> Dict[str, Any]:
    """Normalize a single clause into the canonical schema."""
    raw_type = clause.get("clause_type", "miscellaneous")
    norm_type = normalize_clause_type(raw_type)
    norm_stages = normalize_startup_stages(clause.get("startup_stage"))
    norm_jurisdiction, is_adaptable = normalize_jurisdiction(
        clause.get("jurisdiction", "general")
    )
    text = clause.get("text", "").strip()

    return {
        "clause_id": f"NORM_{seq_id:04d}",
        "original_id": clause.get("id", f"unknown_{seq_id}"),
        "source_document": clause.get("source_document", "unknown"),
        "document_type": clause.get("document_type", "nda"),
        "nda_subtype": clause.get("nda_subtype", "generic_nda"),
        "clause_name": clause.get("clause_name", "Unnamed Clause"),
        "clause_type": norm_type,
        "original_clause_type": raw_type,
        "risk_level": normalize_risk_level(clause.get("risk_level", "medium")),
        "importance": normalize_importance(clause.get("importance", "standard")),
        "startup_stage": norm_stages,
        "jurisdiction": norm_jurisdiction,
        "jurisdiction_adaptable": is_adaptable or detect_adaptable_text(text),
        "india_adaptation_notes": "",
        "tags": normalize_tags(clause.get("tags")),
        "text": text,
        "text_hash": hashlib.md5(text.encode()).hexdigest(),
    }


# ── Deduplication ────────────────────────────────────────────────────────────

def _clause_quality_score(clause: Dict[str, Any]) -> float:
    """Score a clause for dedup ranking. Higher = better quality."""
    text = clause["text"]
    length = len(text)

    # Prefer medium-length clauses (not too short, not too long)
    if length < 50:
        length_score = 0.2
    elif length < 150:
        length_score = 0.6
    elif length < 500:
        length_score = 1.0
    elif length < 1000:
        length_score = 0.8
    else:
        length_score = 0.6

    # Prefer critical/important clauses
    importance_scores = {"critical": 1.0, "important": 0.7, "standard": 0.4}
    imp_score = importance_scores.get(clause["importance"], 0.4)

    # Prefer clauses without excessive placeholders
    placeholder_count = text.count("___") + text.count("[")
    placeholder_penalty = max(0, 1.0 - placeholder_count * 0.15)

    # Prefer clauses with more specific (non-generic) text
    specificity_bonus = 0.1 if clause["nda_subtype"] != "generic_nda" else 0

    return length_score * 0.4 + imp_score * 0.3 + placeholder_penalty * 0.2 + specificity_bonus * 0.1


def deduplicate_clauses(
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Remove duplicate boilerplate clauses, keeping the best variant per subtype.

    Non-boilerplate clauses are always kept.
    For boilerplate types, we keep the highest-quality version per nda_subtype.
    Additionally, exact text duplicates (same hash) are always removed.
    """
    # Phase 1: Remove exact text duplicates
    seen_hashes: Set[str] = set()
    unique_clauses: List[Dict[str, Any]] = []
    exact_dupes = 0

    for clause in clauses:
        h = clause["text_hash"]
        if h in seen_hashes:
            exact_dupes += 1
            continue
        seen_hashes.add(h)
        unique_clauses.append(clause)

    logger.info(f"  Exact text duplicates removed: {exact_dupes}")

    # Phase 2: For boilerplate types, keep best per (clause_type, nda_subtype)
    boilerplate: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    non_boilerplate: List[Dict[str, Any]] = []

    for clause in unique_clauses:
        if clause["clause_type"] in BOILERPLATE_TYPES:
            key = (clause["clause_type"], clause["nda_subtype"])
            boilerplate[key].append(clause)
        else:
            non_boilerplate.append(clause)

    # For each boilerplate group, keep the best one
    kept_boilerplate: List[Dict[str, Any]] = []
    boilerplate_removed = 0
    for key, group in boilerplate.items():
        if len(group) == 1:
            kept_boilerplate.append(group[0])
        else:
            scored = sorted(group, key=_clause_quality_score, reverse=True)
            kept_boilerplate.append(scored[0])
            boilerplate_removed += len(scored) - 1

    logger.info(f"  Boilerplate variants deduplicated: {boilerplate_removed}")

    result = non_boilerplate + kept_boilerplate
    return result


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_normalization() -> List[Dict[str, Any]]:
    """Run the full normalization pipeline."""
    logger.info("=" * 60)
    logger.info("CLAUSE LIBRARY NORMALIZATION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load
    logger.info("\n[1/5] Loading raw clause files...")
    raw_clauses = load_raw_clauses()

    # Step 2: Validate
    logger.info("\n[2/5] Validating schema consistency...")
    valid_clauses = []
    invalid_count = 0
    for i, clause in enumerate(raw_clauses):
        if validate_clause(clause, i):
            valid_clauses.append(clause)
        else:
            invalid_count += 1
    logger.info(f"  Valid: {len(valid_clauses)}, Invalid: {invalid_count}")

    # Step 3: Normalize
    logger.info("\n[3/5] Normalizing metadata...")
    normalized = []
    for i, clause in enumerate(valid_clauses, start=1):
        normalized.append(normalize_single_clause(clause, i))

    # Log clause type distribution
    type_dist = Counter(c["clause_type"] for c in normalized)
    logger.info("  Clause type distribution:")
    for ct, count in type_dist.most_common():
        logger.info(f"    {ct:30s} → {count}")

    # Log jurisdiction distribution
    jur_dist = Counter(c["jurisdiction"] for c in normalized)
    logger.info("  Jurisdiction distribution:")
    for j, count in jur_dist.most_common():
        logger.info(f"    {j:30s} → {count}")

    # Log unmapped clause types
    unmapped = set()
    for clause in raw_clauses:
        raw_type = clause.get("clause_type", "")
        if raw_type and raw_type not in CLAUSE_TYPE_MAP:
            unmapped.add(raw_type)
    if unmapped:
        logger.warning(f"  Unmapped clause types: {unmapped}")

    # Step 4: Deduplicate
    logger.info("\n[4/5] Deduplicating boilerplate clauses...")
    deduped = deduplicate_clauses(normalized)
    logger.info(f"  Before dedup: {len(normalized)}, After: {len(deduped)}")

    # Re-assign sequential IDs after dedup
    for i, clause in enumerate(deduped, start=1):
        clause["clause_id"] = f"NORM_{i:04d}"

    # Step 5: Save
    logger.info("\n[5/5] Saving normalized clause library...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    logger.info(f"  Output: {OUTPUT_FILE}")
    logger.info(f"  Total normalized clauses: {len(deduped)}")

    # Final stats
    logger.info("\n" + "=" * 60)
    logger.info("NORMALIZATION COMPLETE")
    logger.info("=" * 60)
    final_types = Counter(c["clause_type"] for c in deduped)
    logger.info(f"  Total clauses: {len(deduped)}")
    logger.info(f"  Unique clause types: {len(final_types)}")
    logger.info(f"  NDA subtypes: {len(set(c['nda_subtype'] for c in deduped))}")
    logger.info(f"  Source documents: {len(set(c['source_document'] for c in deduped))}")

    return deduped


if __name__ == "__main__":
    run_normalization()
