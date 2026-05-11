"""Embedding Bootstrap Script.

Loads the normalized clause library and embeds all clauses into ChromaDB.
Can be re-run anytime to rebuild the index.

Usage:
    python -m scripts.embed_clauses
    python -m scripts.embed_clauses --reset  # clear and rebuild
"""
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent to path so we can import services
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

NORMALIZED_FILE = BASE_DIR / "data" / "normalized" / "normalized_clause_library.json"


def main():
    """Load normalized clauses and embed into ChromaDB."""
    reset = "--reset" in sys.argv

    logger.info("=" * 60)
    logger.info("CLAUSE EMBEDDING PIPELINE")
    logger.info("=" * 60)

    # Step 1: Load normalized clauses
    logger.info("\n[1/3] Loading normalized clause library...")
    if not NORMALIZED_FILE.exists():
        logger.error(
            f"Normalized file not found: {NORMALIZED_FILE}\n"
            "Run 'python -m scripts.normalize_clauses' first."
        )
        sys.exit(1)

    with open(NORMALIZED_FILE, "r", encoding="utf-8") as f:
        clauses = json.load(f)
    logger.info(f"  Loaded {len(clauses)} normalized clauses")

    # Step 2: Initialize vector store
    logger.info("\n[2/3] Initializing ChromaDB vector store...")
    from services.vector_store import ClauseVectorStore

    store = ClauseVectorStore()
    store.initialize()

    if reset:
        logger.info("  Resetting collection (--reset flag)...")
        store.reset()

    # Step 3: Embed clauses
    logger.info("\n[3/3] Embedding clauses...")
    added = store.add_clauses(clauses)

    # Print stats
    stats = store.get_collection_stats()
    logger.info("\n" + "=" * 60)
    logger.info("EMBEDDING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Total clauses embedded: {stats['total_clauses']}")
    logger.info(f"  Collection: {stats['collection_name']}")
    logger.info(f"  Persist directory: {stats['persist_dir']}")
    logger.info(f"  Embedding model: {stats['embedding_model']}")
    if "unique_clause_types" in stats:
        logger.info(f"  Clause types: {len(stats['unique_clause_types'])}")
        for ct in stats["unique_clause_types"]:
            logger.info(f"    - {ct}")
    if "unique_nda_subtypes" in stats:
        logger.info(f"  NDA subtypes: {len(stats['unique_nda_subtypes'])}")
    if "unique_jurisdictions" in stats:
        logger.info(f"  Jurisdictions: {stats['unique_jurisdictions']}")

    # Quick retrieval test
    logger.info("\n--- Quick Retrieval Test ---")
    test_results = store.retrieve(
        query="confidentiality obligation for protecting trade secrets in a startup",
        top_k=3,
    )
    for r in test_results:
        meta = r.get("metadata", {})
        logger.info(
            f"  [{r['clause_id']}] score={r['score']:.4f} "
            f"type={meta.get('clause_type', '?')} "
            f"subtype={meta.get('nda_subtype', '?')}"
        )
        logger.info(f"    {r['text'][:120]}...")

    logger.info("\nDone. ChromaDB is ready for retrieval.")


if __name__ == "__main__":
    main()
