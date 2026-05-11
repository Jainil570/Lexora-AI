"""ChromaDB-backed vector store for clause library.

Provides persistent local storage, embedding, and metadata-aware
semantic retrieval of legal clauses using BAAI/bge-small-en-v1.5.

Usage:
    from services.vector_store import ClauseVectorStore
    store = ClauseVectorStore()
    store.initialize()
    results = store.retrieve("confidentiality for seed stage startup", top_k=5)
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DEFAULT_PERSIST_DIR = str(BASE_DIR / "data" / "chroma_db")
DEFAULT_COLLECTION = "lexora_clauses"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
NORMALIZED_FILE = BASE_DIR / "data" / "normalized" / "normalized_clause_library.json"


class ClauseVectorStore:
    """ChromaDB-backed clause vector store with local embeddings.

    Handles:
    - Persistent storage and retrieval
    - Sentence-transformer embeddings (local, no API key)
    - Metadata filtering (clause_type, document_type, startup_stage, etc.)
    - Collection lifecycle (create, populate, query, reset)
    """

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = model_name
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection = None
        self._embedding_fn = None

    def initialize(self) -> None:
        """Initialize ChromaDB client, embedding function, and collection."""
        logger.info(f"Initializing ChromaDB at {self.persist_dir}")
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=self.model_name,
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        count = self._collection.count()
        logger.info(
            f"ChromaDB collection '{self.collection_name}' ready. "
            f"Existing documents: {count}"
        )

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if self._collection is None:
            raise RuntimeError(
                "ClauseVectorStore not initialized. Call initialize() first."
            )

    def add_clauses(self, clauses: List[Dict[str, Any]]) -> int:
        """Embed and upsert clauses into the collection.

        Args:
            clauses: List of normalized clause dicts.

        Returns:
            Number of clauses added.
        """
        self._ensure_initialized()

        if not clauses:
            logger.warning("No clauses to add.")
            return 0

        # Prepare batches (ChromaDB recommends batches ≤ 5000)
        batch_size = 500
        total_added = 0

        for i in range(0, len(clauses), batch_size):
            batch = clauses[i : i + batch_size]

            ids = []
            documents = []
            metadatas = []

            for clause in batch:
                clause_id = clause["clause_id"]
                text = clause["text"]

                # Build metadata dict — ChromaDB metadata values must be
                # str, int, float, or bool. Lists need to be comma-joined.
                metadata = {
                    "clause_id": clause_id,
                    "original_id": clause.get("original_id", ""),
                    "clause_type": clause.get("clause_type", ""),
                    "document_type": clause.get("document_type", ""),
                    "nda_subtype": clause.get("nda_subtype", ""),
                    "startup_stage": ",".join(clause.get("startup_stage", [])),
                    "jurisdiction": clause.get("jurisdiction", ""),
                    "risk_level": clause.get("risk_level", ""),
                    "importance": clause.get("importance", ""),
                    "tags": ",".join(clause.get("tags", [])),
                    "source_document": clause.get("source_document", ""),
                    "clause_name": clause.get("clause_name", ""),
                    "jurisdiction_adaptable": clause.get(
                        "jurisdiction_adaptable", True
                    ),
                }

                ids.append(clause_id)
                documents.append(text)
                metadatas.append(metadata)

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            total_added += len(batch)
            logger.info(
                f"  Embedded batch {i // batch_size + 1}: "
                f"{len(batch)} clauses (total: {total_added})"
            )

        logger.info(f"Total clauses embedded: {total_added}")
        return total_added

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 8,
    ) -> List[Dict[str, Any]]:
        """Retrieve clauses by semantic similarity + metadata filters.

        Args:
            query: Natural language query text.
            filters: ChromaDB where-clause dict for metadata filtering.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts with keys: clause_id, text, metadata, score.
        """
        self._ensure_initialized()

        query_params = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count()),
        }
        if filters:
            query_params["where"] = filters

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            return []

        # Parse results into a friendlier format
        parsed: List[Dict[str, Any]] = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []

            for j, cid in enumerate(ids):
                # ChromaDB returns cosine distance; convert to similarity
                distance = distances[j] if j < len(distances) else 1.0
                similarity = 1.0 - distance  # cosine similarity

                parsed.append({
                    "clause_id": cid,
                    "text": documents[j] if j < len(documents) else "",
                    "metadata": metadatas[j] if j < len(metadatas) else {},
                    "score": round(similarity, 4),
                })

        return parsed

    def retrieve_by_type(
        self,
        clause_type: str,
        top_k: int = 5,
        additional_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve clauses filtered by clause_type.

        Uses a generic query with type-specific metadata filter.
        """
        filters = {"clause_type": clause_type}
        if additional_filters:
            filters = {"$and": [filters, additional_filters]}

        # Use a generic descriptive query for the type
        type_queries = {
            "confidentiality": "confidentiality and non-disclosure obligations",
            "definitions": "definition of confidential information",
            "governing_law": "governing law and jurisdiction",
            "remedies": "breach remedies and enforcement",
            "return_of_information": "return or destruction of confidential information",
            "exclusions": "exclusions and exceptions to confidentiality",
            "term": "term duration and confidentiality period",
            "assignment": "assignment and transfer restrictions",
            "severability": "severability of agreement provisions",
            "attorneys_fees": "attorney fees and litigation costs",
            "entire_agreement": "entire agreement and integration clause",
            "compelled_disclosure": "compelled disclosure by law or court order",
            "no_license": "no license or rights granted",
            "indemnification": "indemnification and hold harmless",
            "injunctive_relief": "injunctive and equitable relief",
            "permitted_disclosure": "permitted disclosure and purpose limitation",
            "recitals": "recitals and purpose of agreement",
            "non_compete": "non-compete and non-solicitation",
            "relationship_disclaimer": "relationship disclaimer no partnership",
            "miscellaneous": "miscellaneous provisions",
        }
        query = type_queries.get(clause_type, clause_type.replace("_", " "))

        return self.retrieve(query=query, filters=filters, top_k=top_k)

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        self._ensure_initialized()

        count = self._collection.count()

        # Sample metadata to get unique values
        stats: Dict[str, Any] = {
            "total_clauses": count,
            "collection_name": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedding_model": self.model_name,
        }

        if count > 0:
            # Get all metadata
            try:
                all_data = self._collection.get(
                    include=["metadatas"],
                    limit=count,
                )
                if all_data and all_data["metadatas"]:
                    clause_types = set()
                    nda_subtypes = set()
                    jurisdictions = set()
                    for meta in all_data["metadatas"]:
                        clause_types.add(meta.get("clause_type", ""))
                        nda_subtypes.add(meta.get("nda_subtype", ""))
                        jurisdictions.add(meta.get("jurisdiction", ""))
                    stats["unique_clause_types"] = sorted(clause_types)
                    stats["unique_nda_subtypes"] = sorted(nda_subtypes)
                    stats["unique_jurisdictions"] = sorted(jurisdictions)
            except Exception as e:
                logger.warning(f"Failed to gather detailed stats: {e}")

        return stats

    def reset(self) -> None:
        """Delete and recreate the collection."""
        self._ensure_initialized()
        logger.warning(f"Resetting collection '{self.collection_name}'")
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection reset complete.")

    def is_populated(self) -> bool:
        """Check if the collection has any documents."""
        self._ensure_initialized()
        return self._collection.count() > 0


# ── Module-level singleton ───────────────────────────────────────────────────
_store_instance: Optional[ClauseVectorStore] = None


def get_vector_store() -> ClauseVectorStore:
    """Get or create the singleton ClauseVectorStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ClauseVectorStore()
        _store_instance.initialize()

        # Auto-populate if empty
        if not _store_instance.is_populated():
            logger.info("Collection empty. Auto-populating from normalized library...")
            if NORMALIZED_FILE.exists():
                with open(NORMALIZED_FILE, "r", encoding="utf-8") as f:
                    clauses = json.load(f)
                _store_instance.add_clauses(clauses)
                logger.info(f"Auto-populated {len(clauses)} clauses.")
            else:
                logger.warning(
                    f"Normalized clause library not found at {NORMALIZED_FILE}. "
                    "Run 'python -m scripts.normalize_clauses' first."
                )
    return _store_instance
