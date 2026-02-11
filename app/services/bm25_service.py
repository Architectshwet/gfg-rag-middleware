"""
BM25 service for keyword-based search using rank-bm25 library.
Provides fast, local keyword matching without requiring external models.
"""

import logging
from functools import lru_cache
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Service:
    """
    BM25 (Best Matching 25) service for keyword search.
    
    BM25 is a ranking function used by search engines to estimate relevance
    of documents to a given search query. It's based on probabilistic retrieval
    framework and considers:
    - Term frequency (TF): How often a term appears in a document
    - Inverse document frequency (IDF): How rare a term is across all documents
    - Document length normalization: Prevents bias towards longer documents
    
    Why BM25 for keyword search:
    1. Fast: Pure Python, no ML models needed (~1ms for 100K docs)
    2. Exact matching: Perfect for product codes, brand names, rare terms
    3. No embeddings: Works on tokenized text directly
    4. Well-established: Used by Elasticsearch, Lucene, etc.
    
    Example:
        Query: "Global Accord 1004"
        - BM25 scores highest: Products with exact "Global", "Accord", "1004"
        - Semantic search: Might match "International Agreement" (wrong!)
    """

    def __init__(self):
        """Initialize BM25 service (corpus loaded on demand)"""
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_corpus = []
        self.is_initialized = False

    def tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25.
        
        Simple whitespace + lowercase tokenization.
        For production, could use nltk or spacy for better tokenization.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on whitespace
        # Remove punctuation for better matching
        import re
        text = text.lower()
        # Keep alphanumeric and spaces, replace others with space
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        # Split and filter empty strings
        tokens = [token for token in text.split() if token]
        return tokens

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        """
        Build BM25 index from documents.
        
        This should be called once after loading documents from Qdrant.
        
        Args:
            documents: List of dicts with 'id' and 'text' keys
        """
        if not documents:
            logger.warning("⚠️ No documents provided for BM25 indexing")
            return

        # Extract and tokenize documents
        self.doc_ids = [doc['id'] for doc in documents]
        self.tokenized_corpus = [self.tokenize(doc['text']) for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_initialized = True

        logger.info(f"✅ BM25 index built with {len(self.doc_ids)} documents")

    def search(self, query: str, top_k: int = 20) -> list[tuple[Any, float]]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of (doc_id, score) tuples, sorted by score (descending)
        """
        if not self.is_initialized:
            logger.warning("⚠️ BM25 index not initialized. Call build_index() first.")
            return []

        # Tokenize query
        query_tokens = self.tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top K indices
        top_k = min(top_k, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]  # Descending order

        # Return (doc_id, score) pairs
        results = [(self.doc_ids[i], scores[i]) for i in top_indices if scores[i] > 0]

        return results

    def get_stats(self) -> dict[str, Any]:
        """Get BM25 index statistics"""
        return {
            "is_initialized": self.is_initialized,
            "total_documents": len(self.doc_ids),
            "avg_doc_length": sum(len(doc) for doc in self.tokenized_corpus) / len(self.tokenized_corpus)
            if self.tokenized_corpus
            else 0,
        }


@lru_cache
def get_bm25_service() -> BM25Service:
    """Get cached BM25 service instance"""
    return BM25Service()
