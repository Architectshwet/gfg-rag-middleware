"""Search service for hybrid product search (semantic + keyword)"""

import logging
from functools import lru_cache
from typing import Any

from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from app.core.config import settings
from app.core.mongo import get_mongo_db
from app.services.embedding_service import get_embedding_service
from app.services.qdrant_service import get_qdrant_service
from app.services.query_analyzer import get_query_analyzer
from app.services.bm25_service import get_bm25_service

logger = logging.getLogger(__name__)


class SearchService:
    """
    Handles hybrid product search logic.
    
    Two-component hybrid search:
    1. Semantic Search: OpenAI embeddings + Qdrant with metadata filtering
       (understands meaning + applies exact filters during search)
    2. Keyword Search: BM25 algorithm (exact matching for product codes, brands)
    
    Results are fused using RRF (Reciprocal Rank Fusion) for optimal relevance.
    
    Note: Metadata filtering happens INSIDE semantic search (Qdrant applies filters),
    not as a separate post-processing step.
    """

    def __init__(
        self,
        query_analyzer,
        embedding_service,
        qdrant_service,
        bm25_service,
    ):
        self.query_analyzer = query_analyzer
        self.embedding_service = embedding_service
        self.qdrant_service = qdrant_service
        self.bm25_service = bm25_service
        self.db = get_mongo_db()
        self.collection = self.db[settings.PRODUCTS_COLLECTION]
        
        # Initialize BM25 index on first use
        self._bm25_initialized = False

    def _ensure_bm25_initialized(self):
        """Ensure BM25 index is built (lazy initialization)"""
        if not self._bm25_initialized:
            logger.info("🔄 Initializing BM25 index (first search)...")
            documents = self.qdrant_service.get_all_documents()
            self.bm25_service.build_index(documents)
            self._bm25_initialized = True

    async def search_products(self, user_query: str, use_hybrid: bool = True) -> dict[str, Any]:
        """
        Execute hybrid product search with semantic + keyword.

        Search Flow:
        1. Query Analysis: Extract search terms + filters (LLM-based)
        2. Parallel Retrieval:
           - Semantic: OpenAI embedding → Qdrant vector search WITH metadata filters (top 20)
           - Keyword: BM25 on full text (top 20)
        3. Fusion: Combine results using RRF (Reciprocal Rank Fusion)
        4. Return top K results

        Note: Metadata filtering (price, category, dimensions) happens DURING semantic search
        in Qdrant, not as a separate step. This is more efficient.

        Args:
            user_query: Natural language search query
            use_hybrid: If True, uses semantic+keyword; if False, semantic only

        Returns:
            dict with analyzed_query, filters_detected, search_method, and results
        """
        # Step 1: Analyze query to extract search terms and filters
        analysis = self.query_analyzer.analyze_query(user_query)
        search_query = analysis["search_query"]
        filters = analysis["filters"]

        # Step 2: Convert filters to Qdrant format
        qdrant_filter = None
        if filters:
            qdrant_filter = self._convert_filters_to_qdrant(filters)

        logger.info(f"🔍 Search Query: {search_query}")
        logger.info(f"🔍 Filters: {filters}")
        logger.info(f"🔍 Hybrid Mode: {use_hybrid}")

        if use_hybrid:
            # HYBRID SEARCH: Semantic (with metadata filtering) + Keyword
            
            # Step 3a: Semantic search WITH metadata filtering (Qdrant applies filters during search)
            query_embedding = self.embedding_service.create_embedding(search_query)
            semantic_results = self.qdrant_service.query(
                query_embedding=query_embedding,
                n_results=settings.HYBRID_RETRIEVAL_SIZE,  # Get 20-50 candidates with filters applied
                filters=qdrant_filter,  # Metadata filtering happens HERE in Qdrant
            )

            # Step 3b: Keyword search (BM25) - no metadata filtering (operates on all docs)
            self._ensure_bm25_initialized()
            bm25_results = self.bm25_service.search(search_query, top_k=settings.HYBRID_RETRIEVAL_SIZE)

            logger.info(f"✅ Semantic (with filters): {len(semantic_results['ids'])} results")
            logger.info(f"✅ BM25 (keyword): {len(bm25_results)} results")

            # Step 4: Fuse results using RRF (combines filtered semantic + keyword results)
            fused_results = self._fuse_results_rrf(
                semantic_results=semantic_results,
                bm25_results=bm25_results,
                top_k=settings.SEARCH_TOP_K,
            )

            search_method = "hybrid (semantic+metadata+keyword)"

        else:
            # SEMANTIC ONLY with metadata filtering
            query_embedding = self.embedding_service.create_embedding(search_query)
            semantic_results = self.qdrant_service.query(
                query_embedding=query_embedding,
                n_results=settings.SEARCH_TOP_K,
                filters=qdrant_filter,  # Metadata filtering happens in Qdrant
            )
            fused_results = semantic_results
            search_method = "semantic+metadata only"

        # Step 5: Format and return results
        formatted_results = await self._format_results(fused_results)

        return {
            "analyzed_query": search_query,
            "filters_detected": filters,
            "search_method": search_method,
            "results": formatted_results,
            "total_results": len(formatted_results),
        }

    def _fuse_results_rrf(
        self,
        semantic_results: dict,
        bm25_results: list[tuple[Any, float]],
        top_k: int = 5,
        k: int = 60,
    ) -> dict:
        """
        Fuse semantic and keyword search results using Reciprocal Rank Fusion (RRF).

        RRF Formula: score(d) = Σ 1 / (k + rank(d))
        - k: constant (typically 60) to reduce impact of high rankings
        - rank(d): position of document d in each ranking (1-indexed)

        Why RRF:
        1. No score normalization needed (different scales for semantic vs BM25)
        2. Emphasizes agreement across methods (documents ranked high in both)
        3. Simple and effective (used by Elasticsearch, Vespa)

        Args:
            semantic_results: Results from vector search
            bm25_results: Results from BM25 search (list of (id, score))
            top_k: Number of final results
            k: RRF constant (default 60)

        Returns:
            Fused results in same format as semantic_results
        """
        # Build RRF scores
        rrf_scores = {}

        # Add semantic results
        for rank, doc_id in enumerate(semantic_results["ids"], start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Add BM25 results
        for rank, (doc_id, _score) in enumerate(bm25_results, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1 / (k + rank)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Build result dictionary in expected format
        fused_ids = []
        fused_documents = []
        fused_metadatas = []
        fused_distances = []

        # Create lookup for semantic results
        semantic_lookup = {
            doc_id: {
                "document": semantic_results["documents"][i],
                "metadata": semantic_results["metadatas"][i],
                "distance": semantic_results["distances"][i],
            }
            for i, doc_id in enumerate(semantic_results["ids"])
        }

        for doc_id, rrf_score in sorted_ids:
            if doc_id in semantic_lookup:
                fused_ids.append(doc_id)
                fused_documents.append(semantic_lookup[doc_id]["document"])
                fused_metadatas.append(semantic_lookup[doc_id]["metadata"])
                fused_distances.append(rrf_score)  # Use RRF score as "distance"

        logger.info(f"✅ RRF Fusion: {len(fused_ids)} final results")

        return {
            "ids": fused_ids,
            "documents": fused_documents,
            "metadatas": fused_metadatas,
            "distances": fused_distances,
        }

    def _convert_filters_to_qdrant(self, mongo_filters: dict) -> Filter | None:
        """
        Convert query analyzer filters to Qdrant Filter object

        Qdrant supports:
        - Arrays in metadata (categories stored as list)
        - MatchAny for array matching (like $in)
        - Range for numeric comparisons
        - MatchValue for exact matches

        Args:
            mongo_filters: Filters from query analyzer

        Returns:
            Qdrant Filter object
        """
        if not mongo_filters:
            return None

        must_conditions = []

        # Handle product_code filter
        if "product_code" in mongo_filters:
            must_conditions.append(
                FieldCondition(key="product_code", match=MatchValue(value=mongo_filters["product_code"]))
            )

        # Handle base_price filters
        if "base_price" in mongo_filters:
            price_filter = mongo_filters["base_price"]
            if isinstance(price_filter, dict):
                # Qdrant supports range queries
                range_params = {}
                if "$gte" in price_filter:
                    range_params["gte"] = price_filter["$gte"]
                if "$lte" in price_filter:
                    range_params["lte"] = price_filter["$lte"]
                if "$gt" in price_filter:
                    range_params["gt"] = price_filter["$gt"]
                if "$lt" in price_filter:
                    range_params["lt"] = price_filter["$lt"]

                if range_params:
                    must_conditions.append(FieldCondition(key="base_price", range=Range(**range_params)))
            else:
                must_conditions.append(FieldCondition(key="base_price", match=MatchValue(value=price_filter)))

        # Handle categories filter - ✅ Qdrant supports array matching with MatchAny!
        if "categories" in mongo_filters:
            categories = mongo_filters["categories"]
            if isinstance(categories, list) and categories:
                # MatchAny: matches if ANY of the provided values is in the array field
                must_conditions.append(FieldCondition(key="categories", match=MatchAny(any=categories)))
            elif isinstance(categories, str):
                must_conditions.append(FieldCondition(key="categories", match=MatchAny(any=[categories])))

        # Handle dimension filters
        dimension_fields = ["height_value", "width_value", "depth_value", "weight_value", "volume_value"]
        for dim_field in dimension_fields:
            if dim_field in mongo_filters:
                dim_filter = mongo_filters[dim_field]
                if isinstance(dim_filter, dict):
                    range_params = {}
                    if "$gte" in dim_filter:
                        range_params["gte"] = dim_filter["$gte"]
                    if "$lte" in dim_filter:
                        range_params["lte"] = dim_filter["$lte"]
                    if "$gt" in dim_filter:
                        range_params["gt"] = dim_filter["$gt"]
                    if "$lt" in dim_filter:
                        range_params["lt"] = dim_filter["$lt"]

                    if range_params:
                        must_conditions.append(FieldCondition(key=dim_field, range=Range(**range_params)))
                else:
                    must_conditions.append(FieldCondition(key=dim_field, match=MatchValue(value=dim_filter)))

        # Return None if no conditions
        if not must_conditions:
            return None

        # Return Filter with must conditions (all must be satisfied = AND logic)
        return Filter(must=must_conditions)

    async def _format_results(self, qdrant_results: dict) -> list[dict[str, Any]]:
        """
        Format Qdrant results into complete product information.
        
        Enriches vector search results with full product details from MongoDB,
        including series descriptions and feature specifications.

        Args:
            qdrant_results: Raw results from Qdrant vector search

        Returns:
            List of formatted product results with complete details
        """
        formatted_results = []

        if not qdrant_results["ids"]:
            return formatted_results

        # Extract product codes for batch lookup
        product_codes = [qdrant_results["metadatas"][i].get("product_code") 
                        for i in range(len(qdrant_results["ids"]))]
        
        # Batch fetch product details from MongoDB (async)
        products_cursor = self.collection.find(
            {"product_code": {"$in": product_codes}},
            {"product_code": 1, "series": 1, "features": 1}
        )
        
        # Convert async cursor to dict for quick lookup
        products_map = {}
        async for p in products_cursor:
            products_map[p["product_code"]] = p

        for i, product_id in enumerate(qdrant_results["ids"]):
            # Extract search result metadata
            metadata = qdrant_results["metadatas"][i]
            distance = qdrant_results["distances"][i] if "distances" in qdrant_results else 0
            score = distance

            # Get core product information from vector search
            product_code = metadata.get("product_code", "")
            categories = metadata.get("categories", [])
            
            # Enrich with MongoDB product details
            series = None
            features = []
            
            if product_code in products_map:
                product = products_map[product_code]
                
                # Extract series information
                if series_data := product.get("series"):
                    series = series_data.get("description")
                
                # Extract feature specifications (codes and descriptions only)
                if features_data := product.get("features"):
                    for feature in features_data:
                        feature_item = {}
                        if feature_code := feature.get("feature_code"):
                            feature_item["feature_code"] = feature_code
                        if feature_desc := feature.get("feature_description"):
                            feature_item["feature_description"] = feature_desc
                        if feature_item:
                            features.append(feature_item)

            formatted_results.append(
                {
                    "product_code": product_code,
                    "description": metadata.get("description", ""),
                    "base_price": metadata.get("base_price"),
                    "categories": categories,
                    "series": series,
                    "features": features,
                    "score": round(score, 4),
                }
            )

        return formatted_results

    def get_health_status(self) -> dict[str, Any]:
        """Get search service health status"""
        try:
            count = self.qdrant_service.get_count()
            return {
                "status": "healthy",
                "vectors_count": count,
                "collection": settings.QDRANT_COLLECTION_NAME,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


@lru_cache
def get_search_service() -> SearchService:
    """Get cached search service instance"""
    return SearchService(
        query_analyzer=get_query_analyzer(),
        embedding_service=get_embedding_service(),
        qdrant_service=get_qdrant_service(),
        bm25_service=get_bm25_service(),
    )
