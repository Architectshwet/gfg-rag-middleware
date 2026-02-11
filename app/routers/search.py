from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.search_service import get_search_service

router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    """Search request with optional hybrid mode toggle"""
    query: str = Field(..., description="Search query text", min_length=1)
    use_hybrid: bool = Field(
        default=False,
        description="Use hybrid search (semantic + metadata filtering + keyword). Set to False for semantic-only search with metadata filtering.",
    )


class ProductResult(BaseModel):
    """Single product search result"""
    product_code: str
    description: str
    base_price: float | None = None
    categories: list[str]
    series: str | None = None
    features: list[dict[str, str]] | None = None  # List of {feature_code, feature_description}
    score: float


class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    analyzed_query: str  # What was actually searched
    filters_detected: dict[str, Any]  # Auto-detected filters
    search_method: str  # "hybrid (semantic + keyword)" or "semantic only"
    results: list[ProductResult]
    total_results: int


@router.post("", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Hybrid product search with semantic + keyword.
    
    🔍 **Architecture:**
    
    1. **Query Analysis** (GPT-4o-mini):
       - Extracts search terms: "chair" from "office chair under $500"
       - Extracts filters: price ≤ 500
    
    2. **Parallel Retrieval**:
       - **Semantic Search**: OpenAI embeddings + Qdrant with metadata filtering
         → Understands: "comfortable" ≈ "ergonomic" ≈ "cushioned"
         → Applies: price, category, dimension filters DURING search
       - **Keyword Search**: BM25 algorithm
         → Exact match: product codes, brand names, rare terms
    
    3. **Result Fusion** (RRF - Reciprocal Rank Fusion):
       - Combines semantic + keyword rankings
       - No score normalization needed
       - Emphasizes agreement between methods
    
    🎯 **Why Hybrid?**
    - Semantic alone: Misses exact product codes, brand names
    - Keyword alone: Misses synonyms, semantic meaning
    - Hybrid: Best of both worlds!
    
    📝 **Examples:**
        ```json
        # Semantic understanding + metadata filtering + keyword matching
        {"query": "comfortable office chair under $500"}
        
        # Exact product code (keyword excels here)
        {"query": "product 1004"}
        
        # Category + dimensions (filters applied in semantic search)
        {"query": "workplace chair height over 40 inches"}
        
        # Semantic only (disable hybrid keyword search)
        {"query": "ergonomic seating", "use_hybrid": false}
        ```

    Args:
        request: SearchRequest with query and optional use_hybrid flag

    Returns:
        SearchResponse with top 5 products, search method used, and detected filters
    """
    try:
        search_service = get_search_service()
        result = await search_service.search_products(request.query, use_hybrid=request.use_hybrid)
        
        # Convert dict results to Pydantic models
        product_results = [ProductResult(**product) for product in result["results"]]
        
        return SearchResponse(
            query=request.query,
            analyzed_query=result["analyzed_query"],
            filters_detected=result["filters_detected"],
            search_method=result["search_method"],
            results=product_results,
            total_results=result["total_results"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/health")
async def search_health():
    """Check if search service is healthy"""
    try:
        search_service = get_search_service()
        return search_service.get_health_status()
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
