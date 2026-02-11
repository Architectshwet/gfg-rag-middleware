from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Simple search request - just the query"""
    query: str = Field(..., description="Natural language search query", min_length=1)


class ProductResult(BaseModel):
    """Single product search result"""
    product_code: str
    description: str
    base_price: float | None = None
    categories: list[str]
    score: float


class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    analyzed_query: str  # What was actually searched after analysis
    filters_detected: dict[str, Any]  # Auto-detected filters from query
    results: list[ProductResult]
    total_results: int
