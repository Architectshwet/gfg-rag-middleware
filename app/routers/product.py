from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.mongo import get_mongo_db
from app.schemas.product import (
    CategoryProductResponse,
    ChairSeriesResponse,
    FamilyCodeResponse,
    ProductCodeResponse,
    ProductSeriesResponse,
)
from app.services.product_service import ProductService

router = APIRouter(prefix="/products", tags=["Products"])


def get_product_service() -> ProductService:
    db = get_mongo_db()
    return ProductService(db)


# ============================================================================
# GFG Business Query Endpoints (Must be defined BEFORE /{product_id})
# ============================================================================


@router.get("/family-code", response_model=list[FamilyCodeResponse])
async def get_family_code_from_fabric_code(
    fabric_code: str = Query(..., description="Fabric code to search for"),
    service: ProductService = Depends(get_product_service),
):
    """Get family code from fabric code."""
    results = await service.get_family_code_by_fabric_code(fabric_code)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No family code found for fabric code: {fabric_code}",
        )
    return results


@router.get("/chair-series", response_model=list[ChairSeriesResponse])
async def get_chair_series_from_family_code(
    family_code: str = Query(..., description="Family code to search for"),
    service: ProductService = Depends(get_product_service),
):
    """Get chair series using family code."""
    results = await service.get_chair_series_by_family_code(family_code)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chair series found for family code: {family_code}",
        )
    return results


@router.get("/product-series", response_model=list[ProductSeriesResponse])
async def get_product_series_by_descriptions(
    series: list[str] = Query(..., description="Product series to search for (can be repeated)"),
    service: ProductService = Depends(get_product_service),
):
    """Get product series using product/chair series list. Use: ?series=Chap&series=File Buddy"""
    # Filter out empty and whitespace-only values
    filtered_series = [s.strip() for s in series if s.strip()]
    if not filtered_series:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one product series must be provided",
        )
    results = await service.get_product_series_by_descriptions(filtered_series)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No product series found for the provided series descriptions",
        )
    return results


@router.get("/product-codes", response_model=list[ProductCodeResponse])
async def get_product_codes_by_series_description(
    series_description: str = Query(..., description="Series description to search for"),
    service: ProductService = Depends(get_product_service),
):
    """Get product codes using series description."""
    results = await service.get_product_codes_by_series_description(series_description)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No product codes found for series description: {series_description}",
        )
    return results


@router.get("/filter-by-category", response_model=list[CategoryProductResponse])
async def filter_by_category(
    category: list[str] = Query(..., description="Categories to filter by (can be repeated)"),
    service: ProductService = Depends(get_product_service),
):
    """Filter products by categories. Use: ?category=Work and Task Seating&category=Workplace"""
    # Filter out empty values
    filtered_categories = [c.strip() for c in category if c.strip()]
    if not filtered_categories:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one category must be provided",
        )
    results = await service.filter_by_category(filtered_categories)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No products found for categories: {', '.join(filtered_categories)}",
        )
    return results


@router.get("/product-details")
async def get_product_code_details(
    product_code: str = Query(..., description="Product code to search for"),
    service: ProductService = Depends(get_product_service),
):
    """Get product code details - returns raw data without validation."""
    result = await service.get_product_code_details(product_code)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No product found for product code: {product_code}",
        )
    return result


# ============================================================================
# Decision Trees Endpoints
# ============================================================================


@router.get("/decision-trees")
async def get_all_decision_trees(
    service: ProductService = Depends(get_product_service),
):
    """Get all decision trees from the collection."""
    results = await service.get_all_decision_trees()
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No decision trees found",
        )
    return results
