from datetime import datetime

from pydantic import BaseModel, Field


class ProductBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    price: float = Field(..., gt=0)
    sku: str = Field(..., min_length=1, max_length=100)
    quantity: int = Field(default=0, ge=0)


class ProductCreate(ProductBase):
    pass


class ProductUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None
    price: float | None = Field(None, gt=0)
    sku: str | None = Field(None, min_length=1, max_length=100)
    quantity: int | None = Field(None, ge=0)


class ProductResponse(ProductBase):
    id: str
    created_at: datetime
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


# ============================================================================
# GFG Business Query Response Models
# ============================================================================


class FamilyCodeResponse(BaseModel):
    """Response model for family code"""

    family_code: str


class ChairSeriesResponse(BaseModel):
    """Response model for chair series"""

    chair_series: str


class ProductSeriesResponse(BaseModel):
    """Response model for product series"""

    series_description: str


class ProductCodeResponse(BaseModel):
    """Response model for product codes"""

    product_code: str
    description: str
    series_description: str | None = None


class CategoryProductResponse(BaseModel):
    """Response model for category filtered products"""

    product_code: str
    description: str
    series_description: str | None = None


class ProductDetailsResponse(BaseModel):
    """Response model for product details"""

    product_code: str | None = None
    base_price: str | None = None
    category: str | None = None
    description: str | None = None
    dimensions: str | None = None
    features: str | None = None
    price_list_ref: str | None = None
    processed_at: str | None = None
    series_name: str | None = None
    sub_category: str | None = None
    category_description: str | None = None
    sub_category_description: str | None = None
    series_description: str | None = None

    class Config:
        from_attributes = True
        extra = "allow"  # Allow additional fields from database
