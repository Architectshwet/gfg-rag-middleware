from functools import lru_cache
from typing import Any

from openai import OpenAI

from app.core.config import settings


class EmbeddingService:
    """Service for creating embeddings from product data using OpenAI"""

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL

    def extract_product_text(self, product: dict[str, Any]) -> str:
        """
        Extract searchable text from product document according to requirements:
        - product_code
        - base_price
        - All category descriptions (not codes)
        - Main product description
        - All dimension values
        - Feature descriptions only (codes are internal IDs not useful for search)
        - Series description
        """
        text_parts = []

        # Product code
        if product_code := product.get("product_code"):
            text_parts.append(f"Product Code: {product_code}")

        # Base price
        if base_price := product.get("base_price"):
            text_parts.append(f"Price: ${base_price}")

        # Categories - get all descriptions
        if categories := product.get("categories"):
            category_descriptions = [cat.get("description", "") for cat in categories if cat.get("description")]
            if category_descriptions:
                text_parts.append(f"Categories: {', '.join(category_descriptions)}")

        # Main product description
        if description := product.get("description"):
            text_parts.append(f"Description: {description}")

        # Dimensions - extract all values with units
        if dimensions := product.get("dimensions"):
            dimension_values = []
            for key, value in dimensions.items():
                if value and isinstance(value, dict):
                    # Handle structured dimensions: {'value': 26.0, 'unit': 'LBS'}
                    dim_value = value.get('value')
                    dim_unit = value.get('unit', '')
                    if dim_value:
                        dimension_values.append(f"{key}: {dim_value} {dim_unit}".strip())
                elif value:
                    # Handle simple string dimensions (fallback)
                    dimension_values.append(f"{key}: {value}")
            if dimension_values:
                text_parts.append(f"Dimensions: {', '.join(dimension_values)}")

        # Features - only include descriptions (feature codes are internal IDs)
        if features := product.get("features"):
            feature_descriptions = []
            for feature in features:
                if feature_desc := feature.get("feature_description"):
                    feature_descriptions.append(feature_desc)
            if feature_descriptions:
                text_parts.append(f"Features: {'; '.join(feature_descriptions)}")

        # Series description
        if series := product.get("series"):
            if series_desc := series.get("description"):
                text_parts.append(f"Series: {series_desc}")

        return " | ".join(text_parts)

    def create_embedding(self, text: str) -> list[float]:
        """Create embedding vector from text using OpenAI API"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            encoding_format="float"
        )
        return response.data[0].embedding

    def create_product_embedding(self, product: dict[str, Any]) -> tuple[str, list[float]]:
        """
        Extract text and create embedding for a product

        Returns:
            tuple of (text_representation, embedding_vector)
        """
        text = self.extract_product_text(product)
        embedding = self.create_embedding(text)
        return text, embedding


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get cached embedding service instance"""
    return EmbeddingService()
