"""Service for processing products and creating embeddings with metadata"""

import logging
import sys
from typing import Any

from app.services.qdrant_service import get_qdrant_service
from app.services.embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


class ProductEmbeddingService:
    """Service to process products and prepare them for Qdrant storage"""

    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.qdrant_service = get_qdrant_service()

    def prepare_product_metadata(self, product: dict[str, Any]) -> dict[str, Any]:
        """
        Extract and structure product metadata for vector storage.
        
        Prepares metadata fields that will be stored alongside embeddings in Qdrant,
        enabling efficient filtering during semantic search operations.

        Args:
            product: MongoDB product document

        Returns:
            Metadata dictionary optimized for Qdrant storage
        """
        metadata = {
            "product_code": product.get("product_code", "unknown"),
            "base_price": float(product.get("base_price", 0)) if product.get("base_price") else 0,
            "description": product.get("description", "")[:500],
        }

        # Store categories as array for efficient matching
        if categories := product.get("categories"):
            category_list = [cat.get("description", "") for cat in categories if cat.get("description")]
            metadata["categories"] = category_list
        else:
            metadata["categories"] = []

        # Extract dimensional attributes for range-based filtering
        if dimensions := product.get("dimensions"):
            for dim_key, dim_data in dimensions.items():
                if isinstance(dim_data, dict) and dim_data.get("value") is not None:
                    metadata[f"{dim_key}_value"] = float(dim_data["value"])
                    metadata[f"{dim_key}_unit"] = dim_data.get("unit", "")

        # Store series information
        if series := product.get("series"):
            if series_desc := series.get("description"):
                metadata["series"] = series_desc[:100]

        # Store series flag
        metadata["multiple_series_flag"] = product.get("multiple_series_flag", 0)

        return metadata

    def process_product(self, product: dict[str, Any]) -> tuple[str, str, list[float], dict[str, Any]]:
        """
        Transform product document into vector embedding with metadata.
        
        Generates searchable text representation, creates embedding vector,
        and prepares structured metadata for the vector database.

        Args:
            product: MongoDB product document

        Returns:
            Tuple of (product_id, text_representation, embedding_vector, metadata)
        """
        product_id = str(product["_id"])
        product_code = product.get("product_code", "unknown")

        # Generate embedding from product text
        text_representation, embedding = self.embedding_service.create_product_embedding(product)

        # Extract metadata for search filtering
        metadata = self.prepare_product_metadata(product)

        # Display processing information
        logger.info(f"{'='*100}")
        logger.info(f"🔄 PROCESSING PRODUCT: {product_code}")
        logger.info(f"{'='*100}")
        logger.info(f"📝 TEXT REPRESENTATION (being embedded):")
        logger.info(f"{text_representation}")
        logger.info(f"🏷️  METADATA (being stored in Qdrant):")
        for key, value in metadata.items():
            logger.info(f"  {key}: {value}")
        logger.info(f"✅ Created embedding with {len(embedding)} dimensions")
        logger.info(f"{'='*100}")

        return product_id, text_representation, embedding, metadata

    def store_batch(
        self, ids: list[str], embeddings: list[list[float]], documents: list[str], metadatas: list[dict]
    ) -> int:
        """
        Store a batch of embeddings in Qdrant

        Args:
            ids: List of product IDs
            embeddings: List of embedding vectors
            documents: List of text representations
            metadatas: List of metadata dictionaries

        Returns:
            Number of items stored
        """
        if not ids:
            return 0

        self.qdrant_service.add_embeddings(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

        return len(ids)


def get_product_embedding_service() -> ProductEmbeddingService:
    """Get product embedding service instance"""
    return ProductEmbeddingService()
