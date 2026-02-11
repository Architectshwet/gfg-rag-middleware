import logging
import sys
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.mongo import get_mongo_db
from app.services.qdrant_service import get_qdrant_service
from app.services.product_embedding_service import get_product_embedding_service

router = APIRouter(prefix="/embeddings", tags=["embeddings"])
logger = logging.getLogger(__name__)


class EmbeddingRequest(BaseModel):
    """Request to create embeddings"""

    limit: int = Field(default=10, ge=1, le=10000, description="Number of products to process")
    skip: int = Field(default=0, ge=0, description="Number of products to skip")
    force_update: bool = Field(
        default=False, description="Force update even if embeddings already exist"
    )


class EmbeddingStats(BaseModel):
    """Embedding statistics"""

    total_products: int
    products_with_embeddings: int
    products_without_embeddings: int
    progress_percentage: float


class EmbeddingResponse(BaseModel):
    """Response from embedding creation"""

    processed: int
    successful: int
    failed: int
    errors: list[dict[str, str]]
    stats: EmbeddingStats


@router.post("/create", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """
    Create embeddings for products and store in local Qdrant

    This endpoint processes products and creates embeddings using OpenAI's text-embedding-3-small model,
    then stores them in a local Qdrant vector database.

    Args:
        request: EmbeddingRequest with limit, skip, and force_update options

    Returns:
        EmbeddingResponse with processing statistics

    Example:
        ```json
        {
            "limit": 10,
            "skip": 0,
            "force_update": false
        }
        ```
    """
    try:
        db = get_mongo_db()
        collection = db[settings.PRODUCTS_COLLECTION]
        product_service = get_product_embedding_service()

        # Fetch products from MongoDB (excluding features.options)
        projection = {"features.options": 0}
        cursor = collection.find({}, projection).skip(request.skip).limit(request.limit)

        processed = 0
        successful = 0
        failed = 0
        errors = []

        # Batch data for Qdrant
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        async for product in cursor:
            try:
                # Process product using service (extracts text, creates embedding, prepares metadata)
                product_id, text_rep, embedding, metadata = product_service.process_product(product)

                # Collect for batch insert
                ids.append(product_id)
                embeddings.append(embedding)
                documents.append(text_rep)
                metadatas.append(metadata)

                processed += 1

            except Exception as e:
                failed += 1
                errors.append({"product_code": product.get("product_code", "unknown"), "error": str(e)})

        # Batch insert into Qdrant using service
        if ids:
            try:
                logger.info(f"{'🔥'*50}")
                logger.info(f"📦 BATCH INSERT: Storing {len(ids)} embeddings in Qdrant...")
                successful = product_service.store_batch(ids, embeddings, documents, metadatas)
                logger.info(f"✅ Successfully stored {successful} embeddings!")
                logger.info(f"{'🔥'*50}")
            except Exception as e:
                logger.error(f"❌ Batch insert failed: {str(e)}")
                errors.append({"batch_insert": str(e)})
                failed = len(ids)
                successful = 0

        # Get statistics
        stats = await get_embedding_statistics()

        return EmbeddingResponse(
            processed=processed, successful=successful, failed=failed, errors=errors, stats=stats
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding creation failed: {str(e)}")


@router.get("/preview")
async def preview_text_extraction(limit: int = 5):
    """
    Preview text extraction for products WITHOUT creating embeddings
    
    This shows you what the extracted text looks like for each product,
    helping to understand what filters can be applied.
    
    Args:
        limit: Number of products to preview (default: 5)
    
    Returns:
        List of products with their extracted text and metadata
    """
    try:
        db = get_mongo_db()
        collection = db[settings.PRODUCTS_COLLECTION]
        product_service = get_product_embedding_service()
        
        # Exclude features.options when fetching products
        projection = {"features.options": 0}
        cursor = collection.find({}, projection).limit(limit)
        
        previews = []
        
        async for product in cursor:
            product_code = product.get("product_code", "unknown")
            
            # Extract text (without creating embedding)
            text_representation = product_service.embedding_service.extract_product_text(product)
            
            # Prepare metadata (what will be stored in Qdrant)
            metadata = product_service.prepare_product_metadata(product)
            
            # Build preview
            preview = {
                "product_code": product_code,
                "text_representation": text_representation,
                "metadata": metadata,
            }
            
            previews.append(preview)
        
        return {
            "total_previewed": len(previews),
            "previews": previews,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.get("/stats", response_model=EmbeddingStats)
async def get_embedding_stats():
    """
    Get embedding statistics from Qdrant

    Returns statistics about how many products have embeddings in local Qdrant.

    Returns:
        EmbeddingStats with counts and progress percentage
    """
    try:
        qdrant_service = get_qdrant_service()
        db = get_mongo_db()
        collection = db[settings.PRODUCTS_COLLECTION]
        
        total = await collection.count_documents({})
        with_embeddings = qdrant_service.get_count()
        without_embeddings = max(0, total - with_embeddings)
        progress = (with_embeddings / total * 100) if total > 0 else 0.0

        return EmbeddingStats(
            total_products=total,
            products_with_embeddings=with_embeddings,
            products_without_embeddings=without_embeddings,
            progress_percentage=round(progress, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.delete("/clear")
async def clear_embeddings():
    """
    Clear all embeddings from Qdrant

    ⚠️ WARNING: This will remove all embeddings from local Qdrant.
    Use with caution!

    Returns:
        Number of products cleared
    """
    try:
        qdrant_service = get_qdrant_service()
        count_before = qdrant_service.get_count()
        
        qdrant_service.clear()
        
        # Reinitialize
        get_qdrant_service.cache_clear()

        return {
            "success": True,
            "cleared_count": count_before,
            "message": f"Cleared {count_before} embeddings from Qdrant"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear embeddings: {str(e)}")


@router.get("/inspect")
async def inspect_embeddings(limit: int = 5, offset: int = 0):
    """
    Inspect actual stored embeddings and metadata in Qdrant
    
    Shows the actual metadata structure as stored in the vector database
    
    Args:
        limit: Number of embeddings to retrieve (default: 5)
        offset: Offset for pagination (default: 0)
    
    Returns:
        List of embeddings with their metadata, documents, and IDs
    """
    try:
        qdrant_service = get_qdrant_service()
        
        # Get points from Qdrant
        result = qdrant_service.get_points(limit=limit, offset=offset)
        
        if not result["points"]:
            return {
                "message": "No embeddings found in Qdrant",
                "total_embeddings": 0,
                "showing": 0,
                "embeddings": []
            }
        
        # Format the results
        embeddings_data = []
        for point in result["points"]:
            payload = point["payload"]
            embeddings_data.append({
                "id": point["id"],
                "document": payload.get("document", ""),
                "metadata": {k: v for k, v in payload.items() if k != "document"},
            })
        
        return {
            "total_embeddings": result["total_count"],
            "showing": len(embeddings_data),
            "embeddings": embeddings_data,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspection failed: {str(e)}")


async def get_embedding_statistics() -> EmbeddingStats:
    """Helper function to get embedding statistics"""
    qdrant_service = get_qdrant_service()
    db = get_mongo_db()
    collection = db[settings.PRODUCTS_COLLECTION]

    total = await collection.count_documents({})
    with_embeddings = qdrant_service.get_count()
    without_embeddings = max(0, total - with_embeddings)
    progress = (with_embeddings / total * 100) if total > 0 else 0.0

    return EmbeddingStats(
        total_products=total,
        products_with_embeddings=with_embeddings,
        products_without_embeddings=without_embeddings,
        progress_percentage=round(progress, 2),
    )
