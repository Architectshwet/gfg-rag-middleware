"""
Qdrant vector database service for storing and querying product embeddings.
"""

import logging
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.core.config import settings

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector database"""

    def __init__(self):
        """Initialize Qdrant client and collection"""
        # Use local persistent storage
        self.client = QdrantClient(path=settings.QDRANT_PATH)
        self.collection_name = settings.QDRANT_COLLECTION_NAME

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Create collection if it doesn't exist"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]

            if self.collection_name not in collection_names:
                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIMENSION, distance=Distance.COSINE
                    ),
                )
                logger.info(f"✅ Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"✅ Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"❌ Error initializing collection: {e}")
            raise

    def add_embeddings(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        """
        Add embeddings to Qdrant collection

        Args:
            ids: List of unique IDs for each embedding (MongoDB ObjectId strings)
            embeddings: List of embedding vectors
            metadatas: List of metadata dictionaries
            documents: List of text documents
        """
        try:
            points = []
            for i in range(len(ids)):
                # Combine metadata and document into payload
                payload = metadatas[i].copy()
                payload["document"] = documents[i]
                payload["mongodb_id"] = ids[i]  # Store original MongoDB ID in payload

                # Convert MongoDB ObjectId string to integer for Qdrant
                # MongoDB ObjectId is a 24-char hex string, we convert to int
                point_id = int(ids[i], 16)

                point = PointStruct(id=point_id, vector=embeddings[i], payload=payload)
                points.append(point)

            # Upsert points (insert or update if exists)
            self.client.upsert(collection_name=self.collection_name, points=points)

        except Exception as e:
            raise Exception(f"Failed to add embeddings to Qdrant: {str(e)}")

    def get_count(self) -> int:
        """Get total number of embeddings in collection"""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return collection_info.points_count
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            return 0

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        filters: dict | None = None,
    ) -> dict:
        """
        Query similar embeddings from Qdrant using semantic search only.

        Args:
            query_embedding: Query vector
            n_results: Number of results to return
            filters: Optional Qdrant filters (Filter object)

        Returns:
            Dictionary with ids, documents, metadatas, and distances
        """
        try:
            # Use query_points method (correct Qdrant API)
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=n_results,
                query_filter=filters,  # Qdrant Filter object
            ).points

            # Format results to match expected structure
            ids = []
            documents = []
            metadatas = []
            distances = []

            for result in results:
                ids.append(result.id)
                
                # Extract document from payload
                payload = result.payload
                documents.append(payload.get("document", ""))
                
                # Remove document from metadata
                metadata = {k: v for k, v in payload.items() if k != "document"}
                metadatas.append(metadata)
                
                # Qdrant returns score (higher = more similar)
                distances.append(result.score)

            return {
                "ids": ids,
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
            }

        except Exception as e:
            raise Exception(f"Failed to query Qdrant: {str(e)}")

    def get_all_documents(self) -> list[dict]:
        """
        Retrieve all documents from collection for BM25 indexing.
        
        Returns:
            List of dicts with 'id' and 'text' keys
        """
        try:
            documents = []
            offset = None
            batch_size = 1000

            # Scroll through all points
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )

                if not points:
                    break

                # Extract id and document text
                for point in points:
                    if "document" in point.payload:
                        documents.append({"id": point.id, "text": point.payload["document"]})

                if offset is None:
                    break

            logger.info(f"✅ Retrieved {len(documents)} documents from Qdrant for BM25 indexing")
            return documents

        except Exception as e:
            raise Exception(f"Failed to retrieve documents: {str(e)}")

    def clear(self) -> None:
        """Delete and recreate collection"""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self._initialize_collection()
            logger.info(f"✅ Cleared collection: {self.collection_name}")
        except Exception as e:
            raise Exception(f"Failed to clear collection: {str(e)}")

    def get_points(self, limit: int = 10, offset: int = 0) -> dict:
        """
        Get raw points from collection for inspection

        Args:
            limit: Number of points to fetch
            offset: Offset for pagination

        Returns:
            Dictionary with points data
        """
        try:
            # Scroll through points
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False,  # Don't include vectors for inspection
            )

            formatted_points = []
            for point in points:
                formatted_points.append({
                    "id": point.id,
                    "payload": point.payload,
                })

            return {
                "total_count": self.get_count(),
                "points": formatted_points,
            }

        except Exception as e:
            raise Exception(f"Failed to get points: {str(e)}")


@lru_cache
def get_qdrant_service() -> QdrantService:
    """Dependency injection for Qdrant service"""
    return QdrantService()
