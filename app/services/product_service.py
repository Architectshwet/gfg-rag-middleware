import re
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase


class ProductService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db["products"]
        # GFG Collections
        self.fabric_families_collection = db["fabric_families"]
        self.fabric_to_series_mapping_collection = db["fabric_to_series_mapping"]
        self.products_canada_collection = db["products_canada"]
        self.decision_trees_collection = db["decision_trees"]

    # ============================================================================
    # GFG Business Query Methods
    # ============================================================================

    async def get_family_code_by_fabric_code(self, fabric_code: str) -> list[dict[str, Any]]:
        """Get family code from fabric code."""
        escaped_fabric_code = re.escape(fabric_code)
        pipeline = [
            {"$match": {"fabric_code": {"$regex": f"^{escaped_fabric_code}$", "$options": "i"}}},
            {"$project": {"family_code": 1, "_id": 0}},
        ]
        cursor = self.fabric_families_collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_chair_series_by_family_code(self, family_code: str) -> list[dict[str, Any]]:
        """Get chair series using family code."""
        escaped_family_code = re.escape(family_code)
        pipeline = [
            {
                "$match": {
                    "family_code": {"$regex": f"^{escaped_family_code}$", "$options": "i"},
                    "fabric_status": {"$regex": "^Included$", "$options": "i"},
                }
            },
            {"$project": {"chair_series": 1, "_id": 0}},
        ]
        cursor = self.fabric_to_series_mapping_collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_product_series_by_descriptions(self, product_series: list[str]) -> list[dict[str, Any]]:
        """Get product series using product/chair series list."""
        # MongoDB $in doesn't support regex patterns, so we use $or instead
        escaped_patterns = [re.escape(series) for series in product_series]
        or_conditions = [
            {"series.description": {"$regex": f"^{pattern}$", "$options": "i"}} for pattern in escaped_patterns
        ]

        pipeline = [
            {"$match": {"$or": or_conditions}},
            {"$project": {"series_description": "$series.description", "_id": 0}},
        ]
        cursor = self.products_canada_collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_product_codes_by_series_description(self, series_description: str) -> list[dict[str, Any]]:
        """Get product codes using series description."""
        escaped_series_description = re.escape(series_description)
        pipeline = [
            {"$match": {"series.description": {"$regex": f"^{escaped_series_description}$", "$options": "i"}}},
            {"$project": {"product_code": 1, "description": 1, "series_description": "$series.description", "_id": 0}},
        ]
        cursor = self.products_canada_collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def filter_by_category(self, categories: list[str]) -> list[dict[str, Any]]:
        """Filter products by categories - matches if any category description matches."""
        # Build $or conditions for each category description to match against categories array
        escaped_categories = [re.escape(cat) for cat in categories]
        category_conditions = [
            {"categories.description": {"$regex": f"^{pattern}$", "$options": "i"}} for pattern in escaped_categories
        ]
        
        pipeline = [
            {"$match": {"$or": category_conditions}},
            {"$project": {"product_code": 1, "description": 1, "series_description": "$series.description", "_id": 0}},
        ]
        cursor = self.products_canada_collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_product_code_details(self, product_code: str) -> dict[str, Any] | None:
        """Get product code details using direct filter."""
        escaped_product_code = re.escape(product_code)
        filter_query = {"product_code": {"$regex": f"^{escaped_product_code}$", "$options": "i"}}

        # Exclude embedding-related fields from the response
        projection = {
            "_id": 0,
            "embedding_text": 0,
            "embedding_model": 0,
            "embedding_includes_features": 0,
            "embedding_features_depth": 0,
            "description_embedding": 0,
        }

        return await self.products_canada_collection.find_one(filter_query, projection)

    # ============================================================================
    # Decision Trees Methods
    # ============================================================================

    async def get_all_decision_trees(self) -> list[dict[str, Any]]:
        """Get all decision trees from the collection."""
        cursor = self.decision_trees_collection.find({}, {"_id": 0})
        return await cursor.to_list(length=None)
