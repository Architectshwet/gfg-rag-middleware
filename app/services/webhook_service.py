from datetime import UTC, datetime
from typing import Any

import xmltodict
from bson import ObjectId
from bson.errors import InvalidId
from motor.motor_asyncio import AsyncIOMotorDatabase


class WebhookService:
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db["webhooks"]

    def _to_object_id(self, id_str: str) -> ObjectId | None:
        """Convert string to ObjectId, returning None if invalid."""
        try:
            return ObjectId(id_str)
        except InvalidId:
            return None

    def parse_xml(self, xml_data: str) -> dict[str, Any]:
        """Parse XML string to dictionary."""
        return xmltodict.parse(xml_data)

    async def store_webhook(self, source: str, xml_data: str, parsed_data: dict[str, Any]) -> str:
        """Store webhook data in MongoDB."""
        document = {
            "source": source,
            "raw_xml": xml_data,
            "parsed_data": parsed_data,
            "received_at": datetime.now(UTC),
            "processed": False,
        }
        result = await self.collection.insert_one(document)
        return str(result.inserted_id)

    async def get_webhook(self, webhook_id: str) -> dict[str, Any] | None:
        """Retrieve a webhook by ID."""
        obj_id = self._to_object_id(webhook_id)
        if obj_id is None:
            return None
        document = await self.collection.find_one({"_id": obj_id})
        if document:
            document["id"] = str(document.pop("_id"))
        return document

    async def get_webhooks(self, source: str | None = None, skip: int = 0, limit: int = 100) -> list[dict[str, Any]]:
        """Retrieve webhooks with optional filtering."""
        query = {}
        if source:
            query["source"] = source

        cursor = self.collection.find(query).skip(skip).limit(limit).sort("received_at", -1)
        webhooks = await cursor.to_list(length=limit)

        for webhook in webhooks:
            webhook["id"] = str(webhook.pop("_id"))

        return webhooks

    async def mark_processed(self, webhook_id: str) -> bool:
        """Mark a webhook as processed."""
        obj_id = self._to_object_id(webhook_id)
        if obj_id is None:
            return False
        result = await self.collection.update_one(
            {"_id": obj_id}, {"$set": {"processed": True, "processed_at": datetime.now(UTC)}}
        )
        return result.modified_count > 0
