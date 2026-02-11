from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app.core.mongo import get_mongo_db
from app.services.webhook_service import WebhookService

router = APIRouter(
    prefix="/webhooks",
    tags=["Webhooks"],
)


def get_webhook_service() -> WebhookService:
    db = get_mongo_db()
    return WebhookService(db)


@router.post("/receive/{source}", status_code=status.HTTP_201_CREATED)
async def receive_webhook(source: str, request: Request, service: WebhookService = Depends(get_webhook_service)):
    """Receive and store XML webhook data."""
    content_type = request.headers.get("content-type", "")

    if "xml" not in content_type.lower() and "text/plain" not in content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be application/xml or text/xml",
        )

    try:
        xml_data = await request.body()
        xml_string = xml_data.decode("utf-8")

        parsed_data = service.parse_xml(xml_string)
        webhook_id = await service.store_webhook(source, xml_string, parsed_data)

        return {"message": "Webhook received successfully", "webhook_id": webhook_id, "source": source}
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Failed to parse XML: {str(e)}") from e


@router.get("/")
async def get_webhooks(
    source: str | None = None,
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    service: WebhookService = Depends(get_webhook_service),
):
    """Retrieve stored webhooks."""
    webhooks = await service.get_webhooks(source=source, skip=skip, limit=limit)
    return {"webhooks": webhooks, "count": len(webhooks)}


@router.get("/{webhook_id}")
async def get_webhook(webhook_id: str, service: WebhookService = Depends(get_webhook_service)):
    """Retrieve a specific webhook by ID."""
    webhook = await service.get_webhook(webhook_id)
    if not webhook:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Webhook with id {webhook_id} not found")
    return webhook


@router.patch("/{webhook_id}/process")
async def mark_webhook_processed(webhook_id: str, service: WebhookService = Depends(get_webhook_service)):
    """Mark a webhook as processed."""
    success = await service.mark_processed(webhook_id)
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Webhook with id {webhook_id} not found")
    return {"message": "Webhook marked as processed", "webhook_id": webhook_id}
