from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.mongo import close_mongo, connect_mongo
from app.routers.product import router as product_router
from app.routers.webhook import router as webhook_router
from app.routers.embeddings import router as embeddings_router
from app.routers.search import router as search_router

# Setup logging (must be done before any other imports that use logging)
setup_logging(log_level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_mongo()
    yield
    # Shutdown
    await close_mongo()


app = FastAPI(
    title="GFG Middleware API",
    description="Product management, webhook processing, and RAG search API",
    version="1.0.0",
    lifespan=lifespan,
)

# Middleware routes under /middleware
app.include_router(product_router, prefix="/middleware/api/v1")
app.include_router(webhook_router, prefix="/middleware")

# RAG routes under /middleware/api/v1
app.include_router(search_router, prefix="/middleware/api/v1")
app.include_router(embeddings_router, prefix="/middleware/api/v1")


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.APP_NAME}"}


@app.get("/middleware")
async def middleware_root():
    return {"message": f"Welcome to {settings.APP_NAME} - Middleware"}


@app.get("/middleware/health")
async def health_check():
    return {"status": "healthy"}
