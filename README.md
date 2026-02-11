# GFG RAG Middleware

FastAPI middleware for GFG product data, webhook ingestion, embedding generation, and hybrid RAG search.

## What Problem This Solves

Product discovery in furniture/catalog data needs more than keyword search:
- Users ask in natural language ("comfortable workplace chair under $900")
- Product metadata includes structured filters (price, dimensions, categories)
- Exact-match terms like product codes must still work

This service solves that by combining:
- Semantic search (OpenAI embeddings + Qdrant vector retrieval)
- Exact keyword retrieval (BM25)
- Query understanding (LLM-based filter extraction)

## Core Use Cases

- Build embeddings for product catalog documents from MongoDB
- Inspect and validate extracted text and stored vector metadata
- Run semantic-only or hybrid (semantic + BM25) product search
- Serve product/business lookup APIs (family code, series, product details, categories)
- Receive and store XML webhooks for downstream processing

## High-Level Architecture

1. `FastAPI` routes requests via modular routers (`products`, `embeddings`, `search`, `webhooks`)
2. `Motor` provides async MongoDB access for source data and enrichment
3. `EmbeddingService` converts product text into vectors using OpenAI embeddings
4. `ProductEmbeddingService` prepares searchable text + metadata and performs batch upserts
5. `QdrantService` stores vectors locally and executes similarity search with metadata filters
6. `QueryAnalyzer` uses GPT model to convert natural language into search query + filter JSON
7. `BM25Service` provides local keyword search for exact token matching
8. `SearchService` fuses semantic and BM25 results using RRF (Reciprocal Rank Fusion)

## Embedding Pipeline

1. Read products from MongoDB (`products_canada` by default)
2. Extract searchable text fields:
   - `product_code`, `base_price`, categories, description, dimensions, feature descriptions, series
3. Create embeddings with `text-embedding-3-small` (1536 dimensions)
4. Build metadata payload (price/category/dimension fields for filterable search)
5. Batch upsert vectors into local Qdrant collection (`gfg_products`)

Note: current implementation is batch-oriented in-request processing (no separate queue worker).

## Search Pipeline

1. Analyze user query with LLM (`gpt-4o-mini`) into:
   - normalized `search_query`
   - structured filters (`base_price`, `categories`, dimension ranges, etc.)
2. Run semantic retrieval in Qdrant with filters applied during vector query
3. Optionally run BM25 keyword retrieval over stored document text
4. Fuse rankings with RRF and return top results with enriched product details

## API Surface (Main Endpoints)

Base prefix: `/middleware/api/v1`

- `POST /embeddings/create` - Generate and store embeddings in batch
- `GET /embeddings/preview` - Preview extracted text + metadata without embedding
- `GET /embeddings/stats` - Vector coverage vs total products
- `DELETE /embeddings/clear` - Clear Qdrant collection
- `GET /embeddings/inspect` - Inspect stored payload/documents
- `POST /search` - Semantic or hybrid product search
- `GET /search/health` - Search subsystem health
- `GET /products/family-code` - Fabric code -> family code
- `GET /products/chair-series` - Family code -> chair series
- `GET /products/product-series` - Series lookup
- `GET /products/product-codes` - Product codes by series description
- `GET /products/filter-by-category` - Category-based product filtering
- `GET /products/product-details` - Full product details by product code
- `GET /products/decision-trees` - Fetch decision trees

Webhook endpoints (prefix `/middleware/webhooks`):
- `POST /receive/{source}` - Receive XML payload
- `GET /` and `GET /{webhook_id}` - Retrieve webhook records
- `PATCH /{webhook_id}/process` - Mark webhook processed

## Configuration

Create `.env` from `.env.example` and set:
- `MONGO_URL`
- `MONGO_DB_NAME`
- `OPENAI_API_KEY`

Important defaults in `app/core/config.py`:
- `PRODUCTS_COLLECTION=products_canada`
- `EMBEDDING_MODEL=text-embedding-3-small`
- `QUERY_ANALYZER_MODEL=gpt-4o-mini`
- `QDRANT_PATH=./qdrant_db`
- `QDRANT_COLLECTION_NAME=gfg_products`

## Run Locally

Prerequisites:
- Python 3.12+
- `uv` package manager
- MongoDB access
- OpenAI API key

Commands:
```bash
uv sync
uv run uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Health check:
- `GET http://localhost:8001/middleware/health`
- Swagger: `http://localhost:8001/docs`

## Run With Docker

```bash
docker compose up --build
```

Dev (auto-reload):
```bash
docker compose -f docker-compose.dev.yml up --build
```

## Implementation Notes

- Qdrant runs in local persistent mode via filesystem path (`./qdrant_db`)
- Mongo connection is opened/closed through FastAPI lifespan hooks
- Services are cached where appropriate (`lru_cache`) for reuse
- Search response is enriched with series/features fetched back from MongoDB
