from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    APP_NAME: str = "GFG Middleware API"
    DEBUG: bool = False

    # MongoDB settings
    MONGO_URL: str
    MONGO_DB_NAME: str = "gfg_dev_mongo"

    # Collection names
    PRODUCTS_COLLECTION: str = "products_canada"
    
    # OpenAI settings
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI embedding model
    EMBEDDING_DIMENSION: int = 1536  # Dimension for text-embedding-3-small
    QUERY_ANALYZER_MODEL: str = "gpt-4o-mini"  # Model for query analysis
    
    # Qdrant settings
    QDRANT_PATH: str = "./qdrant_db"  # Local storage path
    QDRANT_COLLECTION_NAME: str = "gfg_products"
    
    # Search settings
    SEARCH_TOP_K: int = 5  # Final number of results to return
    HYBRID_RETRIEVAL_SIZE: int = 20  # Candidates from each method (semantic + BM25) before fusion
    SEARCH_ALPHA: float = 0.7  # Deprecated (using RRF for fusion now)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
