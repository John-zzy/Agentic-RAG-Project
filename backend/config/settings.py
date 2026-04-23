import json
import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / ".chroma"
SQLITE_PATH = BASE_DIR / "memory" / "sessions.db"
ENV_FILE = BASE_DIR / ".env"
MODEL_ROUTING_FILE = BASE_DIR / "config" / "model_routing.json"

load_dotenv(ENV_FILE)
ENV_VALUES = dotenv_values(ENV_FILE)


class ModelEndpointConfig(BaseModel):
    provider: str
    model_name: str
    api_base: str | None = None
    api_key: str | None = None
    supports_streaming: bool = False
    timeout_seconds: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_env_value(key: str) -> str | None:
    value = os.getenv(key)
    if value is not None:
        return value
    return ENV_VALUES.get(key)


def load_model_routing_config() -> dict[str, dict[str, dict[str, object]]]:
    if not MODEL_ROUTING_FILE.exists():
        return {
            "models": {
                "simple": {
                    "provider": "dashscope",
                    "model_name": "qwen-turbo",
                    "api_base": DASHSCOPE_API_BASE,
                },
                "moderate": {
                    "provider": "dashscope",
                    "model_name": "qwen-plus",
                    "api_base": DASHSCOPE_API_BASE,
                },
                "complex": {
                    "provider": "dashscope",
                    "model_name": "qwen-max",
                    "api_base": DASHSCOPE_API_BASE,
                },
            }
        }

    with MODEL_ROUTING_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_api_keys() -> dict[str, str | None]:
    return {
        "simple": get_env_value("AI_RAG_MODELS__SIMPLE__API_KEY"),
        "moderate": get_env_value("AI_RAG_MODELS__MODERATE__API_KEY"),
        "complex": get_env_value("AI_RAG_MODELS__COMPLEX__API_KEY"),
    }


def parse_env_int(key: str, default: int) -> int:
    value = get_env_value(key)
    return int(value) if value else default


def parse_env_bool(key: str, default: bool) -> bool:
    value = get_env_value(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_vector_store_config() -> dict[str, object]:
    chroma_directory = get_env_value("AI_RAG_VECTOR_STORE__CHROMA__PERSIST_DIRECTORY")

    return {
        "provider": get_env_value("AI_RAG_VECTOR_STORE__PROVIDER") or "chroma",
        "top_k": parse_env_int("AI_RAG_VECTOR_STORE__TOP_K", 5),
        "products": {
            "collection_name": get_env_value("AI_RAG_VECTOR_STORE__PRODUCTS__COLLECTION_NAME") or "products",
            "index_name": get_env_value("AI_RAG_VECTOR_STORE__PRODUCTS__INDEX_NAME") or "ai-rag-products",
        },
        "reviews": {
            "collection_name": get_env_value("AI_RAG_VECTOR_STORE__REVIEWS__COLLECTION_NAME") or "reviews",
            "index_name": get_env_value("AI_RAG_VECTOR_STORE__REVIEWS__INDEX_NAME") or "ai-rag-reviews",
        },
        "chroma": {
            "persist_directory": Path(chroma_directory) if chroma_directory else CHROMA_DIR,
        },
        "elasticsearch": {
            "url": get_env_value("AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL") or "http://localhost:9200",
            "username": get_env_value("AI_RAG_VECTOR_STORE__ELASTICSEARCH__USERNAME"),
            "password": get_env_value("AI_RAG_VECTOR_STORE__ELASTICSEARCH__PASSWORD"),
            "api_key": get_env_value("AI_RAG_VECTOR_STORE__ELASTICSEARCH__API_KEY"),
            "verify_certs": parse_env_bool("AI_RAG_VECTOR_STORE__ELASTICSEARCH__VERIFY_CERTS", True),
            "request_timeout_seconds": parse_env_int(
                "AI_RAG_VECTOR_STORE__ELASTICSEARCH__REQUEST_TIMEOUT_SECONDS",
                30,
            ),
            "index_prefix": get_env_value("AI_RAG_VECTOR_STORE__ELASTICSEARCH__INDEX_PREFIX") or "ai-rag",
        },
    }


MODEL_ROUTING_CONFIG = load_model_routing_config()["models"]
MODEL_API_KEYS = load_api_keys()
VECTOR_STORE_VALUES = load_vector_store_config()


class ModelRoutingConfig(BaseModel):
    simple: ModelEndpointConfig
    moderate: ModelEndpointConfig
    complex: ModelEndpointConfig
    fallback_order: tuple[Literal["simple", "moderate", "complex"], ...] = (
        "simple",
        "moderate",
        "complex",
    )


class ChromaConfig(BaseModel):
    persist_directory: Path = CHROMA_DIR


class ElasticsearchConfig(BaseModel):
    url: str = "http://localhost:9200"
    username: str | None = None
    password: str | None = None
    api_key: str | None = None
    verify_certs: bool = True
    request_timeout_seconds: int = Field(default=30, ge=1)
    index_prefix: str = "ai-rag"


class VectorNamespaceConfig(BaseModel):
    collection_name: str
    index_name: str


class VectorStoreConfig(BaseModel):
    provider: Literal["chroma", "elasticsearch"] = "chroma"
    top_k: int = Field(default=5, ge=1)
    products: VectorNamespaceConfig = Field(
        default_factory=lambda: VectorNamespaceConfig(
            collection_name="products",
            index_name="ai-rag-products",
        )
    )
    reviews: VectorNamespaceConfig = Field(
        default_factory=lambda: VectorNamespaceConfig(
            collection_name="reviews",
            index_name="ai-rag-reviews",
        )
    )
    chroma: ChromaConfig = ChromaConfig()
    elasticsearch: ElasticsearchConfig = ElasticsearchConfig()


class SessionConfig(BaseModel):
    sqlite_path: Path = SQLITE_PATH
    timeout_minutes: int = Field(default=30, ge=1)
    window_size: int = Field(default=10, ge=1)
    cleanup_batch_size: int = Field(default=100, ge=1)


class AppSettings(BaseSettings):
    app_name: str = "ai-rag-project"
    environment: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    data_dir: Path = DATA_DIR
    vector_store: VectorStoreConfig = Field(default_factory=lambda: VectorStoreConfig(**VECTOR_STORE_VALUES))
    session: SessionConfig = SessionConfig()
    models: ModelRoutingConfig = Field(
        default_factory=lambda: ModelRoutingConfig(
            simple=ModelEndpointConfig(
                provider=str(MODEL_ROUTING_CONFIG["simple"]["provider"]),
                model_name=str(MODEL_ROUTING_CONFIG["simple"]["model_name"]),
                api_base=str(MODEL_ROUTING_CONFIG["simple"]["api_base"]),
                api_key=MODEL_API_KEYS["simple"],
                supports_streaming=bool(MODEL_ROUTING_CONFIG["simple"].get("supports_streaming", False)),
            ),
            moderate=ModelEndpointConfig(
                provider=str(MODEL_ROUTING_CONFIG["moderate"]["provider"]),
                model_name=str(MODEL_ROUTING_CONFIG["moderate"]["model_name"]),
                api_base=str(MODEL_ROUTING_CONFIG["moderate"]["api_base"]),
                api_key=MODEL_API_KEYS["moderate"],
                supports_streaming=bool(MODEL_ROUTING_CONFIG["moderate"].get("supports_streaming", False)),
            ),
            complex=ModelEndpointConfig(
                provider=str(MODEL_ROUTING_CONFIG["complex"]["provider"]),
                model_name=str(MODEL_ROUTING_CONFIG["complex"]["model_name"]),
                api_base=str(MODEL_ROUTING_CONFIG["complex"]["api_base"]),
                api_key=MODEL_API_KEYS["complex"],
                supports_streaming=bool(MODEL_ROUTING_CONFIG["complex"].get("supports_streaming", False)),
            ),
        )
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="AI_RAG_",
        extra="ignore",
    )


settings = AppSettings()
