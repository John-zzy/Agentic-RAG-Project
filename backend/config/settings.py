import json
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / ".chroma"
SQLITE_PATH = BASE_DIR / "memory" / "sessions.db"
ENV_FILE = BASE_DIR / ".env"
MODEL_ROUTING_FILE = BASE_DIR / "config" / "model_routing.json"

load_dotenv(ENV_FILE)


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
    env_values = dotenv_values(ENV_FILE)
    return {
        "simple": env_values.get("AI_RAG_MODELS__SIMPLE__API_KEY"),
        "moderate": env_values.get("AI_RAG_MODELS__MODERATE__API_KEY"),
        "complex": env_values.get("AI_RAG_MODELS__COMPLEX__API_KEY"),
    }


MODEL_ROUTING_CONFIG = load_model_routing_config()["models"]
MODEL_API_KEYS = load_api_keys()


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
    products_collection: str = "products"
    reviews_collection: str = "reviews"
    top_k: int = Field(default=5, ge=1)


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
    chroma: ChromaConfig = ChromaConfig()
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
