import json
import os
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FILES_DIR = DATA_DIR / "files"
CHROMA_DIR = DATA_DIR / ".chroma"
LEGACY_SQLITE_PATH = BASE_DIR / "memory" / "sessions.db"
SQLITE_PATH = DATA_DIR / "sessions.db"
ENV_FILE = BASE_DIR / ".env"
MODEL_ROUTING_FILE = Path(__file__).resolve().parent / "model_routing.json"
MODEL_ROUTING_KEYS = ("simple", "moderate", "complex", "embedding", "rerank")

load_dotenv(ENV_FILE)
ENV_VALUES = dotenv_values(ENV_FILE)


class ModelEndpointConfig(BaseModel):
    provider: str
    model_name: str
    api_base: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    supports_streaming: bool = False
    timeout_seconds: int = Field(default=30, ge=1)
    max_tokens: int = Field(default=1024, ge=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class EmbeddingModelConfig(BaseModel):
    provider: str
    model_name: str
    api_base: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    dimensions: int = Field(default=256, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)


class RerankModelConfig(BaseModel):
    provider: str
    model_name: str
    api_base: str | None = None
    api_key_env: str | None = None
    api_key: str | None = None
    top_n: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=30, ge=1)


DASHSCOPE_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_env_value(key: str) -> str | None:
    """优先从系统环境变量读取配置，缺失时回退到 .env。"""
    value = os.getenv(key)
    if value is not None:
        return value
    return ENV_VALUES.get(key)


def load_model_routing_config() -> dict[str, dict[str, dict[str, object]]]:
    """加载模型路由配置；缺失时显式失败。"""
    if not MODEL_ROUTING_FILE.exists():
        raise FileNotFoundError(f"Model routing config file is required: {MODEL_ROUTING_FILE}")

    with MODEL_ROUTING_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_model_api_key_envs(
    models: dict[str, dict[str, object]],
) -> dict[str, str]:
    """从模型路由配置解析每个模型的 API Key 环境变量名。"""
    api_key_envs: dict[str, str] = {}
    for model_key in MODEL_ROUTING_KEYS:
        model_config = models.get(model_key)
        if model_config is None:
            raise KeyError(f"Missing model routing entry: {model_key}")
        api_key_env = model_config.get("api_key_env")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError(f"Missing api_key_env for model routing entry: {model_key}")
        api_key_envs[model_key] = api_key_env
    return api_key_envs


def load_api_keys(models: dict[str, dict[str, object]] | None = None) -> dict[str, str | None]:
    """读取模型路由涉及的 API Key。"""
    resolved_models = models or load_model_routing_config()["models"]
    api_key_envs = resolve_model_api_key_envs(resolved_models)
    return {model_key: get_env_value(api_key_env) for model_key, api_key_env in api_key_envs.items()}


def parse_env_int(key: str, default: int) -> int:
    """将环境变量解析为整数，缺失时返回默认值。"""
    value = get_env_value(key)
    return int(value) if value else default


def parse_env_bool(key: str, default: bool) -> bool:
    """将环境变量解析为布尔值，支持常见 truthy 文本。"""
    value = get_env_value(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_env_json(key: str) -> dict[str, object] | None:
    """将 JSON 环境变量解析为字典，缺失时返回 None。"""
    value = get_env_value(key)
    if value is None or not value.strip():
        return None
    return json.loads(value)


def resolve_backend_runtime_path(value: str | Path | None, default: Path) -> Path:
    """解析运行时路径；相对路径默认以 backend 目录为基准。"""
    if value is None or not str(value).strip():
        return default

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    if path.parts and path.parts[0] == BASE_DIR.name:
        return BASE_DIR.parent / path
    return BASE_DIR / path


def load_vector_store_config() -> dict[str, object]:
    """汇总向量库相关配置（provider、命名空间、后端参数）。"""
    chroma_directory = get_env_value("AI_RAG_VECTOR_STORE__CHROMA__PERSIST_DIRECTORY")
    knowledge_sources = parse_env_json("AI_RAG_VECTOR_STORE__KNOWLEDGE_SOURCES")

    return {
        "provider": get_env_value("AI_RAG_VECTOR_STORE__PROVIDER") or "chroma",
        "top_k": parse_env_int("AI_RAG_VECTOR_STORE__TOP_K", 5),
        "knowledge_sources": knowledge_sources
        or {
            "products": {
                "collection_name": get_env_value("AI_RAG_VECTOR_STORE__PRODUCTS__COLLECTION_NAME") or "products",
                "index_name": get_env_value("AI_RAG_VECTOR_STORE__PRODUCTS__INDEX_NAME") or "ai-rag-products",
            },
            "reviews": {
                "collection_name": get_env_value("AI_RAG_VECTOR_STORE__REVIEWS__COLLECTION_NAME") or "reviews",
                "index_name": get_env_value("AI_RAG_VECTOR_STORE__REVIEWS__INDEX_NAME") or "ai-rag-reviews",
            },
            "orders": {
                "collection_name": get_env_value("AI_RAG_VECTOR_STORE__ORDERS__COLLECTION_NAME") or "orders",
                "index_name": get_env_value("AI_RAG_VECTOR_STORE__ORDERS__INDEX_NAME") or "ai-rag-orders",
            },
        },
        "documents": {
            "index_name": get_env_value("AI_RAG_VECTOR_STORE__DOCUMENTS__INDEX_NAME") or "documents",
        },
        "chunks": {
            "index_name": get_env_value("AI_RAG_VECTOR_STORE__CHUNKS__INDEX_NAME") or "chunks",
        },
        "chroma": {
            "persist_directory": resolve_backend_runtime_path(chroma_directory, CHROMA_DIR),
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


def load_app_runtime_config() -> dict[str, object]:
    """汇总应用运行时配置。"""
    active_scene = get_env_value("AI_RAG_APP__ACTIVE_SCENE")
    return {
        "active_scene": active_scene or "generic_assistant",
    }

class ModelRoutingConfig(BaseModel):
    simple: ModelEndpointConfig
    moderate: ModelEndpointConfig
    complex: ModelEndpointConfig
    embedding: EmbeddingModelConfig
    rerank: RerankModelConfig
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


class DocumentIndexConfig(BaseModel):
    """描述文档管理索引的基础命名配置。"""

    index_name: str


class VectorStoreConfig(BaseModel):
    provider: Literal["chroma", "elasticsearch"] = "chroma"
    top_k: int = Field(default=5, ge=1)
    knowledge_sources: dict[str, VectorNamespaceConfig] = Field(
        default_factory=lambda: {
            "products": VectorNamespaceConfig(
                collection_name="products",
                index_name="ai-rag-products",
            ),
            "reviews": VectorNamespaceConfig(
                collection_name="reviews",
                index_name="ai-rag-reviews",
            ),
            "orders": VectorNamespaceConfig(
                collection_name="orders",
                index_name="ai-rag-orders",
            ),
        }
    )
    documents: DocumentIndexConfig = Field(
        default_factory=lambda: DocumentIndexConfig(index_name="documents")
    )
    chunks: DocumentIndexConfig = Field(default_factory=lambda: DocumentIndexConfig(index_name="chunks"))
    chroma: ChromaConfig = ChromaConfig()
    elasticsearch: ElasticsearchConfig = ElasticsearchConfig()


class SessionConfig(BaseModel):
    sqlite_path: Path = SQLITE_PATH
    timeout_minutes: int = Field(default=30, ge=1)
    window_size: int = Field(default=10, ge=1)
    cleanup_batch_size: int = Field(default=100, ge=1)


class AppRuntimeConfig(BaseModel):
    """描述应用运行时装配配置。"""

    active_scene: str = "generic_assistant"


class AppSettings(BaseSettings):
    app_name: str = "ai-rag-project"
    environment: str = "development"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)

    data_dir: Path = DATA_DIR
    app: AppRuntimeConfig = Field(default_factory=lambda: AppRuntimeConfig(**load_app_runtime_config()))
    vector_store: VectorStoreConfig = Field(default_factory=lambda: VectorStoreConfig(**load_vector_store_config()))
    session: SessionConfig = SessionConfig()
    models: ModelRoutingConfig = Field(
        default_factory=lambda: build_model_routing_settings()
    )

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix="AI_RAG_",
        extra="ignore",
    )


def build_model_routing_settings() -> ModelRoutingConfig:
    """根据当前文件与环境变量构建模型路由配置。"""
    models = load_model_routing_config()["models"]
    api_key_envs = resolve_model_api_key_envs(models)
    api_keys = load_api_keys(models)
    # 统一从 model_routing.json 读取模型声明，环境变量只负责注入密钥值。
    return ModelRoutingConfig(
        simple=ModelEndpointConfig(
            provider=str(models["simple"]["provider"]),
            model_name=str(models["simple"]["model_name"]),
            api_base=str(models["simple"]["api_base"]),
            api_key_env=api_key_envs["simple"],
            api_key=api_keys["simple"],
            supports_streaming=bool(models["simple"].get("supports_streaming", False)),
        ),
        moderate=ModelEndpointConfig(
            provider=str(models["moderate"]["provider"]),
            model_name=str(models["moderate"]["model_name"]),
            api_base=str(models["moderate"]["api_base"]),
            api_key_env=api_key_envs["moderate"],
            api_key=api_keys["moderate"],
            supports_streaming=bool(models["moderate"].get("supports_streaming", False)),
        ),
        complex=ModelEndpointConfig(
            provider=str(models["complex"]["provider"]),
            model_name=str(models["complex"]["model_name"]),
            api_base=str(models["complex"]["api_base"]),
            api_key_env=api_key_envs["complex"],
            api_key=api_keys["complex"],
            supports_streaming=bool(models["complex"].get("supports_streaming", False)),
        ),
        embedding=EmbeddingModelConfig(
            provider=str(models["embedding"]["provider"]),
            model_name=str(models["embedding"]["model_name"]),
            api_base=str(models["embedding"].get("api_base") or DASHSCOPE_API_BASE),
            api_key_env=api_key_envs["embedding"],
            api_key=api_keys["embedding"],
            dimensions=int(models["embedding"].get("dimensions", 256)),
            timeout_seconds=int(models["embedding"].get("timeout_seconds", 30)),
            max_retries=int(models["embedding"].get("max_retries", 3)),
        ),
        rerank=RerankModelConfig(
            provider=str(models["rerank"]["provider"]),
            model_name=str(models["rerank"]["model_name"]),
            api_base=str(models["rerank"].get("api_base") or DASHSCOPE_API_BASE),
            api_key_env=api_key_envs["rerank"],
            api_key=api_keys["rerank"],
            top_n=int(models["rerank"].get("top_n", 3)),
            timeout_seconds=int(models["rerank"].get("timeout_seconds", 30)),
        ),
    )


settings = AppSettings()
