from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import Any, TypeVar, cast

import chromadb
from chromadb.api.models.Collection import Collection
from pydantic import BaseModel, Field

from backend.config.settings import AppSettings, VectorNamespaceConfig, settings

try:
    from elasticsearch import Elasticsearch
except ModuleNotFoundError:  # pragma: no cover
    Elasticsearch = None  # type: ignore[assignment]


VectorMetadata = dict[str, Any]
MetadataValue = str | int | float | bool
SUPPORTED_NAMESPACES = ("products", "reviews")


class VectorStoreDocument(BaseModel):
    """描述进入向量库的一条标准化文档。"""

    id: str
    content: str
    metadata: VectorMetadata = Field(default_factory=dict)
    embedding: list[float] | None = None


class VectorSearchResult(BaseModel):
    """描述一次向量检索命中的文档和得分。"""

    document: VectorStoreDocument
    score: float | None = None


class VectorStoreHealth(BaseModel):
    """描述向量后端可用性探活结果。"""

    provider: str
    available: bool
    detail: str | None = None


class LocalHashingEmbedder:
    """提供无外部依赖的本地哈希向量化实现。"""

    def __init__(self, dimensions: int = 256) -> None:
        """初始化本地哈希向量器的维度。"""
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """将文本编码为归一化稀疏向量。"""
        normalized = text.strip().lower()
        vector = [0.0] * self.dimensions
        tokens = self._tokenize(normalized)

        if not tokens:
            vector[0] = 1.0
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector

        return [value / magnitude for value in vector]

    def _tokenize(self, text: str) -> list[str]:
        """将文本切分为英文 token、中文单字与 n-gram。"""
        ascii_tokens = re.findall(r"[a-z0-9]+", text)
        cjk_sequences = re.findall(r"[\u4e00-\u9fff]+", text)
        cjk_chars = [char for sequence in cjk_sequences for char in sequence]
        ngrams: list[str] = []

        for sequence in cjk_sequences:
            if len(sequence) > 1:
                ngrams.append(sequence)
            for size in (2, 3):
                if len(sequence) < size:
                    continue
                for index in range(len(sequence) - size + 1):
                    ngrams.append(sequence[index : index + size])

        return ascii_tokens + cjk_chars + ngrams


class VectorStore(ABC):
    """统一抽象不同向量后端的能力接口。"""

    def __init__(self, app_settings: AppSettings) -> None:
        """初始化向量库基类配置与内置嵌入器。"""
        self.settings = app_settings
        self.config = app_settings.vector_store
        self._embedder = LocalHashingEmbedder()

    @abstractmethod
    def ensure_collections(self) -> None:
        """创建或校验后端所需命名空间。"""

    @abstractmethod
    def upsert_documents(self, namespace: str, documents: list[VectorStoreDocument]) -> None:
        """在指定命名空间写入或更新文档。"""

    @abstractmethod
    def search(
        self,
        namespace: str,
        query: str,
        top_k: int | None = None,
        filters: VectorMetadata | None = None,
    ) -> list[VectorSearchResult]:
        """在指定命名空间执行语义检索。"""

    @abstractmethod
    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        """按文档 ID 删除指定命名空间下的数据。"""

    @abstractmethod
    def healthcheck(self) -> VectorStoreHealth:
        """返回向量后端可用性与连通性信息。"""

    def resolve_namespace_config(self, namespace: str) -> VectorNamespaceConfig:
        """解析命名空间对应的配置对象。"""
        if namespace not in SUPPORTED_NAMESPACES:
            raise ValueError(
                f"Unsupported namespace '{namespace}'. Expected one of: {', '.join(SUPPORTED_NAMESPACES)}."
            )
        return cast(VectorNamespaceConfig, getattr(self.config, namespace))

    def build_embedding(self, text: str) -> list[float]:
        """构建文本向量。"""
        return self._embedder.embed(text)

    def normalize_metadata(self, metadata: VectorMetadata) -> dict[str, MetadataValue]:
        """将 metadata 规范化为后端可序列化的标量字典。"""
        normalized: dict[str, MetadataValue] = {}
        for key, value in metadata.items():
            if isinstance(value, bool | str | int | float):
                normalized[key] = value
            elif value is not None:
                normalized[key] = str(value)
        return normalized


VectorStoreType = TypeVar("VectorStoreType", bound=VectorStore)


class ChromaVectorStore(VectorStore):
    """Chroma 向量库实现。"""

    def __init__(self, app_settings: AppSettings) -> None:
        """初始化 Chroma 客户端与集合缓存。"""
        super().__init__(app_settings)
        persist_directory = self.config.chroma.persist_directory
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collections: dict[str, Collection] = {}

    def ensure_collections(self) -> None:
        """确保 Chroma 中存在 products/reviews 集合。"""
        for namespace in SUPPORTED_NAMESPACES:
            namespace_config = self.resolve_namespace_config(namespace)
            self._collections[namespace] = self._client.get_or_create_collection(
                name=namespace_config.collection_name
            )

    def upsert_documents(self, namespace: str, documents: list[VectorStoreDocument]) -> None:
        """批量 upsert 文档到 Chroma 集合。"""
        collection = self._get_collection(namespace)
        if not documents:
            return
        ids = [document.id for document in documents]
        text_documents = [document.content for document in documents]
        metadatas = [self.normalize_metadata(document.metadata) for document in documents]
        embeddings = [
            document.embedding if document.embedding is not None else self.build_embedding(document.content)
            for document in documents
        ]
        collection.upsert(ids=ids, documents=text_documents, metadatas=metadatas, embeddings=embeddings)

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int | None = None,
        filters: VectorMetadata | None = None,
    ) -> list[VectorSearchResult]:
        """在 Chroma 中执行向量检索并转换统一结果格式。"""
        collection = self._get_collection(namespace)
        query_embedding = self.build_embedding(query)
        query_result = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k or self.config.top_k,
            where=self.normalize_metadata(filters or {}) or None,
        )
        ids = query_result.get("ids", [[]])[0]
        documents = query_result.get("documents", [[]])[0]
        metadatas = query_result.get("metadatas", [[]])[0]
        distances = query_result.get("distances", [[]])[0]

        results: list[VectorSearchResult] = []
        for index, document_id in enumerate(ids):
            distance = distances[index] if index < len(distances) else None
            score = None if distance is None else 1.0 / (1.0 + float(distance))
            content = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
            results.append(
                VectorSearchResult(
                    document=VectorStoreDocument(
                        id=document_id,
                        content=content,
                        metadata=cast(dict[str, MetadataValue], metadata),
                    ),
                    score=score,
                )
            )
        return results

    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        """从 Chroma 集合删除给定文档 ID。"""
        if not ids:
            return
        collection = self._get_collection(namespace)
        collection.delete(ids=ids)

    def healthcheck(self) -> VectorStoreHealth:
        """检查 Chroma 客户端可用性。"""
        try:
            self._client.list_collections()
        except Exception as exc:
            return VectorStoreHealth(provider="chroma", available=False, detail=str(exc))
        return VectorStoreHealth(provider="chroma", available=True)

    def _get_collection(self, namespace: str) -> Collection:
        """获取集合实例，不存在时自动初始化。"""
        if namespace not in self._collections:
            self.ensure_collections()
        return self._collections[namespace]


class ElasticsearchVectorStore(VectorStore):
    """Elasticsearch 向量库实现。"""

    def __init__(self, app_settings: AppSettings, client: Any | None = None) -> None:
        """初始化 Elasticsearch 客户端。"""
        super().__init__(app_settings)
        self._client = client or self._build_client()

    def ensure_collections(self) -> None:
        """确保 Elasticsearch 中目标索引存在。"""
        for namespace in SUPPORTED_NAMESPACES:
            self._ensure_index(namespace)

    def upsert_documents(self, namespace: str, documents: list[VectorStoreDocument]) -> None:
        """通过 bulk API 批量写入文档。"""
        if not documents:
            return
        index_name = self._ensure_index(namespace)
        operations: list[dict[str, Any]] = []
        for document in documents:
            operations.append({"index": {"_index": index_name, "_id": document.id}})
            operations.append(
                {
                    "content": document.content,
                    "embedding": document.embedding or self.build_embedding(document.content),
                    "metadata": self.normalize_metadata(document.metadata),
                    "namespace": namespace,
                }
            )
        response = self._client.bulk(operations=operations, refresh=True)
        if response.get("errors"):
            raise RuntimeError(f"Elasticsearch bulk upsert failed: {response}")

    def search(
        self,
        namespace: str,
        query: str,
        top_k: int | None = None,
        filters: VectorMetadata | None = None,
    ) -> list[VectorSearchResult]:
        """执行基于 cosineSimilarity 的向量检索。"""
        index_name = self._ensure_index(namespace)
        response = self._client.search(
            index=index_name,
            query={
                "script_score": {
                    "query": self._build_filter_query(filters),
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": self.build_embedding(query)},
                    },
                }
            },
            size=top_k or self.config.top_k,
            source=["content", "metadata", "namespace"],
        )
        results: list[VectorSearchResult] = []
        for hit in response.get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            results.append(
                VectorSearchResult(
                    document=VectorStoreDocument(
                        id=str(hit.get("_id", "")),
                        content=str(source.get("content", "")),
                        metadata=cast(dict[str, MetadataValue], source.get("metadata", {})),
                    ),
                    score=float(hit["_score"]) if isinstance(hit.get("_score"), int | float) else None,
                )
            )
        return results

    def delete_documents(self, namespace: str, ids: list[str]) -> None:
        """通过 bulk delete 删除索引中的文档。"""
        if not ids:
            return
        index_name = self.resolve_index_name(namespace)
        if not self._client.indices.exists(index=index_name):
            return
        operations = [{"delete": {"_index": index_name, "_id": doc_id}} for doc_id in ids]
        response = self._client.bulk(operations=operations, refresh=True)
        if response.get("errors"):
            failed = [
                item.get("delete", {}).get("_id", "unknown")
                for item in response.get("items", [])
                if "error" in item.get("delete", {})
            ]
            raise RuntimeError(f"Elasticsearch delete failed for IDs: {failed}")

    def healthcheck(self) -> VectorStoreHealth:
        """通过 ping 检查 Elasticsearch 可用性。"""
        try:
            available = bool(self._client.ping())
        except Exception as exc:
            return VectorStoreHealth(provider="elasticsearch", available=False, detail=str(exc))
        detail = None if available else "Elasticsearch ping returned False."
        return VectorStoreHealth(provider="elasticsearch", available=available, detail=detail)

    def resolve_index_name(self, namespace: str) -> str:
        """根据命名空间配置与前缀规则计算索引名。"""
        namespace_config = self.resolve_namespace_config(namespace)
        configured_name = namespace_config.index_name.strip()
        prefix = self.config.elasticsearch.index_prefix.strip("-")
        if prefix and configured_name in {namespace, namespace_config.collection_name}:
            return f"{prefix}-{configured_name}"
        return configured_name

    def _ensure_index(self, namespace: str) -> str:
        """确保索引存在，不存在则按映射创建。"""
        index_name = self.resolve_index_name(namespace)
        if self._client.indices.exists(index=index_name):
            return index_name
        self._client.indices.create(
            index=index_name,
            mappings={
                "properties": {
                    "content": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": self._embedder.dimensions, "index": False},
                    "metadata": {"type": "flattened"},
                    "namespace": {"type": "keyword"},
                }
            },
            settings={"number_of_shards": 1, "number_of_replicas": 0},
        )
        return index_name

    def _build_client(self) -> Any:
        """按配置构造 Elasticsearch 客户端实例。"""
        if Elasticsearch is None:
            raise ModuleNotFoundError(
                "The 'elasticsearch' package is not installed. Install backend/requirements.txt "
                "or inject a client when constructing ElasticsearchVectorStore."
            )
        elasticsearch_config = self.config.elasticsearch
        client_kwargs: dict[str, Any] = {
            "hosts": [elasticsearch_config.url],
            "verify_certs": elasticsearch_config.verify_certs,
            "request_timeout": elasticsearch_config.request_timeout_seconds,
        }
        if elasticsearch_config.api_key:
            client_kwargs["api_key"] = elasticsearch_config.api_key
        elif elasticsearch_config.username:
            client_kwargs["basic_auth"] = (
                elasticsearch_config.username,
                elasticsearch_config.password or "",
            )
        return Elasticsearch(**client_kwargs)

    def _build_filter_query(self, filters: VectorMetadata | None) -> dict[str, Any]:
        """将 metadata 过滤条件转换为 Elasticsearch bool 过滤语句。"""
        normalized_filters = self.normalize_metadata(filters or {})
        if not normalized_filters:
            return {"match_all": {}}
        return {"bool": {"filter": [{"term": {f"metadata.{key}": value}} for key, value in normalized_filters.items()]}}


class VectorStoreFactory:
    """维护 provider 到具体向量库实现的注册关系。"""

    _registry: dict[str, type[VectorStore]] = {}

    @classmethod
    def register(cls, provider: str, store_cls: type[VectorStoreType]) -> None:
        """注册向量后端实现。"""
        cls._registry[provider] = store_cls

    @classmethod
    def create(cls, app_settings: AppSettings | None = None) -> VectorStore:
        """按配置创建对应的向量后端实例。"""
        resolved_settings = app_settings or settings
        provider = resolved_settings.vector_store.provider
        store_cls = cls._registry.get(provider)
        if store_cls is None:
            raise NotImplementedError(
                f"Vector store provider '{provider}' is not registered yet. "
                "Implement the provider and register it with VectorStoreFactory.register()."
            )
        return store_cls(resolved_settings)


VectorStoreFactory.register("chroma", ChromaVectorStore)
VectorStoreFactory.register("elasticsearch", ElasticsearchVectorStore)
