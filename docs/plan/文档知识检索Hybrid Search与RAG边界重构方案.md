# 文档知识检索 Hybrid Search 与 RAG 边界重构方案

## Summary

本次方案同时解决两个问题：

- 为文档知识检索补齐正式的 Hybrid Search 关键词召回能力
- 重新划清 `platform.knowledge` 与 `platform.rag` 的领域边界

新的固定原则如下：

- `platform.knowledge` 只负责知识库管理与存储相关职责：数据处理、切块、文档发布、向量存储适配、底层仓储访问
- `platform.rag` 统一负责知识召回相关职责：检索 API、召回算法、查询编排、Hybrid Search、LangChain retriever 适配
- 一期只落地 `documents/chunks` 的 Hybrid Search
- 后端完整透出检索调试字段，前端本期可以不展示

这不是“把现有检索代码挪位置”的小调整，而是一次按读写职责拆分的边界治理：知识管理归 `knowledge`，知识召回归 `rag`。

## Key Changes

### 1. `knowledge` 与 `rag` 的职责边界重划分

`platform.knowledge` 保留：

- 文档加载、预处理、切块、版本发布、索引生命周期管理
- `VectorStoreDocument` 这一类存储模型
- Chroma / Elasticsearch 的底层 provider 适配
- 面向存储的仓储接口

`platform.knowledge` 移除：

- 文档召回入口，例如 `search_document_chunks()`
- Hybrid Search 排序逻辑
- 关键词召回逻辑
- embedding 策略与查询改写类召回算法职责

`platform.rag` 新增并统一承接：

- 文档知识检索统一入口
- 语义召回器
- 关键词召回器
- 融合排序器
- LangChain `BaseRetriever` 适配
- 后续 `products/reviews/orders` 的统一召回扩展模式

固定边界约束：

- `knowledge` 可以回答“数据怎么存、怎么读底层记录”
- `rag` 才能回答“数据怎么召回、怎么排序、怎么暴露检索 API 给上层”

### 2. `knowledge` 层重构为纯仓储与知识管理层

当前 `store.py` 中混合的职责需要拆开，改成“底层仓储 + provider 实现”的结构。

需要保留或新增的接口：

- 文档写侧仓储接口
  - 文档主记录写入、读取、列表
  - chunk 写入、激活、失活、删除
- 文档向量读侧仓储接口
  - 接收 `query_embedding`、`top_k`、`namespace`
  - 返回 chunk 与原始向量分数
- 文档源文本读侧仓储接口
  - 枚举活跃 chunk 文本与元数据
  - 供关键词召回器构建 BM25 索引

provider 层要求：

- Chroma / Elasticsearch 都实现相同语义的底层仓储接口
- provider 只做数据访问，不做 Hybrid Search
- provider 层不再直接暴露文档召回业务 API

`VectorStoreFactory` 继续保留，但职责调整为：

- 创建文档管理仓储
- 创建底层向量查询仓储
- 创建活跃 chunk 文本源仓储

不再只返回一个“大而全”的复合 store 给上层直接当检索器使用。

### 3. `rag` 层新增文档知识检索统一入口

在 `platform.rag` 下新增文档检索模块，固定组合如下：

- `EmbeddingStrategy`
  - 迁入 `rag`
  - 一期默认继续使用本地 hashing embedding，保证本地可运行

- `DocumentSemanticRetriever`
  - 依赖 `knowledge` 提供的文档向量读侧仓储
  - 负责 query embedding 与向量召回

- `DocumentKeywordRetriever`
  - 依赖 `knowledge` 提供的活跃 chunk 文本源仓储
  - 使用 LangChain `BM25Retriever`
  - 新增依赖：`langchain-community`、`rank-bm25`

- `HybridFusionRanker`
  - 从 `knowledge` 迁入 `rag`
  - 只负责两路结果融合，不负责召回
  - 默认权重固定为 `vector_weight=0.65`、`keyword_weight=0.35`

- `DocumentHybridRetriever`
  - 作为 LangChain `BaseRetriever`
  - 组合 semantic / keyword / fusion 三部分
  - 成为文档知识检索的标准入口

- `DocumentRetrievalService`
  - 面向 scene 与 tool 暴露统一文档检索 API
  - scene/tool 不再直接依赖 `knowledge` 中的文档检索方法

### 4. 文档知识检索一期行为定义

一期仅覆盖 `documents/chunks`。

固定行为如下：

- 向量召回：按当前 `top_k` 返回语义命中结果
- 关键词召回：基于活跃 chunk 文本集合构建 BM25 检索
- BM25 候选规模：`max(top_k * 10, 20)`
- 融合策略：沿用当前权重 `0.65 / 0.35`
- 低相关过滤：继续保留，默认按融合分数过滤

检索结果必须保留：

- `document_id`
- `chunk_id`
- `chunk_index`
- `score`
- `vector_score`
- `keyword_score`
- `vector_rank`
- `keyword_rank`
- `matched_by`

其中：

- `score` 保持兼容，表示最终融合分数
- `matched_by` 取值固定为：
  - `["vector"]`
  - `["keyword"]`
  - `["vector", "keyword"]`

### 5. Scene / Tool / API 调用面调整

调用边界统一改为：

- `scenes/*` 只依赖 `platform.rag` 的 retriever 或 retrieval service
- 文档检索工具 `knowledge_document_search` 只调用 `platform.rag`
- `knowledge_service.search_document_chunks()` 不再保留

上层行为不变要求：

- `/chat` 主流程不改
- `generic_assistant` 仍以文档检索为默认首轮
- 电商场景若复用文档检索，也必须走 `platform.rag` 的文档检索入口

## Public APIs / Types

需要调整或新增的公开契约如下：

- 统一检索结果结构新增字段：
  - `vector_score`
  - `keyword_score`
  - `vector_rank`
  - `keyword_rank`
  - `matched_by`

- `/chat` 的 `citations` 追加上述可选调试字段

- `GET /sessions/{session_id}` 返回的 `retrieval_snippets` 追加上述可选字段

- 历史兼容策略固定为：
  - 旧记录缺失新字段时，后端归一化补 `None` 或空数组
  - 不破坏旧 session 详情读取

内部类型调整要求：

- 检索结果模型迁入 `platform.rag`
- `knowledge` 中保留存储模型，不再保留召回结果融合模型
- provider 返回原始命中数据，`rag` 负责映射为统一检索结果结构

## Test Plan

必须覆盖以下场景：

- `knowledge` 仓储测试
  - provider 只验证底层数据访问能力
  - 文档主记录与 chunk 生命周期保持正确
  - 向量查询与活跃 chunk 枚举返回稳定结果

- `rag` 检索测试
  - 关键词强匹配问题能召回预期 chunk
  - 纯向量命中时 `matched_by=["vector"]`
  - 纯关键词命中时 `matched_by=["keyword"]`
  - 双路命中时 `matched_by=["vector","keyword"]`
  - 融合后仍保留 `document_id/chunk_id/chunk_index`

- scene / tool 测试
  - `knowledge_document_search` 改走 `platform.rag`
  - `generic_assistant` 仍能完成文档问答
  - 电商场景复用文档检索时不回退到 `knowledge` 检索入口

- API 与 session 测试
  - `/chat` 返回 `citations` 包含新增调试字段
  - `retrieval_snippets` 能写入并读回新增字段
  - 旧格式 `retrieval_snippets` 仍能兼容读取

- 验收测试
  - 准备唯一关键词文档 chunk
  - 用关键词强匹配问题执行一次文档检索
  - 确认结果中出现预期文档或片段
  - 确认 `matched_by` 至少包含 `keyword`

## Assumptions

- 一期只做文档知识检索，不扩展 `products/reviews/orders` 的 Hybrid Search 实现
- 但二期扩展要求必须记录到文档中：
  - 这些知识源的召回 API 也必须统一进入 `platform.rag`
  - 不允许再把召回算法塞回 `knowledge` 或 scene service
- 前端本期不展示调试字段，但后端接口与历史存储必须完整支持
- 继续保持本地可运行，不引入在线 embedding 依赖前提
- 文档说明后续还需同步到 `README.md`、`docs/api-list.md`、`docs/data-model.md` 与相关架构图

## Implementation Conclusion

截至 2026-05-20，`documents/chunks` 的一期 Hybrid Search 已经落地，且 `/chat`、scene tool 与 session history 已接入并通过测试。

已确认落地的实现点：

- `platform.rag` 已新增文档检索模块：
  - `DocumentEmbeddingStrategy`
  - `DocumentSemanticRetriever`
  - `DocumentKeywordRetriever`
  - `HybridFusionRanker`
  - `DocumentHybridRetriever`
  - `DocumentRetrievalService`
- `DocumentKeywordRetriever` 已使用 LangChain `BM25Retriever`，并通过活跃 chunk 文本源构建关键词候选。
- `generic_assistant` 与 `ecommerce` 复用的 `knowledge_document_search` 已统一改走 `platform.rag.DocumentRetrievalService`。
- `/chat` 的 `citations` 与 `GET /sessions/{session_id}` 的 `retrieval_snippets` 已透传 `vector_score`、`keyword_score`、`vector_rank`、`keyword_rank`、`matched_by`。
- 测试已覆盖 provider 分层行为、Hybrid Search、chat/session 兼容与关键词强匹配验收场景。

对应测试文件：

- `backend/tests/test_document_hybrid_retrieval.py`
- `backend/tests/test_knowledge_chroma.py`
- `backend/tests/test_knowledge_elasticsearch.py`
- `backend/tests/test_chat_api.py`
- `backend/tests/test_session_store.py`
- `backend/tests/test_agentic_retrieval.py`

## Implementation Deviations

本次实现已经完成“一期文档 Hybrid Search 可用”和“scene/tool 不再直接依赖文档召回业务 API”这两个主要目标，但 `knowledge` 与 `rag` 的边界还没有达到设计文档最理想的“彻底解耦”状态，当前仍有以下偏差：

- `DocumentEmbeddingStrategy` 仍复用 `backend/platform/knowledge/base/store.py` 中的 `LocalHashingEmbedder`，embedding 具体实现尚未完全移出 `knowledge.base`。
- `DocumentKeywordScoreCalculator` 仍复用 `LocalHashingEmbedder._tokenize()` 作为 fallback overlap scoring 的分词来源。
- `DocumentRetrievalService` 仍依赖 `backend/platform/knowledge/base/relevance.py` 中的低相关过滤与受管文档过滤辅助函数。
- `VectorStoreFactory` 虽然已经提供 `create_document_repository()`、`create_document_chunk_vector_repository()`、`create_active_document_chunk_source()` 等分职责工厂方法，但当前 provider 仍返回同一个复合 `VectorStore` 实例，而不是物理分离的独立仓储实现。
- `backend/platform/knowledge/base/store.py` 仍保留 `KnowledgeRetriever` 抽象与 `VectorStore(KnowledgeRetriever, KnowledgeDocumentRepository, DocumentChunkVectorRepository, ActiveDocumentChunkSource)` 这一复合类型，因此底层代码层面的聚合抽象尚未完全拆除。

这意味着当前状态更准确的表述是：

- 文档召回入口、Hybrid Search 算法与对外检索契约已经迁入 `platform.rag`
- `knowledge` 不再作为 scene/chat 的文档召回入口
- 但 `rag` 仍复用少量 `knowledge.base` 的基础实现与 helper，底层 provider 也仍以单对象多接口方式承载多个职责

## Phase 2 Constraint

二期如果为 `products`、`reviews`、`orders` 增加关键词召回、Hybrid Search、融合排序或 rerank：

- 新能力必须进入 `platform.rag`
- scene service 只能依赖 `platform.rag` 暴露的统一检索入口或 retriever/service
- 不允许把召回算法重新塞回 `platform.knowledge`
- 不允许把商品、评论、订单的 Hybrid Search 直接堆叠在 `scenes/ecommerce/*` 服务中
