# ReRank 模型接入功能最小闭环实施方案

## Summary

- 目标：完成真实 ReRank 最小闭环，并同步把 embedding / rerank 纳入 `model_routing.json` 统一模型配置。
- 技术选择：ReRank 使用 LangChain `DashScopeRerank` 包装器；embedding 使用 LangChain/OpenAI-compatible embedding 客户端，由项目 `llm` 模型层统一创建。
- 默认行为：`rerank_enabled=false` 时保持现有排序、citations、no-hit fallback 不变；开启后只对已过滤候选重排，并按 `rerank_top_n` 截断证据。
- 兼容策略：embedding 默认用 `text-embedding-v4` 且 `dimensions=256`，保持当前向量维度边界；已有旧 embedding 数据需要重新发布/重建索引后才能保证语义一致。

## Key Changes

- 扩展 `backend/platform/config/model_routing.json`：
  - 在 `models` 下新增 `embedding`：`provider=dashscope`、`model_name=text-embedding-v4`、`api_base=https://dashscope.aliyuncs.com/compatible-mode/v1`、`dimensions=256`。
  - 在 `models` 下新增 `rerank`：`provider=dashscope`、`model_name=qwen3-rerank`、`top_n` 可选，API key 从 `AI_RAG_MODELS__RERANK__API_KEY` 读取。
  - 保留 `simple/moderate/complex` 现有 LLM 配置和行为。
- 扩展模型层 `backend/platform/models/llm/`：
  - 保留 `ModelClient` 聊天模型职责，新增 embedding client/factory 与 rerank compressor factory。
  - 新增通用模型路由读取能力，支持按 `chat/simple/moderate/complex`、`embedding`、`rerank` 获取配置。
  - API key 缺失时在实际调用模型前抛出明确错误，不静默回退。
- 接入 embedding：
  - 将知识入库、文档分块索引、语义检索查询向量统一改为模型层提供的 embedding strategy。
  - 移除业务路径中对 `LocalHashingEmbedder` 的默认依赖；测试可通过注入 fake embedding 或测试配置使用本地策略。
  - Elasticsearch mapping 维度继续按 embedding strategy 的 `dimensions` 创建。
- 接入 ReRank：
  - 新增真实 `RetrievalReranker` 实现，内部使用 LangChain `DashScopeRerank`。
  - 输入为 query + 过滤后的 `RetrievalResult` 候选；输出按 rerank 结果重排 `records/documents/citations`。
  - 写入 `rerank_score` 到 record、document metadata、citation metadata。
  - `RerankTrace` 增加 `model`、`fallback_reason`、`error` 字段；失败、超时、空结果全部回退原候选顺序并记录 trace/log。
  - `AgenticRetriever._apply_rerank()` 捕获 reranker 异常，不影响 no-hit fallback 和 citations 可信边界。

## Public Interfaces

- `/chat` 与 SSE 的 `retrieval_trace.rounds[].rerank` 将包含：
  - `enabled`、`provider`、`model`、`applied`、`input_count`、`output_count`、`top_n`、`fallback_reason`、`error`。
- citations metadata 可能新增：
  - `rerank_score`，仅在真实 rerank 成功应用时出现。
- `.env.example` 新增：
  - `AI_RAG_MODELS__EMBEDDING__API_KEY`
  - `AI_RAG_MODELS__RERANK__API_KEY`

## Test Plan

- 模型配置：
  - 验证 `model_routing.json` 可加载 `embedding` 和 `rerank`。
  - 验证缺 API key 时 embedding/rerank client 抛出清晰错误。
- Embedding：
  - 用 fake embedding client 验证入库、查询、Elasticsearch mapping 维度来自模型配置。
  - 验证现有 Hybrid Search 流程仍走语义召回 + keyword + fusion + relevance filter。
- ReRank：
  - fake LangChain reranker 返回倒序结果，验证 records/documents/citations 同步重排并写入 `rerank_score`。
  - `rerank_enabled=false` 验证现有顺序、citations、no-hit fallback 不变。
  - `rerank_enabled=true + rerank_top_n=1` 验证只保留 1 条证据进入 citations。
  - reranker 异常、超时、空结果验证回退原顺序，并在 trace/log 中记录原因。
- API/SSE：
  - 验证 `/chat` 和 streaming `tool/done` 均暴露 rerank trace。
  - 验证低相关过滤后无证据时仍返回 `knowledge_used=false` 和空 citations。

## Assumptions

- ReRank provider 固定先接 DashScope LangChain 包装器，不直接调用裸 HTTP 或 SDK。
- embedding 本次同步迁移到统一模型路由；为了降低存储兼容风险，默认维度选 `256`。
- 已有 Chroma/Elasticsearch 向量数据在切换真实 embedding 后需要重新发布或重建，否则旧 hashing 向量与新模型向量不可比较。
- 参考资料：
  - DashScope embedding 文档：https://platform.qianwenai.com/docs/api-reference/text-embedding/dashscope-embedding
  - LangChain reranker 集成模式：https://docs.langchain.com/oss/python/integrations/document_transformers/cross_encoder_reranker
