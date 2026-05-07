## Context

当前项目已经具备以下基础：

- `backend/knowledge/base/store.py` 提供 `VectorStore` 抽象，并支持 `chroma` 与 `elasticsearch`
- `backend/knowledge/ecommerce/loader.py` 已经从 `backend/data` 下读取 JSON 数据，并直接注册为商品/评价知识库
- `backend/memory/base/session_store.py` 中的 SQLite 仅承担会话记忆，不承担知识库存储

因此，本次设计不引入新的数据库类型，而是在现有基础上明确“我的文档知识库”的存储边界、索引组织方式和文档生命周期。

## Goals / Non-Goals

**Goals**

- 保留当前“直接注册进知识库”的主流程，不要求先上传再入库
- 原始 JSON 文档统一放在 `backend/data/files` 或其子目录
- 向量检索、chunk 元数据和文档主记录统一放在 Elasticsearch
- 检索结果至少追溯到“源文件中的具体记录”
- 默认覆盖式更新；显式指定时支持保留历史版本
- 删除知识库文档时默认不删除原始 JSON 文件

**Non-Goals**

- 不引入 SQLite 作为知识库文档元数据主存储
- 不实现用户认证、多租户、权限控制和审计
- 不在本次变更中扩展 PDF、Word 等复杂格式解析
- 不改动现有聊天 API 的请求/响应契约

## Decisions

### 1. 采用“文件系统 + Elasticsearch”双层存储

- 原始 JSON 文档保存在 `backend/data/files/<namespace>/...`
- Elasticsearch 承担文档主记录、版本摘要、chunk 元数据和向量检索索引
- SQLite 不参与“我的文档知识库”的文档或 chunk 存储

这样设计的原因是：

- 与当前项目已有 JSON 数据组织方式一致，迁移成本低
- 文件系统更适合保存原始依据
- ES 已经是向量检索后端，继续承载文档管理元数据可以避免双写漂移

### 2. 保持“直接注册进知识库”为主入口

- 现有项目中的 JSON 文件可以直接注册进知识库
- 后续即使增加上传接口，也只作为补充入口，不改变当前主流程

这样可以保证现有演示链路和数据组织方式不被打断。

### 3. `document_id` 按 `namespace + source_path` 稳定生成

- 同一路径文档在同一 `namespace` 下始终映射到同一个 `document_id`
- 默认覆盖更新时不新建 `document_id`
- 版本变化通过 `document_version` 表达

这是最简单的实现方式，因为它不需要额外维护“路径到文档 ID”的映射表，并天然适配当前“按文件注册”的主流程。

### 4. 默认覆盖，显式保留版本

- 文档唯一性默认按 `namespace + source_path` 识别
- 重复注册同一路径文档时，默认覆盖旧活动版本
- 若显式指定保留版本，则创建新的 `document_version`，旧版本保留但默认不参与召回

这是一个折中设计：默认行为简单，版本能力可用，但不会把所有日常更新都复杂化。

### 5. 追溯粒度至少到记录级

每个 chunk 必须带上以下最小追溯字段：

- `document_id`
- `document_version`
- `namespace`
- `source_type`
- `source_path`
- `source_record_id`
- `chunk_id`
- `chunk_index`
- `updated_at`

其中 `source_record_id` 必须稳定，能定位到源文件中的具体记录。这样检索结果才能明确回答“这一段内容来自哪个文件中的哪条记录”。

### 6. Elasticsearch 使用双索引组织

采用两个索引，保持职责单一：

- `documents` 索引：只存文档主记录和活动版本摘要
- `chunks` 索引：只存可检索 chunk、向量和追溯 metadata

不采用“主记录和 chunk 混在一个索引”的方案，因为那会让列表、详情、版本切换和删除逻辑更复杂。

### 7. 删除知识库文档时默认不删原始文件

- 删除操作默认只清理 Elasticsearch 中的文档记录和 chunk
- `backend/data/files` 下原始 JSON 文件保留

这样可以避免误删原始依据，也便于后续重新注册和调试。

## Data Model

### Documents Index

`documents` 索引中的主记录建议包含：

- `document_id`
- `namespace`
- `source_type`
- `source_path`
- `status`
- `active_version`
- `chunk_count`
- `created_at`
- `updated_at`
- `last_error`

### Document Version Summary

版本摘要建议作为文档记录的一部分，或以单独子结构保存：

- `document_version`
- `is_active`
- `chunk_size`
- `chunk_overlap`
- `chunk_count`
- `source_snapshot_hash`
- `created_at`

### Chunks Index

`chunks` 索引中的记录建议包含：

- `chunk_id`
- `document_id`
- `document_version`
- `namespace`
- `content`
- `embedding`
- `source_path`
- `source_record_id`
- `chunk_index`
- `is_active`
- `metadata`

## Operational Flow

### Register

1. 输入 `namespace`、`source_path`、`chunk_size`、`chunk_overlap`、`keep_version`
2. 读取 `backend/data/files/...` 下 JSON
3. 为每条源记录生成稳定的 `source_record_id`
4. 切分记录内容并生成稳定的 `chunk_id`
5. 若 `keep_version=false`，先失活或删除旧活动版本 chunk
6. 将新版本 chunk 写入 `chunks` 索引
7. 更新 `documents` 索引中的 `active_version`、`chunk_count` 和状态

### Delete

1. 按 `document_id` 查询文档主记录和活动版本
2. 删除或失活活动版本关联 chunk
3. 删除或标记失效文档主记录
4. 不删除 `backend/data/files` 下原始文件

### Rechunk

1. 输入新的 `chunk_size`、`chunk_overlap`
2. 基于原始文件重新生成 chunk
3. 默认覆盖旧活动版本
4. 若显式保留版本，则保留旧版本并切换新的 `active_version`

## Risks / Trade-offs

- 同步注册大文件会阻塞请求。MVP 阶段可通过限制文件大小和记录数控制风险。
- 覆盖式更新如果“删旧成功、写新失败”，可能导致短暂不可检索。需要用状态位和明确错误信息暴露失败状态。
- ES 同时承担文档记录和 chunk 存储，查询模型更统一，但需要额外注意索引结构和过滤条件一致性。
## Implementation Notes

本变更已在 `codex/manage-knowledge-documents` 分支实现，最终实现补充了以下工程约束：

- 文档服务发布新版本时采用失败安全顺序：先写入新 chunk，再发布新文档记录，再失活旧活动版本 chunk。若 chunk 写入、记录发布、旧 chunk 失活或回滚清理任一步失败，服务会尽量恢复旧活动记录与旧活动 chunk，避免出现“文档记录指向新版本但无可检索 chunk”的状态。
- `keep_version=true` 时会保留历史版本记录，但默认活动检索仅保留新版本 chunk 为 active；旧版本 chunk 会失活，避免默认检索命中历史版本。
- FastAPI 只在访问 `/knowledge/documents` 或测试显式注入时懒加载 `KnowledgeDocumentService`，避免默认聊天 API 启动路径被 Elasticsearch 文档索引初始化阻塞。
- API 错误响应使用稳定结构化 detail，区分参数校验、文档不存在、存储后端错误和未知后端错误；未知错误不会把原始异常文本直接暴露给静态测试页。
- 默认测试仍不依赖本地 Elasticsearch；Elasticsearch live 行为继续通过 `@pytest.mark.integration` 测试覆盖。
