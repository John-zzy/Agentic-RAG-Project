# API List

本文基于 `backend/application/runtime/api/*/routes.py` 与对应 `schemas.py` 整理当前项目实际暴露的 REST 接口。

## Chat And Session

### `GET /health`

- 一句话说明：健康检查接口，确认服务已启动。
- 主要入参：无。
- 返回结构：`{ status }`，其中 `status` 固定为 `"ok"`。

### `POST /chat`

- 一句话说明：统一聊天入口，执行场景绑定的检索增强问答并返回答案。
- 主要入参：
  - Body `message`: 用户输入文本，必填，1-4000 字符。
  - Body `session_id`: 会话 ID，可选；不传时由服务侧按默认逻辑处理。
  - Body `stream`: 是否流式，布尔值；传 `false` 或不传时返回 JSON，传 `true` 时返回 `text/event-stream`。
- 返回结构：
  - 非流式 `stream=false`：
    - `session_id`: 会话 ID。
    - `request_id`: 本次请求 ID。
    - `answer`: 模型回答；有 citations 时应包含可见引用编号，若模型未生成编号，服务会在尾部补 `参考来源：[1][2]`。
    - `knowledge_used`: 是否使用了知识检索结果。
    - `scene`: 当前响应所属场景。
    - `agent`: 代理/角色标识，可为空。
    - `citations`: 统一引用列表，每项包含 `index`、`citation_id`、`namespace`、`source_kind`、`source_name`、`source_path`、`document_id`、`chunk_id`、`chunk_index`、`snippet`、`score`、`vector_score`、`keyword_score`、`vector_rank`、`keyword_rank`、`matched_by`、`rank`。
  - 流式 `stream=true`：
    - 响应头为 `Content-Type: text/event-stream`。
    - 事件类型为 `start`、`history`、`tool`、`chunk`、`done`、`error`，且 `data` 一律为 JSON。
    - `start.data` 包含 `session_id`、`request_id`、`knowledge_used`、`scene`、`agent`。
    - `history.data` 包含 `session_id`、`request_id`、`window_size`、`message_count`、`messages`，用于暴露本轮注入模型前的历史消息窗口；`messages` 每项包含 `type` 与 `content`。
    - `tool.data` 包含 retrieval 阶段的结构化结果，固定补充 `session_id`、`request_id`、`knowledge_used`、`citations`；当前常见字段还包括 `stage`、`mode`、`retrieval_policy`、`candidate_tools`、`documents`、`rounds`。
    - `chunk.data` 包含 `delta`，表示最终回答文本增量。
    - `done.data` 与非流式 `ChatResponse` 结构一致，客户端应以 `done.answer` 作为最终权威文本。
    - `error.data` 包含 `code`、`message`、`request_id`。

补充说明：

- `stream=true` 只会对“最终回答生成阶段”按 `chunk` 推送文本增量；但在生成回答前，服务端会先发送 `history` 和 `tool` 事件用于暴露可观测上下文。
- `/chat` 请求体不再接收检索数量参数；检索数量、最低相关性阈值、召回策略和 ReRank 接入位由当前 scene 的 `retrieval_policy` 控制。
- `tool.data.retrieval_policy` 只包含可观测的策略配置摘要：`top_k`、`min_relevance_score`、`recall_strategy`、`no_hit_strategy`、`rerank_enabled`、`rerank_top_n`。
- 命中知识时，成功路径事件顺序通常为 `start -> history -> tool -> chunk... -> done`。
- 无知识命中时，仍返回 SSE 成功态，事件顺序为 `start -> history -> tool -> chunk -> done`，其中 `tool.documents = 0`，`done.knowledge_used = false`。
- 失败路径事件顺序为 `start -> history -> tool -> error`。

### `GET /scenes`

- 一句话说明：返回当前运行时支持的场景列表和默认场景。
- 主要入参：无。
- 返回结构：
  - `default_scene`: 默认场景标识。
  - `scenes`: 场景列表，每项包含 `scene`、`name`、`description`、`is_default`。

### `POST /sessions`

- 一句话说明：创建新会话，并将会话绑定到指定或默认场景。
- 主要入参：
  - Body `scene`: 场景标识，可选；不传则使用默认场景。
  - Body `mounted_knowledge_sources`: 会话允许使用的知识源列表，可选；当前支持 `documents`、`ecommerce`，默认 `["documents"]`。
- 返回结构：
  - `session_id`: 新建会话 ID。
  - `scene`: 会话绑定场景。
  - `mounted_knowledge_sources`: 规范化后的挂载知识源列表，去重并按稳定顺序返回。
- 常见错误：
  - `400 UNKNOWN_SCENE`: 请求的 `scene` 未注册。
  - `400 INVALID_MOUNTED_KNOWLEDGE_SOURCES`: 传入了未知知识源值。

### `GET /sessions/{session_id}`

- 一句话说明：查询指定会话的详情和最近消息视图。
- 主要入参：
  - Path `session_id`: 会话 ID。
  - Query `limit`: 返回最近多少条消息，默认 `20`，范围 `1-100`。
- 返回结构：
  - `session_id`: 会话 ID。
  - `scene`: 会话所属场景。
  - `mounted_knowledge_sources`: 会话挂载的知识源列表；历史会话缺失该字段时默认回填 `["documents"]`。
  - `total_messages`: 历史总消息数。
  - `messages`: 最近消息列表，每项包含：
    - `type`: 消息类型，例如 `human`、`ai`。
    - `content`: 消息内容。
    - `request_id`: 所属请求 ID。
    - `timestamp`: 写入时间。
    - `knowledge_used`: 仅 assistant 消息可用，表示该回答是否使用了知识检索。
    - `citations`: 仅 assistant 消息可用，对应结构化引用列表；兼容旧历史中的 `retrieval_snippets`，会按当前 citation 契约做字段补齐。

补充说明：

- 当前上传文档检索已经由 `platform.rag.DocumentRetrievalService` 统一承接。
- `documents/chunks` 会执行 Hybrid Search：语义召回 + BM25 关键词召回 + 融合排序。
- 当前 session 详情已经贴近 `chat_messages` 主数据模型，而不是旧的 turn 兼容视图。
- 新增调试字段主要用于后端引用透传、message history 持久化和排障，本期前端页面可不展示。

### `DELETE /sessions/{session_id}`

- 一句话说明：删除指定会话及其全部历史消息。
- 主要入参：
  - Path `session_id`: 会话 ID。
- 返回结构：
  - `session_id`: 被删除的会话 ID。
  - `deleted_messages`: 被删除的消息数量。

## Evals

统一前缀：`/evals`

### `GET /evals/latest`

- 一句话说明：读取最新 eval artifact 的 UI 安全视图。
- 主要入参：无。
- 返回结构：
  - `run`: sanitized eval run payload。
- 常见错误：
  - `404 EVAL_LATEST_NOT_FOUND`: 尚未生成 `backend/data/evals/latest.json`。

### `GET /evals/runs`

- 一句话说明：列出历史 eval run。
- 主要入参：无。
- 返回结构：
  - `runs`: 历史 run 摘要列表；无 `runs/index.json` 时返回空列表。

### `GET /evals/runs/{run_id}`

- 一句话说明：读取指定历史 eval run 的 UI 安全视图。
- 主要入参：
  - Path `run_id`: run ID，仅允许普通文件名字符，不允许路径穿越。
- 返回结构：
  - `run`: sanitized eval run payload。
- 常见错误：
  - `404 EVAL_RUN_NOT_FOUND`: run 不存在或 `run_id` 非法。

### `GET /evals/runs/{run_id}/status`

- 一句话说明：查询后台 eval run 状态。
- 主要入参：
  - Path `run_id`: run ID。
- 返回结构：
  - `run_id`: run ID。
  - `sample_set`: 样本集，可为空。
  - `status`: `queued`、`running`、`succeeded`、`failed` 或 `not_found`。
  - `started_at`、`finished_at`、`error`: 状态辅助字段。

### `POST /evals/runs`

- 一句话说明：后台触发 allowlist 内的 eval run，默认运行 `retrieval_benchmark`。
- 主要入参：
  - Body `sample_set`: 可选，仅允许 `minimal` 或 `retrieval_benchmark`。
- 返回结构：
  - 与 status 响应一致，HTTP 状态码为 `202`。
- 常见错误：
  - `422 EVAL_SAMPLE_SET_NOT_ALLOWED`: 样本集不在 allowlist。
  - `409 EVAL_RUN_ALREADY_RUNNING`: 当前进程已有 eval run queued/running。

补充说明：

- `/evals` API 只读取 `backend/data/evals/latest.json`、`backend/data/evals/runs/index.json` 和 `backend/data/evals/runs/<run_id>.json`，不会读取任意路径。
- API 响应会递归移除 `snippet`、`content`、`prompt`、`reason`、`rewrite_reason`、`raw_fixture_content`、完整 `answer` 等文本字段；CLI artifact 仍可保留完整调试信息。
- `POST /evals/runs` 的 `base_url` 从当前请求推导，客户端不能传入任意外部 URL。

## File Management

### `POST /files/upload`

- 一句话说明：上传本地知识文件到服务端文件目录。
- 主要入参：
  - Form `file`: 上传文件本体，支持扩展名 `json`、`txt`、`md`、`csv`、`pdf`、`docx`、`xlsx`。
- 返回结构：
  - `filename`: 原始文件名。
  - `file_path`: 相对文件路径。
  - `file_size`: 文件大小，字节数。
  - `content_type`: 文件 MIME 类型。
  - `upload_time`: 上传时间，ISO 格式。

### `GET /files/`

- 一句话说明：列出当前已上传且受支持的文件。
- 主要入参：无。
- 返回结构：
  - `files`: 文件列表，每项包含 `filename`、`file_path`、`file_size`、`content_type`、`created_time`。

### `DELETE /files/{filename}`

- 一句话说明：删除指定上传文件。
- 主要入参：
  - Path `filename`: 文件名。
- 返回结构：
  - `success`: 是否删除成功。
  - `message`: 文本说明。
  - `filename`: 被删除文件名。

### `GET /files/download/{filename}`

- 一句话说明：下载指定上传文件。
- 主要入参：
  - Path `filename`: 文件名。
- 返回结构：文件流响应，按文件类型返回对应 `media_type`，下载文件名为原始文件名。

## Knowledge Documents

统一前缀：`/knowledge/documents`

### `POST /knowledge/documents/preprocess-preview`

- 一句话说明：在正式入库前预览知识文档的预处理结果。
- 主要入参：
  - Body `namespace`: 知识命名空间，必填。
  - Body `source_path`: 源文件路径，必填。
  - Body `processing_rules`: 本次启用的规则 ID 列表，可选，默认空列表。
  - Body `chunk_size`: 预览使用的切块大小，可选，`> 0`；缺省时使用数据预处理模块默认值。
  - Body `chunk_overlap`: 预览使用的切块重叠长度，可选，`>= 0` 且必须小于 `chunk_size`；缺省时使用数据预处理模块默认值。
- 返回结构：
  - `namespace`、`source_path`、`source_type`、`chunk_size`、`chunk_overlap`。
  - `supported_rules`: 当前文件类型支持的规则列表，每项包含 `rule_id`、`display_name`、`description`、`supported_source_types`、`level`。
  - `selected_rules`: 本次实际生效的规则定义列表。
  - `processing_stats`: 处理统计，包含 `raw_record_count`、`processed_record_count`、`removed_record_count`、`raw_char_count`、`processed_char_count`。
  - `original_samples`、`processed_samples`: 预览样本列表，每项包含 `sample_index`、`source_record_id`、`record_index`、`content`、`content_hash`、`applied_rules`、`dropped`。
  - `can_index`: 当前文件是否允许继续入库。
  - `warnings`: 结构化 warning 列表，每项包含 `code`、`message`、`severity`、`source_record_id`、`record_index`。

### `POST /knowledge/documents`

- 一句话说明：按给定预处理规则注册知识文档并建立索引版本。
- 主要入参：
  - Body `namespace`: 知识命名空间，必填。
  - Body `source_path`: 源文件路径，必填。
  - Body `processing_rules`: 本次启用的规则 ID 列表，可选，默认空列表。
  - Body `chunk_size`: 切块大小，可选，`> 0`；缺省时使用数据预处理模块默认值。
  - Body `chunk_overlap`: 切块重叠长度，可选，`>= 0` 且必须小于 `chunk_size`；缺省时使用数据预处理模块默认值。
  - Body `keep_version`: 是否保留旧版本，默认 `false`。
- 返回结构：
  - 文档详情字段：`document_id`、`namespace`、`source_path`、`status`、`active_version`、`chunk_count`、`updated_at`、`source_type`、`chunk_size`、`chunk_overlap`、`processing_rules`、`processing_stats`、`provenance_enabled`、`last_error`、`versions`。
  - 额外字段 `document_version`: 本次生成的文档版本号。

### `GET /knowledge/documents`

- 一句话说明：按命名空间筛选并列出知识文档。
- 主要入参：
  - Query `namespace`: 命名空间，可选。
- 返回结构：
  - `documents`: 文档列表，每项包含 `document_id`、`namespace`、`source_path`、`status`、`source_type`、`processing_rules`、`processing_stats`、`provenance_enabled`、`active_version`、`chunk_count`、`updated_at`。

### `GET /knowledge/documents/files`

- 一句话说明：按上传文件维度聚合展示索引状态。
- 主要入参：
  - Query `namespace`: 命名空间，可选。
- 返回结构：
  - `items`: 文件索引状态列表，每项包含 `filename`、`source_path`、`file_size`、`created_at`、`namespace`、`document_id`、`indexed`、`status`、`active_version`、`chunk_count`、`updated_at`、`last_error`、`can_index`。
  - 状态补充：可处理但未入库文件返回 `awaiting_processing`；当前不支持预处理的文件返回 `unsupported`。

### `GET /knowledge/documents/{document_id}`

- 一句话说明：读取单个知识文档详情。
- 主要入参：
  - Path `document_id`: 文档 ID。
- 返回结构：
  - `document_id`、`namespace`、`source_path`、`status`、`active_version`、`chunk_count`、`updated_at`。
  - `source_type`、`chunk_size`、`chunk_overlap`、`processing_rules`、`processing_stats`、`provenance_enabled`、`last_error`。
  - `versions`: 版本列表，每项包含 `document_version`、`status`、`chunk_count`、`chunk_size`、`chunk_overlap`、`created_at`、`source_type`、`processing_rules`、`processing_stats`、`provenance_enabled`、`last_error`。

### `DELETE /knowledge/documents/{document_id}`

- 一句话说明：软删除指定知识文档。
- 主要入参：
  - Path `document_id`: 文档 ID。
- 返回结构：
  - 与知识文档写操作响应一致，包含文档详情字段和 `document_version`，用于表示删除后的最新状态。

### `POST /knowledge/documents/{document_id}/rechunk`

- 一句话说明：沿用当前活动版本的处理规则，按新的切块参数重建指定知识文档的分块与索引版本。
- 主要入参：
  - Path `document_id`: 文档 ID。
  - Body `chunk_size`: 新切块大小，可选，`> 0`；缺省时使用数据预处理模块默认值。
  - Body `chunk_overlap`: 新切块重叠长度，可选，`>= 0` 且必须小于 `chunk_size`；缺省时使用数据预处理模块默认值。
  - Body `keep_version`: 是否保留旧版本，默认 `false`。
- 返回结构：
  - 与注册接口一致，返回最新文档详情和 `document_version`。

### `POST /knowledge/documents/{document_id}/reprocess`

- 一句话说明：按新的预处理规则或切块参数重跑指定知识文档，并生成新的活动版本。
- 主要入参：
  - Path `document_id`: 文档 ID。
  - Body `processing_rules`: 本次启用的规则 ID 列表，可选，默认空列表。
  - Body `chunk_size`: 新切块大小，可选，`> 0`；缺省时使用数据预处理模块默认值。
  - Body `chunk_overlap`: 新切块重叠长度，可选，`>= 0` 且必须小于 `chunk_size`；缺省时使用数据预处理模块默认值。
  - Body `keep_version`: 是否保留旧版本，默认 `false`。
- 返回结构：
  - 与注册接口一致，返回最新文档详情和 `document_version`。
