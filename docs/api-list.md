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
  - Body `stream`: 是否流式，布尔值；当前接口保留字段，传 `true` 会报未支持。
  - Body `top_k`: 检索条数上限，可选，1-20。
- 返回结构：
  - `session_id`: 会话 ID。
  - `request_id`: 本次请求 ID。
  - `answer`: 模型回答。
  - `knowledge_used`: 是否使用了知识检索结果。
  - `scene`: 当前响应所属场景。
  - `agent`: 代理/角色标识，可为空。
  - `citations`: 引用列表，每项包含 `citation_id`、`namespace`、`snippet`、`score`。

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
- 返回结构：
  - `session_id`: 新建会话 ID。
  - `scene`: 会话绑定场景。

### `GET /sessions/{session_id}`

- 一句话说明：查询指定会话的详情和历史轮次。
- 主要入参：
  - Path `session_id`: 会话 ID。
  - Query `limit`: 返回最近多少轮会话，默认 `20`，范围 `1-100`。
- 返回结构：
  - `session_id`: 会话 ID。
  - `scene`: 会话所属场景。
  - `total_turns`: 历史总轮数。
  - `turns`: 轮次列表，每项包含 `request_id`、`user_message`、`assistant_answer`、`retrieval_snippets`、`timestamp`。

### `DELETE /sessions/{session_id}`

- 一句话说明：删除指定会话及其全部历史消息。
- 主要入参：
  - Path `session_id`: 会话 ID。
- 返回结构：
  - `session_id`: 被删除的会话 ID。
  - `deleted_turns`: 被删除的历史轮次数量。

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
