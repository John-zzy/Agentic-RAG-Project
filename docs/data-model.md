# Data Model

本文基于以下来源整理项目核心数据模型：

- 持久化表结构：`backend/platform/memory/base/session_store.py`
- 知识文档模型：`backend/platform/knowledge/documents/*.py`、`backend/platform/knowledge/base/store.py`
- API DTO：`backend/application/runtime/api/*/schemas.py`
- 电商演示数据：`backend/data/orders.json`、`backend/data/products.json`、`backend/data/reviews.json`

## 总览

这个项目没有传统 ORM 实体层，也没有独立的数据库建表 SQL 文件。

核心数据主要分为三类：

1. SQLite 会话数据
   - 用于聊天会话、轮次历史、LangChain message 历史。
2. 知识文档索引数据
   - 用于知识文档主记录、版本信息、文档分块及向量检索元数据。
3. 电商场景演示业务数据
   - 用于 `ecommerce` 场景的商品、评论、订单和售后工单。

## 一、SQLite 持久化模型

说明：

- 这些表由 `SQLiteSessionStore._ensure_schema()` 在运行时创建。
- SQL 中没有显式 `FOREIGN KEY` 约束，但存在明确的逻辑关联。

### 1. `sessions`

一句话说明：聊天会话主表，记录会话归属场景、状态和活跃时间。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `session_id` | `TEXT` | PK | 会话主键，UUID 字符串。 |
| `scene` | `TEXT` |  | 会话绑定场景，如 `generic_assistant`、`ecommerce`。 |
| `status` | `TEXT` | 枚举 | 会话状态。 |
| `created_at` | `TEXT` |  | 会话创建时间，ISO 8601。 |
| `updated_at` | `TEXT` |  | 最近一次更新记录时间。 |
| `last_active_at` | `TEXT` |  | 最近活跃时间，用于过期清理。 |
| `expired_at` | `TEXT` | 可空 | 会话过期时间。 |

枚举值：

- `status`: `active`、`expired`

### 2. `chat_turns`

一句话说明：按“一问一答”保存聊天轮次以及引用片段。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INTEGER` | PK | 自增主键。 |
| `session_id` | `TEXT` | 逻辑 FK -> `sessions.session_id` | 所属会话。 |
| `request_id` | `TEXT` |  | 本轮请求 ID。 |
| `user_message` | `TEXT` |  | 用户输入。 |
| `assistant_answer` | `TEXT` |  | 助手回答。 |
| `retrieval_snippets` | `TEXT` |  | JSON 字符串，保存引用片段列表。 |
| `created_at` | `TEXT` |  | 本轮创建时间，ISO 8601。 |

### 3. `chat_messages`

一句话说明：为 LangChain `BaseChatMessageHistory` 适配而持久化的消息级历史表。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `id` | `INTEGER` | PK | 自增主键。 |
| `session_id` | `TEXT` | 逻辑 FK -> `sessions.session_id` | 所属会话。 |
| `request_id` | `TEXT` |  | 对应一次请求或一次消息批次。 |
| `message_type` | `TEXT` |  | 消息类型，如 human / ai。 |
| `message_payload` | `TEXT` |  | LangChain 消息 JSON。 |
| `created_at` | `TEXT` |  | 写入时间。 |
| `sequence_index` | `INTEGER` | 唯一索引一部分 | 同一 `request_id` 内的顺序。 |

补充约束：

- 唯一索引：`(session_id, request_id, sequence_index)`

## 二、知识文档模型

说明：

- 这部分没有单独关系型表，而是抽象为“文档主记录 + 文档版本 + 文档分块”。
- 主记录和分块最终会落到 Chroma 或 Elasticsearch。
- 文档版本在实现上嵌入 `versions` 数组中，不是独立物理表；下文按逻辑实体描述。

### 4. `KnowledgeDocumentRecord`

一句话说明：知识文档的主记录，描述某个源文件当前激活版本和索引状态。

来源：

- `KnowledgeDocumentService._build_record()`
- `VectorStore.upsert_document_record()`

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_id` | `str` | PK | 文档主键，基于 `namespace + source_path` 的 SHA256。 |
| `namespace` | `str` |  | 业务命名空间，如 `products`、`reviews`、`orders`。 |
| `source_type` | `str` |  | 源文件类型；当前实现写死为 `json`。 |
| `source_path` | `str` |  | 相对数据根目录的源文件路径。 |
| `status` | `str` | 枚举 | 文档当前状态。 |
| `active_version` | `int` |  | 当前激活版本号。 |
| `chunk_count` | `int` |  | 当前激活版本的分块数。 |
| `chunk_size` | `int` |  | 当前激活版本的切块大小。 |
| `chunk_overlap` | `int` |  | 当前激活版本的切块重叠长度。 |
| `created_at` | `str` |  | 文档首次入库时间。 |
| `updated_at` | `str` |  | 最近一次版本切换时间。 |
| `last_error` | `str \| null` | 可空 | 最近一次失败原因。 |
| `versions` | `list[KnowledgeDocumentVersion]` | 逻辑子实体 | 版本历史列表。 |

枚举值：

- `status`: `active`、`failed`、`deleted`

### 5. `KnowledgeDocumentVersion`

一句话说明：知识文档某个版本的分块参数和结果摘要。

来源：

- `KnowledgeDocumentVersionSummary`
- `KnowledgeDocumentService._build_version()`

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_version` | `int` | 逻辑复合键一部分 | 版本号，从 1 递增。 |
| `status` | `str` | 枚举 | 该版本状态。 |
| `chunk_count` | `int` |  | 该版本产生的分块数。 |
| `chunk_size` | `int` |  | 该版本切块大小。 |
| `chunk_overlap` | `int` |  | 该版本切块重叠长度。 |
| `created_at` | `str` |  | 版本创建时间。 |
| `last_error` | `str \| null` | 可空 | 该版本失败原因。 |

枚举值：

- `status`: `active`、`failed`、`deleted`

### 6. `DocumentRecord`

一句话说明：从单个源文件中读取出的标准化原始记录，是切块前的中间模型。

来源：

- `backend/platform/knowledge/documents/schemas.py`
- `load_document_records()`

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `namespace` | `str` |  | 所属命名空间。 |
| `source_path` | `str` |  | 源文件路径。 |
| `source_record_id` | `str` | PK（逻辑） | 源记录稳定 ID，基于路径、位置和内容生成。 |
| `record_index` | `int` |  | 在源文件中的记录序号。 |
| `content` | `str` |  | 参与向量化和切块的文本内容。 |
| `record` | `dict[str, Any]` |  | 原始结构化记录。 |

### 7. `DocumentChunk`

一句话说明：文档切块后的标准模型，是写入向量库的最小单元。

来源：

- `backend/platform/knowledge/documents/schemas.py`
- `build_document_chunks()`

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `chunk_id` | `str` | PK | 分块主键，基于文档、版本、源记录和内容生成。 |
| `chunk_index` | `int` |  | 分块序号。 |
| `content` | `str` |  | 分块正文。 |
| `metadata` | `dict[str, Any]` |  | 分块追踪元数据。 |

`metadata` 关键字段：

| 元字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_id` | `str` | 逻辑 FK -> `KnowledgeDocumentRecord.document_id` | 所属文档。 |
| `document_version` | `int` | 逻辑 FK -> `KnowledgeDocumentVersion.document_version` | 所属版本。 |
| `namespace` | `str` |  | 所属命名空间。 |
| `source_type` | `str` |  | 源类型。 |
| `source_path` | `str` |  | 源文件路径。 |
| `source_record_id` | `str` | 逻辑 FK -> `DocumentRecord.source_record_id` | 来源记录。 |
| `chunk_id` | `str` |  | 当前分块 ID。 |
| `chunk_index` | `int` |  | 当前分块序号。 |
| `updated_at` | `str` |  | 最近更新时间。 |
| `is_active` | `bool` | 运行时附加 | 是否为当前激活分块，写入向量库时追加。 |

## 三、关键 API DTO

说明：

- 这些不是数据库实体，但确实是项目对外最关键的数据契约。
- 这里覆盖当前对外 REST 接口涉及的主要 DTO。

### 8. `HealthcheckResponse`

一句话说明：健康检查接口的内联响应结构。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `status` | `str` |  | 健康状态，当前固定返回 `"ok"`。 |

### 9. `ChatRequest`

一句话说明：统一聊天接口请求 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `message` | `str` | 必填 | 用户消息文本。 |
| `session_id` | `str \| null` | 可空 | 会话 ID。 |
| `stream` | `bool` |  | 是否流式。 |
| `top_k` | `int \| null` | 可空 | 检索数量上限。 |

### 10. `Citation`

一句话说明：聊天回答中的引用片段 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `citation_id` | `str` |  | 引用 ID。 |
| `namespace` | `str` |  | 引用所属命名空间。 |
| `snippet` | `str` |  | 引用摘要文本。 |
| `score` | `float \| null` | 可空 | 检索得分。 |

### 11. `ChatResponse`

一句话说明：统一聊天接口响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `session_id` | `str` |  | 会话 ID。 |
| `request_id` | `str` |  | 请求 ID。 |
| `answer` | `str` |  | 回答文本。 |
| `knowledge_used` | `bool` |  | 是否命中知识。 |
| `scene` | `str` |  | 当前场景。 |
| `agent` | `str \| null` | 可空 | 当前代理标识。 |
| `citations` | `list[Citation]` |  | 引用片段列表。 |

### 12. `SceneSummary`

一句话说明：场景列表项 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `scene` | `str` |  | 场景标识。 |
| `name` | `str` |  | 场景名称。 |
| `description` | `str` |  | 场景说明。 |
| `is_default` | `bool` |  | 是否默认场景。 |

### 13. `SceneListResponse`

一句话说明：场景列表接口响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `default_scene` | `str` |  | 默认场景标识。 |
| `scenes` | `list[SceneSummary]` |  | 场景列表。 |

### 14. `SessionCreateRequest`

一句话说明：会话创建请求 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `scene` | `str \| null` | 可空 | 期望绑定的场景。 |

### 15. `SessionCreateResponse`

一句话说明：会话创建响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `session_id` | `str` |  | 新建会话 ID。 |
| `scene` | `str` |  | 绑定场景。 |

### 16. `SessionTurnResponse`

一句话说明：单轮会话历史 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `request_id` | `str` |  | 请求 ID。 |
| `user_message` | `str` |  | 用户消息。 |
| `assistant_answer` | `str` |  | 助手回答。 |
| `retrieval_snippets` | `list[dict[str, Any]]` |  | 引用片段原始列表。 |
| `timestamp` | `str` |  | 轮次时间。 |

### 17. `SessionDetailResponse`

一句话说明：会话详情响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `session_id` | `str` |  | 会话 ID。 |
| `scene` | `str` |  | 会话场景。 |
| `total_turns` | `int` |  | 历史总轮数。 |
| `turns` | `list[SessionTurnResponse]` |  | 历史轮次列表。 |

### 18. `SessionDeleteResponse`

一句话说明：会话删除响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `session_id` | `str` |  | 被删除会话 ID。 |
| `deleted_turns` | `int` |  | 删除的轮次数量。 |

### 19. `FileUploadResponse`

一句话说明：文件上传响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `filename` | `str` |  | 原始文件名。 |
| `file_path` | `str` |  | 相对文件路径。 |
| `file_size` | `int` |  | 文件大小。 |
| `content_type` | `str` |  | MIME 类型。 |
| `upload_time` | `str` |  | 上传时间。 |

### 20. `FileInfo`

一句话说明：文件列表中的单文件 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `filename` | `str` |  | 文件名。 |
| `file_path` | `str` |  | 相对路径。 |
| `file_size` | `int` |  | 文件大小。 |
| `content_type` | `str` |  | MIME 类型。 |
| `created_time` | `str` |  | 创建时间。 |

### 21. `FileListResponse`

一句话说明：文件列表响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `files` | `list[FileInfo]` |  | 文件列表。 |

### 22. `FileDeleteResponse`

一句话说明：文件删除响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `success` | `bool` |  | 是否成功。 |
| `message` | `str` |  | 删除结果说明。 |
| `filename` | `str` |  | 被删除文件名。 |

### 23. `KnowledgeDocumentRegisterRequest`

一句话说明：知识文档注册/索引构建 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `namespace` | `str` | 必填 | 目标命名空间。 |
| `source_path` | `str` | 必填 | 源文件路径。 |
| `chunk_size` | `int \| null` | `> 0` | 切块大小；缺省时使用数据预处理模块默认值。 |
| `chunk_overlap` | `int \| null` | `>= 0` | 切块重叠长度，且必须小于 `chunk_size`；缺省时使用数据预处理模块默认值。 |
| `keep_version` | `bool` |  | 是否保留旧版本。 |

### 24. `KnowledgeDocumentRechunkRequest`

一句话说明：知识文档重分块请求 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | `int \| null` | `> 0` | 新切块大小；缺省时使用数据预处理模块默认值。 |
| `chunk_overlap` | `int \| null` | `>= 0` | 新切块重叠长度，且必须小于 `chunk_size`；缺省时使用数据预处理模块默认值。 |
| `keep_version` | `bool` |  | 是否保留旧版本。 |

### 25. `KnowledgeDocumentVersionResponse`

一句话说明：知识文档版本响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_version` | `int` |  | 版本号。 |
| `status` | `str` | 枚举 | 版本状态。 |
| `chunk_count` | `int` |  | 分块数。 |
| `chunk_size` | `int` |  | 切块大小。 |
| `chunk_overlap` | `int` |  | 切块重叠。 |
| `created_at` | `str` |  | 版本创建时间。 |
| `last_error` | `str \| null` | 可空 | 错误信息。 |

### 26. `KnowledgeDocumentSummaryResponse`

一句话说明：知识文档列表项 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_id` | `str` |  | 文档 ID。 |
| `namespace` | `str` |  | 命名空间。 |
| `source_path` | `str` |  | 源路径。 |
| `status` | `str` | 枚举 | 当前状态。 |
| `active_version` | `int` |  | 当前版本。 |
| `chunk_count` | `int` |  | 当前分块数。 |
| `updated_at` | `str` |  | 更新时间。 |

### 27. `KnowledgeDocumentListResponse`

一句话说明：知识文档列表响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `documents` | `list[KnowledgeDocumentSummaryResponse]` |  | 文档列表。 |

### 28. `KnowledgeFileIndexSummaryResponse`

一句话说明：按上传文件聚合的索引状态 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `filename` | `str` |  | 文件名。 |
| `source_path` | `str` |  | 源路径。 |
| `file_size` | `int \| null` | 可空 | 文件大小。 |
| `created_at` | `str \| null` | 可空 | 文件创建时间。 |
| `namespace` | `str \| null` | 可空 | 命名空间。 |
| `document_id` | `str \| null` | 可空 | 对应文档 ID。 |
| `indexed` | `bool` |  | 是否已建索引。 |
| `status` | `str` | 枚举 | 索引状态。 |
| `active_version` | `int \| null` | 可空 | 当前激活版本。 |
| `chunk_count` | `int \| null` | 可空 | 当前分块数。 |
| `updated_at` | `str \| null` | 可空 | 更新时间。 |
| `last_error` | `str \| null` | 可空 | 错误信息。 |
| `can_index` | `bool` |  | 当前文件是否允许建索引。 |

### 29. `KnowledgeFileIndexListResponse`

一句话说明：文件索引状态列表响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `items` | `list[KnowledgeFileIndexSummaryResponse]` |  | 文件索引状态列表。 |

### 30. `KnowledgeDocumentDetailResponse`

一句话说明：知识文档详情响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_id` | `str` |  | 文档 ID。 |
| `namespace` | `str` |  | 命名空间。 |
| `source_path` | `str` |  | 源路径。 |
| `status` | `str` | 枚举 | 当前状态。 |
| `active_version` | `int` |  | 当前版本。 |
| `chunk_count` | `int` |  | 当前分块数。 |
| `updated_at` | `str` |  | 更新时间。 |
| `source_type` | `str` |  | 源类型。 |
| `chunk_size` | `int` |  | 切块大小。 |
| `chunk_overlap` | `int` |  | 切块重叠。 |
| `last_error` | `str \| null` | 可空 | 错误信息。 |
| `versions` | `list[KnowledgeDocumentVersionResponse]` |  | 版本列表。 |

### 31. `KnowledgeDocumentOperationResponse`

一句话说明：知识文档写操作响应 DTO，返回文档当前状态和版本信息。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_id` | `str` |  | 文档 ID。 |
| `namespace` | `str` |  | 命名空间。 |
| `source_path` | `str` |  | 源路径。 |
| `status` | `str` | 枚举 | 当前状态。 |
| `active_version` | `int` |  | 当前激活版本。 |
| `chunk_count` | `int` |  | 当前分块数。 |
| `updated_at` | `str` |  | 更新时间。 |
| `source_type` | `str` |  | 源类型。 |
| `chunk_size` | `int` |  | 切块大小。 |
| `chunk_overlap` | `int` |  | 切块重叠。 |
| `last_error` | `str \| null` | 可空 | 最近错误。 |
| `versions` | `list[KnowledgeDocumentVersionResponse]` |  | 版本历史。 |
| `document_version` | `int` |  | 本次操作对应版本。 |

### 32. `KnowledgeDocumentDeleteResponse`

一句话说明：知识文档删除响应 DTO。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `document_id` | `str` |  | 文档 ID。 |
| `namespace` | `str` |  | 命名空间。 |
| `source_path` | `str` |  | 源路径。 |
| `status` | `str` | 枚举 | 删除后的状态。 |
| `active_version` | `int` |  | 当前版本。 |
| `chunk_count` | `int` |  | 当前分块数。 |
| `updated_at` | `str` |  | 更新时间。 |
| `source_type` | `str` |  | 源类型。 |
| `chunk_size` | `int` |  | 切块大小。 |
| `chunk_overlap` | `int` |  | 切块重叠。 |
| `last_error` | `str \| null` | 可空 | 错误信息。 |
| `versions` | `list[KnowledgeDocumentVersionResponse]` |  | 版本历史。 |
| `document_version` | `int` |  | 当前操作对应版本。 |

## 四、电商场景演示业务模型

说明：

- 这些模型来自 `backend/data/*.json` 和 `commerce_tools.py`。
- 它们是示例业务数据，不是统一数据库实体。

### 33. `Product`

一句话说明：电商商品主数据，包含规格和库存。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `product_id` | `str` | PK | 商品 ID。 |
| `name` | `str` |  | 商品名称。 |
| `category` | `str` |  | 商品分类。 |
| `description` | `str` |  | 商品描述。 |
| `price` | `number` |  | 商品价格。 |
| `currency` | `str` |  | 币种，如 `CNY`。 |
| `specs` | `object` |  | 规格字典。 |
| `inventory` | `object` |  | 库存字典。 |

`inventory.status` 枚举值：

- `in_stock`
- `low_stock`

### 34. `Review`

一句话说明：商品评论数据。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `review_id` | `str` | PK | 评论 ID。 |
| `product_id` | `str` | FK -> `Product.product_id` | 关联商品。 |
| `rating` | `int` |  | 评分。 |
| `title` | `str` |  | 评论标题。 |
| `content` | `str` |  | 评论正文。 |
| `user_name` | `str` |  | 评论用户昵称。 |
| `created_at` | `str` |  | 评论时间。 |

### 35. `Order`

一句话说明：订单主数据，包含用户、配送和订单状态。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `order_id` | `str` | PK | 订单 ID。 |
| `user_id` | `str` |  | 用户 ID。 |
| `status` | `str` | 枚举 | 订单状态。 |
| `created_at` | `str` |  | 创建时间。 |
| `paid_at` | `str \| null` | 可空 | 支付时间。 |
| `payment_deadline` | `str \| null` | 可空 | 待付款订单截止时间。 |
| `shipped_at` | `str \| null` | 可空 | 发货时间。 |
| `delivered_at` | `str \| null` | 可空 | 签收时间。 |
| `tracking_no` | `str \| null` | 可空 | 运单号。 |
| `carrier` | `str \| null` | 可空 | 物流公司。 |
| `shipping_address` | `str` |  | 收货地址。 |
| `items` | `list[OrderItem]` |  | 订单项列表。 |
| `total_amount` | `number` |  | 订单总金额。 |
| `currency` | `str` |  | 币种。 |

样例枚举值：

- `待付款`
- `已发货`
- `运输中`
- `已签收`

### 36. `OrderItem`

一句话说明：订单中的商品明细行。

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `product_id` | `str` | FK -> `Product.product_id` | 商品 ID。 |
| `name` | `str` |  | 商品名称快照。 |
| `quantity` | `int` |  | 购买数量。 |
| `unit_price` | `number` |  | 下单单价。 |

### 37. `ServiceTicket`

一句话说明：售后或投诉工单，由工具运行时写入 `service_tickets.json`。

来源：

- `return_ticket_create`
- `complaint_ticket_create`

| 字段 | 类型 | 约束 | 说明 |
| --- | --- | --- | --- |
| `ticket_id` | `str` | PK | 工单 ID。 |
| `ticket_type` | `str` | 枚举 | 工单类型。 |
| `order_id` | `str` | FK -> `Order.order_id` | 关联订单。 |
| `reason` | `str \| null` | 可空 | 退换货原因。 |
| `items` | `list[str] \| null` | 可空 | 退货涉及商品。 |
| `message` | `str \| null` | 可空 | 投诉内容。 |
| `contact` | `str \| null` | 可空 | 联系方式。 |
| `status` | `str` | 枚举 | 工单状态。 |
| `created_at` | `str` |  | 工单创建时间。 |

枚举值：

- `ticket_type`: `return`、`complaint`
- `status`: `open`

## 五、关系总结

### 持久化主链路

- `sessions` 1 对多 `chat_turns`
- `sessions` 1 对多 `chat_messages`
- `KnowledgeDocumentRecord` 1 对多 `KnowledgeDocumentVersion`
- `KnowledgeDocumentRecord` 1 对多 `DocumentChunk`
- `DocumentRecord` 1 对多 `DocumentChunk`

### 电商示例链路

- `Product` 1 对多 `Review`
- `Order` 1 对多 `OrderItem`
- `Product` 1 对多 `OrderItem`
- `Order` 1 对多 `ServiceTicket`

## 六、建模备注

- SQLite 只对 `sessions.session_id` 声明了物理主键；其余跨表关系目前是逻辑外键，不是数据库级约束。
- 知识文档的“版本”是嵌套在主记录里的版本数组，不是独立关系表。
- 知识文档主记录和分块可落在 Chroma 或 Elasticsearch，因此字段是统一逻辑模型，不依赖单一后端。
- 电商场景数据当前以 JSON 文件形式存在，更接近示例主数据而不是强约束事务模型。

## 七、与 `docs/api-list.md` 的一致性检查

对照范围：

- `docs/api-list.md`
- `backend/application/runtime/api/chat/schemas.py`
- `backend/application/runtime/api/file/schemas.py`
- `backend/application/runtime/api/knowledge/schemas.py`

### 原始不一致点

在本次修复前，`docs/api-list.md` 中提到但 `docs/data-model.md` 未定义或未显式定义的接口实体包括：

- `HealthcheckResponse`
- `Citation`
- `SceneSummary`
- `SceneListResponse`
- `SessionCreateRequest`
- `SessionCreateResponse`
- `SessionTurnResponse`
- `SessionDetailResponse`
- `SessionDeleteResponse`
- `FileUploadResponse`
- `FileInfo`
- `FileListResponse`
- `FileDeleteResponse`
- `KnowledgeDocumentRechunkRequest`
- `KnowledgeDocumentVersionResponse`
- `KnowledgeDocumentSummaryResponse`
- `KnowledgeDocumentListResponse`
- `KnowledgeFileIndexSummaryResponse`
- `KnowledgeFileIndexListResponse`
- `KnowledgeDocumentDetailResponse`
- `KnowledgeDocumentDeleteResponse`

### 验证结果

- 上述实体都能在对应 `schemas.py` 中找到实际定义，不是“接口文档误写”。
- 问题根因是 `docs/data-model.md` 之前只覆盖了部分“关键 DTO”，没有覆盖全部对外接口 DTO。
- 代码层无须修改；需要修复的是文档覆盖范围。

### 修复结果

已将上述缺失 DTO 全部补入本文第三部分“关键 API DTO”，并按代码字段逐项对齐。
