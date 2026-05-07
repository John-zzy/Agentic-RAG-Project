## ADDED Requirements

### Requirement: 文档注册与入库

系统 SHALL 允许将 `backend/data/files` 中的受支持 JSON 文档直接注册进知识库，并 SHALL 按给定切分参数生成可检索 chunk 写入 Elasticsearch。

#### Scenario: 成功注册 JSON 文档

- **WHEN** 调用方选择一个受支持的 JSON 文档，并提供合法的 `namespace`、`chunk_size` 和 `chunk_overlap`
- **THEN** 系统保存原始文件路径信息，创建文档主记录和活动版本记录，生成一个或多个 chunk，写入 Elasticsearch，并返回 `document_id`、`document_version`、`status`、`namespace` 和 `chunk_count`

#### Scenario: 拒绝非法注册参数

- **WHEN** 调用方提供不支持的 `namespace`、不存在的文件路径、空内容、非正数 `chunk_size`，或大于等于 `chunk_size` 的 `chunk_overlap`
- **THEN** 系统返回参数校验错误，且不得写入任何文档记录或 chunk

### Requirement: 文档 ID 稳定生成

系统 SHALL 按 `namespace + source_path` 为同一源文档稳定生成同一个 `document_id`。

#### Scenario: 重复注册同一路径文档

- **WHEN** 调用方重复注册同一 `namespace + source_path` 的文档
- **THEN** 系统应复用既有 `document_id`，而不是新建新的文档 ID

### Requirement: 检索结果可追溯到源记录

系统 SHALL 确保每个可检索 chunk 都能定位到源文件中的具体记录。

#### Scenario: 检索命中后追溯源记录

- **WHEN** 检索结果命中某个 chunk
- **THEN** 返回结果中应包含足够的追溯信息，使调用方能定位到 `source_path` 和 `source_record_id`

#### Scenario: 记录级追溯字段完整

- **WHEN** 系统写入 chunk 到 Elasticsearch
- **THEN** 每个 chunk 的 metadata 至少包含 `document_id`、`document_version`、`namespace`、`source_path`、`source_record_id`、`chunk_id` 和 `chunk_index`

### Requirement: 文档列表与详情

系统 SHALL 提供 API 用于列出已管理文档，并查看单个文档的主记录、活动版本和处理状态。

#### Scenario: 列出文档

- **WHEN** 调用方请求文档列表，并可选提供 `namespace` 过滤
- **THEN** 系统返回文档的 `document_id`、`namespace`、`source_type`、`source_path`、`status`、`chunk_count`、`active_version` 和 `updated_at`

#### Scenario: 获取文档详情

- **WHEN** 调用方使用已存在的 `document_id` 请求详情
- **THEN** 系统返回主记录、活动版本信息、最近一次切分参数和最近一次错误信息

### Requirement: 默认覆盖，显式保留版本

系统 SHALL 默认按 `namespace + source_path` 覆盖已存在文档；当调用方显式要求保留版本时，SHALL 创建新的版本记录并保留历史版本。

#### Scenario: 默认覆盖注册

- **WHEN** 调用方重复注册同一 `namespace + source_path` 的文档，且未指定保留版本
- **THEN** 系统应删除或失活旧活动版本的 chunk，写入新活动版本，并更新 `active_version`

#### Scenario: 显式保留版本

- **WHEN** 调用方重复注册同一 `namespace + source_path` 的文档，并显式要求保留版本
- **THEN** 系统应创建新的 `document_version`，保留旧版本记录，并确保默认检索只命中新活动版本

### Requirement: Elasticsearch 双索引组织

系统 SHALL 将文档主记录与检索 chunk 分别存储在独立索引中。

#### Scenario: 文档记录与 chunk 分离存储

- **WHEN** 系统写入知识库文档数据
- **THEN** 文档主记录写入 `documents` 索引，chunk 与向量写入 `chunks` 索引

### Requirement: 文档删除

系统 SHALL 允许删除已管理文档，并 SHALL 清理其活动版本相关的 Elasticsearch chunk。

#### Scenario: 删除文档

- **WHEN** 调用方删除一个已存在的文档
- **THEN** 系统删除或失活文档主记录，移除其活动版本关联 chunk，并使其不再出现在默认列表和默认检索结果中

#### Scenario: 删除文档不删除原始文件

- **WHEN** 调用方删除知识库文档
- **THEN** 系统不得默认删除 `backend/data/files` 下的原始 JSON 文件

### Requirement: 文档重新切分

系统 SHALL 允许使用新的切分参数重新处理已存在文档，并 SHALL 生成新的活动版本。

#### Scenario: 重新切分文档

- **WHEN** 调用方对一个已存在文档发起重新切分，并提供合法参数
- **THEN** 系统应基于原始文件重新生成 chunk，创建新的活动版本，并使默认检索只使用新版本
### Requirement: 版本发布失败安全

系统 SHALL 在注册或重新切分切换活动版本时保持文档记录与 active chunk 状态一致。

#### Scenario: 新版本发布中途失败

- **WHEN** 系统在写入新 chunk、发布新文档记录、失活旧活动 chunk 或回滚清理任一步发生后端存储失败
- **THEN** 系统 SHALL 尽量恢复到失败前的活动版本记录和活动 chunk 状态，并 SHALL 返回结构化错误，而不是留下活动版本指向不可检索 chunk 的状态

### Requirement: API 懒加载与结构化错误

系统 SHALL 只在访问知识库文档管理接口时初始化文档管理服务，并 SHALL 对文档管理 API 返回稳定的结构化错误。

#### Scenario: 仅使用聊天 API

- **WHEN** 应用启动并只访问聊天或健康检查接口
- **THEN** 系统 SHALL 不因知识库文档 Elasticsearch 索引初始化失败而阻塞聊天接口可用性

#### Scenario: 文档管理后端错误

- **WHEN** 文档管理接口遇到存储后端错误或未知后端错误
- **THEN** 系统 SHALL 返回包含稳定 `code` 和安全 `message` 的错误 detail，且不直接暴露原始后端异常文本
