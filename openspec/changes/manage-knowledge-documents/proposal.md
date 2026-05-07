## Why

当前 `frontend/api-tester.html` 只能验证对话链路，无法管理“我的文档知识库”。现有项目已经具备将电商 JSON 数据直接注册进知识库的能力，但缺少一套明确、可展示、可追溯的文档管理方案，导致新增、重建、删除和验证都需要依赖脚本或直接改数据。

本变更用于补齐“知识库文档管理”能力，并统一方案边界：
- 原始 JSON 文档保存在文件系统 `backend/data/files`。
- 向量检索、chunk 元数据和知识检索相关数据写入 Elasticsearch。
- 保持当前“直接注册进知识库”的主流程，不强制改成上传优先。
- 检索命中的 chunk 必须能够追溯到源文件中的具体记录。

## What Changes

- 新增“我的文档知识库”存储设计，采用“文件系统保存原始文档 + Elasticsearch 保存文档记录与 chunk 检索数据”的双层方案。
- 新增文档注册、列表、详情、删除和重新切分能力，支持当前项目中的 JSON 文件直接注册。
- 默认采用覆盖式更新；当调用方显式要求保留版本时，保留旧版本并切换新的活动版本。
- 在 chunk metadata 中保存 `source_path`、`source_record_id`、`chunk_id`、`chunk_index` 等追溯字段，确保召回结果可以定位到原始依据。
- 更新静态测试页，使其能演示文档注册、查看、删除和重新切分。
- 增加对应服务层和 API 测试。

## Capabilities

### New Capabilities

- `knowledge-document-management`: 通过后端 API 和静态测试页管理“我的文档知识库”，支持注册、列表、详情、删除、重建切分和版本控制。

### Modified Capabilities

- `knowledge retrieval`: 检索结果新增对原始文档记录的追溯能力。

## Impact

- 后端 API：新增知识库文档管理路由、Schema 和服务层逻辑。
- 知识库模块：扩展文档注册、chunk 构建、删除和重建流程。
- 存储设计：原始 JSON 保存在 `backend/data/files`，ES 保存文档主记录、版本信息、chunk 元数据和向量。
- 前端：更新 `frontend/api-tester.html`，加入知识库管理区域。
- 测试：新增 `backend/tests/` 下的单元测试和集成测试约束。
## Implementation Status

本变更已实现于 `codex/manage-knowledge-documents` 分支，覆盖后端服务、API、Elasticsearch 存储扩展、测试和 `frontend/api-tester.html` 静态测试页。

已完成的主要交付：

- 新增 `backend/knowledge/documents/` 文档管理模块，支持 JSON 文档校验、加载、切分、稳定 ID、注册、列表、详情、删除和重新切分。
- 扩展 `backend/knowledge/base/store.py`，为 Elasticsearch 增加 `documents` 与 `chunks` 双索引，以及文档记录、版本、chunk active 状态管理能力。
- 新增 `/knowledge/documents` API，覆盖注册、列表、详情、删除和 rechunk，并提供结构化错误响应。
- 更新 `frontend/api-tester.html`，加入文档知识库管理区域，同时保留原聊天测试交互。
- 新增服务层、API 层和 Elasticsearch fake/live 相关测试；默认测试不依赖本地 Elasticsearch。

最终验证：

- `python -m pytest backend\tests -q -c backend\tests\pytest.ini` -> `79 passed, 4 deselected`
- `python backend\run.py` 后访问 `/health` -> `{"status":"ok"}`

实现中补充的关键约束：

- 版本发布具备失败回滚保护，避免 chunk 写入、记录发布、旧 chunk 失活或清理失败导致活动记录与可检索 chunk 不一致。
- `keep_version=true` 保留历史版本记录，但默认活动检索只使用最新活动版本 chunk。
- FastAPI 懒加载 `KnowledgeDocumentService`，避免仅使用聊天 API 时被文档索引初始化阻塞。
