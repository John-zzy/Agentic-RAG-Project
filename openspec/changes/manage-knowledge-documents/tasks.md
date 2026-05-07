## 1. 文档存储与切分
- [x] 1.1 新增“我的文档知识库”存储方案：原始 JSON 文档统一落在 `backend/data/files` 或其子目录；文档主记录、版本摘要、chunk 映射与检索 metadata 统一写入 Elasticsearch；SQLite 不承担知识库存储。
- [x] 1.2 新增 `namespace`、文件路径、文件内容、`chunk_size` 和 `chunk_overlap` 的校验工具。
- [x] 1.3 新增面向 JSON 文档的切分服务，为每条源记录生成稳定的 `source_record_id`，并为每个 chunk 生成稳定的 `chunk_id` 与追溯 metadata。
- [x] 1.4 新增服务层测试，覆盖文档引用、Elasticsearch 中的文档记录、版本切换、chunk metadata 持久化，以及非法切分参数拒绝。

## 2. 知识库文档服务
- [x] 2.1 实现文档注册处理：读取 `backend/data/files` 中的原始 JSON，切分内容，写入向量存储并更新处理状态。
- [x] 2.2 实现文档列表与详情查询，支持按 `namespace` 过滤，并返回最小可追溯字段集合。
- [x] 2.3 实现文档删除，删除或失活活动版本关联 chunk，并在默认列表结果中排除已删除文档。
- [x] 2.4 实现文档重新切分：校验新参数，默认覆盖旧活动版本；当显式要求保留版本时，保留旧版本并切换新活动版本。
- [x] 2.5 新增服务层测试，覆盖成功注册、列表、详情、删除、重新切分、显式保留版本、文档不存在、向量存储失败状态，以及版本切换失败回滚。

## 3. API 路由
- [x] 3.1 新增 `backend/api/knowledge/` schema 和路由，覆盖注册、列表、详情、删除和重新切分操作。
- [x] 3.2 在 FastAPI app 中注册知识库文档路由，并保持现有聊天 API 路由不变。文档服务采用懒加载，避免影响仅使用聊天 API 的启动路径。
- [x] 3.3 新增 API 测试，覆盖合法请求、参数校验失败、文档不存在、默认覆盖、显式保留版本、后端错误结构化响应和聊天 API 回归。

## 4. Elasticsearch 组织
- [x] 4.1 为知识库文档新增 `documents` 索引，存储文档主记录和活动版本摘要。
- [x] 4.2 为知识库文档新增 `chunks` 索引，存储可检索 chunk、向量和追溯 metadata。
- [x] 4.3 实现默认仅对活动版本 chunk 进行检索和删除过滤，并补充旧版本/回滚场景测试。

## 5. 静态测试页 UI
- [x] 5.1 更新 `frontend/api-tester.html`，新增知识库文档管理区域，包含 `namespace`、文件路径、`chunk_size`、`chunk_overlap`、注册、刷新、删除和重新切分操作。
- [x] 5.2 将页面 JavaScript 接入新 API，并展示文档状态、活动版本、chunk 数量、时间戳和 API 错误。
- [x] 5.3 验证 UI 修改后现有聊天测试交互仍可正常使用，静态脚本语法检查通过。

## 6. 验证
- [x] 6.1 运行 `python -m pytest backend\tests -q -c backend\tests\pytest.ini`，结果为 `79 passed, 4 deselected`。
- [x] 6.2 手动启动 `python backend\run.py`，验证 `/health` 返回 `{"status":"ok"}`，确认路由注册和应用启动正常。
- [x] 6.3 确认 Elasticsearch 专属行为仍由 `integration` 标记测试覆盖；默认测试配置不会要求本地 Elasticsearch 可用。
