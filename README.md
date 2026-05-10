# AI RAG Project

一个通用智能体 / RAG 平台示例项目。当前代码已经从“电商客服 MVP”重构为三层架构：

- `platform`：通用平台能力
- `application`：运行时装配与 API 挂载
- `scenes`：具体场景实现

默认场景是 `generic_assistant`，同时保留 `ecommerce` 作为电商演示场景。

## 当前能力

- 基于 `FastAPI` 提供 `/chat`、`/sessions/*`、`/files/*`、`/knowledge/documents/*` 接口
- 支持上传本地知识文件，并管理文档索引
- 支持 `Chroma` 与 `Elasticsearch` 两种向量存储
- 支持 `SQLite` 会话记忆
- 支持场景切换：
  - `generic_assistant`：仅依赖上传文档与会话记忆
  - `ecommerce`：包含商品、评价、订单、库存与客服工具链

## 目录结构

```text
.
├── backend/
│   ├── application/
│   │   └── runtime/            # 运行时装配、API、启动引导
│   ├── platform/               # 平台通用能力：config / models / memory / knowledge / rag / tools
│   ├── scenes/                 # 场景实现：generic_assistant / ecommerce
│   ├── tests/
│   ├── data/
│   ├── .env
│   ├── .env.example
│   ├── requirements.txt
│   └── run.py
├── frontend/
├── docs/elasticsearch/
├── openspec/
├── AGENTS.md
└── README.md
```

## 环境准备

以下命令默认在仓库根目录执行：

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

## 启动后端

不要直接运行裸 `python`。本项目开发与测试统一使用 `backend\.venv\Scripts\python.exe`。

```powershell
backend\.venv\Scripts\python.exe backend\run.py
```

默认地址：

- API: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- 对话测试页: `http://127.0.0.1:8000/frontend/api-tester.html`
- 知识库管理页: `http://127.0.0.1:8000/frontend/knowledge-manager.html`

## 场景切换

推荐使用：

```env
AI_RAG_APP__ACTIVE_SCENE=generic_assistant
```

可选值：

- `generic_assistant`
- `ecommerce`

PowerShell 临时切换示例：

```powershell
$env:AI_RAG_APP__ACTIVE_SCENE = "ecommerce"  # 仅修改新会话默认场景
backend\.venv\Scripts\python.exe backend\run.py
```

> 当前推荐通过前端对话页下拉框，或调用 `GET /scenes` 与 `POST /sessions` 来切换会话场景。`AI_RAG_APP__ACTIVE_SCENE` 现在用于设置“新会话默认场景”，不再是日常切换场景的主要方式。

## 向量存储

默认使用 `chroma`：

```env
AI_RAG_VECTOR_STORE__PROVIDER=chroma
AI_RAG_VECTOR_STORE__CHROMA__PERSIST_DIRECTORY=backend/data/.chroma
```

切换到 Elasticsearch：

```env
AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch
AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL=http://127.0.0.1:9200
```

本地启动 Elasticsearch：

```powershell
docker compose -f docs\elasticsearch\docker-compose.yml up -d
```

## 知识文档管理

支持上传的文件类型：

- `.json`
- `.txt`
- `.md`
- `.csv`
- `.pdf`
- `.docx`
- `.xlsx`

当前支持建索引的文件类型：

- `.json`
- `.txt`
- `.md`
- `.csv`

相关接口：

- `POST /files/upload`
- `GET /files`
- `DELETE /files/{filename}`
- `POST /knowledge/documents`
- `GET /knowledge/documents`
- `GET /knowledge/documents/files`
- `GET /knowledge/documents/{document_id}`
- `POST /knowledge/documents/{document_id}/rechunk`
- `DELETE /knowledge/documents/{document_id}`

## 测试

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

如果只跑单个文件：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

## 开发约束

- 不要重新引入旧顶层代码包：`backend/api`、`backend/config`、`backend/knowledge`、`backend/tools`、`backend/models`、`backend/memory`
- 后端顶层业务结构保持为：`application`、`platform`、`scenes`
- 修改文件时优先做真实迁移，不要保留兼容 re-export 壳
- Windows 下不要假设 `python` 指向正确环境；统一用 `backend\.venv\Scripts\python.exe`
- 写文件时不要用临时重定向或脚本覆盖导致内容损坏；小中型改动统一走精确 patch
- `__init__.py` 保持最小导出，避免把运行时、API、scene 装配耦合进包初始化里，防止循环导入

## 当前验证状态

本次三层架构收口后已验证：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

结果：

```text
86 passed, 3 skipped, 4 deselected
```
