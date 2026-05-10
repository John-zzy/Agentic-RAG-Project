# AI RAG Project Agent Guide

本文件用于帮助 AI 快速理解仓库结构与修改入口。若与用户当前指令冲突，以用户指令为准。

## 项目概览

这是一个多场景智能助手 / RAG 示例项目，后端当前采用三层结构：

- `platform`：平台级通用能力
- `application`：运行时装配与 API
- `scenes`：场景实现

默认场景是 `generic_assistant`，同时保留 `ecommerce` 作为电商演示场景。

## 目录速查

### `backend/application/runtime`

负责运行时装配与 API 暴露，优先在这些场景下查看这里：

- 启动流程、依赖装配：`bootstrap.py`、`service.py`
- FastAPI 应用与路由注册：`api/app.py`
- 聊天与会话接口：`api/chat/`
- 文件接口：`api/file/`
- 知识文档接口：`api/knowledge/`

### `backend/platform`

负责通用基础能力，优先按职责定位：

- 配置与环境变量：`config/`
- 模型路由与 LLM 客户端：`models/`
- 会话记忆与上下文：`memory/`
- 通用知识文档处理：`knowledge/`
- RAG 核心逻辑：`rag/`
- 通用工具协议：`tools/`

### `backend/scenes`

负责具体场景能力：

- `generic_assistant/`：通用助手定义
- `ecommerce/`：电商场景定义、知识服务、检索工具、业务工具
- `base.py`：场景基础抽象

### 其他常用目录

- `backend/tests/`：后端测试
- `backend/data/`：本地数据与持久化目录
- `frontend/`：调试页面
- `backend/run.py`：后端启动入口

## 常用命令

所有命令默认在仓库根目录执行。

### 安装依赖

```powershell
python -m venv backend\.venv
backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

### 启动后端

```powershell
backend\.venv\Scripts\python.exe backend\run.py
```

### 运行测试

全量：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

单文件：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

## 环境变量

优先关注：

```env
AI_RAG_APP__ACTIVE_SCENE=generic_assistant
```

这个变量当前表示“新会话默认场景”，不是日常切换场景的唯一入口。场景切换优先走会话级 API 或前端选择。

## 修改时的注意事项

- 使用 `apply_patch` 做手工文件修改
- 避免大面积无关格式化 diff
- 修改 `__init__.py` 时保持最小化，避免引入运行时依赖导致循环导入
- 如果改动了架构、启动方式、环境变量或测试命令，要同步检查 `README.md`、`AGENTS.md`、`backend/.env.example`

## 高频错误

- 使用了错误的 Python 解释器，而不是 `backend\.venv\Scripts\python.exe`
- 改完实际代码后，没有同步更新文档和环境样例
- 在 `__init__.py` 中引入运行时装配，导致循环导入
- 用不精确的覆盖式写文件方式修改内容，导致文件损坏或内容串乱
- 架构相关改动后，没有补跑受影响测试或全量测试
