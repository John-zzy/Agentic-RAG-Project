# 常见坑与排障

本文记录项目开发和 Agent 协作中最高频的错误。遇到异常时先检查这些项，避免把环境问题误判成业务缺陷。

## Python 环境

- 运行测试和后端时优先使用 `backend\.venv\Scripts\python.exe`，不要混用系统 Python。
- 安装依赖后仍报包缺失，先确认当前 shell 是否激活了 `backend\.venv`。
- 后端命令默认在仓库根目录执行；相对路径依赖这一约定。

## 配置与模型 Key

- 本地启动前需要从 `backend\.env.example` 复制 `backend\.env`。
- DashScope 相关 key 至少要覆盖 simple、moderate、complex、embedding；启用 rerank 时还要配置 rerank key。
- `AI_RAG_APP__ACTIVE_SCENE` 只控制新会话默认 scene，不是日常切换场景的主流程。

## 文档和索引

- 架构、接口、数据模型、运行方式或状态语义变化后，要同步检查 `README.md`、`AGENTS.md` 和 `docs/documents/` 下的模块文档。
- API 字段变化优先同步 `docs/documents/reference/api-list.md` 和 `docs/documents/reference/data-model.md`。
- Mermaid 图移动或改名后，要同时维护 `.mmd` 和对应 `.svg`，并更新 README / AGENTS / 文档索引链接。
- Docker Compose、本地基础设施和部署脚本统一放到 `devops/`；
- 历史方案文档在 `docs/plan/`，不要把旧方案当成当前实现事实；当前事实优先看 `docs/documents/` 和代码。

## 编码与 PowerShell

- 本仓库源码和文档按 UTF-8 处理。
- 在 Windows PowerShell 5.1 中，读取中文文档要显式使用 `Get-Content -Raw -Encoding UTF8 <file>`。
- 如果终端输出出现乱码，先判断是不是读取编码错误，不要直接认定文件已损坏。
- 不要基于乱码输出制作 patch；重新用 UTF-8 读取后再修改。

## Imagegen 生图命令

- 用户明确要求 `gpt-image-2` / `gpt-image2` 时，走 `imagegen` skill 的 CLI fallback：`C:\Users\zzy\.codex\skills\.system\imagegen\scripts\image_gen.py`；普通生图默认才优先内置 `image_gen`。
- Windows 上不要直接用系统 `python`，优先用 `backend\.venv\Scripts\python.exe`。
- CLI fallback 需要 `OPENAI_API_KEY`，并要求当前 Python 环境已有 `openai`；后处理、尺寸校验或升采样需要 `Pillow`。
- 中文、多行、带引号的复杂 prompt 不要直接塞进 PowerShell 命令参数。先写入 `tmp\imagegen\<name>-prompt.txt`，再用 `--prompt-file` 传入，避免参数被拆开或中文在错误输出中乱码。
- PowerShell 5.1 下不要把 imagegen CLI 的 stderr 轻易 `2>&1` 合并后判断成失败；CLI 会把 `OPENAI_API_KEY is set.` 等状态写到 stderr，合并后可能显示成 `NativeCommandError`。优先看 CLI 是否真正写出目标文件。
- `gpt-image-2` 不支持 `--background transparent`；透明图不要静默降级到 `gpt-image-1.5`，除非用户明确确认。
- `--size 3840x2160` 是请求参数，不保证最终 PNG 一定按该尺寸落盘。生成后用 Pillow 检查 `im.size`；如果只是文档插图需要高分辨率版本，可以本地 LANCZOS 升采样并保留原图。
- 输出不要覆盖既有资产，除非用户明确要求替换；新图优先使用语义化文件名，例如 `<source-name>-infographic.png` 和 `<source-name>-infographic-3840x2160.png`。

## Git 工作区

- 工作区可能已有用户或其他任务留下的未提交改动，不要执行 `git reset --hard` 或 `git checkout --` 回退它们。
- 做代码或文档修改前先看 `git status --short`，只处理当前任务相关文件。
- 如果同一个文件已有脏改，先读取当前内容再 patch，不要用旧片段覆盖。

## 架构边界

- 不要把运行时装配逻辑放进 `platform` 或 `__init__.py`。
- `platform.knowledge` 只负责知识管理和底层存储，不要新增面向 chat 的检索入口。
- 文档检索统一走 `DocumentRetrievalService`。
- scene 负责 prompt、scene policy 和业务工具装配；runtime 不硬编码业务工具选择。
- ToolRegistry 只维护工具注册、分组、白名单和 MCP 暴露标记，不承载业务逻辑。

## Workflow / HITL 状态

- `sessions.status` 只表示聊天会话 active/expired，不能替代 workflow run state。
- `waiting_user` 是等待人工输入，不是失败。
- `reject` 或 cancel 进入 `cancelled`，不是 `failed`。
- `succeeded / failed / cancelled` 是终态，不能继续 resume、retry 或写回 running。
- `/chat/resume` 必须校验最新 checkpoint 中的 `interrupt_id`，不能接受旧等待点。

## 测试

- 全量测试命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

- 改 `/chat`、SSE、HITL 或 Workflow State Machine 后，至少跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_langgraph_runtime.py backend\tests\test_generic_assistant_hitl.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```

- 改 RAG 检索、rerank、citation 或 trace 后，至少跑：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_document_hybrid_retrieval.py backend\tests\test_chat_api.py -q -c backend\tests\pytest.ini
```


