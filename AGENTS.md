# AI RAG Project Agent Guide

本文件约束后续在本仓库内工作的智能体行为。若与用户当前指令冲突，以用户指令为准。

## 项目现状

仓库已经完成三层架构收口，`backend` 顶层只保留：

- `application`
- `platform`
- `scenes`
- `tests`
- `data`
- 运行入口与环境文件

不要再把代码放回旧目录，不要再创建这些旧顶层包：

- `backend/api`
- `backend/config`
- `backend/knowledge`
- `backend/tools`
- `backend/models`
- `backend/memory`
- `backend/agents`
- `backend/mcp`
- `backend/monitoring`

## 当前职责边界

- `backend/platform`
  - 通用配置
  - 模型路由
  - 会话记忆
  - 通用知识文档能力
  - RAG 协议
  - 通用工具协议
- `backend/application/runtime`
  - API 路由
  - 运行时装配
  - active scene 选择
  - 启动引导
- `backend/scenes`
  - `generic_assistant`
  - `ecommerce`
  - 场景提示词、检索策略、场景工具、场景知识模型

## 正确命令

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

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

不要直接运行：

```powershell
python backend\run.py
python -m pytest ...
```

本仓库开发过程中，裸 `python` 可能不是 `backend/.venv`，直接使用是错误的。

## 环境变量约定

优先使用：

```env
AI_RAG_APP__ACTIVE_SCENE=generic_assistant
```

该变量当前表示“新会话默认场景”。
如果实现前端/API 场景切换，优先走会话级切换，不要再把它当作日常手工切换场景的唯一入口。

## 编码与改动约束

- 使用 `apply_patch` 做手工文件修改
- 不要用 Python 脚本或 shell 重定向粗暴覆盖文件，除非明确必要
- 不要制造大面积无关格式化 diff
- 不要保留“兼容层假迁移”
- 做目录迁移时要同步修正导入、测试和启动链路
- 修改 `__init__.py` 时保持最小化，避免在包初始化阶段引入重量依赖
- 对运行时/API 代码尤其注意循环导入

## 文档与命令维护

如果你修改了架构、目录、启动方式、环境变量或测试命令，必须同步检查并更新：

- `README.md`
- `AGENTS.md`
- `backend/.env.example`

不要让文档继续引用旧目录、旧命令、旧环境变量。

## 测试要求

- 功能改动后至少运行受影响测试
- 架构迁移、导入调整、启动链路调整后必须跑全量后端测试
- 在声明完成前，必须给出真实执行过的命令与结果

标准全量命令：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests -q -c backend\tests\pytest.ini
```

## 本项目的高频错误，后续不要再犯

- 把代码放回旧顶层目录
- 使用错误的 Python 解释器
- 改完架构不改文档
- 用兼容 re-export 冒充真实迁移
- 在 `__init__.py` 里引入运行时装配导致循环导入
- 用不精确的写文件方式覆盖内容，导致文件损坏或自引用错误
