# runtime渐进式LCEL与LangChain Memory改造方案

> 状态：已完成，已于当前主线代码落地。

## Summary

本方案用于把当前 `application/runtime` 的 `/chat` 主链，逐步从：

- `history_text` 字符串拼接
- `invoke_template()` / `stream_template()`
- runtime 手工控制历史注入

演进为：

- LCEL 主导的回答生成链
- LangChain message-based prompt
- 最终接入 `RunnableWithMessageHistory`

这次不再追求一步到位大改，而是按 **4 个小模块** 渐进推进。当前 4 个模块均已完成。每个模块最初要求：

- 可独立提交
- 可独立回归
- 改完后可以单独讲解代码设计

固定节奏为：

1. 改 1 个模块
2. 跑受影响测试
3. 讲清楚代码职责和写法
4. 再进入下一个模块

一句话概括本方案：

“先把 runtime 的回答执行方式切到 LCEL，再把历史从字符串改成 message history，最后再让 session API 与最终 memory 主模型对齐。”

## Key Changes

### 模块 1：先把 `/chat` 生成链改成 LCEL

状态：已完成

目标：

- 只改 `backend/application/runtime/service.py`
- 只改 `backend/application/runtime/api/chat/prompts.py`
- 不碰 memory 主模型
- 不改 `/sessions` 接口
- 不改 SSE 事件结构

核心改动：

- `ChatService` 不再直接依赖 `invoke_template()` / `stream_template()`
- 改为通过 `model.get_runnable(...)` 执行最终回答链
- prompt 变量先保持现状：
  - `history`
  - `input`
  - `context`
- `history` 此阶段仍然允许来自 `history_text`
- `RetrievalChainModel` 协议从“模板 helper 优先”改成“`get_runnable()` 优先”

这一模块的边界非常明确：

- 只改“执行方式”
- 不改“历史存储方式”

这样做的原因是：

- `platform.models` 已经 LCEL First
- runtime 仍然停留在旧 helper 心智
- 先把执行层和模型层的接口对齐，后续 message history 改造才不会回头返工

实际结果：

- `/chat` 同步与流式路径均已改为通过 runnable 执行最终回答阶段
- runtime 已不再依赖 `invoke_template()` / `stream_template()` 作为主执行入口

你会学到：

- `prompt | model | parser` 为什么是 LCEL 的基本结构
- 为什么先改 execution，再改 memory

### 模块 2：把 history 从字符串改成 message history

状态：已完成

目标：

- 只改 runtime 和现有 `SQLiteChatMessageHistory` 接入点
- 暂时不改 `/sessions/{id}` 对外响应结构

核心改动：

- `build_rag_answer_prompt_template()` 改成：
  - `system`
  - `MessagesPlaceholder("history")`
  - `human`
- runtime 不再拼 `history_text`
- 改为从 `SQLiteChatMessageHistory` 读取 LangChain messages
- 正式接入 `RunnableWithMessageHistory`
- `PromptContextBuilder` 正式用于裁剪 message window

此阶段的固定原则：

- `/chat` 的外部响应先不变
- memory 主模型开始切到 `chat_messages`
- runtime 不再自己承担历史拼接职责

实际结果：

- prompt 已改为 `system + MessagesPlaceholder("history") + human`
- runtime 已不再拼 `history_text`
- `RunnableWithMessageHistory` 已接入 `/chat` 主链
- `PromptContextBuilder` 已参与 message window 裁剪

你会学到：

- `MessagesPlaceholder` 的语义
- `RunnableWithMessageHistory` 是如何把历史自动注入到链里的
- 为什么 message history 比 `history_text` 更适合作为长期主模型

### 模块 3：补齐流式可观测事件

状态：已完成

目标：

- 只改 `/chat` 的 stream 路径
- 不改 session detail API

核心改动：

- 保留现有：
  - `start`
  - `chunk`
  - `done`
- 新增：
  - `history`
  - `tool`
- `history` 用于暴露本轮注入模型前的历史消息快照
- `tool` 用于暴露 retrieval 阶段或检索工具阶段的结构化结果

此阶段的边界：

- 只增强流式可观测性
- 不顺手改 session API
- 不顺手删旧存储模型

实际结果：

- `/chat` 流式路径已稳定输出 `start`、`history`、`tool`、`chunk`、`done`
- 错误路径继续通过 SSE `error` 事件对外暴露
- `history` 事件用于调试本轮注入模型前的历史窗口
- `tool` 事件用于暴露 retrieval 阶段的结构化执行结果

你会学到：

- 为什么业务事件流不能直接等同于模型 token 流
- runtime 如何把 LCEL 执行过程转成更友好的前端事件

### 模块 4：把 `/sessions/{id}` 改成 messages 视图

状态：已完成

目标：

- 最后才动外部历史接口
- 这一步正式让外部视图和 memory 主模型统一

核心改动：

- `SessionDetailResponse` 从：
  - `total_turns`
  - `turns`
  改成：
  - `total_messages`
  - `messages`
- 新增 `SessionMessageResponse`
- `GET /sessions/{id}` 直接从 `chat_messages` 读取
- `DELETE /sessions/{id}` 的返回语义改成 `deleted_messages`
- AI message 上挂：
  - `citations`
  - `request_id`
  - `timestamp`
  - `knowledge_used`

实际结果：

- `GET /sessions/{id}` 已切到 message 视图
- `SessionDetailResponse` 已改为 `total_messages` 和 `messages`
- `DELETE /sessions/{id}` 已返回 `deleted_messages`
- assistant message 已暴露 `request_id`、`timestamp`、`knowledge_used`、`citations`

此阶段允许发生对外 API 视图变化。

原因是：

- 前 3 个模块先把内部主链稳定下来
- 最后再让 API 与主数据模型对齐
- 避免一开始就把“内部重构”和“外部接口升级”绑死在一起

你会学到：

- message payload 如何映射成对外 API
- 为什么 API 最终应该贴近主数据模型，而不是长期维持旧兼容视图

## Implementation Strategy

整个改造按以下顺序落地：

1. 先改执行链
2. 再改历史注入方式
3. 再补流式可观测能力
4. 最后再升级 session detail 视图

明确不建议的做法：

- 不建议一开始同时改 runtime、memory、API
- 不建议先动 `/sessions` 视图
- 不建议先删 legacy turn 模型再改回答链

因为这样会让：

- 回归面过大
- 出错时难定位
- 很难边改边讲清楚设计

## Test Plan

每个模块只跑受影响测试，不一次性全量推进。

### 模块 1

至少运行：

- `backend/tests/test_chat_api.py`

验收点：

- `/chat` 流式与非流式都不回归
- `ChatResponse` 不变
- runnable 链真正被 runtime 使用

### 模块 2

至少运行：

- `backend/tests/test_chat_api.py`
- `backend/tests/test_session_store.py`

验收点：

- 多轮对话能保留上下文
- 历史窗口仍然受 `window_size` 控制
- runtime 不再自己拼 `history_text`

### 模块 3

至少运行：

- `/chat` stream 相关测试

验收点：

- `start/chunk/done` 仍可用
- `history` / `tool` 事件增加后顺序稳定
- no-hit / hit / error 三条路径都能正确输出事件

### 模块 4

至少运行：

- session detail / delete 相关测试
- `backend/tests/test_chat_api.py`

验收点：

- `/sessions/{id}` 正式返回 message 视图
- 删除接口语义与 message store 一致
- 文档同步完成

## Landed State

当前 runtime 主链已经完成以下收敛：

- `platform.models` 负责模型路由和 LCEL 执行
- `application/runtime` 通过 runnable 驱动最终回答链
- prompt 使用 LangChain message-based 结构，而不是 `history_text` 拼接
- 历史注入通过 `RunnableWithMessageHistory` 和 `SQLiteChatMessageHistory` 完成
- session 外部视图已经和 `chat_messages` 主模型对齐

当前实现明确不再采用旧式 `ConversationBufferMemory` 或 `ConversationBufferWindowMemory` 作为 `/chat` 主链 memory 模型。

## Migration Notes

这轮改造后，运行时职责边界变为：

- `platform.models`：只负责模型选择、runnable 构建和 LCEL 执行
- `application/runtime`：负责场景路由、retrieval 编排、history 注入、SSE 事件组织
- `platform.memory`：负责会话持久化、LangChain message history 适配、session message 视图

仍然保留但不再作为主路径的旧心智：

- `history_text` 字符串式历史拼接
- runtime 直接调用模板 helper 执行模型
- turn-first 的 session API 视图

## Teaching Goal

本方案不是单纯追求“功能改完”，而是要求每个模块完成后都能反向讲清楚：

1. 这一模块改动的职责边界是什么
2. 核心类和函数如何协作
3. 为什么这样写，而不是另一种写法
4. 以后如果继续扩展，同类代码应该怎么组织

也就是说，这份方案的最终目标不是只得到一套能跑的代码，而是让后续维护者能真正理解：

- LCEL 是怎么进入 runtime 的
- LangChain memory 为什么应该在这一层接入
- 为什么要按模块渐进改，而不是一次性推翻

## Assumptions

- 当前 `platform.models` 已经完成 LCEL First 改造
- 下一阶段先只改 `application/runtime`
- 每个模块完成后暂停，不连续跳过讲解直接进入下一个模块
- `memory` 包的彻底收敛与兼容层清理，不在这份方案的第一轮实施范围内


