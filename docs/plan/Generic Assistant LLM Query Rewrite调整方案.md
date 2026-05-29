# Generic Assistant LLM Query Rewrite调整方案

## Summary

本方案用于调整 `generic_assistant` 的 query rewrite 策略：从固定字符串拼接改为 LLM 改写。

当前问题是：无命中问题进入第二轮检索时，旧 rewriter 会追加“相关文档、定义、字段、配置、数据模型、表结构、常见问题”等通用词。这些词可能被 keyword 检索高分命中，导致陌生问题被误判为 `knowledge_used=true`，并返回伪 citations。

目标是让 LLM 只做“更适合检索的保守改写”，而不是凭空扩展领域词：

- normal-hit 问题仍能命中文档
- no-hit 问题不因 rewrite 泛词产生伪引用
- `/chat` 和 SSE 的 `retrieval_trace` 能看到 LLM rewrite 或 fallback 原因
- Evaluation Harness 的 `minimal` 样本重新通过

本次不改变 `/chat` 响应结构，不改变 SSE 事件结构，不接 ReRank，不重做前端。

## Key Changes

### 1. 将拼接式 rewriter 改为 LLM rewriter

调整 `GenericAssistantQueryRewriter`：

- 复用现有 `ModelClient.get_runnable()` 和 `ModelClient.invoke_runnable()`
- 默认使用 `simple` complexity
- 不直接依赖 DashScope、OpenAI 或其他具体 provider SDK
- LLM 调用失败时回退到原始 query，不让 `/chat` 整体失败

旧行为删除：

```text
query + " 相关文档 定义 说明 字段 配置 流程 接口 规则 数据模型 表结构 常见问题"
```

新行为固定为：

```text
当前检索无命中
-> 调 LLM 生成短检索 query
-> 校验 LLM 输出是否安全
-> 安全则使用 LLM query
-> 不安全则回退原始 query
```

### 2. 新增 query rewrite prompt

为 `generic_assistant` 增加专用 prompt，只让模型改写检索 query，不让模型回答问题。

Prompt 规则：

- 保留原问题里的实体名、版本号、错误码、英文缩写、订单号、代码型 token
- 不添加原问题没有表达的领域词
- 不把随机字符串解释成某个业务概念
- 如果问题像未知暗号、随机编号、无上下文 token，直接返回原 query
- 输出必须是 JSON

建议输出格式：

```json
{
  "query": "用于下一轮检索的短 query",
  "reason": "为什么这样改写"
}
```

示例：

```text
输入：VOID-ALPHA-7788 secret handshake?
输出：{"query": "VOID-ALPHA-7788 secret handshake?", "reason": "未知代码型问题缺少上下文，保留原始 token。"}
```

```text
输入：开始使用前需要先安装什么版本的 Python？
输出：{"query": "开始使用 安装 Python 版本", "reason": "保留 Python 并压缩为安装版本相关检索 query。"}
```

### 3. 增加 LLM 输出安全校验

LLM 输出不能直接信任。解析后必须做安全校验。

回退到原始 query 的情况：

- LLM 返回非 JSON
- JSON 中没有 `query`
- `query` 为空
- `query` 过长
- `query` 删除了原问题中的关键 token
- `query` 追加了明显无依据的通用扩展词
- 模型调用异常或返回空内容

关键 token 至少覆盖：

- 英文缩写：`MFA`
- 版本号：`Python 3.11`
- 错误码 / 代码型 token：`VOID-ALPHA-7788`
- 数字编号：`7788`

回退时返回：

```python
QueryRewrite(
    query=normalized_original_query,
    reason="LLM rewrite unavailable or unsafe; fallback to original query.",
    metadata={
        "strategy": "llm_fallback",
        "fallback": True,
        "fallback_reason": "..."
    },
)
```

### 4. 保持 trace 可解释

`retrieval_trace.rewritten_query` 继续表示最终用于下一轮检索的 query。

`QueryRewrite.metadata` 建议记录：

```python
{
    "strategy": "llm",
    "fallback": False,
    "preserved_tokens": ["Python", "3.11"]
}
```

如果触发兜底：

```python
{
    "strategy": "llm_fallback",
    "fallback": True,
    "fallback_reason": "missing_preserved_token",
    "preserved_tokens": ["VOID-ALPHA-7788"]
}
```

这样演示时可以说明：

- LLM 是否参与了 query rewrite
- LLM 输出是否被接受
- 为什么 no-hit 没有继续误召回无关文档

### 5. 不改变公共接口

本次保持外部协议稳定：

- `/chat` 请求体不变
- `/chat` 响应体不变
- SSE event name 不变
- `retrieval_trace` schema 不做破坏性修改
- 前端 Trace 抽屉不需要重做

如果需要展示更多 metadata，优先放在已有 trace 的 round 信息或调试日志中，不新增必填 API 字段。

## Implementation Notes

建议实现位置：

- `backend/scenes/generic_assistant/definition.py`
  - 改造 `GenericAssistantQueryRewriter`
  - 增加 query rewrite prompt 构造
  - 增加 LLM JSON 解析与安全校验 helper
- `backend/tests/test_agentic_retrieval.py`
  - 更新旧的拼接式 rewrite 测试
  - 增加 LLM rewrite、fallback、token 保留测试
- `backend/tests/test_chat_api.py`
  - 增加 no-hit + unrelated docs 的非流式和 SSE 回归测试
- `backend/evals/evaluation-harness.md` 或 `README.md`
  - 补充一条 LLM rewrite trace 的演示说明

模型依赖建议通过构造参数注入，测试中使用 fake model，避免单元测试真实调用 LLM。

## Performance Impact

LLM query rewrite 会带来额外延迟，但影响范围应控制在 no-hit 或弱命中请求中。

触发条件固定为：

```text
第一轮检索没有足够证据
-> judge 决定 rewrite
-> 调 LLM 改写 query
-> 第二轮再检索
```

因此：

- 第一轮正常命中文档时，不调用 LLM rewrite，基本无额外性能影响。
- no-hit 问题会多一次 `simple` 模型调用和一次检索。
- 模糊问题可能多一次 LLM rewrite，用于提高第二轮检索质量。
- SSE 请求会在开始输出最终答案前完成 retrieval / rewrite，因此 no-hit 或弱命中流式请求的首包时间会变长。

性能控制措施：

- rewrite prompt 保持短小，只传用户 query 和改写规则，不传文档正文。
- 使用 `simple` complexity，避免占用更贵、更慢的复杂模型。
- 每次请求最多触发一次 LLM rewrite；第二轮仍无可靠结果就走 no-hit fallback。
- LLM 调用失败、超时、输出非法或不安全时，直接回退原 query，不阻断 `/chat`。
- 后续如有需要，可以对完全相同的 query rewrite 做短期缓存，但本次不作为必要实现。

验收时需要关注：

- normal-hit 请求不应新增 LLM rewrite 调用。
- no-hit 请求允许多一次 rewrite 调用，但必须返回 `knowledge_used=false` 和空 citations。
- SSE no-hit 样本可以多一点首包延迟，但最终 `done` 事件语义必须正确。

## Test Plan

运行后端回归：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_chat_api.py backend\tests\test_eval_assets.py -q -c backend\tests\pytest.ini
```

必须补齐以下测试：

- LLM 返回合法 JSON 时，使用 LLM query
- LLM 返回非法 JSON 时，回退原 query
- LLM 返回空 query 时，回退原 query
- LLM 删除代码型 token 时，回退原 query
- LLM 输出包含无依据通用扩展词时，回退原 query
- `VOID-ALPHA-7788 secret handshake?` 不应被改写出 `数据模型`、`表结构`、`常见问题`
- normal-hit 非流式 `/chat` 仍返回 `knowledge_used=true` 和 citations
- no-hit 非流式 `/chat` 返回 `knowledge_used=false`、`citations=[]`
- no-hit SSE `done` 事件返回 `knowledge_used=false`、`citations=[]`

端到端回放：

```powershell
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\latest.json
```

验收目标：

- `4/4 samples passed`
- `2/2 stream samples passed`
- `no_hit_fallback` 的 `knowledge_used=false`
- `no_hit_fallback` 的 `citations=[]`
- `no_hit_fallback` 的 `retrieval_trace.filtered_candidates_count=0`

## Acceptance

完成后必须满足：

1. `GenericAssistantQueryRewriter` 使用 LLM，而不是固定拼接字符串
2. LLM rewrite 不会凭空追加通用文档词
3. LLM 输出不安全时会回退原 query
4. no-hit 样本不会因为 rewrite 产生伪 citations
5. normal-hit 样本不退化
6. `/chat` 非流式和 SSE 的 trace 仍能展示 rewrite 结果
7. `minimal` eval 回放通过

## Assumptions

- LLM rewrite 是检索增强步骤，不是 `/chat` 成功的硬依赖。
- LLM 调用失败时选择保守回退，不向用户返回 502。
- 本次只改 `generic_assistant` 的 docs-first query rewrite。
- 不新增环境变量；继续使用现有模型路由配置。
- 不在本次实现 ReRank 或额外检索平台。
