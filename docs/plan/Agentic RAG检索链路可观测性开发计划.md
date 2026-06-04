# Agentic RAG检索链路可观测性开发计划

## Summary

本计划用于补齐当前 `/chat` 主链路的检索可观测性，在不改变回答语义的前提下，为每次 Agentic RAG 检索增加结构化 `retrieval_trace`。

目标是让一次请求返回后，外部可以清楚看到：

- 原始 query 和改写后 query
- Agentic RAG 调用了几次 tool、调用了什么 tool
- 每轮初召回多少片段、过滤后剩多少片段
- 最终 top chunks 的分数、来源和命中方式
- citations 来自哪些 chunk
- no-hit 时为什么 `knowledge_used=false`

本次只做最小本地可观测，不做前端 debug 页面、不接 OpenTelemetry / Jaeger、不接真实 ReRank。

## Key Changes

### 1. 新增 `/chat` 响应字段

在 `ChatResponse` 中新增可选字段 `retrieval_trace`：

```json
{
  "retrieval_trace": {
    "original_query": "string",
    "final_query": "string | null",
    "rewritten_query": "string | null",
    "tool_call_count": 1,
    "candidate_tools": ["knowledge_document_search"],
    "exit_reason": "sufficient | ask_user | max_rounds_reached | finished_by_judge",
    "knowledge_used": true,
    "raw_candidates_count": 5,
    "filtered_candidates_count": 3,
    "top_k_chunks": [],
    "citations": [],
    "rounds": []
  }
}
```

固定规则：

- `answer` 不变
- `knowledge_used` 不变
- `citations` 不变
- `retrieval_trace` 只新增观测信息，不参与主链判断

### 2. 每轮 Agentic tool 调用都记录 trace

`retrieval_trace.rounds[]` 固定包含：

```json
{
  "round_index": 1,
  "tool_name": "knowledge_document_search",
  "query": "本轮实际检索 query",
  "rewritten_query": "下一轮改写 query，没有则 null",
  "decision": "finish | rewrite | switch_tool | ask_user",
  "is_sufficient": true,
  "reason": "judge 给出的原因",
  "raw_candidates_count": 5,
  "filtered_candidates_count": 3,
  "document_count": 3,
  "success": true,
  "error": null,
  "top_k_chunks": []
}
```

这样可以直接回答：

- Agentic RAG 调用了几次 tool：看 `rounds.length`
- 每次调用的 tool：看 `rounds[].tool_name`
- 每次用的 query：看 `rounds[].query`
- 是否发生 query rewrite：看 `rounds[].rewritten_query`
- judge 为什么继续或停止：看 `rounds[].decision` 和 `rounds[].reason`

### 3. 文档检索服务补齐 raw / filtered 计数

在文档检索链路中补充 trace 数据：

- `raw_candidates_count`：`DocumentRetrievalService._retrieve_by_strategy()` 返回后的数量，即过滤前候选数
- `filtered_candidates_count`：低相关过滤 + 托管文档过滤后的最终数量
- `top_k_chunks`：最终进入 prompt / citations 候选池的 chunk 摘要

`top_k_chunks` 固定保留安全字段：

```json
{
  "rank": 1,
  "citation_id": "string",
  "document_id": "string | null",
  "chunk_id": "string | null",
  "chunk_index": 0,
  "source_name": "README.md",
  "source_path": "string | null",
  "score": 0.86,
  "vector_score": 0.81,
  "keyword_score": 0.45,
  "vector_rank": 1,
  "keyword_rank": 2,
  "matched_by": ["vector", "keyword"]
}
```

不在 `top_k_chunks` 中额外放完整正文；正文片段仍只沿用现有 `citations[].snippet`。

### 4. SSE 流式路径与非流式保持一致

`stream=false`：

- 直接在 JSON 响应中返回 `retrieval_trace`

`stream=true`：

- `tool` 事件返回 `retrieval_trace`
- `done` 事件也返回同一个 `retrieval_trace`
- `done.retrieval_trace` 与普通 `/chat` 响应结构一致

成功事件仍是：

```text
start -> history -> tool -> chunk... -> done
```

no-hit 事件仍是：

```text
start -> history -> tool -> chunk -> done
```

no-hit 时固定满足：

```json
{
  "knowledge_used": false,
  "citations": [],
  "retrieval_trace": {
    "knowledge_used": false,
    "filtered_candidates_count": 0,
    "citations": []
  }
}
```

### 5. Evaluation Harness 写入 trace

扩展 `backend/evals/run_http_eval.py`：

- `results[].observed.retrieval_trace` 保存非流式响应 trace
- `results[].stream.observed.retrieval_trace` 保存 SSE `done` trace
- `results[].stream.policy_evidence` 继续保留现有安全 policy 摘要
- minimal 样本新增 trace 存在性断言

固定验收：

- `no_hit_fallback`
  - `knowledge_used=false`
  - `citations=[]`
  - `retrieval_trace.knowledge_used=false`
  - `retrieval_trace.filtered_candidates_count=0`
- normal-hit
  - `knowledge_used=true`
  - `citations` 非空
  - `retrieval_trace.top_k_chunks` 非空
  - citation 的 `chunk_id/citation_id` 能在 `top_k_chunks` 中找到来源

### 6. README / Demo Path

README 增加一节“查看单次检索 trace”：

- 普通请求：查看 `/chat` JSON 响应里的 `retrieval_trace`
- 流式请求：查看 SSE `tool` 事件或最终 `done.retrieval_trace`
- Evaluation Harness：查看 `backend/data/evals/latest.json -> results[].observed.retrieval_trace`

同时把 README 近期规划中的“可观测性”更新为已落地或说明当前完成范围。

## Test Plan

运行针对性测试：

```powershell
backend\.venv\Scripts\python.exe -m pytest backend\tests\test_agentic_retrieval.py backend\tests\test_document_hybrid_retrieval.py backend\tests\test_eval_assets.py -q -c backend\tests\pytest.ini
```

补充或更新测试覆盖：

- 非流式 `/chat` 返回 `retrieval_trace`
- SSE `tool` 和 `done` 都返回 `retrieval_trace`
- SSE `done.retrieval_trace` 与非流式结构一致
- no-hit fallback 保持 `knowledge_used=false`、`citations=[]`
- normal-hit trace 中能解释 citations 来源
- raw count 大于等于 filtered count
- `top_k_chunks` 包含 score、chunk id、source、matched_by

端到端回放：

```powershell
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\latest.json
```

验收重点：

- `latest.json` 中 no-hit 样本 trace 存在且不产生伪引用
- normal-hit 样本 trace 存在且能解释 citation 来源
- stream 样本的 `done.retrieval_trace` 存在
- 原有 `answer`、`knowledge_used`、`citations` 断言不退化

## Acceptance

完成后，以下标准必须全部满足：

1. 能指出 retrieval trace 各字段在代码中的暴露位置
2. `/chat` 响应中包含 query rewrite 前后对比、召回数、过滤数和 top-k chunk score
3. SSE `tool` 事件和 `done` 事件都能看到与非流式一致的 trace
4. `no_hit_fallback` 回放通过，且 `knowledge_used=false`、`citations=[]`
5. normal-hit 回放通过，且 trace 能解释 citations 来源
6. README / Demo Path 说明如何查看一次请求的检索过程

## Assumptions

- `raw_candidates_count` 定义为过滤前的召回候选数量，不拆分 vector / keyword 两路数量。
- 本次不新增前端页面，只通过结构化字段和 eval artifact 查看 trace。
- 本次不改变 session 数据库结构，不把完整 trace 落库到历史消息。
- 本次不改变 retrieval 排序、过滤阈值、query rewrite 策略或 no-hit fallback 策略。
- `retrieval_trace` 默认作为开发和面试讲解用字段暴露；后续如果需要生产环境开关，再单独加配置。


