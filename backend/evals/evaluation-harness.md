# Evaluation Harness 基础版

## 概览

`Evaluation Harness 基础版` 是当前仓库的最小可重复评测入口，目标不是做大规模 benchmark，也不是引入 LLM-as-a-judge，而是给 `generic_assistant + documents` 主链补上一套固定样本、真实回放命令、SSE 流式回放、可读日志、基本指标表格和样本级 pass/fail 断言。

当前入口位于 `backend/evals/`，通过独立 HTTP replay 脚本调用已经运行的本地后端，复用现有：

- `POST /files/upload`
- `POST /knowledge/documents`
- `POST /sessions`
- `POST /chat`
- `POST /chat` with `stream=true`

## 为什么它是通用型评测集

这套样本刻意对齐“通用文档 RAG”而不是垂直业务知识库：

- 固定走 `generic_assistant` scene
- 固定只挂载 `documents`
- 样本文档覆盖 `quickstart`、`IT policy`、`support FAQ` 三种常见文体
- 样本不依赖商品、订单、库存、SKU 或业务主键
- 输出重点是回放日志、引用来源观察和基础指标，不依赖复杂自动评分

因此它更适合作为“平台级 RAG 主链验证”，而不是某个行业 demo 的效果展示。

## 目录与入口

评测资产固定放在：

- `backend/evals/run_http_eval.py`
- `backend/evals/samples/minimal.json`
- `backend/evals/fixtures/eval-harness-quickstart.md`
- `backend/evals/fixtures/eval-harness-it-policy.md`
- `backend/evals/fixtures/eval-harness-support-faq.md`

运行命令：

```powershell
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\latest.json
```

baseline / candidate 对比命令：

```powershell
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\baseline.json
# 调整 scene retrieval policy、阈值或 ReRank 实现后再跑 candidate
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\candidate.json --compare-to backend\data\evals\baseline.json
```

脚本流程固定为：

1. 检查 `/health`
2. 清理旧的 `eval-harness-*` 上传文件
3. 清理旧的评测文档记录
4. 上传 3 份固定 fixture
5. 重新走 `/knowledge/documents` 入库
6. 每条样本创建独立 session
7. 调 `/chat` 回放普通响应
8. 对配置了 `eval_stream=true` 的样本调 `/chat stream=true` 回放 SSE
9. 输出 `latest.json` 和 `latest.md`
10. 如果任一样本断言失败，以非 0 退出码结束
11. 如果传入 `--compare-to`，在 candidate 输出旁生成 `*.compare.json` 和 `*.compare.md`

## 样本与文体映射

| sample_id | 文档体裁 | query | 主要检查点 |
| --- | --- | --- | --- |
| `quickstart_setup_requirement` | quickstart guide | `开始使用前需要先安装什么版本的 Python？` | 观察是否命中文档、答案是否包含 `Python 3.11`、引用里是否出现目标来源 |
| `it_policy_mfa_rule` | security policy | `远程访问公司系统时有什么安全要求？` | 观察是否命中文档、答案是否包含 `MFA` 或 `双重认证`、引用里是否出现目标来源 |
| `faq_response_sla` | support FAQ | `普通支持请求通常多久会得到首次响应？` | 观察是否命中文档、答案是否包含 `1 个工作日`、引用里是否出现 support FAQ |
| `no_hit_fallback` | 无 | `VOID-ALPHA-7788 secret handshake?` | 严格断言 no-hit fallback：`knowledge_used=false`、`citations=[]`、回答语义为缺乏可靠资料 |

其中 `quickstart_setup_requirement` 和 `no_hit_fallback` 固定启用 `eval_stream=true`，用于覆盖一条 normal-hit 流式路径和一条 no-hit 流式路径。

## 输出 JSON 结构

输出文件默认写到 `backend/data/evals/latest.json`，同时会生成 `backend/data/evals/latest.md` 表格报告。

JSON 字段目前固定包含：

- `run_id`
- `executed_at`
- `base_url`
- `sample_set`
- `summary`
- `results`

每条 `results[]` 目前包含：

- `sample_id`
- `query`
- `target`
- `passed`
- `failure_reasons`
- `assertions`
- `observed`
- `metrics`
- `stream`
- `status`

`observed` 至少保存：

- `answer`
- `answer_preview`
- `knowledge_used`
- `citation_count`
- `citation_sources`
- `citations`
- `session_id`
- `request_id`

其中 `assertions[]` 会逐项记录：

- `name`
- `expected`
- `actual`
- `passed`

如果样本失败，`failure_reasons[]` 会把失败断言的预期值和实际值写出来。例如 no-hit fallback 如果回归成伪引用，报告会暴露实际的 `knowledge_used` 和 `citations`。

配置了 `eval_stream=true` 的样本会额外记录 `stream` 字段：

- `stream.passed`
- `stream.failure_reasons`
- `stream.assertions`
- `stream.observed`
- `stream.metrics`
- `stream.event_types`
- `stream.chunk_count`
- `stream.chunk_text_preview`
- `stream.policy_evidence`

流式断言以 SSE `done` 事件作为最终权威结果；`chunk` 只用于验证流式链路确实有增量输出。

SSE `tool` 事件还会暴露当前 scene retrieval policy 的安全摘要，用于对比不同策略下的检索行为。该摘要只包含 `top_k`、`min_relevance_score`、`recall_strategy`、`no_hit_strategy`、`rerank_enabled`、`rerank_top_n` 等配置字段，不包含 prompt、密钥或原始业务数据。

`stream.policy_evidence` 从 SSE `tool` 事件提取安全字段：

- `mode`
- `retrieval_policy`
- `candidate_tools`
- `documents`
- `exit_reason`
- `success`
- `rounds[].tool_name`
- `rounds[].decision`
- `rounds[].result_count`
- `rounds[].document_count`
- `rounds[].rerank`

它不会保存 query、prompt、citation snippet 或原始私有文档内容。

## baseline / candidate 对比报告

传入 `--compare-to <baseline.json>` 后，脚本仍会正常生成 candidate 的 `candidate.json` 和 `candidate.md`，并额外生成：

- `candidate.compare.json`
- `candidate.compare.md`

对比按稳定的 `sample_id` 匹配样本，不依赖列表位置。报告会显式标记：

- baseline 里有、candidate 里没有的 `missing_candidate`
- candidate 里有、baseline 里没有的 `missing_baseline`
- 两边都有但关键字段变化的 `changed`
- 两边关键字段一致的 `same`

样本级对比字段包括：

- pass 状态
- `knowledge_used`
- citation 数量
- citation 来源名称
- no-hit 场景下 `citations=[]` 是否仍成立
- stream pass 状态、事件类型、最终 `knowledge_used` 和最终 citation 数量
- SSE `tool` 事件捕获到的 policy evidence

典型用法是先保存一份 baseline，再调整 `SceneRetrievalPolicy`、相关性阈值或真实 ReRank provider，随后跑 candidate 并使用 `--compare-to`。如果 `no_hit_fallback` 从 `knowledge_used=false/citations=[]` 变成了伪引用，candidate 自身断言会失败，对比报告也会在 `no_hit_citations_empty`、`knowledge_used` 和 `citation_count` 中暴露差异。

## 当前指标口径

- `completion_rate = 成功完成调用的样本数 / 总样本数`
- `sample_pass_rate = 断言通过样本数 / 总样本数`
- `knowledge_hit_rate = knowledge_used=true 的样本数 / 成功调用样本数`
- `citation_presence_rate = citations 非空的样本数 / 成功调用样本数`
- `answer_keyword_hit_rate = 答案命中预设关键词的样本数 / 成功调用样本数`
- `expected_source_hit_rate = citations 中出现目标来源的样本数 / 成功调用样本数`
- `stream_pass_rate = stream=true 回放断言通过样本数 / 启用流式回放样本数`

## no-hit fallback 回归口径

`minimal` 样本集固定包含 `no_hit_fallback`。该样本在上传 fixture 之前回放，避免固定评测文档对陌生 query 产生干扰。

该样本的验收口径是：

- `passed=true`
- `observed.knowledge_used=false`
- `observed.citation_count=0`
- `observed.citations=[]`
- `metrics.fallback_like=true`

如果后续修改 ReRank、scene retrieval policy、query rewrite、检索阈值或 citation 组装逻辑，导致 no-hit 又返回 citations，则该样本应失败。定位时优先看：

- `results[].failure_reasons`
- `results[].assertions`
- `results[].observed.knowledge_used`
- `results[].observed.citations`

## SSE stream=true 回归口径

`minimal` 样本集当前固定对 `quickstart_setup_requirement` 和 `no_hit_fallback` 启用流式回放。

流式回放验收口径是：

- SSE 事件流必须包含 `done`
- SSE 事件流必须包含至少 1 个 `chunk`
- `done.answer` 满足样本答案断言
- `done.knowledge_used` 满足样本预期
- `done.citations` 满足样本引用断言
- `done.knowledge_used` 与普通响应的 `knowledge_used` 保持一致
- `done.citations` 数量与普通响应的 `citations` 数量保持一致
- no-hit 流式样本仍必须满足 `knowledge_used=false`、`citations=[]`

如果后续修改 SSE 编码、前端流式适配、主链响应组装或 fallback 逻辑，导致 `done` 缺失、`error` 事件出现，或最终引用语义和普通响应不一致，则该样本应失败。定位时优先看：

- `results[].stream.failure_reasons`
- `results[].stream.assertions`
- `results[].stream.event_types`
- `results[].stream.observed`

## 验证记录

### 2026-05-25

- 评测入口：`backend/evals/run_http_eval.py`
- 样本集：`minimal`
- 验证方式：
  - 静态校验：`backend/tests/test_eval_assets.py`
  - 真实回放：本地启动 `backend/run.py` 后执行 HTTP replay
- 结果记录：
  - 结构化输出文件：`backend/data/evals/latest.json`
  - 表格输出文件：`backend/data/evals/latest.md`
  - 本文档记录当前口径与样本定义

如果本地环境已启动并跑过回放，可直接打开 `backend/data/evals/latest.md` 做演示，再用 `backend/data/evals/latest.json` 查看明细。

## README / Demo Path

推荐演示路径：

1. 启动 `backend/run.py`
2. 运行 `backend/evals/run_http_eval.py`
3. 打开 `backend/data/evals/latest.md`
4. 先展示 Sample Table 里的 `pass`、`stream`、`knowledge`、`citations` 和 `failure`
5. 说明主链顺序：Hybrid Search 先召回与融合，scene retrieval policy 控制数量、阈值和 ReRank 接入位，后续可用同一评测集做策略对比
6. 针对 `no_hit_fallback` 确认 `pass=yes`、`knowledge=no`、`citations=0`
7. 查看 `SSE Stream Evidence`，确认关键样本 `stream_pass=yes` 且事件包含 `done`
8. 再打开 `backend/data/evals/latest.json` 查看单条样本明细、`assertions`、`observed.knowledge_used`、`observed.citations` 和 `stream.observed`
9. 调整 scene 策略或 ReRank 实现后，用 `--compare-to backend\data\evals\baseline.json` 生成 candidate 对比报告，展示策略变更是否影响 normal-hit、no-hit 和 SSE 关键语义

## JD 证明点

- 不是只会做 RAG 主链，还补了固定样本与验证入口
- 有可重复 replay 命令，不靠临场手工提问演示
- 能把“命中文档”“引用来源”“基础回退表现”落成结构化日志和表格
- 能把 `stream=true` 的真实对话路径纳入回归，避免流式响应和普通响应语义漂移
- 后续可以继续接入 scene retrieval policy 对比、rerank、更多样本、更多场景，而不需要重做入口

## 简历素材草稿

可表述为：

> 为 scene-based RAG 后端补齐基础版 Evaluation Harness：设计固定样本集、实现 HTTP replay 与 SSE stream replay，复用 `/files` `/knowledge/documents` `/sessions` `/chat` 主链，输出回放日志、结构化结果和基础指标表格，用于展示文档命中、引用来源、no-hit fallback 和流式回答稳定性。
