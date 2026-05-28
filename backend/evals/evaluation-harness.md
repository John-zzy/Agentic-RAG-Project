# Evaluation Harness 基础版

## 概览

`Evaluation Harness` 是当前仓库的可重复评测入口。`minimal` 继续作为端到端回归门禁，覆盖固定样本、真实 HTTP 回放、SSE 流式回放、可读日志、基本指标表格和样本级 pass/fail 断言；`retrieval_benchmark` 新增 qrels 驱动的检索 benchmark，用 eval-only probe 收集安全 ranked list，并计算 Precision@k / Recall@k / MRR / NDCG 等 IR 指标。

当前 CLI 入口位于 `backend/evals/`，通过独立 HTTP replay 脚本调用已经运行的本地后端，复用现有：

- `POST /files/upload`
- `POST /knowledge/documents`
- `POST /sessions`
- `POST /chat`
- `POST /chat` with `stream=true`

面向 UI 的读取和触发接口挂载在 `/evals`，只读取 eval artifact 或后台触发 allowlist 内的样本集，不修改 `/chat` 响应与 SSE 事件结构。

## 三维度覆盖结论

对照“检索质量、生成质量、系统性能”三类 RAG 评估目标，当前 harness 的定位应理解为**最小回归评测**，不是完整效果 benchmark。

| 维度 | 当前已覆盖 | 当前未覆盖 | 结论 |
| --- | --- | --- | --- |
| 检索质量 | `minimal` 覆盖 `knowledge_used`、citation 数量、预期来源命中、正文引用标记、no-hit 不返回伪引用、SSE `tool` 事件中的 retrieval policy evidence；`retrieval_benchmark` 覆盖 qrels、safe ranked list、Precision@k、Recall@k、MRR、NDCG@k、document recall、no-hit false positive rate | 尚未覆盖大规模样本、多场景、人工相关性复核和性能压测 | 能同时做端到端回归和首版检索排序量化 |
| 生成质量 | 答案关键词命中、引用来源命中、fallback 语义、正文引用标记、普通响应与 stream `done` 最终语义一致 | CR/AR/F 的 LLM-as-a-judge 或人工评分；多级事实性/忠实性评分；开放式答案质量评分 | 能做低成本生成回归门禁，还不能证明答案质量全面优秀 |
| 系统性能 | HTTP 回放完成率、错误样本数、SSE 是否完成并产生 chunk | 延迟分位数、吞吐量、并发压测、错误率时间序列、各阶段耗时拆分 | 能发现链路是否可用，还不能评估性能容量 |

因此，当前评估能给出的效果结论是：

> 在 `minimal` 固定样本集上，当前 `generic_assistant + documents` 主链可以稳定完成回放；命中类问题能返回预期来源和最小正确答案；no-hit 问题能避免伪引用；stream=true 与普通响应的关键语义一致。

但它不能替代完整 RAG 调优评估。如果要评估生成质量或性能容量，下一步仍应给生成结果增加 LLM judge 或人工评分，输出 CR / AR / F；给 HTTP replay 增加耗时记录，输出平均延迟、P95、错误率。

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
- `backend/evals/samples/retrieval_benchmark.json`
- `backend/evals/qrels/retrieval_benchmark.json`
- `backend/evals/fixtures/eval-harness-quickstart.md`
- `backend/evals/fixtures/eval-harness-it-policy.md`
- `backend/evals/fixtures/eval-harness-support-faq.md`
- `backend/evals/fixtures/eval-benchmark-*.md`
- `backend/evals/retrieval_probe.py`
- `backend/evals/retrieval_metrics.py`

运行命令：

```powershell
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set minimal --output backend\data\evals\latest.json
```

检索 benchmark 命令：

```powershell
backend\.venv\Scripts\python.exe backend\evals\run_http_eval.py --base-url http://127.0.0.1:8000 --sample-set retrieval_benchmark --output backend\data\evals\latest.json
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
9. 如果样本集声明 `qrels_path`，在 eval 进程内运行 `retrieval_probe.py`，只读复用 `generic_assistant` scene、`DocumentRetrievalService`、`AgenticRetriever` 和 scene retrieval policy
10. 对 qrels 与 safe ranked list 计算检索指标
11. 输出 `latest.json`、`latest.md`、`runs/<run_id>.json` 和 `runs/index.json`
12. 如果任一样本断言失败，以非 0 退出码结束
13. 如果传入 `--compare-to`，在 candidate 输出旁生成 `*.compare.json` 和 `*.compare.md`

## 样本与文体映射

| sample_id | 文档体裁 | query | 主要检查点 |
| --- | --- | --- | --- |
| `quickstart_setup_requirement` | quickstart guide | `开始使用前需要先安装什么版本的 Python？` | 观察是否命中文档、答案是否包含 `Python 3.11`、引用里是否出现目标来源 |
| `it_policy_mfa_rule` | security policy | `远程访问公司系统时有什么安全要求？` | 观察是否命中文档、答案是否包含 `MFA` 或 `双重认证`、引用里是否出现目标来源 |
| `faq_response_sla` | support FAQ | `普通支持请求通常多久会得到首次响应？` | 观察是否命中文档、答案是否包含 `1 个工作日`、引用里是否出现 support FAQ |
| `no_hit_fallback` | 无 | `VOID-ALPHA-7788 secret handshake?` | 严格断言 no-hit fallback：`knowledge_used=false`、`citations=[]`、回答语义为缺乏可靠资料 |

其中 `quickstart_setup_requirement` 和 `no_hit_fallback` 固定启用 `eval_stream=true`，用于覆盖一条 normal-hit 流式路径和一条 no-hit 流式路径。

## 输出 JSON 结构

输出文件默认写到 `backend/data/evals/latest.json`，同时会生成 `backend/data/evals/latest.md` 表格报告。每次运行还会写入 `backend/data/evals/runs/<run_id>.json`，并更新 `backend/data/evals/runs/index.json`，供 UI 列表读取历史 run。

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

如果样本集声明 qrels，payload 会额外包含：

- `retrieval_benchmark.qrels`: 原始 qrels 标注。
- `retrieval_benchmark.probe`: eval-only probe 的安全输出。
- `retrieval_benchmark.sample_metrics`: 按 `sample_id` 对齐的检索指标。
- `retrieval_benchmark.aggregate_metrics`: 聚合后的 Precision@k、Recall@k、MRR、NDCG@k、document recall、expected document hit、no-hit false positive rate。
- `results[].retrieval.qrels`: 当前样本的 qrels。
- `results[].retrieval.ranked_list`: 只包含 `rank`、`source_doc`、`document_id`、`chunk_id`、`chunk_index`、`score`、`matched_by` 的 ranked list。
- `results[].retrieval.metrics`: 当前样本的检索指标。
- `results[].retrieval.failure_reasons`: probe 或指标计算失败原因。

`retrieval_probe.py` 和 `/evals` API 都会过滤 snippet、content、prompt、reason、rewrite reason、raw fixture content 等文本字段。CLI 的 HTTP 回放 artifact 仍保留 `observed.answer` 供本地调试；UI API 会移除完整 answer，仅保留 preview 和结构化摘要。

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

## retrieval_benchmark 指标口径

`retrieval_benchmark` 首版包含 5 份 benchmark fixture、16 条样本和 qrels。chunk 级 qrels 使用 `source_doc + chunk_index`，document 级 qrels 使用 `source_doc`，相关性分值为 `1=相关`、`2=强相关`。

默认计算 `k=[1,3,5]`：

- `precision_at_k`: top-k 中相关 chunk 的比例。
- `recall_at_k`: top-k 覆盖相关 chunk 的比例。
- `mrr`: 第一个相关 chunk 排名的倒数。
- `ndcg_at_k`: 使用 qrels relevance 加权的排序质量。
- `document_recall_at_k`: top-k 覆盖预期 source document 的比例。
- `expected_document_hit`: ranked list 是否命中任一预期文档。
- `no_hit_false_positive_rate`: no-hit 样本仍返回 ranked chunk 的比例。

no-hit 样本 qrels 为空，只参与 `no_hit_false_positive_rate`，不参与核心 IR 指标平均。ranked list 会按 `source_doc + chunk_index` 优先去重，其次按 `chunk_id` 去重。

## `/evals` API

`/evals` API 面向后续 UI 消费 eval artifact：

- `GET /evals/latest`: 返回当前 `latest.json` 的 sanitized 视图；不存在时返回结构化 404。
- `GET /evals/runs`: 从 `runs/index.json` 返回历史 run 列表；不存在时返回空列表。
- `GET /evals/runs/{run_id}`: 返回单个 run artifact 的 sanitized 视图；拒绝路径穿越。
- `GET /evals/runs/{run_id}/status`: 返回 `queued`、`running`、`succeeded`、`failed` 或 `not_found`。
- `POST /evals/runs`: 后台触发 allowlist 内的 eval，默认 `retrieval_benchmark`，仅允许 `minimal` 和 `retrieval_benchmark`。`base_url` 从当前请求推导，不接受任意外部 URL。

同一进程同时只允许一个后台 eval run；如果已有 run queued/running，触发接口返回 `409 EVAL_RUN_ALREADY_RUNNING`。后台状态是进程内状态，服务重启后状态接口会优先从已落盘 artifact 推断历史 run。

## 当前指标口径

- `completion_rate = 成功完成调用的样本数 / 总样本数`
- `sample_pass_rate = 断言通过样本数 / 总样本数`
- `knowledge_hit_rate = knowledge_used=true 的样本数 / 成功调用样本数`
- `citation_presence_rate = citations 非空的样本数 / 成功调用样本数`
- `answer_keyword_hit_rate = 答案命中预设关键词的样本数 / 成功调用样本数`
- `expected_source_hit_rate = citations 中出现目标来源的样本数 / 成功调用样本数`
- `stream_pass_rate = stream=true 回放断言通过样本数 / 启用流式回放样本数`

这些指标里，`expected_source_hit_rate` 最接近检索相关性判断，但它只判断 citation 是否指向预期文档，不等价于 Precision@k / Recall@k / MRR / NDCG。`answer_keyword_hit_rate` 最接近答案相关性判断，但它只是关键词级断言，不等价于 CR / AR / F 的完整生成质量评分。`completion_rate` 和 `stream_pass_rate` 是链路可用性指标，不等价于延迟、吞吐量和错误率压测。

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
4. 先看 `Conclusion`：确认本次是否可以作为当前 RAG 主链通过的证据
5. 再看 `Current Scorecard`、`Retrieval Quality`、`Generation Quality`、`System Quality`：关注当前已经能自动评测的效果指标
6. 查看 `Sample Results`、`no-hit Boundary`、`SSE Evidence`：确认 3 条命中样本、1 条 no-hit 样本和 stream=true 路径各自结果
7. 如果失败，再看 `Failure Focus`，并打开 `backend/data/evals/latest.json` 查看单条样本明细、`assertions`、`observed.knowledge_used`、`observed.citations` 和 `stream.observed`
8. 最后看 `Benchmark Gaps` 和 `Pending Development Items`：这些是不满足完整 benchmark 的待开发项，需要人工确认后再实现
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
