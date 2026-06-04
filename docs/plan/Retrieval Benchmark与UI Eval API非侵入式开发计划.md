# Retrieval Benchmark + UI Eval API Plan

## Summary

实现非侵入式 retrieval benchmark，并补充面向 UI 的 eval artifact 读写接口。检索评测仍通过 eval-only probe 复用现有检索组件，不修改 `/chat` 响应、不修改 SSE 事件结构、不改变业务主链路。

UI 接口只暴露 eval 能力：读取历史 run、读取单个 run、获取 latest、触发一次后台 benchmark run。触发接口不直接阻塞等待完整评测完成。

## Key Changes

### 新增 eval-only retrieval probe

- 放在 `backend/evals/retrieval_probe.py`
- 复用 `generic_assistant` scene、`DocumentRetrievalService`、`AgenticRetriever`、现有 scene retrieval policy
- 只读访问同一套 vector store，不改 `application/runtime/service.py` 的业务执行链路
- ranked list 只保留安全字段：`rank`、`source_doc`、`document_id`、`chunk_id`、`chunk_index`、`score`、`matched_by`
- 不输出 snippet、正文、prompt、rewrite reason 等文本泄露字段

### 新增检索指标模块

- 放在 `backend/evals/retrieval_metrics.py`
- 默认 `k=[1,3,5]`
- chunk 级指标：`precision_at_k`、`recall_at_k`、`mrr`、`ndcg_at_k`
- document 级指标：`document_recall_at_k`、`expected_document_hit`
- no-hit 样本单独统计 `no_hit_false_positive_rate`，不参与核心 IR 平均

### 扩展 eval runner

- `minimal` 保持原有 HTTP replay 行为
- sample set 声明 `qrels_path` 时，完成 fixture 上传和知识入库后额外运行 retrieval probe
- 将 qrels、ranked list、per-sample metrics、aggregate metrics 写入结果
- 保留 `backend/data/evals/latest.json` 和 `latest.md`
- 新增 `backend/data/evals/runs/<run_id>.json`
- 新增 `backend/data/evals/runs/index.json`

### 新增 UI Eval API

- 新建 `backend/application/runtime/api/evals/routes.py` 和 `schemas.py`
- 在 `create_app()` 中 include 新 router
- API 前缀使用 `/evals`，tags 使用 `evals`
- `GET /evals/latest`：返回 `latest.json` 的安全摘要和完整指标结构
- `GET /evals/runs`：读取 `runs/index.json`，返回历史 run 列表
- `GET /evals/runs/{run_id}`：读取 `runs/<run_id>.json`
- `POST /evals/runs`：后台触发 eval run，默认 sample set 为 `retrieval_benchmark`
- `GET /evals/runs/{run_id}/status`：返回 `queued/running/succeeded/failed/not_found`
- 单进程内只允许一个 eval run 运行；并发触发返回 `409 EVAL_RUN_ALREADY_RUNNING`
- 只允许运行 `minimal` 和 `retrieval_benchmark`，禁止任意路径或任意命令参数
- API 不暴露 raw fixture content、snippet、prompt、secret、完整 answer 正文；如需展示回答，只返回已有 preview 字段

## Sample Dataset Changes

### 保留 minimal sample set

- 保留 `backend/evals/samples/minimal.json`
- 继续覆盖 3 条命中样本、1 条 no-hit 样本
- 继续验证 `/chat`、citation、fallback、stream 一致性

### 新增 retrieval benchmark sample set

- 新增 `backend/evals/samples/retrieval_benchmark.json`
- `sample_set: "retrieval_benchmark"`
- `namespace: "documents"`
- `append_eval_anchors: false`
- `qrels_path: "retrieval_benchmark.json"`

### 新增 qrels

- 新增 `backend/evals/qrels/retrieval_benchmark.json`
- document qrels 使用 `source_doc`
- chunk qrels 使用 `source_doc + chunk_index`
- relevance 使用 `1=相关`、`2=强相关`
- no-hit 样本 qrels 为空

### 新增 benchmark fixtures

- `eval-benchmark-quickstart.md`
- `eval-benchmark-security-policy.md`
- `eval-benchmark-support-faq.md`
- `eval-benchmark-release-runbook.md`
- `eval-benchmark-access-control.md`

### retrieval_benchmark 首版样本

- 首版放 16 条样本
- 覆盖单文档精确命中、相似干扰、多文档、英文查询、no-hit
- fixture 内容定稿后，用当前 chunker 生成 chunk index，再填写 qrels
- 后续修改 fixture 必须同步校验 qrels

## Test Plan

### 单元测试 retrieval metrics

- `Precision@k`、`Recall@k`、`MRR`、`NDCG@k`
- 多相关 chunk、无命中、空 ranked list、重复 chunk 去重
- document 级与 chunk 级 qrels 分别计算正确

### 单元测试 retrieval probe

- probe 不返回 snippet/content/prompt/rewrite reason
- ranked list 字段稳定且安全
- no-hit query 返回空 ranked list 或不命中 qrels

### 扩展 eval asset 测试

- `retrieval_benchmark.json` manifest 合法
- qrels 的 `sample_id` 都存在
- qrels 的 `source_doc` 都存在于 fixtures
- hit 样本必须有 qrels，no-hit 样本 qrels 必须为空

### 新增 UI Eval API 测试

- `GET /evals/latest` 在无 artifact 时返回结构化 404
- `GET /evals/runs` 可读取 index
- `GET /evals/runs/{run_id}` 拒绝路径穿越，只读 runs 目录内 JSON
- `POST /evals/runs` 只接受 allowlist sample set
- 并发触发第二个 run 返回 409
- API 响应不包含 snippet/content/prompt/reason 等敏感字段

### 回归验证

- 跑 `minimal`，确认原有 4/4 回归不变
- 跑 `retrieval_benchmark`，确认 JSON/Markdown 输出包含 IR 指标
- 确认 `backend/data/evals/runs/index.json` 可列出历史 run
- 跑受影响测试：`test_eval_assets.py`、新增 metrics/probe/API 测试、`test_chat_api.py`

## Assumptions

- UI 接口本次只做后端 API，不实现前端页面。
- 触发 run 使用 FastAPI 后台任务或等价的单进程后台执行，不引入队列、Redis 或任务调度系统。
- eval run 的 `base_url` 默认由当前请求推导，例如 `http://host:port`，不允许 UI 传任意外部 URL。
- 本次不新增生成质量 judge、不做性能压测、不覆盖 ecommerce。
- 非侵入式探针允许 eval 代码 import 现有业务组件，但不得要求业务组件新增 eval-only 参数或字段。


