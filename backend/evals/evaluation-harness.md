# Evaluation Harness 基础版

## 概览

`Evaluation Harness 基础版` 是当前仓库的最小可重复评测入口，目标不是做大规模 benchmark，也不是今天就做完整自动评分，而是给 `generic_assistant + documents` 主链补上一套固定样本、真实回放命令、可读日志和基本指标表格。

当前入口位于 `backend/evals/`，通过独立 HTTP replay 脚本调用已经运行的本地后端，复用现有：

- `POST /files/upload`
- `POST /knowledge/documents`
- `POST /sessions`
- `POST /chat`

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

脚本流程固定为：

1. 检查 `/health`
2. 清理旧的 `eval-harness-*` 上传文件
3. 清理旧的评测文档记录
4. 上传 3 份固定 fixture
5. 重新走 `/knowledge/documents` 入库
6. 每条样本创建独立 session
7. 调 `/chat` 回放
8. 输出 `latest.json` 和 `latest.md`

## 样本与文体映射

| sample_id | 文档体裁 | query | 主要检查点 |
| --- | --- | --- | --- |
| `quickstart_setup_requirement` | quickstart guide | `开始使用前需要先安装什么版本的 Python？` | 观察是否命中文档、答案是否包含 `Python 3.11`、引用里是否出现目标来源 |
| `it_policy_mfa_rule` | security policy | `远程访问公司系统时有什么安全要求？` | 观察是否命中文档、答案是否包含 `MFA` 或 `双重认证`、引用里是否出现目标来源 |
| `faq_response_sla` | support FAQ | `普通支持请求通常多久会得到首次响应？` | 观察是否命中文档、答案是否包含 `1 个工作日`、引用里是否出现 support FAQ |
| `no_hit_fallback` | 无 | `VOID-ALPHA-7788 secret handshake?` | 观察模型在陌生 query 下的回退表达、知识使用情况和引用情况 |

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
- `observed`
- `metrics`
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

## 当前指标口径

- `completion_rate = 成功完成调用的样本数 / 总样本数`
- `knowledge_hit_rate = knowledge_used=true 的样本数 / 成功调用样本数`
- `citation_presence_rate = citations 非空的样本数 / 成功调用样本数`
- `answer_keyword_hit_rate = 答案命中预设关键词的样本数 / 成功调用样本数`
- `expected_source_hit_rate = citations 中出现目标来源的样本数 / 成功调用样本数`

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
4. 先展示 Sample Table 和 Metrics Table
5. 再打开 `backend/data/evals/latest.json` 查看单条样本明细和 `citations`

## JD 证明点

- 不是只会做 RAG 主链，还补了固定样本与验证入口
- 有可重复 replay 命令，不靠临场手工提问演示
- 能把“命中文档”“引用来源”“基础回退表现”落成结构化日志和表格
- 后续可以继续接入 rerank、更多样本、更多场景，而不需要重做入口

## 简历素材草稿

可表述为：

> 为 scene-based RAG 后端补齐基础版 Evaluation Harness：设计固定样本集、实现 HTTP replay 脚本、复用 `/files` `/knowledge/documents` `/sessions` `/chat` 主链，输出回放日志、结构化结果和基础指标表格，用于展示文档命中、引用来源和回答稳定性。
