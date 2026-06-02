# 项目文档索引

这个目录是项目长期文档入口，目标是让人和 LLM 都能先找到“应该读哪份”，再按模块逐步展开细节。README 和 AGENTS 只保留入口与导航，详细说明统一放在这里。

## 推荐阅读顺序

1. 先读 [系统架构图](./architecture/system-overview.svg)，理解 `platform / application / scenes` 三层关系。
2. 如果关注对话主链路，读 [Agentic RAG 设计说明](./rag/agentic-rag.md) 和 [Agentic RAG 流程图](./rag/agentic-rag-retrieval-flow.svg)。
3. 如果关注 LangGraph、HITL、状态治理，读 [LangGraph Runtime 图](./architecture/langgraph-runtime-current.svg)。
4. 如果关注知识库入库、预处理和索引，读 [知识管理流程图](./knowledge/knowledge-document-flow.svg)。
5. 如果要对接接口或排查字段，读 [API 文档](./reference/api-list.md) 和 [数据模型](./reference/data-model.md)。
6. 如果要运行 Elasticsearch 或排查本地问题，读 [运维与排障](./operations/common-pitfalls.md) 和 [Elasticsearch 本地运行](../../devops/elasticsearch/README.md)。

## 架构文档

- [系统架构图 SVG](./architecture/system-overview.svg)
- [系统架构图 Mermaid](./architecture/system-overview.mmd)
- [LangGraph Runtime 图 SVG](./architecture/langgraph-runtime-current.svg)
- [LangGraph Runtime 图 Mermaid](./architecture/langgraph-runtime-current.mmd)

## RAG 与 Agentic Retrieval

- [Agentic RAG 设计说明](./rag/agentic-rag.md)
- [Agentic RAG 工具清单说明](./rag/agentic-rag-tools.md)
- [Agentic RAG 流程图 SVG](./rag/agentic-rag-retrieval-flow.svg)
- [Agentic RAG 流程图 Mermaid](./rag/agentic-rag-retrieval-flow.mmd)

## 知识管理

- [知识管理流程图 SVG](./knowledge/knowledge-document-flow.svg)
- [知识管理流程图 Mermaid](./knowledge/knowledge-document-flow.mmd)

## 接口与数据模型

- [API 文档](./reference/api-list.md)
- [数据模型](./reference/data-model.md)

## 运维与排障

- [常见坑与排障](./operations/common-pitfalls.md)
- [Elasticsearch 本地运行](../../devops/elasticsearch/README.md)
- [Elasticsearch docker-compose](../../devops/elasticsearch/docker-compose.yml)

说明：Docker Compose、本地基础设施和部署脚本统一归档到 `devops/`。`docs/documents/` 只维护说明文档索引和项目架构资料。

## 资源文件

- [对话工作台截图](./assets/images/api-tester-ui.png)
- [知识库管理截图](./assets/images/knowledge-manager-ui.png)
- [设计稿资源目录](./assets/design/)

## 历史计划

历史设计方案仍保留在 `docs/plan/`，用于追溯需求演进和实施过程。这里不把所有 plan 平铺进主索引，避免 LLM 在查当前接口或架构时被旧方案干扰。
