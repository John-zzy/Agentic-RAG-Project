# Elasticsearch Local Runtime

这个目录提供项目本地开发用的 Elasticsearch 单节点 `docker-compose`。

## 用法

在仓库根目录执行：

```bash
docker compose -f docs/elasticsearch/docker-compose.yml up -d
```

停止并清理容器：

```bash
docker compose -f docs/elasticsearch/docker-compose.yml down
```

## 默认约定

- HTTP 地址：`http://127.0.0.1:9200`
- 安全认证：关闭，便于本地开发直接接入当前项目默认配置
- 数据目录：挂载到 `backend/data/elasticsearch/`

## 对应后端环境变量

```env
AI_RAG_VECTOR_STORE__PROVIDER=elasticsearch
AI_RAG_VECTOR_STORE__ELASTICSEARCH__URL=http://127.0.0.1:9200
```
