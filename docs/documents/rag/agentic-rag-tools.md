# Agentic RAG 工具清单说明

本文说明当前项目中，供 `AgenticRetriever` 调用的 retrieval tools 有多少个、分别做什么，以及它们在运行时如何被筛选。

## 1. 先说结论

当前 `AgenticRetriever` 注册的 retrieval tools 总数是：

```text
6 个
```

分别是：

1. `knowledge_document_search`
2. `product_semantic_search`
3. `review_semantic_search`
4. `order_semantic_search`
5. `inventory_lookup`
6. `product_detail_lookup`

这些工具由 `build_agentic_retrieval_tools()` 统一构建，代码位置见：

- `backend/scenes/ecommerce/retrieval_tools.py`

其中实际返回的工具集合位于：

- `build_agentic_retrieval_tools()`：`backend/scenes/ecommerce/retrieval_tools.py:368`

## 2. 为什么是 6 个

`build_agentic_retrieval_tools()` 当前返回的是一个固定长度的工具元组：

- 3 个语义检索工具
- 1 个文档检索工具
- 2 个结构化精确查询工具

对应代码中的返回顺序为：

1. `product_semantic_search`
2. `review_semantic_search`
3. `order_semantic_search`
4. `knowledge_document_search`
5. `inventory_lookup`
6. `product_detail_lookup`

也就是说，从“已注册到 AgenticRetriever 的工具总量”这个口径来看，当前就是 6 个。

## 3. 每个工具的作用

### `knowledge_document_search`

作用：

- 查询用户上传的文档知识
- 内部调用 `DocumentRetrievalService.retrieve()`
- 支持语义召回 + 关键词召回 + Hybrid Fusion + 过滤

适合问题：

- 文档问答
- FAQ、制度、规则、手册、说明类问题

### `product_semantic_search`

作用：

- 在商品知识中做语义检索
- 用于先定位候选商品

适合问题：

- “AeroPhone X 怎么样？”
- “有没有适合拍照的手机？”

### `review_semantic_search`

作用：

- 在评论/评价知识中做语义检索
- 补充口碑、优缺点、推荐理由

适合问题：

- “这款手机值不值得买？”
- “用户评价怎么样？”

### `order_semantic_search`

作用：

- 在订单知识中做语义检索
- 用于物流、订单状态、订单相关语义匹配

适合问题：

- “订单 O202604210010 到哪了？”
- “我的包裹什么时候送到？”

### `inventory_lookup`

作用：

- 按 `product_id` 做结构化精确查询
- 返回库存状态、库存数量、仓库等信息

适合问题：

- “AeroPhone X 现在有货吗？”

说明：

- 它通常不是第一轮直接使用
- 更常见路径是先 `product_semantic_search` 找到商品，再切到 `inventory_lookup`

### `product_detail_lookup`

作用：

- 按 `product_id` 做结构化精确查询
- 返回价格、规格、描述等详情

适合问题：

- “这款手机多少钱？”
- “这款手机的配置是什么？”

说明：

- 它通常也是在商品语义检索之后作为补查工具使用

## 4. 注册数量不等于单次请求可用数量

虽然 `AgenticRetriever` 注册了 6 个工具，但一次 `/chat` 请求里，真正允许使用哪些工具，不是只看注册，还要看 session 的：

```text
mounted_knowledge_sources
```

运行时筛选逻辑在：

- `backend/application/runtime/service.py:291`

当前规则是：

- 挂载 `documents` 时，加入 `knowledge_document_search`
- 挂载 `ecommerce` 时，加入：
  - `product_semantic_search`
  - `review_semantic_search`
  - `order_semantic_search`
  - `inventory_lookup`
  - `product_detail_lookup`

## 5. 所以“当前有多少 tool 可调用”要分三种口径

### 口径一：注册到 AgenticRetriever 的总数

```text
6 个
```

这是最稳定、最完整的口径。

### 口径二：默认新会话实际可用数量

默认新会话挂载的是：

```text
["documents"]
```

所以默认情况下，单次请求实际可用的 retrieval tool 数量是：

```text
1 个
```

即：

1. `knowledge_document_search`

### 口径三：当会话挂载 `["documents", "ecommerce"]` 时

这时实际可用的 retrieval tool 数量是：

```text
6 个
```

即文档工具 1 个，加上电商工具 5 个。

### 补充：如果会话只挂载 `["ecommerce"]`

那么实际可用数量会是：

```text
5 个
```

即：

1. `product_semantic_search`
2. `review_semantic_search`
3. `order_semantic_search`
4. `inventory_lookup`
5. `product_detail_lookup`

## 6. 两个 scene 下是否一样

当前 `generic_assistant` 和 `ecommerce` 两个 scene，在 Agentic Retrieval 层面都调用了同一个：

```text
build_agentic_retrieval_tools()
```

所以从“注册了多少个 tools”这个角度看，二者当前是一致的，都是 6 个。

对应代码位置：

- `backend/scenes/generic_assistant/definition.py:230`
- `backend/scenes/ecommerce/definition.py:381`

两者差异主要不在工具总数，而在：

- scene 的系统提示词
- fallback 行为
- session 当前挂载了哪些知识源
- `SufficiencyJudge` 在不同问题上的决策路径

## 7. 与 Agentic RAG 流程的关系

这些 tools 不是最终回答工具，而是 Agentic RAG 在 retrieval 阶段可轮流调用的“证据收集工具”。

大致流程是：

1. `AgenticRetriever` 选一个 tool
2. tool 执行一次 `retrieve()`
3. 返回 `RetrievalResult`
4. `SufficiencyJudge` 判断证据够不够
5. 不够则 `switch_tool` 或 `rewrite`
6. 够了再进入最终回答生成

所以：

- `tool 数量` 决定了 Agentic RAG 可切换的证据收集能力上限
- `mounted_knowledge_sources` 决定了本次会话真正开放给 Agentic RAG 的工具范围

## 8. 一句话总结

当前项目中，`AgenticRetriever` 注册的 retrieval tools 总数是 **6 个**；但默认新会话通常只实际开放 **1 个** 文档工具。只有当会话挂载了 `ecommerce` 知识源后，另外 **5 个** 电商相关工具才会进入本次 Agentic RAG 的可调用集合。


