# NeSy Reasoning MCP Server — 完整设计规格文档 v2.0

> 文档状态：设计规格草案  
> 目标协议：MCP Specification 2025-11-25  
> 目标实现：Python 3.11+、MCP Python SDK、NetworkX/SQLite  
> 适用对象：Claude Code、Claude Desktop、Cursor、Codex、其他支持 MCP 的 Agent harness

---

## 0. 摘要

NeSy Reasoning MCP Server 是一个面向 AI Agent 的确定性符号推理能力层。它把 LLM 不稳定的自然语言推理任务拆成两段：

1. LLM 负责提出候选命题、候选因果关系、候选约束。
2. MCP Server 负责在结构化关系图上执行可追溯、可验证、可测试的符号推理。

本规格保留原始方案的完整范围：

- 因果关系分类：充分、必要、充要、未知、矛盾。
- 多步传递链验证。
- 逻辑矛盾检测。
- 反事实推理。
- 关系加载、导出、清空、列举。
- 显式互斥关系声明。
- Claude Code Hook 集成。
- 路线图、技术栈、测试、未来扩展。

同时，本规格修正原始文档中的三个关键问题：

- 不再把“没有找到替代路径”当成“必然阻断”。默认采用开放世界语义。
- 不再把“存在另一个充分条件”当成“原条件不是必要条件”。除非存在独立性或互斥反例证明。
- 不再只写业务 JSON 示例，而是定义 MCP tool metadata、inputSchema、outputSchema、CallToolResult 约定。

---

## 1. 项目概述

### 1.1 要解决的问题

当前主流 AI Agent harness 层普遍缺少可外部验证的确定性逻辑推理能力。LLM 可以生成看似合理的推理链，但不能稳定保证以下基本逻辑性质：

```text
A -> B 且 B -> C  推出  A -> C
A 是 B 的必要条件，且 A 不成立  推出  B 不成立
X 是 Y 的充分条件，X 不成立  不能推出  Y 不成立
存在 C -> Y  不能推出  X 不是 Y 的必要条件
```

这类失败在因果分析、商业策略、代码审查、安全评估、系统架构推演中会造成两类问题：

1. **假阳性推理**：LLM 推出逻辑上并不成立的结论。
2. **自洽性缺失**：同一回答内同时出现互相冲突的约束或因果声明。

NeSy Reasoning MCP Server 的目标不是取代 LLM，而是把 LLM 的高风险推理步骤外包给一个结构化、确定性、可追溯的符号推理模块。

### 1.2 方案定位

本项目实现一个通用 MCP Server，提供以下能力：

| 能力 | 说明 |
|---|---|
| 关系声明 | 将自然语言命题转换后的结构化关系写入关系图 |
| 因果分类 | 判断 source 与 target 之间是充分、必要、充要、未知还是矛盾 |
| 链路验证 | 验证多步传递链是否成立，返回路径和断裂点 |
| 矛盾检测 | 检查直接、循环、传递、互斥目标导致的逻辑矛盾 |
| 反事实推理 | 假设某条件不成立，分析必然阻断、可能阻断、仍可能成立、未知 |
| 持久化 | 支持内存、JSON 文件、SQLite 三种状态管理方式 |
| Hook 集成 | 通过 Claude Code hooks 做上下文注入和输出前逻辑检查 |

本 Server 只处理已经结构化的命题和关系，不负责可靠地从自然语言中自动抽取关系。自然语言抽取可以作为 Hook 或上游 Agent 的辅助步骤，但不属于符号引擎的核心确定性职责。

### 1.3 非目标

本项目第一阶段不做以下事情：

- 不做通用自然语言理解。
- 不自动判断“利润增加”和“利润减少”是否互斥，除非用户显式声明互斥组。
- 不把概率相关性等同于因果关系。
- 不把置信度低的关系当成可证明逻辑真理。
- 不提供交易、医疗、法律等高风险领域的自动决策。
- 不要求 SWI-Prolog、Z3、TLA+ 等外部推理器作为基础依赖。

### 1.4 理论来源

| 来源 | 借鉴点 | 对应设计 |
|---|---|---|
| 圈论推理体系 | 用集合包含描述充分、必要、充要关系 | 关系语义与传递规则 |
| ProSLM | LLM + domain KB + logic validation | 结构化知识库和验证工具 |
| LINC / Logic-LM | LLM 将自然语言转成符号表达，外部 solver 做推理 | LLM 与符号推理器职责拆分 |
| Reliable Reasoning Beyond Natural Language | 把迭代推理转给 Prolog 等符号系统 | 可执行逻辑层 |
| Project Chimera | LLM strategist + symbolic constraint engine + causal inference | 三层架构、可测试性、消融实验 |
| GSM-Symbolic | 展示 LLM 对无关信息和符号变体的推理脆弱性 | 架构动机 |

### 1.5 规范性参考

- MCP Specification 2025-11-25：`https://modelcontextprotocol.io/specification/2025-11-25`
- MCP Tools：`https://modelcontextprotocol.io/specification/2025-11-25/server/tools`
- MCP Transports：`https://modelcontextprotocol.io/specification/2025-11-25/basic/transports`
- Claude Code Hooks：`https://code.claude.com/docs/en/hooks`
- ProSLM：`https://arxiv.org/abs/2409.11589`
- LINC：`https://arxiv.org/abs/2310.15164`
- Reliable Reasoning Beyond Natural Language：`https://arxiv.org/abs/2407.11373`
- GSM-Symbolic：`https://arxiv.org/abs/2410.05229`
- Project Chimera：`https://arxiv.org/abs/2510.23682`

---

## 2. MCP 协议契约

### 2.1 Server identity

```json
{
  "name": "nesy-reasoning",
  "version": "2.0.0",
  "description": "Deterministic neuro-symbolic reasoning tools for causal classification, chain verification, contradiction checking, and counterfactual analysis."
}
```

### 2.2 Server capabilities

Server 必须声明 `tools` capability。

```json
{
  "capabilities": {
    "tools": {
      "listChanged": false
    }
  }
}
```

如果实现支持动态启用/禁用工具，`listChanged` 可以设为 `true`，并在工具列表变化时发送 `notifications/tools/list_changed`。

### 2.3 Transport

支持两种部署模式。

#### 模式 A：stdio，本地单 Agent 使用

适合 Claude Desktop、Claude Code 直接启动 MCP server。

要求：

- Server 从 `stdin` 读取 JSON-RPC 消息。
- Server 只向 `stdout` 写合法 MCP JSON-RPC 消息。
- 日志只能写 `stderr`。
- `stdout` 不能打印 debug 文本、banner、进度条。
- Server 进程生命周期可作为 session 生命周期。

#### 模式 B：Streamable HTTP，本地 daemon 或多客户端共享

适合以下场景：

- Claude Code hooks 与 MCP client 需要共享同一关系图。
- 多个 Agent 同时访问同一个推理状态。
- 需要长期运行的 SQLite store。

要求：

- 默认绑定 `127.0.0.1`。
- 必须校验 `Origin` header。
- 远程访问必须启用认证。
- 所有请求必须有超时和速率限制。

### 2.4 Tool 命名

所有工具使用 `nesy.` 前缀：

```text
nesy.assert_relations
nesy.classify
nesy.verify_chain
nesy.check_contradictions
nesy.counterfactual
nesy.assert_exclusive
nesy.list_relations
nesy.clear_relations
nesy.load_relations
nesy.export_relations
nesy.summarize_graph
```

兼容旧版文档中的名称：

| 旧名称 | 新名称 | 说明 |
|---|---|---|
| `assert_relation` | `nesy.assert_relations` | 单条和批量统一为数组输入 |
| `chain_verify` | `nesy.verify_chain` | 动词命名统一 |
| `contradiction_check` | `nesy.check_contradictions` | 动词命名统一 |
| `counterfactual` | `nesy.counterfactual` | 仅加前缀 |
| `load_relations` | `nesy.load_relations` | 仅加前缀 |
| `clear_relations` | `nesy.clear_relations` | 仅加前缀 |
| `list_relations` | `nesy.list_relations` | 仅加前缀 |
| `assert_exclusive` | `nesy.assert_exclusive` | 仅加前缀 |

实现可以在一个小版本周期内保留旧名称作为 alias，但 `tools/list` 中推荐只暴露新名称，避免 LLM 混用。

### 2.5 Tool result 约定

每个工具调用返回 MCP `CallToolResult`：

```json
{
  "content": [
    {
      "type": "text",
      "text": "{...serialized structuredContent...}"
    }
  ],
  "structuredContent": {
    "status": "ok",
    "...": "..."
  },
  "isError": false
}
```

规则：

- 所有业务输出必须放在 `structuredContent`。
- 为兼容客户端，`content[0].text` 应包含同一结构化结果的 JSON 字符串。
- 输入验证失败、逻辑冲突、文件加载失败属于 tool execution error，使用 `isError: true`。
- 未知工具、非法 JSON-RPC、MCP 协议层错误使用 JSON-RPC protocol error。
- 每个工具必须有 `inputSchema`。
- 推荐每个工具都有 `outputSchema`。

### 2.6 通用输出字段

所有工具的 `structuredContent` 至少包含以下字段：

```json
{
  "status": "ok",
  "trace": [],
  "diagnostics": [],
  "graph_stats": {
    "relations": 0,
    "propositions": 0,
    "exclusive_groups": 0,
    "contexts": 0
  }
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | `ok` / `error` / `warning` | 工具执行状态 |
| `trace` | array | 可追溯推理步骤，机器可读或短文本均可 |
| `diagnostics` | array | 警告、冲突、输入修正、语义限制 |
| `graph_stats` | object | 当前关系图统计 |

---

## 3. 核心形式语义

### 3.1 基本对象

本系统处理的是命题，而不是自然语言句子。每个节点表示一个 proposition：

```text
P = "降价"
Q = "销量增加"
R = "市场份额增长"
```

关系以 `source` 和 `target` 表达：

```text
source 是 target 的某种条件
```

不要使用 JSON 字段名 `from`，因为它在 Python 中是关键字。统一使用：

```json
{
  "source": "降价",
  "target": "销量增加"
}
```

### 3.2 关系类型

#### sufficient

```text
sufficient(source, target)
含义：source 是 target 的充分条件
形式：source -> target
集合：source ⊆ target
自然语言：只要 source 成立，target 就成立
```

示例：

```text
降价 sufficient 销量增加
等价于：降价 -> 销量增加
```

#### necessary

```text
necessary(source, target)
含义：source 是 target 的必要条件
形式：target -> source
集合：target ⊆ source
自然语言：target 要成立，source 必须成立；没有 source，则 target 不成立
```

示例：

```text
登录 necessary 下单
等价于：下单 -> 登录
```

#### equivalent

```text
equivalent(source, target)
含义：source 是 target 的充要条件
形式：source -> target 且 target -> source
集合：source = target
自然语言：source 与 target 互相蕴含
```

示例：

```text
通过身份认证 equivalent 拥有有效会话
等价于：通过身份认证 -> 拥有有效会话，且 拥有有效会话 -> 通过身份认证
```

### 3.3 Canonical implication graph

内部推理不直接在 `sufficient` / `necessary` 文本标签上做路径组合，而是统一转换为 implication edge：

| 输入关系 | 内部 implication edge |
|---|---|
| `sufficient(A, B)` | `A -> B` |
| `necessary(A, B)` | `B -> A` |
| `equivalent(A, B)` | `A -> B` 和 `B -> A` |

所有传递推理都在 implication graph 上执行。

之后根据两个方向是否可达，把结果映射回外部关系：

| `source -> target` | `target -> source` | 分类 |
|---|---|---|
| true | false | `sufficient` |
| false | true | `necessary` |
| true | true | `equivalent` |
| false | false | `unknown` |

### 3.4 开放世界语义

默认采用开放世界假设：

```text
没有证据证明 X，不等于证明 not X。
没有找到路径，不等于路径不存在。
没有找到替代充分条件，不等于替代充分条件不存在。
```

因此：

```text
X -> Y
not X
```

不能推出：

```text
not Y
```

只有以下情况才能推出 `not Y`：

```text
Y -> X
not X
```

也就是 X 是 Y 的必要条件。

### 3.5 闭合世界语义

部分业务场景可以启用 closed-world mode：

```json
{
  "world_mode": "closed"
}
```

闭合世界只在满足以下条件时使用：

- 用户明确声明某个 target 的充分原因已经穷尽。
- 或导入的关系集对某个 context 标记 `causal_completeness: true`。
- 或调用工具时显式设置 `world_mode: "closed"`。

闭合世界下，如果 target 的所有已知充分原因都被阻断，可以推出 target 被阻断。默认不开启。

### 3.6 互斥和否定

系统不自动理解自然语言语义矛盾。必须显式声明互斥组：

```json
{
  "group_id": "profit_state",
  "members": ["利润增加", "利润减少", "利润不变"]
}
```

互斥组含义：同一 context、同一时间窗口、兼容 assumptions 下，一个互斥组内至多一个命题为真。

内部可把互斥视为：

```text
A exclusive B  =>  A -> not B, B -> not A
```

但实现不需要创建显式负节点，除非高级版本支持 negation graph。

### 3.7 置信度

置信度是证据可靠性，不是逻辑有效性。

每个关系可有 `confidence`：

```json
{
  "confidence": 0.95
}
```

默认路径置信度策略：

```text
product_independent: cumulative_confidence = Π edge.confidence
```

其他可选策略：

| 策略 | 公式 | 使用场景 |
|---|---|---|
| `product_independent` | 连乘 | 默认，假设证据近似独立 |
| `min` | 取最小边 | 保守链路质量 |
| `max_path` | 多路径中取最高路径 | 查询最可信路径 |
| `no_aggregation` | 不聚合 | 只返回每条边置信度 |

输出必须同时区分：

```json
{
  "logic_validity": true,
  "evidence_confidence": 0.8075,
  "confidence_policy": "product_independent"
}
```

### 3.8 Context、时间和 assumptions

同一组命题在不同上下文中可能有不同关系：

```text
降价 -> 利润增加  context=清库存
降价 -> 利润减少  context=毛利承压
```

这不一定是矛盾。

所有关系都可以带：

```json
{
  "context_id": "ecommerce_q3",
  "temporal": {
    "delay": "short",
    "valid_from": "2026-07-01",
    "valid_to": "2026-09-30"
  },
  "assumptions": ["same_market", "no_stockout"]
}
```

矛盾检测、分类、链路验证、反事实推理默认只在 context 与 assumptions 兼容的关系集合内执行。

### 3.9 传递规则

因为内部统一为 implication graph，传递规则可简化为：

```text
A -> B 且 B -> C  推出 A -> C
```

映射回关系类型时：

| 左关系 | 右关系 | 是否直接得到 source 对 final target 的同类关系 |
|---|---|---|
| sufficient | sufficient | sufficient |
| necessary | necessary | necessary |
| equivalent | equivalent | equivalent |
| equivalent | sufficient | sufficient |
| sufficient | equivalent | sufficient |
| equivalent | necessary | necessary |
| necessary | equivalent | necessary |
| sufficient | necessary | unknown/broken |
| necessary | sufficient | unknown/broken |

注意：最后两行不是因为 implication 不能传递，而是因为按原始 `source/target` 条件语义组合时方向断裂。实现层通过 canonical graph 检测两个方向的可达性，避免人工组合表出错。

### 3.10 反事实语义

输入：

```text
if_not = X
```

默认开放世界下，输出分为四类：

| 类别 | 含义 | 证明条件 |
|---|---|---|
| `necessarily_blocked` | 必然阻断 | target -> X，且 X 不成立 |
| `possibly_blocked` | 可能阻断 | X -> target，但不能证明替代路径足够独立 |
| `still_possible` | 仍可能成立 | 存在 alternative -> target，且 alternative 不依赖 X |
| `unknown` | 无法判断 | 当前图不足以推出阻断或不受影响 |

不再使用强语义的 `unaffected` 作为默认输出。若为了兼容旧版需要，可以输出 `not_derivably_affected`，含义只是“当前图不能推出受影响”。

### 3.11 非必要性的证明

要证明 `source` 不是 `target` 的必要条件，不能只靠“还有另一个充分条件”。需要一个反例：

```text
C -> target
C does_not_imply source
或 C independent_of source
或 C exclusive source
```

因此 `classify` 中的 `not_necessary` 只有在存在显式或可推导的独立反例时才返回。否则返回：

```json
{
  "necessity_status": "unknown",
  "reason": "Alternative sufficient causes do not disprove necessity unless independence or counterexample is established."
}
```

---

## 4. 架构设计

### 4.1 总体架构

```text
┌───────────────────────────────────────────────────────────┐
│ Agent / Client                                             │
│ Claude Code / Claude Desktop / Cursor / Codex / custom app │
│                                                           │
│  手动工具调用                  Hook 自动触发               │
│  tools/call ──────────────┐    PreToolUse / Stop / ... ──┐ │
└───────────────────────────┼──────────────────────────────┼─┘
                            │                              │
                            ▼                              ▼
┌───────────────────────────────────────────────────────────────┐
│ NeSy Reasoning MCP Server                                      │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ MCP Protocol Layer                                      │  │
│  │ tools/list, tools/call, inputSchema, outputSchema        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌────────────────────┐  ┌────────────────────────────────┐  │
│  │ Validation Layer   │  │ Access / Policy Layer           │  │
│  │ JSON Schema        │  │ roots, allowed paths, limits     │  │
│  │ Pydantic models    │  │ rate limits, audit log           │  │
│  └────────────────────┘  └────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Relation Store                                          │  │
│  │ normalized relation records, contexts, exclusives        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Graph Index                                             │  │
│  │ implication graph, reverse index, exclusivity index      │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌───────────────┐ ┌───────────────┐ ┌───────────────────┐  │
│  │ Classifier    │ │ Chain Verifier │ │ Contradiction Eng.│  │
│  └───────────────┘ └───────────────┘ └───────────────────┘  │
│                                                               │
│  ┌────────────────────┐ ┌─────────────────────────────────┐ │
│  │ Counterfactual Eng. │ │ Persistence                     │ │
│  │ open/closed world   │ │ memory / JSON / SQLite          │ │
│  └────────────────────┘ └─────────────────────────────────┘ │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Hook Bridge                                             │  │
│  │ graph summary, output checks, extraction adapters        │  │
│  └─────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

### 4.2 模块职责

| 模块 | 职责 |
|---|---|
| MCP Protocol Layer | 暴露 tools/list、tools/call，处理 MCP result shape |
| Validation Layer | 校验 JSON Schema，拒绝非法字段和越界参数 |
| Access / Policy Layer | 控制文件访问、context scope、速率限制、最大搜索深度 |
| Relation Store | 保存原始关系、证据、上下文、互斥组 |
| Graph Index | 将关系规范化为 implication edges，支持可达性查询 |
| Classifier | 判断两个命题的充分/必要/充要/未知/矛盾关系 |
| Chain Verifier | 验证显式链或搜索最优链 |
| Contradiction Engine | 检测硬矛盾、软矛盾、上下文隔离冲突 |
| Counterfactual Engine | 执行开放世界或闭合世界反事实推理 |
| Persistence | JSON/SQLite 持久化和导入导出 |
| Hook Bridge | 给 Claude Code hooks 提供简洁接口和摘要 |

### 4.3 状态模型

状态作用域分四层：

| 作用域 | 说明 | 典型用途 |
|---|---|---|
| `process` | MCP server 进程内内存 | Claude Desktop 一次会话 |
| `session` | 按 MCP/Agent session 分隔 | Claude Code 当前项目会话 |
| `workspace` | 按项目目录或 repository 分隔 | 长期项目知识库 |
| `global` | 用户级共享知识库 | 通用规则库 |

每条关系必须带 `store_id` 或被写入默认 store：

```json
{
  "store_id": "default"
}
```

推荐状态策略：

| 场景 | 建议 |
|---|---|
| 单 Agent 临时推理 | memory |
| Claude Code + hooks | SQLite 或 Streamable HTTP daemon |
| 团队共享 | SQLite/Postgres + HTTP + auth |
| 离线评估 | JSON fixture |

### 4.4 为什么不直接用 `nx.DiGraph` 保存全部数据

`nx.DiGraph` 对每个 `(source, target)` 只能保存一组 edge attributes。如果同一对节点存在多个证据、多个 context、多个 relation_type，会覆盖或混淆。

推荐：

```text
RelationStore: 保存完整记录
GraphIndex: 保存推理用 canonical implication edges
```

GraphIndex 可以用 NetworkX，但它只是索引，不是事实源。

### 4.5 设计原则

1. **结构化输入**：工具不从自然语言里猜关系。
2. **可追溯输出**：每个推理结果必须返回路径或不可证明原因。
3. **开放世界默认**：未知不等于否定。
4. **语义边界显式化**：互斥、独立、closed-world completeness 必须显式声明。
5. **置信度与逻辑分离**：logic validity 和 evidence confidence 分开输出。
6. **Context-aware**：矛盾和传递必须考虑上下文兼容性。
7. **安全优先**：文件访问、HTTP、hooks 都必须有边界。

---

## 5. 数据模型

### 5.1 RelationRecord

```json
{
  "id": "rel_01HXYZ",
  "source": "降价",
  "target": "销量增加",
  "relation_type": "sufficient",
  "polarity": "positive",
  "confidence": 0.95,
  "context_id": "ecommerce_q3",
  "store_id": "default",
  "temporal": {
    "delay": "immediate",
    "valid_from": "2026-07-01",
    "valid_to": "2026-09-30"
  },
  "assumptions": ["same_market", "no_stockout"],
  "provenance": {
    "source_type": "user",
    "source_ref": "Q3 sales report",
    "created_by": "agent_or_user",
    "created_at": "2026-05-21T00:00:00Z"
  },
  "metadata": {
    "domain": "ecommerce",
    "note": "observed in prior campaigns"
  }
}
```

字段：

| 字段 | 必填 | 说明 |
|---|---:|---|
| `id` | 否 | 为空时 server 生成 |
| `source` | 是 | 条件命题 |
| `target` | 是 | 结果命题 |
| `relation_type` | 是 | `sufficient` / `necessary` / `equivalent` |
| `polarity` | 否 | 默认 `positive`，预留 `negative` |
| `confidence` | 否 | 0 到 1，默认 1.0 |
| `context_id` | 否 | 默认 `default` |
| `store_id` | 否 | 默认 `default` |
| `temporal` | 否 | 时间延迟和有效期 |
| `assumptions` | 否 | 适用前提 |
| `provenance` | 否 | 证据来源 |
| `metadata` | 否 | 自由元数据 |

### 5.2 CanonicalImplicationEdge

```json
{
  "edge_id": "edge_01HXYZ_A",
  "relation_id": "rel_01HXYZ",
  "antecedent": "降价",
  "consequent": "销量增加",
  "confidence": 0.95,
  "context_id": "ecommerce_q3",
  "store_id": "default",
  "assumptions": ["same_market", "no_stockout"]
}
```

生成规则：

```text
sufficient(A, B) => edge(A, B)
necessary(A, B)  => edge(B, A)
equivalent(A, B) => edge(A, B) and edge(B, A)
```

### 5.3 ExclusiveGroup

```json
{
  "group_id": "profit_state",
  "members": ["利润增加", "利润减少", "利润不变"],
  "context_id": "ecommerce_q3",
  "store_id": "default",
  "scope": "same_context",
  "provenance": {
    "source_type": "user",
    "created_at": "2026-05-21T00:00:00Z"
  }
}
```

规则：

- 同一 group 内任意两个 members 互斥。
- 默认只在相同 context 下互斥。
- 如果 `scope = global`，跨 context 也互斥。

### 5.4 IndependenceRecord

用于证明非必要性或替代路径独立性。

```json
{
  "id": "ind_01",
  "left": "广告增投",
  "right": "降价",
  "relation": "independent_of",
  "context_id": "ecommerce_q3",
  "confidence": 0.9,
  "metadata": {}
}
```

MVP 可以不单独暴露 independence tool，但数据模型预留。第一版可通过 `metadata.independent_of` 或 `assumptions` 实现保守处理。

### 5.5 GraphStats

```json
{
  "relations": 42,
  "propositions": 19,
  "implication_edges": 51,
  "exclusive_groups": 5,
  "contexts": 3,
  "stores": 1
}
```

### 5.6 JSON 持久化格式

```json
{
  "version": "2.0",
  "stores": [
    {
      "store_id": "default",
      "description": "Default reasoning graph"
    }
  ],
  "relations": [
    {
      "source": "降价",
      "target": "销量增加",
      "relation_type": "sufficient",
      "confidence": 0.95,
      "context_id": "ecommerce_q3",
      "temporal": { "delay": "immediate" },
      "assumptions": ["same_market", "no_stockout"],
      "metadata": { "domain": "ecommerce" }
    }
  ],
  "exclusive_groups": [
    {
      "group_id": "profit_state",
      "members": ["利润增加", "利润减少", "利润不变"],
      "context_id": "ecommerce_q3"
    }
  ],
  "context_metadata": {
    "ecommerce_q3": {
      "world_mode": "open",
      "causal_completeness": false
    }
  }
}
```

### 5.7 SQLite schema

推荐表：

```sql
CREATE TABLE relations (
  id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  source_id TEXT,
  target TEXT NOT NULL,
  target_id TEXT,
  relation_type TEXT NOT NULL CHECK (relation_type IN ('sufficient','necessary','equivalent')),
  polarity TEXT NOT NULL DEFAULT 'positive',
  confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
  context_id TEXT NOT NULL DEFAULT 'default',
  store_id TEXT NOT NULL DEFAULT 'default',
  temporal_json TEXT,
  assumptions_json TEXT,
  provenance_json TEXT,
  metadata_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE exclusive_groups (
  group_id TEXT NOT NULL,
  member TEXT NOT NULL,
  context_id TEXT NOT NULL DEFAULT 'default',
  store_id TEXT NOT NULL DEFAULT 'default',
  metadata_json TEXT,
  PRIMARY KEY (group_id, member, context_id, store_id)
);

CREATE TABLE audit_log (
  id TEXT PRIMARY KEY,
  event_type TEXT NOT NULL,
  tool_name TEXT,
  input_hash TEXT,
  result_status TEXT,
  created_at TEXT NOT NULL,
  metadata_json TEXT
);
```

GraphIndex 可在内存中由 SQLite 表重建，不需要单独持久化。

---

## 6. MCP Tool 详细规格

### 6.0 共享 schema 片段

#### RelationInput

```json
{
  "type": "object",
  "properties": {
    "id": { "type": "string" },
    "source": { "type": "string", "minLength": 1 },
    "source_id": { "type": "string", "minLength": 1 },
    "target": { "type": "string", "minLength": 1 },
    "target_id": { "type": "string", "minLength": 1 },
    "relation_type": {
      "type": "string",
      "enum": ["sufficient", "necessary", "equivalent"]
    },
    "confidence": { "type": "number", "minimum": 0, "maximum": 1, "default": 1 },
    "context_id": { "type": "string", "default": "default" },
    "store_id": { "type": "string", "default": "default" },
    "temporal": { "type": "object" },
    "assumptions": { "type": "array", "items": { "type": "string" } },
    "provenance": { "type": "object" },
    "metadata": { "type": "object" }
  },
  "required": ["source", "target", "relation_type"],
  "additionalProperties": false
}
```

`source` and `target` are display labels and remain required for compatibility.
`source_id` and `target_id` are optional stable proposition IDs. If present,
canonical graph reasoning uses the IDs; otherwise it falls back to labels. This
does not implement alias lookup. Temporary `PropositionRecord` inputs can declare
`negates` for `nesy.check_contradictions`; persistence/export of proposition
metadata remains future work.

#### PropositionRecord

```json
{
  "id": { "type": "string", "minLength": 1 },
  "label": { "type": "string", "minLength": 1 },
  "aliases": { "type": "array", "items": { "type": "string" }, "default": [] },
  "negates": { "type": "string", "minLength": 1 },
  "metadata": { "type": "object", "default": {} }
}
```

`negates` is optional. When supplied to `nesy.check_contradictions`, it declares
that this proposition ID is the canonical negation of another proposition ID.
Self-negation is invalid.

#### ContextFilter

```json
{
  "type": "object",
  "properties": {
    "context_id": { "type": "string" },
    "store_id": { "type": "string" },
    "domain": { "type": "string" },
    "assumptions": { "type": "array", "items": { "type": "string" } },
    "valid_at": { "type": "string", "format": "date-time" }
  },
  "additionalProperties": false
}
```

#### Diagnostic

```json
{
  "type": "object",
  "properties": {
    "level": { "type": "string", "enum": ["info", "warning", "error"] },
    "code": { "type": "string" },
    "message": { "type": "string" },
    "related_ids": { "type": "array", "items": { "type": "string" } }
  },
  "required": ["level", "code", "message"],
  "additionalProperties": false
}
```

---

### 6.1 `nesy.assert_relations`

#### 用途

向关系图添加一条或多条结构化关系。替代旧版 `assert_relation`。

#### Tool metadata

```json
{
  "name": "nesy.assert_relations",
  "title": "Assert Logical Relations",
  "description": "Add one or more sufficient, necessary, or equivalent relations to the NeSy reasoning graph.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "relations": {
        "type": "array",
        "items": { "$ref": "#/$defs/RelationInput" },
        "minItems": 1
      },
      "mode": {
        "type": "string",
        "enum": ["append", "upsert", "replace_same_pair"],
        "default": "append"
      },
      "check_contradictions": { "type": "boolean", "default": true },
      "merge_equivalent": {
        "type": "boolean",
        "default": true,
        "description": "Report canonical graph normalization for matching sufficient+necessary evidence without merging or deleting stored evidence records."
      },
      "on_contradiction": {
        "type": "string",
        "enum": ["warn", "reject"],
        "default": "warn",
        "description": "When set to reject, hard contradictions are checked with an effective graph before writing and rejected without storing new records."
      },
      "dry_run": { "type": "boolean", "default": false }
    },
    "required": ["relations"],
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string", "enum": ["ok", "warning", "error"] },
      "added": { "type": "integer" },
      "updated": { "type": "integer" },
      "rejected": { "type": "integer" },
      "relation_ids": { "type": "array", "items": { "type": "string" } },
      "contradictions": { "type": "array" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "added", "updated", "rejected", "relation_ids"]
  }
}
```

#### 示例输入

```json
{
  "relations": [
    {
      "source": "降价",
      "target": "销量增加",
      "relation_type": "sufficient",
      "confidence": 0.95,
      "context_id": "ecommerce_q3",
      "metadata": { "domain": "ecommerce" }
    },
    {
      "source": "销量增加",
      "target": "市场份额增长",
      "relation_type": "sufficient",
      "confidence": 0.85,
      "context_id": "ecommerce_q3"
    }
  ]
}
```

#### 示例 structuredContent

```json
{
  "status": "ok",
  "added": 2,
  "updated": 0,
  "rejected": 0,
  "relation_ids": ["rel_001", "rel_002"],
  "contradictions": [],
  "diagnostics": [],
  "trace": [
    "normalized sufficient(降价, 销量增加) into implication edge 降价 -> 销量增加",
    "normalized sufficient(销量增加, 市场份额增长) into implication edge 销量增加 -> 市场份额增长"
  ],
  "graph_stats": {
    "relations": 2,
    "propositions": 3,
    "implication_edges": 2,
    "exclusive_groups": 0,
    "contexts": 1
  }
}
```

#### 实现规则

- 如果同一 context 下同一 `(source, target)` 同时存在 `sufficient` 和 `necessary`，且 `merge_equivalent=true`，只在 canonical graph/诊断中报告为 `equivalent`；不得合并、删除或改写原始 evidence records。
- 如果新增关系导致已声明互斥目标被同一 source 充分推出，`on_contradiction=warn` 写入后返回 `warning`；`on_contradiction=reject` 必须先用 effective graph 检查，发现 hard contradiction 时返回 `error` 且不写入，即使 `check_contradictions=false` 也不能绕过。
- `dry_run=true` 时只返回将要添加、更新、拒绝的记录，不改变状态。
- `mode=replace_same_pair` 只替换同一 `source/target/context/store` 的关系，不清空其他关系。

---

### 6.2 `nesy.classify`

#### 用途

查询两个命题之间的逻辑关系，返回直接关系、传递推导关系、反向关系、可证明性和 trace。

#### Tool metadata

```json
{
  "name": "nesy.classify",
  "title": "Classify Logical Relation",
  "description": "Classify whether source is sufficient, necessary, equivalent, unknown, or contradictory with respect to target.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source": { "type": "string", "minLength": 1 },
      "target": { "type": "string", "minLength": 1 },
      "context_filter": { "$ref": "#/$defs/ContextFilter" },
      "max_depth": { "type": "integer", "minimum": 1, "maximum": 20, "default": 8 },
      "include_paths": { "type": "boolean", "default": true },
      "require_direct": { "type": "boolean", "default": false },
      "confidence_policy": {
        "type": "string",
        "enum": ["product_independent", "min", "no_aggregation"],
        "default": "product_independent"
      }
    },
    "required": ["source", "target"],
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "source": { "type": "string" },
      "target": { "type": "string" },
      "classification": { "type": "string", "enum": ["sufficient", "necessary", "equivalent", "unknown", "contradictory"] },
      "source_implies_target": { "type": "object" },
      "target_implies_source": { "type": "object" },
      "necessity_status": { "type": "object" },
      "direct_relations": { "type": "array" },
      "paths": { "type": "array" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "source", "target", "classification"]
  }
}
```

#### 示例输入

```json
{
  "source": "降价",
  "target": "市场份额增长",
  "context_filter": { "context_id": "ecommerce_q3" },
  "include_paths": true
}
```

#### 示例 structuredContent

```json
{
  "status": "ok",
  "source": "降价",
  "target": "市场份额增长",
  "classification": "sufficient",
  "source_implies_target": {
    "proven": true,
    "logic_validity": true,
    "evidence_confidence": 0.8075,
    "best_path": ["降价", "销量增加", "市场份额增长"]
  },
  "target_implies_source": {
    "proven": false,
    "logic_validity": false,
    "reason": "No path found from target to source within max_depth under the context filter."
  },
  "necessity_status": {
    "status": "unknown",
    "reason": "No proof that target implies source; absence of proof is not proof of non-necessity."
  },
  "direct_relations": [],
  "paths": [
    {
      "direction": "source_to_target",
      "nodes": ["降价", "销量增加", "市场份额增长"],
      "relation_type": "sufficient",
      "logic_validity": true,
      "evidence_confidence": 0.8075,
      "steps": [
        { "antecedent": "降价", "consequent": "销量增加", "relation_id": "rel_001", "confidence": 0.95 },
        { "antecedent": "销量增加", "consequent": "市场份额增长", "relation_id": "rel_002", "confidence": 0.85 }
      ]
    }
  ],
  "diagnostics": [],
  "trace": [
    "Checked direct relations between 降价 and 市场份额增长: none",
    "Found implication path 降价 -> 销量增加 -> 市场份额增长",
    "No reverse implication path found",
    "Mapped reachability to classification: sufficient"
  ],
  "graph_stats": {
    "relations": 2,
    "propositions": 3,
    "implication_edges": 2,
    "exclusive_groups": 0,
    "contexts": 1
  }
}
```

#### 实现规则

- `classification` 必须由 canonical implication graph 的双向可达性决定。
- 只有 `source -> target` 可达时返回 `sufficient`。
- 只有 `target -> source` 可达时返回 `necessary`。
- 双向可达时返回 `equivalent`。
- 两个方向都不可达时返回 `unknown`。
- 如果能推出 source 会导致 target 的互斥命题，同时也导致 target，返回 `contradictory`。
- 不得仅凭替代充分条件返回 `not_necessary`。必须有独立性、互斥反例或显式 `not_necessary` 证据。

---

### 6.3 `nesy.verify_chain`

#### 用途

验证显式链条或搜索 source 到 target 的推理链。替代旧版 `chain_verify`。

#### Tool metadata

```json
{
  "name": "nesy.verify_chain",
  "title": "Verify Reasoning Chain",
  "description": "Verify an explicit reasoning chain or search for valid implication paths between source and target.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source": { "type": "string", "minLength": 1 },
      "target": { "type": "string", "minLength": 1 },
      "chain": {
        "type": "array",
        "items": { "type": "string" },
        "minItems": 2
      },
      "expected_relation": {
        "type": "string",
        "enum": ["sufficient", "necessary", "equivalent", "any"],
        "default": "any"
      },
      "context_filter": { "$ref": "#/$defs/ContextFilter" },
      "max_depth": { "type": "integer", "minimum": 1, "maximum": 20, "default": 8 },
      "path_strategy": {
        "type": "string",
        "enum": ["best_confidence", "shortest", "all"],
        "default": "best_confidence"
      },
      "max_paths": { "type": "integer", "minimum": 1, "maximum": 50, "default": 5 },
      "confidence_policy": {
        "type": "string",
        "enum": ["product_independent", "min", "no_aggregation"],
        "default": "product_independent"
      }
    },
    "required": ["source", "target"],
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "reachable": { "type": "boolean" },
      "relation_type": { "type": "string" },
      "logic_validity": { "type": "boolean" },
      "best_path": { "type": "object" },
      "paths": { "type": "array" },
      "broken_at": { "type": "object" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "reachable", "logic_validity"]
  }
}
```

#### 示例输入：搜索链条

```json
{
  "source": "降价",
  "target": "市场份额增长",
  "expected_relation": "sufficient",
  "context_filter": { "context_id": "ecommerce_q3" }
}
```

#### 示例 structuredContent

```json
{
  "status": "ok",
  "reachable": true,
  "relation_type": "sufficient",
  "logic_validity": true,
  "best_path": {
    "nodes": ["降价", "销量增加", "市场份额增长"],
    "steps": [
      {
        "antecedent": "降价",
        "consequent": "销量增加",
        "relation_id": "rel_001",
        "source_relation_type": "sufficient",
        "confidence": 0.95
      },
      {
        "antecedent": "销量增加",
        "consequent": "市场份额增长",
        "relation_id": "rel_002",
        "source_relation_type": "sufficient",
        "confidence": 0.85
      }
    ],
    "evidence_confidence": 0.8075,
    "confidence_policy": "product_independent"
  },
  "paths": [],
  "broken_at": null,
  "diagnostics": [],
  "trace": [
    "Searched implication graph from 降价 to 市场份额增长",
    "Found path with 2 edges",
    "Expected relation sufficient matched"
  ],
  "graph_stats": {
    "relations": 2,
    "propositions": 3,
    "implication_edges": 2,
    "exclusive_groups": 0,
    "contexts": 1
  }
}
```

#### 示例输入：验证显式链条

```json
{
  "source": "A",
  "target": "C",
  "chain": ["A", "B", "C"],
  "expected_relation": "sufficient"
}
```

#### 断裂输出示例

```json
{
  "status": "ok",
  "reachable": false,
  "relation_type": "unknown",
  "logic_validity": false,
  "best_path": null,
  "paths": [],
  "broken_at": {
    "index": 1,
    "from": "B",
    "to": "C",
    "reason": "No implication edge B -> C exists. Existing relation necessary(B, C) maps to C -> B, not B -> C."
  },
  "diagnostics": [
    {
      "level": "warning",
      "code": "DIRECTION_MISMATCH",
      "message": "The declared necessary relation points in the reverse implication direction."
    }
  ],
  "trace": [
    "A -> B verified",
    "B -> C not found",
    "Chain is broken at step 1"
  ],
  "graph_stats": {}
}
```

#### 实现规则

- 搜索时使用 BFS 或 Dijkstra-like best path。若采用置信度连乘，最高置信度路径可用 `-log(confidence)` 作为权重。
- 必须使用 visited 集合防止循环。
- `max_depth` 必须强制生效。
- 如果 `chain` 被提供，只验证该链，不自动替换成其他路径；可在 diagnostics 中提示存在替代路径。
- `relation_type` 根据 source-target 双向可达性映射，不从字符串拼接得出。

---

### 6.4 `nesy.check_contradictions`

#### 用途

检测一组输入事实、当前关系图，或两者合并后的逻辑矛盾。

#### Tool metadata

```json
{
  "name": "nesy.check_contradictions",
  "title": "Check Logical Contradictions",
  "description": "Detect direct, cyclic, transitive, and exclusivity-based contradictions in relation facts or the current graph.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "facts": {
        "type": "array",
        "items": { "$ref": "#/$defs/RelationInput" }
      },
      "propositions": {
        "type": "array",
        "items": { "$ref": "#/$defs/PropositionRecord" }
      },
      "mode": {
        "type": "string",
        "enum": ["graph", "facts", "combined"],
        "default": "graph"
      },
      "context_filter": { "$ref": "#/$defs/ContextFilter" },
      "include_soft": { "type": "boolean", "default": true },
      "max_depth": { "type": "integer", "minimum": 1, "maximum": 20, "default": 8 }
    },
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "has_contradictions": { "type": "boolean" },
      "contradictions": { "type": "array" },
      "clean_facts_count": { "type": "integer" },
      "total_facts_count": { "type": "integer" },
      "context_separated": { "type": "array" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "has_contradictions", "contradictions"]
  }
}
```

#### 检测类型

| 类型 | 条件 | 默认 severity |
|---|---|---|
| `exclusive_targets` | 同一 source 充分推出同一互斥组内两个 target | hard |
| `direct_opposition` | 同一命题推出 target 和显式 not target | hard |
| `cycle_to_exclusion` | A -> B 且 B -> not A | hard |
| `transitive_exclusive_targets` | 经传递后同一 source 推出互斥目标 | hard |
| `confidence_tension` | 低置信度或不同证据产生方向冲突 | soft |
| `context_separated_conflict` | 形式上冲突但 context 不兼容 | not_contradiction |

#### 示例输入

```json
{
  "mode": "combined",
  "facts": [
    { "source": "降价", "target": "利润增加", "relation_type": "sufficient", "context_id": "ecommerce_q3" },
    { "source": "降价", "target": "利润减少", "relation_type": "sufficient", "context_id": "ecommerce_q3" }
  ]
}
```

前提：已声明互斥组：

```json
{
  "group_id": "profit_state",
  "members": ["利润增加", "利润减少", "利润不变"],
  "context_id": "ecommerce_q3"
}
```

#### 示例 structuredContent

```json
{
  "status": "warning",
  "has_contradictions": true,
  "contradictions": [
    {
      "type": "exclusive_targets",
      "severity": "hard",
      "source": "降价",
      "targets": ["利润增加", "利润减少"],
      "exclusive_group_id": "profit_state",
      "context_id": "ecommerce_q3",
      "fact_ids": ["input_0", "input_1"],
      "reason": "Under the same context and compatible assumptions, the same source sufficiently implies two mutually exclusive targets."
    }
  ],
  "clean_facts_count": 0,
  "total_facts_count": 2,
  "context_separated": [],
  "diagnostics": [],
  "trace": [
    "Loaded 2 input facts into temporary graph",
    "Resolved exclusive group profit_state",
    "Found source 降价 implies both 利润增加 and 利润减少"
  ],
  "graph_stats": {}
}
```

#### 实现规则

- 未声明互斥组时，不得自动判断两个自然语言 target 矛盾。
- 只有在 context、store、temporal window、assumptions 兼容时才报 hard contradiction。
- 不同 context 下的冲突放入 `context_separated`，不计入 `has_contradictions=true`，除非互斥组 scope 为 `global`。
- `facts` 模式不得永久写入关系图。
- `combined` 模式先构建临时合并图，再检测。

---

### 6.5 `nesy.counterfactual`

#### 用途

假设某个命题不成立，推导必然阻断、可能阻断、仍可能成立和未知结果。

#### Tool metadata

```json
{
  "name": "nesy.counterfactual",
  "title": "Counterfactual Reasoning",
  "description": "Analyze what is necessarily blocked, possibly blocked, still possible, or unknown if a proposition is assumed false.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "if_not": { "type": "string", "minLength": 1 },
      "targets": {
        "type": "array",
        "items": { "type": "string" }
      },
      "context_filter": { "$ref": "#/$defs/ContextFilter" },
      "world_mode": {
        "type": "string",
        "enum": ["open", "closed"],
        "default": "open"
      },
      "max_depth": { "type": "integer", "minimum": 1, "maximum": 20, "default": 8 },
      "include_alternative_paths": { "type": "boolean", "default": true },
      "confidence_policy": {
        "type": "string",
        "enum": ["product_independent", "min", "no_aggregation"],
        "default": "product_independent"
      }
    },
    "required": ["if_not"],
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "if_not": { "type": "string" },
      "world_mode": { "type": "string" },
      "necessarily_blocked": { "type": "array" },
      "possibly_blocked": { "type": "array" },
      "still_possible": { "type": "array" },
      "unknown": { "type": "array" },
      "not_derivably_affected": { "type": "array" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "if_not", "world_mode"]
  }
}
```

#### 示例输入

```json
{
  "if_not": "降价",
  "context_filter": { "context_id": "ecommerce_q3" },
  "world_mode": "open",
  "include_alternative_paths": true
}
```

#### 示例 structuredContent

```json
{
  "status": "ok",
  "if_not": "降价",
  "world_mode": "open",
  "necessarily_blocked": [
    {
      "target": "促销价生效",
      "proof": {
        "type": "necessary_condition",
        "path": ["促销价生效", "降价"],
        "meaning": "促销价生效 implies 降价; therefore not 降价 implies not 促销价生效."
      },
      "logic_validity": true,
      "evidence_confidence": 0.9
    }
  ],
  "possibly_blocked": [
    {
      "target": "销量增加",
      "reason": "降价 is a sufficient path to the target, but absence of 降价 does not logically imply absence of the target under open-world semantics.",
      "blocked_path": ["降价", "销量增加"],
      "alternative_paths": [
        {
          "nodes": ["广告增投", "销量增加"],
          "evidence_confidence": 0.8,
          "independence_from_if_not": "unknown"
        }
      ]
    }
  ],
  "still_possible": [],
  "unknown": [
    {
      "target": "品牌信任",
      "reason": "No necessary dependency on 降价 and no sufficient path from 降价 was found. This is unknown, not proven unaffected."
    }
  ],
  "not_derivably_affected": ["品牌信任"],
  "diagnostics": [
    {
      "level": "info",
      "code": "OPEN_WORLD_DEFAULT",
      "message": "No alternative path found is not treated as proof of impossibility."
    }
  ],
  "trace": [
    "Assume not 降价",
    "Search targets that imply 降价: found 促销价生效 -> 降价",
    "Search targets implied by 降价: found 降价 -> 销量增加",
    "Classified 销量增加 as possibly_blocked, not necessarily_blocked"
  ],
  "graph_stats": {}
}
```

#### 实现规则

- `necessarily_blocked` 只在 target 可推出 `if_not` 时成立。
- `possibly_blocked` 用于 `if_not` 可推出 target，但 target 不一定依赖 `if_not`。
- `still_possible` 需要存在不依赖 `if_not` 的替代充分路径。
- 替代路径是否不依赖 `if_not`，默认必须证明；不能证明则标记 `independence_from_if_not: "unknown"`。
- `world_mode=closed` 下，可以在关系集声明 causal completeness 的范围内，把“所有充分原因都被阻断”升级为 `necessarily_blocked`，但 trace 必须说明使用了闭合世界假设。

---

### 6.6 `nesy.assert_exclusive`

#### 用途

声明互斥组，供矛盾检测和反事实推理使用。

#### Tool metadata

```json
{
  "name": "nesy.assert_exclusive",
  "title": "Assert Exclusive Groups",
  "description": "Declare propositions that cannot all be true together under a context.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "groups": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "group_id": { "type": "string" },
            "members": {
              "type": "array",
              "items": { "type": "string" },
              "minItems": 2
            },
            "context_id": { "type": "string", "default": "default" },
            "store_id": { "type": "string", "default": "default" },
            "scope": { "type": "string", "enum": ["same_context", "global"], "default": "same_context" },
            "metadata": { "type": "object" }
          },
          "required": ["members"],
          "additionalProperties": false
        },
        "minItems": 1
      }
    },
    "required": ["groups"],
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "added_groups": { "type": "integer" },
      "updated_groups": { "type": "integer" },
      "group_ids": { "type": "array", "items": { "type": "string" } },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "added_groups", "updated_groups", "group_ids"]
  }
}
```

#### 示例输入

```json
{
  "groups": [
    {
      "group_id": "profit_state",
      "members": ["利润增加", "利润减少", "利润不变"],
      "context_id": "ecommerce_q3"
    }
  ]
}
```

#### 示例 structuredContent

```json
{
  "status": "ok",
  "added_groups": 1,
  "updated_groups": 0,
  "group_ids": ["profit_state"],
  "diagnostics": [],
  "trace": ["Registered exclusive group profit_state with 3 members"],
  "graph_stats": {
    "relations": 2,
    "propositions": 5,
    "implication_edges": 2,
    "exclusive_groups": 1,
    "contexts": 1
  }
}
```

---

### 6.7 `nesy.list_relations`

#### 用途

列出当前关系图中的关系，支持过滤和分页。

#### Tool metadata

```json
{
  "name": "nesy.list_relations",
  "title": "List Relations",
  "description": "List stored relation records with optional filtering and pagination.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "filter": {
        "type": "object",
        "properties": {
          "source": { "type": "string" },
          "target": { "type": "string" },
          "relation_type": { "type": "string", "enum": ["sufficient", "necessary", "equivalent"] },
          "context_id": { "type": "string" },
          "store_id": { "type": "string" },
          "domain": { "type": "string" }
        },
        "additionalProperties": false
      },
      "include_implication_edges": { "type": "boolean", "default": false },
      "include_exclusive_groups": { "type": "boolean", "default": false },
      "limit": { "type": "integer", "minimum": 1, "maximum": 500, "default": 100 },
      "cursor": { "type": "string" }
    },
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "relations": { "type": "array" },
      "implication_edges": { "type": "array" },
      "exclusive_groups": { "type": "array" },
      "total": { "type": "integer" },
      "next_cursor": { "type": ["string", "null"] },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "relations", "total"]
  }
}
```

#### 示例输入

```json
{
  "filter": {
    "source": "降价",
    "context_id": "ecommerce_q3"
  },
  "include_implication_edges": true
}
```

---

### 6.8 `nesy.clear_relations`

#### 用途

清空全部或部分关系图。

#### Tool metadata

```json
{
  "name": "nesy.clear_relations",
  "title": "Clear Relations",
  "description": "Remove relation records and optionally exclusive groups by scope or filter.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "scope": {
        "type": "string",
        "enum": ["all", "store", "context", "filter"],
        "default": "context"
      },
      "store_id": { "type": "string", "default": "default" },
      "context_id": { "type": "string", "default": "default" },
      "filter": { "type": "object" },
      "include_exclusive_groups": { "type": "boolean", "default": false },
      "dry_run": { "type": "boolean", "default": false }
    },
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "removed_relations": { "type": "integer" },
      "removed_exclusive_groups": { "type": "integer" },
      "dry_run": { "type": "boolean" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "removed_relations", "removed_exclusive_groups", "dry_run"]
  }
}
```

#### 安全规则

- `scope=all` 可以清空所有 store，客户端应提示用户确认。
- Server 可配置禁止 LLM 自动执行 `scope=all`。
- 所有清空操作写入 audit log。

---

### 6.9 `nesy.load_relations`

#### 用途

从内联 JSON、受信任文件路径或 MCP resource URI 加载关系集。

#### Tool metadata

```json
{
  "name": "nesy.load_relations",
  "title": "Load Relations",
  "description": "Load relation records and exclusive groups from inline JSON, an allowed local file, or a resource URI.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "source_type": {
        "type": "string",
        "enum": ["inline", "file", "resource_uri"]
      },
      "data": { "type": "object" },
      "path": { "type": "string" },
      "resource_uri": { "type": "string" },
      "mode": {
        "type": "string",
        "enum": ["append", "upsert", "replace_store"],
        "default": "append"
      },
      "store_id": { "type": "string", "default": "default" },
      "validate_only": { "type": "boolean", "default": false },
      "check_contradictions": { "type": "boolean", "default": true }
    },
    "required": ["source_type"],
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "loaded_relations": { "type": "integer" },
      "loaded_exclusive_groups": { "type": "integer" },
      "rejected": { "type": "integer" },
      "conflicts": { "type": "array" },
      "validate_only": { "type": "boolean" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "loaded_relations", "loaded_exclusive_groups", "rejected"]
  }
}
```

#### 示例输入：inline

```json
{
  "source_type": "inline",
  "data": {
    "version": "2.0",
    "relations": [
      {
        "source": "降价",
        "target": "销量增加",
        "relation_type": "sufficient",
        "confidence": 0.95
      }
    ],
    "exclusive_groups": [
      {
        "group_id": "profit_state",
        "members": ["利润增加", "利润减少"]
      }
    ]
  }
}
```

#### 文件安全规则

- `source_type=file` 只允许读取配置的 `allowed_roots` 下的文件。
- 禁止跟随越界 symlink。
- 默认最大文件大小：5 MB。
- 只允许 `.json`、`.jsonl`。
- 必须先做 schema validation，再写入 store。
- 加载失败时不得部分写入，除非 `mode` 显式允许 partial import。

---

### 6.10 `nesy.export_relations`

#### 用途

导出当前关系图或过滤后的关系集。

#### Tool metadata

```json
{
  "name": "nesy.export_relations",
  "title": "Export Relations",
  "description": "Export relation records and exclusive groups as JSON or JSONL, either inline or to an allowed path.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "format": { "type": "string", "enum": ["json", "jsonl"], "default": "json" },
      "filter": { "type": "object" },
      "include_exclusive_groups": { "type": "boolean", "default": true },
      "include_metadata": { "type": "boolean", "default": true },
      "destination": { "type": "string", "enum": ["inline", "file"], "default": "inline" },
      "path": { "type": "string" },
      "max_inline_bytes": { "type": "integer", "minimum": 1000, "maximum": 1000000, "default": 100000 }
    },
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "format": { "type": "string" },
      "relation_count": { "type": "integer" },
      "exclusive_group_count": { "type": "integer" },
      "data": { "type": ["object", "string", "null"] },
      "path": { "type": ["string", "null"] },
      "bytes": { "type": "integer" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "format", "relation_count", "exclusive_group_count"]
  }
}
```

#### 实现规则

- 文件写入也必须限制在 `allowed_roots`。
- 如果 inline 输出超过 `max_inline_bytes`，返回 error 或建议改为 `destination=file`。
- 输出应能被 `nesy.load_relations` 重新加载。

---

### 6.11 `nesy.summarize_graph`

#### 用途

为 Hook 或 Agent 提供关系图摘要，不返回过大的完整图。

#### Tool metadata

```json
{
  "name": "nesy.summarize_graph",
  "title": "Summarize Reasoning Graph",
  "description": "Return a compact summary of the current reasoning graph for context injection and diagnostics.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "focus_terms": {
        "type": "array",
        "items": { "type": "string" }
      },
      "context_filter": { "$ref": "#/$defs/ContextFilter" },
      "max_relations": { "type": "integer", "minimum": 1, "maximum": 200, "default": 50 },
      "max_chars": { "type": "integer", "minimum": 500, "maximum": 20000, "default": 5000 },
      "include_exclusives": { "type": "boolean", "default": true }
    },
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "object",
    "properties": {
      "status": { "type": "string" },
      "summary": { "type": "string" },
      "relation_count_included": { "type": "integer" },
      "truncated": { "type": "boolean" },
      "diagnostics": { "type": "array" },
      "trace": { "type": "array" },
      "graph_stats": { "type": "object" }
    },
    "required": ["status", "summary", "relation_count_included", "truncated"]
  }
}
```

#### 示例 structuredContent

```json
{
  "status": "ok",
  "summary": "Known reasoning relations in context ecommerce_q3:\n- 降价 sufficient 销量增加 (conf=0.95)\n- 销量增加 sufficient 市场份额增长 (conf=0.85)\nExclusive groups:\n- profit_state: 利润增加 | 利润减少 | 利润不变",
  "relation_count_included": 2,
  "truncated": false,
  "diagnostics": [],
  "trace": ["Selected 2 relations matching focus terms"],
  "graph_stats": {}
}
```

---

## 7. 算法规格

### 7.1 关系规范化

伪代码：

```python
def normalize_relation(rel):
    if rel.relation_type == "sufficient":
        return [Implication(rel.source, rel.target, rel)]
    if rel.relation_type == "necessary":
        return [Implication(rel.target, rel.source, rel)]
    if rel.relation_type == "equivalent":
        return [
            Implication(rel.source, rel.target, rel),
            Implication(rel.target, rel.source, rel),
        ]
```

### 7.2 Context filter

在执行任何推理前，先过滤关系：

```python
def context_compatible(rel, filter):
    if filter.store_id and rel.store_id != filter.store_id:
        return False
    if filter.context_id and rel.context_id != filter.context_id:
        return False
    if filter.domain and rel.metadata.get("domain") != filter.domain:
        return False
    if filter.assumptions:
        return set(filter.assumptions).issubset(set(rel.assumptions))
    if filter.valid_at:
        return temporal_window_contains(rel.temporal, filter.valid_at)
    return True
```

### 7.3 可达性查询

```python
def find_paths(graph, start, end, max_depth):
    queue = [(start, [start], [])]
    visited = set()
    results = []

    while queue:
        node, nodes, edges = queue.pop(0)
        if len(edges) > max_depth:
            continue
        if node == end:
            results.append((nodes, edges))
            continue
        state = (node, len(edges))
        if state in visited:
            continue
        visited.add(state)
        for edge in graph.out_edges(node):
            if edge.consequent not in nodes:
                queue.append((edge.consequent, nodes + [edge.consequent], edges + [edge]))

    return results
```

### 7.4 置信度聚合

```python
def aggregate_confidence(edges, policy):
    values = [e.confidence for e in edges]
    if policy == "product_independent":
        result = 1.0
        for v in values:
            result *= v
        return result
    if policy == "min":
        return min(values) if values else 1.0
    if policy == "no_aggregation":
        return None
```

### 7.5 分类算法

```python
def classify(source, target):
    fwd = find_paths(source, target)
    rev = find_paths(target, source)

    if implies_exclusive_targets(source, target):
        return "contradictory"
    if fwd and rev:
        return "equivalent"
    if fwd:
        return "sufficient"
    if rev:
        return "necessary"
    return "unknown"
```

### 7.6 反事实算法

```python
def counterfactual(if_not, world_mode="open"):
    necessarily_blocked = []
    possibly_blocked = []
    still_possible = []
    unknown = []

    # target -> if_not means if_not is necessary for target
    for target in all_nodes:
        if target == if_not:
            continue

        target_depends_on_x = path_exists(target, if_not)
        x_implies_target = path_exists(if_not, target)

        if target_depends_on_x:
            necessarily_blocked.append(target)
            continue

        if x_implies_target:
            alternatives = find_alternative_sufficient_paths(target, excluding=if_not)
            proven_independent = [p for p in alternatives if independent_from(p, if_not)]

            if proven_independent:
                still_possible.append(target)
            else:
                possibly_blocked.append(target)
            continue

        unknown.append(target)

    if world_mode == "closed":
        upgrade_possible_to_necessary_when_all_causes_blocked()

    return result
```

### 7.7 矛盾检测算法

核心步骤：

1. 构建 context-filtered 临时图。
2. 展开所有 implication paths，不超过 `max_depth`。
3. 对每个 source，收集其可推出的 targets。
4. 对每个 exclusive group，检查 source 是否推出多个 members。
5. 检查循环到互斥：`A -> B` 且 `B -> X`，其中 X 与 A 互斥。
6. 区分 hard、soft、context-separated。

伪代码：

```python
def check_exclusive_targets(graph, exclusive_groups):
    contradictions = []
    for source in graph.nodes:
        reachable = reachable_nodes(source)
        for group in exclusive_groups:
            hits = [m for m in group.members if m in reachable]
            if len(hits) >= 2:
                contradictions.append({
                    "type": "exclusive_targets",
                    "source": source,
                    "targets": hits,
                    "severity": "hard"
                })
    return contradictions
```

---

## 8. 配置规格

### 8.1 配置文件

默认配置文件：

```text
~/.nesy-reasoning/config.json
```

示例：

```json
{
  "server": {
    "transport": "stdio",
    "name": "nesy-reasoning",
    "version": "2.0.0"
  },
  "storage": {
    "backend": "sqlite",
    "sqlite_path": "~/.nesy-reasoning/nesy.db",
    "default_store_id": "default",
    "default_context_id": "default"
  },
  "reasoning": {
    "default_world_mode": "open",
    "default_max_depth": 8,
    "max_nodes": 10000,
    "max_relations": 50000,
    "confidence_policy": "product_independent"
  },
  "security": {
    "allowed_roots": ["~/projects", "~/.nesy-reasoning/relation_sets"],
    "max_file_size_bytes": 5242880,
    "allow_scope_all_clear": false,
    "rate_limit_per_minute": 120
  },
  "logging": {
    "level": "info",
    "audit_log": true
  }
}
```

### 8.2 MCP stdio 配置示例

```json
{
  "mcpServers": {
    "nesy-reasoning": {
      "command": "python3",
      "args": ["/path/to/nesy_mcp_server.py", "--transport", "stdio"],
      "env": {
        "NESY_CONFIG": "/path/to/config.json"
      }
    }
  }
}
```

### 8.3 MCP Streamable HTTP 配置示例

```json
{
  "mcpServers": {
    "nesy-reasoning": {
      "url": "http://127.0.0.1:8765/mcp",
      "headers": {
        "Authorization": "Bearer ${NESY_LOCAL_TOKEN}"
      }
    }
  }
}
```

### 8.4 环境变量

| 变量 | 说明 |
|---|---|
| `NESY_CONFIG` | 配置文件路径 |
| `NESY_STORAGE_BACKEND` | `memory` / `json` / `sqlite` |
| `NESY_SQLITE_PATH` | SQLite 文件路径 |
| `NESY_ALLOWED_ROOTS` | 逗号分隔 allowed roots |
| `NESY_LOG_LEVEL` | `debug` / `info` / `warning` / `error` |
| `NESY_LOCAL_TOKEN` | HTTP daemon 本地认证 token |

---

## 9. Claude Code Hook 集成方案

### 9.1 集成原则

Hook 集成不是 MCP Server 的替代品。Hook 只负责触发：

- 在工具调用前注入当前关系图摘要。
- 在回答结束前检查显式关系图或抽取出的候选关系是否矛盾。
- 在关键工具运行后更新上下文。

符号推理仍由 MCP Server 执行。

### 9.2 状态共享问题

如果 MCP Server 使用 stdio 被 Claude Code 启动，hook 脚本通常是另一个独立进程。Hook 不能默认访问 MCP Server 的内存图。

可选解决方案：

| 方案 | 说明 | 推荐度 |
|---|---|---|
| SQLite 共享 store | MCP Server 和 hook 脚本读写同一个 SQLite 文件 | 高 |
| Streamable HTTP daemon | MCP Client 和 hook 都调用同一个 HTTP 服务 | 高 |
| JSON 文件共享 | 简单但并发能力弱 | 中 |
| 纯内存 | Hook 无法共享 | 低 |

### 9.3 Stop Hook：输出前逻辑检查

#### Claude Code 配置

```json
{
  "hooks": {
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 hooks/nesy_stop_check.py"
          }
        ]
      }
    ]
  }
}
```

#### Hook 输入要点

Stop hook 输入包含：

```json
{
  "session_id": "abc123",
  "transcript_path": "~/.claude/projects/.../session.jsonl",
  "cwd": "/path/to/workspace",
  "permission_mode": "default",
  "hook_event_name": "Stop",
  "stop_hook_active": false,
  "last_assistant_message": "...Claude final response..."
}
```

必须检查 `stop_hook_active`，避免无限阻断。

#### Hook 输出：允许停止

```json
{}
```

或 exit 0 且不输出 JSON。

#### Hook 输出：阻止停止并要求修正

```json
{
  "decision": "block",
  "reason": "NeSy contradiction check found that the answer claims both 降价 -> 利润增加 and 降价 -> 利润减少 under the same context, but these targets are exclusive. Revise the answer or qualify the contexts."
}
```

#### `hooks/nesy_stop_check.py` 逻辑

```text
1. 从 stdin 读取 Stop hook JSON。
2. 如果 stop_hook_active=true：默认放行，或只做一次轻量检查。
3. 读取 last_assistant_message。
4. 提取候选因果声明。
5. 调用 nesy.check_contradictions，mode=facts 或 combined。
6. 如果有 hard contradictions：输出 decision=block 和 reason。
7. 否则放行。
```

#### 自然语言声明提取策略

| 策略 | 说明 | 可靠性 | 推荐阶段 |
|---|---|---:|---|
| A | 不抽取，只检查当前显式关系图 | 高 | 起步 |
| B | 正则匹配“X导致Y”“X是Y的必要条件”等高置信模式 | 中 | Phase 5 |
| C | 调轻量 LLM 输出结构化候选 facts，再由符号层验证 | 中 | Phase 6 |
| D | 项目内约定回答必须附 `NESY_FACTS` JSON 块 | 高 | 推荐实用方案 |

推荐格式：

```text
NESY_FACTS:
[
  {"source":"降价","target":"销量增加","relation_type":"sufficient","context_id":"ecommerce_q3"}
]
```

### 9.4 PreToolUse Hook：工具调用前注入图摘要

#### Claude Code 配置

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash|Edit|Write|Read|Grep|Glob|WebSearch|WebFetch|mcp__nesy-reasoning__.*",
        "hooks": [
          {
            "type": "command",
            "command": "python3 hooks/nesy_context_inject.py"
          }
        ]
      }
    ]
  }
}
```

#### Hook 输入要点

PreToolUse hook 在 Claude 已生成工具参数、工具执行前运行。输入包含：

```json
{
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {
    "command": "npm test"
  },
  "tool_use_id": "toolu_...",
  "cwd": "/path/to/workspace"
}
```

#### Hook 输出：注入上下文

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "NeSy context injected.",
    "additionalContext": "Known NeSy reasoning graph:\n- 登录 necessary 下单\n- 删除用户 sufficient 账户不可恢复\nExclusive groups:\n- account_state: 活跃 | 删除 | 暂停"
  }
}
```

#### `hooks/nesy_context_inject.py` 逻辑

```text
1. 从 stdin 读取 PreToolUse hook JSON。
2. 根据 tool_input、cwd、session_id 选择 context_filter。
3. 调 nesy.summarize_graph 获取短摘要。
4. 返回 hookSpecificOutput.additionalContext。
5. 不修改危险工具权限；权限控制应由独立安全 hook 处理。
```

### 9.5 PostToolBatch Hook：工具批次后更新上下文

可选。适合在一组工具执行后，把新发现的事实提示给 Claude。

```json
{
  "hooks": {
    "PostToolBatch": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 hooks/nesy_post_batch_summary.py"
          }
        ]
      }
    ]
  }
}
```

输出：

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolBatch",
    "additionalContext": "NeSy reminder: current graph contains a necessary dependency 登录 <- 下单. Do not claim orders can be submitted without login unless a new relation overrides this context."
  }
}
```

### 9.6 Hook 安全要求

- Hook 脚本必须有超时。
- Hook 脚本不得无边界扫描 transcript。
- Stop hook 必须处理 `stop_hook_active`。
- Hook 不应把完整关系图无限注入上下文，应使用 `nesy.summarize_graph`。
- Hook 调用 HTTP daemon 时必须使用 localhost 和 token。
- Hook 报错时默认放行还是阻断要可配置。推荐：Stop 检查失败默认放行并记录 warning；安全关键项目可配置为阻断。

---

## 10. 安全规格

### 10.1 输入验证

所有工具必须：

- 校验 JSON Schema。
- 拒绝未知字段，除非字段位于 `metadata`。
- 限制字符串长度，默认单个 proposition 最大 512 字符。
- 限制数组大小，默认单次最多 500 条 relations。
- 限制 `max_depth`，默认最大 20。
- 限制导入文件大小。

### 10.2 文件访问

`load_relations` 和 `export_relations` 的文件模式必须：

- 使用 allowlist roots。
- 解析真实路径后检查仍在 allowed root 内。
- 禁止路径穿越。
- 禁止越界 symlink。
- 限制扩展名。
- 不读取隐藏敏感文件，除非用户显式配置。

### 10.3 HTTP 安全

Streamable HTTP 模式必须：

- 默认只绑定 `127.0.0.1`。
- 校验 Origin。
- 远程访问必须认证。
- 设置请求体大小上限。
- 设置速率限制。
- 设置请求超时。
- 审计所有写操作。

### 10.4 Tool 安全分级

| 工具 | 风险 | 默认是否需要确认 |
|---|---|---|
| `nesy.classify` | 低 | 否 |
| `nesy.verify_chain` | 低 | 否 |
| `nesy.check_contradictions` | 低 | 否 |
| `nesy.counterfactual` | 低 | 否 |
| `nesy.list_relations` | 中，可能泄露知识库 | 视配置 |
| `nesy.summarize_graph` | 中，可能泄露知识库 | 视配置 |
| `nesy.assert_relations` | 中，写状态 | 是 |
| `nesy.assert_exclusive` | 中，写状态 | 是 |
| `nesy.load_relations` | 高，读文件+写状态 | 是 |
| `nesy.export_relations` | 高，写文件/泄露数据 | 是 |
| `nesy.clear_relations` | 高，破坏性 | 是 |

### 10.5 审计日志

写操作必须记录：

```json
{
  "event_id": "audit_001",
  "tool_name": "nesy.assert_relations",
  "session_id": "abc123",
  "store_id": "default",
  "context_id": "ecommerce_q3",
  "input_hash": "sha256:...",
  "result_status": "ok",
  "created_at": "2026-05-21T00:00:00Z"
}
```

### 10.6 Prompt injection 边界

- MCP tool description 不能承诺“自动判断自然语言语义”。
- `metadata.source` 等字段不得被当成指令执行。
- 导入文件中的 `description`、`note`、`provenance` 只作为数据，不作为 prompt。
- Hook 注入 `additionalContext` 时要标明其来源为符号图摘要，不包含可执行命令。

---

## 11. 实现路线图

### Phase 1：协议骨架与数据模型

目标：实现符合 MCP 的工具暴露和基本数据存储。

任务：

- [ ] MCP Server 初始化与 `tools` capability。
- [ ] `tools/list` 中暴露所有 v2 工具 metadata。
- [ ] 通用 CallToolResult 封装：`content` + `structuredContent`。
- [ ] Pydantic/JSON Schema 模型。
- [ ] RelationStore 内存实现。
- [ ] Canonical implication edge 生成。
- [ ] `nesy.assert_relations`。
- [ ] `nesy.list_relations`。
- [ ] `nesy.clear_relations`。

验收标准：

- MCP client 能发现工具。
- 所有工具都有 inputSchema。
- 写入关系后可列出关系和 implication edges。
- stdout 不出现非 MCP 日志。

### Phase 2：分类与链路验证

目标：完成核心推理能力。

任务：

- [ ] `nesy.classify`。
- [ ] `nesy.verify_chain`。
- [ ] BFS/weighted path search。
- [ ] 双向可达映射为 sufficient/necessary/equivalent。
- [ ] cycle protection。
- [ ] confidence aggregation。

验收标准：

- A sufficient B, B sufficient C => A sufficient C。
- A necessary B, B necessary C => A necessary C。
- equivalent 与 sufficient/necessary 混合路径正确。
- sufficient + necessary 不被错误标成 sufficient。

### Phase 3：互斥与矛盾检测

目标：实现显式语义矛盾检测。

任务：

- [ ] `nesy.assert_exclusive`。
- [ ] `nesy.check_contradictions`。
- [ ] mode=graph/facts/combined。
- [ ] context-separated conflict。
- [ ] transitive contradiction。

验收标准：

- 同一 source 推出互斥 targets 报 hard contradiction。
- 不同 context 的相同形式冲突不报 hard contradiction。
- 未声明互斥时不臆断语义矛盾。

### Phase 4：反事实推理

目标：实现开放世界反事实推理。

任务：

- [ ] `nesy.counterfactual`。
- [ ] necessarily_blocked / possibly_blocked / still_possible / unknown 分类。
- [ ] alternative path search。
- [ ] closed-world optional mode。

验收标准：

- X sufficient Y 且 not X，不能推出 not Y。
- X necessary Y 且 not X，能推出 not Y。
- 替代路径不独立时不能标成 still_possible。
- closed-world 使用必须出现在 trace 中。

### Phase 5：持久化与导入导出

目标：支持长期项目使用。

任务：

- [ ] SQLite backend。
- [ ] JSON import/export。
- [ ] `nesy.load_relations`。
- [ ] `nesy.export_relations`。
- [ ] allowed_roots 文件安全。
- [ ] audit log。

验收标准：

- 重启 server 后关系仍存在。
- 导出的 JSON 可重新加载。
- 越界路径被拒绝。

### Phase 6：Claude Code Hook 集成

目标：实现自动触发的推理辅助。

任务：

- [ ] `nesy.summarize_graph`。
- [ ] Stop hook checker。
- [ ] PreToolUse context injector。
- [ ] SQLite 或 HTTP daemon 状态共享。
- [ ] Hook timeout 和 fallback 策略。

验收标准：

- Stop hook 可读取 `last_assistant_message`。
- `stop_hook_active=true` 时不会无限阻断。
- PreToolUse 能通过 `hookSpecificOutput.additionalContext` 注入图摘要。

### Phase 7：评估与消融实验

目标：量化 MCP 对 Agent 推理可靠性的提升。

任务：

- [ ] 构造测试集：分类、传递、反事实、矛盾检测。
- [ ] 对照：LLM only vs LLM + MCP。
- [ ] 工具消融：无 counterfactual、无 contradiction、无 classify。
- [ ] 记录失败案例。
- [ ] 发布 benchmark fixture。

验收标准：

- 每个核心工具都有边际贡献统计。
- 失败案例能转化为 regression tests。

---

## 12. 技术栈

| 组件 | 推荐选型 | 说明 |
|---|---|---|
| 语言 | Python 3.11+ | 类型、异步、生态成熟 |
| MCP SDK | `mcp` Python SDK | 官方协议支持 |
| Schema | Pydantic v2 + JSON Schema | 输入输出校验 |
| 图索引 | NetworkX | 轻量，适合 MVP 到中型图 |
| 持久化 | SQLite | 本地可靠、Hook 共享方便 |
| 测试 | pytest | 标准单元测试 |
| 属性测试 | hypothesis | 随机图推理性质测试 |
| JSON 校验 | jsonschema | 验证 tool schema 和 fixture |
| Lint | ruff | 代码质量 |
| 类型检查 | pyright 或 mypy | 减少 schema/模型错配 |

### 12.1 可选高级后端

| 后端 | 用途 | 何时引入 |
|---|---|---|
| Z3 | SAT/SMT 约束求解 | 复杂互斥、数值约束、组合约束 |
| SWI-Prolog | 规则推理 | 需要 Prolog 生态和递归规则 |
| DuckDB/Postgres | 大型关系库 | 多项目、多用户、分析型查询 |
| TLA+ | 形式验证 | 验证 Agent 策略约束不变量 |

基础版本不依赖这些高级后端。

---

## 13. 测试规格

### 13.1 单元测试矩阵

| 编号 | 输入 | 期望 |
|---|---|---|
| T01 | A sufficient B | classify(A,B)=sufficient |
| T02 | A necessary B | classify(A,B)=necessary |
| T03 | A equivalent B | classify(A,B)=equivalent |
| T04 | A sufficient B, B sufficient C | classify(A,C)=sufficient |
| T05 | A necessary B, B necessary C | classify(A,C)=necessary |
| T06 | A equivalent B, B sufficient C | classify(A,C)=sufficient |
| T07 | A sufficient B, B necessary C | classify(A,C)=unknown |
| T08 | A necessary B, B sufficient C | classify(A,C)=unknown |
| T09 | A sufficient B, C sufficient B | 不能推出 A not necessary B |
| T10 | A necessary B, if_not A | counterfactual => B necessarily_blocked |
| T11 | A sufficient B, if_not A | counterfactual => B possibly_blocked 或 unknown，不是 necessarily_blocked |
| T12 | A sufficient B, C sufficient B, C independent_of A | if_not A => B still_possible |
| T13 | A sufficient B, A sufficient C, B exclusive C | contradiction hard |
| T14 | A sufficient B context X, A sufficient C context Y, B exclusive C same_context | no hard contradiction |
| T15 | A -> B -> C -> A | 搜索不死循环 |
| T16 | 多路径 A -> C | 返回最高 confidence 路径，保留其他路径 |
| T17 | 未声明 exclusives 的“利润增加/利润减少” | 不报 hard contradiction |
| T18 | equivalent 双向路径 | implication edges 数量为 2 |
| T19 | load_relations 越界路径 | error |
| T20 | clear_relations dry_run | 状态不变 |

### 13.2 属性测试

随机生成 DAG 和关系，验证：

- reachability 自反性不应默认产生 equivalent，除非允许零长度路径；分类时 source=target 可单独处理。
- 对所有 A,B,C，如果 A->B 且 B->C，则 A->C。
- 删除一条边后，不能新增可达路径。
- 添加 equivalent(A,B) 后，A->B 和 B->A 都存在。
- confidence aggregation 不超过路径中最大可能值，product 策略下不大于任一边置信度。

### 13.3 MCP 集成测试

- `tools/list` 返回所有工具。
- 每个工具有 `inputSchema`。
- `tools/call` 返回 `content` 和 `structuredContent`。
- outputSchema 与 structuredContent 匹配。
- tool execution error 使用 `isError=true`。
- stdio 模式 stdout 无非法日志。

### 13.4 Hook 集成测试

- Stop hook 能读取 `last_assistant_message`。
- Stop hook 对 hard contradiction 返回 `decision=block`。
- Stop hook 在 `stop_hook_active=true` 时避免循环。
- PreToolUse hook 返回 `hookSpecificOutput.additionalContext`。
- Hook 超时时 fallback 行为符合配置。

### 13.5 Benchmark 设计

数据集：

1. 因果分类题：判断充分/必要/充要/未知。
2. 传递链题：多跳推理和方向混合。
3. 矛盾检测题：互斥目标、上下文隔离、软冲突。
4. 反事实题：开放世界、闭合世界、替代路径。
5. 业务场景题：电商定价、权限系统、部署流程、数据管线。

指标：

| 指标 | 说明 |
|---|---|
| logical accuracy | 逻辑分类正确率 |
| contradiction recall | 矛盾召回率 |
| false contradiction rate | 错报矛盾率 |
| counterfactual conservatism | 是否避免过强结论 |
| trace completeness | 是否返回可解释路径 |
| latency | 工具调用延迟 |

对照组：

```text
LLM only
LLM + tool descriptions but no calls
LLM + NeSy classify only
LLM + classify + verify_chain
LLM + full NeSy MCP
```

---

## 14. 示例关系集

### 14.1 电商定价场景

```json
{
  "version": "2.0",
  "relations": [
    {
      "source": "降价",
      "target": "销量增加",
      "relation_type": "sufficient",
      "confidence": 0.95,
      "context_id": "ecommerce_q3",
      "metadata": { "domain": "ecommerce" }
    },
    {
      "source": "销量增加",
      "target": "市场份额增长",
      "relation_type": "sufficient",
      "confidence": 0.85,
      "context_id": "ecommerce_q3"
    },
    {
      "source": "广告增投",
      "target": "销量增加",
      "relation_type": "sufficient",
      "confidence": 0.80,
      "context_id": "ecommerce_q3"
    },
    {
      "source": "登录",
      "target": "下单",
      "relation_type": "necessary",
      "confidence": 1.0,
      "context_id": "ecommerce_q3"
    }
  ],
  "exclusive_groups": [
    {
      "group_id": "profit_state",
      "members": ["利润增加", "利润减少", "利润不变"],
      "context_id": "ecommerce_q3"
    }
  ]
}
```

### 14.2 权限系统场景

```json
{
  "version": "2.0",
  "relations": [
    {
      "source": "拥有有效会话",
      "target": "访问用户面板",
      "relation_type": "necessary",
      "confidence": 1.0,
      "context_id": "auth_system"
    },
    {
      "source": "管理员角色",
      "target": "删除用户",
      "relation_type": "necessary",
      "confidence": 1.0,
      "context_id": "auth_system"
    },
    {
      "source": "删除用户",
      "target": "账户不可恢复",
      "relation_type": "sufficient",
      "confidence": 1.0,
      "context_id": "auth_system"
    }
  ],
  "exclusive_groups": [
    {
      "group_id": "account_state",
      "members": ["账户活跃", "账户删除", "账户暂停"],
      "context_id": "auth_system"
    }
  ]
}
```

---

## 15. 未来扩展方向

### 15.1 时序推理层

在 `temporal.delay` 基础上支持时序链：

```text
降价 --immediate--> 销量增加 --short--> 市场份额增长
```

输出：

```json
{
  "prediction_window": "1-4w",
  "paths": [
    {
      "nodes": ["降价", "销量增加", "市场份额增长"],
      "estimated_delay": "1-4w",
      "confidence": 0.8075
    }
  ]
}
```

### 15.2 概率因果图

引入更严格的 causal DAG：

- 区分 observation、intervention、counterfactual。
- 支持 `do(X)`。
- 支持 backdoor adjustment 元数据。
- 避免把相关性关系写成因果关系。

### 15.3 SAT/SMT 后端

对复杂约束使用 Z3：

```text
库存 > 0
价格 >= 成本
折扣 <= 30%
利润状态 ∈ {增加, 减少, 不变}
```

用于检测数值约束、区间约束和组合约束。

### 15.4 知识库集成

从 Notion、Google Drive、代码仓库、ADR 文档中抽取候选关系：

```text
提取候选关系 -> 人类确认 -> nesy.assert_relations -> 符号验证
```

抽取层必须和验证层分离。

### 15.5 Capability 注册

将 NeSy Reasoning 作为 Agent capability 注册到能力市场或组织内部工具目录：

```json
{
  "capability": "deterministic_symbolic_reasoning",
  "tools": [
    "nesy.classify",
    "nesy.verify_chain",
    "nesy.check_contradictions",
    "nesy.counterfactual"
  ]
}
```

### 15.6 多 Agent 共享推理记忆

多个 Agent 共享一个 workspace graph：

- Planner 写入任务依赖。
- Coder 写入实现约束。
- Reviewer 检查矛盾。
- Release Agent 做反事实风险分析。

需要更严格的权限和审计。

---

## 16. 兼容性与迁移

### 16.1 旧字段迁移

| 旧字段 | 新字段 |
|---|---|
| `from` | `source` |
| `to` | `target` |
| `type` | `relation_type` |
| `temporal_delay` | `temporal.delay` |
| `metadata.domain` | 可保留 |

迁移示例：

旧：

```json
{
  "from": "降价",
  "to": "销量增加",
  "type": "sufficient",
  "confidence": 0.95,
  "temporal_delay": "immediate"
}
```

新：

```json
{
  "source": "降价",
  "target": "销量增加",
  "relation_type": "sufficient",
  "confidence": 0.95,
  "temporal": { "delay": "immediate" }
}
```

### 16.2 旧输出迁移

旧 `counterfactual.unaffected` 迁移为：

```json
{
  "unknown": [...],
  "not_derivably_affected": [...]
}
```

旧 `classify.reverse_relation = not_necessary` 迁移为：

```json
{
  "necessity_status": {
    "status": "unknown"
  }
}
```

只有在有独立反例时才使用：

```json
{
  "necessity_status": {
    "status": "proven_not_necessary",
    "counterexample": "广告增投",
    "proof": "广告增投 -> 销量增加 and 广告增投 independent_of 降价"
  }
}
```

---

## 17. 最小可发布版本与完整版本边界

本规格不是只定义 MVP。完整目标工具集包括 11 个 MCP tools、持久化、Hook、测试与未来扩展。

为了工程交付，可以把实现拆成版本：

| 版本 | 必须包含 |
|---|---|
| v0.1 | MCP tools/list、assert/list/clear、内存 store |
| v0.2 | classify、verify_chain |
| v0.3 | assert_exclusive、check_contradictions |
| v0.4 | counterfactual |
| v0.5 | load/export、SQLite |
| v0.6 | summarize_graph、Claude Code hooks |
| v1.0 | 完整测试、审计、安全配置、文档 |

但最终 v1.0 的产品范围应覆盖本规格全部核心能力，而不是停留在 MVP。

---

## 18. 术语表

| 术语 | 含义 |
|---|---|
| proposition | 命题，关系图中的节点 |
| source | 条件命题 |
| target | 结果命题 |
| sufficient | source 成立足以推出 target |
| necessary | target 成立必须依赖 source |
| equivalent | source 与 target 互相推出 |
| implication edge | 内部规范化蕴含边 |
| open world | 未知不等于否定 |
| closed world | 在声明完整性的范围内，未列出可视为不存在 |
| exclusive group | 互斥命题集合 |
| context_id | 关系适用上下文 |
| store_id | 关系存储空间 |
| trace | 推理过程记录 |
| evidence confidence | 证据可靠性，不等于逻辑有效性 |

---

## 19. 结论

NeSy Reasoning MCP Server 的核心价值是把 Agent 的高风险推理从自然语言概率模式迁移到结构化符号图上。LLM 仍负责提出假设、组织回答、解释结果；MCP Server 负责执行严格的关系分类、传递验证、矛盾检测和反事实推理。

本 v2.0 规格的关键原则是：

```text
能证明才说 proven。
不能证明就说 unknown。
可能受阻不等于必然受阻。
替代充分条件不等于非必要性证明。
上下文不同不等于逻辑冲突。
置信度不是逻辑有效性。
```

按本规格实现后，该 MCP Server 可以作为 Claude Code、Cursor、Codex 等 Agent harness 的确定性推理层，为复杂任务提供可追溯、可测试、可审计的逻辑能力。
