# NeSy Reasoning MCP

[![CI](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11--3.14-blue)
![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20Streamable%20HTTP-green)
![Version](https://img.shields.io/badge/version-1.0.0-informational)

[English](README.md) | 简体中文

NeSy Reasoning MCP 为 Agent 提供一个符号推理图，用来检查蕴含、必要性、矛盾和反事实影响。

它是 Agent 侧的逻辑审计层。它让关键推理关系显式、可检查，而不是只藏在自然语言总结里。它提升的是推理可审计性，不是搜索质量、来源真实性或召回率。

## 它是什么 / 不是什么

| 它是 | 它不是 |
|---|---|
| 符号推理图 | 搜索引擎 |
| 一致性检查器 | 通用记忆库 |
| 外部推理草稿本 | 向量数据库 |
| 蕴含、必要性、矛盾和反事实 verifier | 文档总结器 |
| 关键推理关系的持久图 | 存放所有相关事实的地方 |

适合用在隐藏推理容易出错的任务：长研究、代码依赖分析、产品或工程决策分析。不适合简单搜索、短总结、闲聊问答，或“记住所有相关事实”的工作流。

## 它给 Agent 带来什么

- **长期推理记忆**：用 SQLite 或 JSON 保存结构化关系，避免聊天或 MCP 进程重启后丢状态。
- **确定性逻辑检查**：判断 `A` 对 `B` 是充分、必要、充要、矛盾，还是未知。
- **可验证推理链**：证明或拒绝 `A -> B -> C` 这类多跳蕴含路径。
- **矛盾护栏**：检测显式互斥、直接否定、循环到否定、软置信度张力。
- **反事实分析**：在开放世界或受保护的封闭世界语义下，分析某个命题被假设为假后哪些目标仍可能成立。
- **Hook 集成**：工具调用前注入紧凑图摘要；最终回答含有显式 `NESY_FACTS` 硬矛盾时阻断。

## 快速开始

从本地 checkout 安装依赖：

```bash
uv sync
```

运行 stdio MCP server：

```bash
uv run nesy-reasoning-mcp --transport stdio
```

使用 SQLite 持久化：

```bash
mkdir -p ~/.nesy-reasoning
NESY_STORAGE_BACKEND=sqlite NESY_SQLITE_PATH=~/.nesy-reasoning/nesy.db \
  uv run nesy-reasoning-mcp --transport stdio
```

运行带认证的本地 Streamable HTTP daemon：

```bash
NESY_LOCAL_TOKEN='change-me' uv run nesy-reasoning-mcp --transport http
```

验证确定性 benchmark：

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

## MCP 客户端配置

可以从这个 stdio 配置开始：

```json
{
  "mcpServers": {
    "nesy-reasoning": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/nesy-reasoning-mcp",
        "run",
        "nesy-reasoning-mcp",
        "--transport",
        "stdio"
      ],
      "env": {
        "PYTHONPATH": "/path/to/nesy-reasoning-mcp/src",
        "NESY_STORAGE_BACKEND": "sqlite",
        "NESY_SQLITE_PATH": "~/.nesy-reasoning/nesy.db"
      }
    }
  }
}
```

更多示例：

- [examples/mcp-config.json](examples/mcp-config.json)
- [examples/nesy-config.json](examples/nesy-config.json)
- [examples/claude-hooks.json](examples/claude-hooks.json)
- [examples/internal-test](examples/internal-test/README.md)
- [Agent 使用策略](docs/agent-usage.md)
- [Agent SDK ingestion 设计](docs/agent-sdk-ingestion.md)

## Claude Code 安装

Claude Code 接入分两步：

1. 用 [examples/mcp-config.json](examples/mcp-config.json) 这类 stdio 配置添加 MCP server。
2. 可选：用 [examples/claude-hooks.json](examples/claude-hooks.json) 或
   [examples/internal-test](examples/internal-test/README.md) 里的 wrapper 添加 hooks。

Hooks 是单独进程，所以要用 SQLite、JSON 或本地 HTTP daemon，让 hook 和 MCP
server 看到同一个图。进程内 memory 不能跨进程共享。

配置 Claude Code 后，跑 internal-test smoke：

```bash
env PYTHONPATH=src uv run python examples/internal-test/smoke.py
```

预期输出：

```text
internal-test smoke ok
```

## 最小推理示例

声明两个充分关系：

```json
{
  "relations": [
    {"source": "A", "target": "B", "relation_type": "sufficient"},
    {"source": "B", "target": "C", "relation_type": "sufficient"}
  ]
}
```

然后用 `nesy.classify` 查询 `A` 和 `C`。Server 会推出 `A -> C`，并返回带路径 trace 的 `classification="sufficient"`。

如果 `B` 和 `C` 被声明为互斥，而同一个 source 可以推出两者，`nesy.check_contradictions` 会返回 hard contradiction。Stop hook 可以在最终回答包含冲突的结构化 `NESY_FACTS` 时阻断。

## 推荐 Agent 用法

不要把 Agent 找到的所有相关事实都写进图。只有来源材料足以支持某个逻辑蕴含、必要性、等价或显式互斥时，才断言关系，并写清 context、confidence 和 provenance。

适合的 prompt：

```text
研究 A 是否真的能解决 B。建立一个小的 NeSy 因果链图，带 provenance，然后验证最终结论。

阅读这个代码库，建模 feature flag、配置依赖和互斥路径。然后回答移除 config X 后什么一定会坏。

比较方案 A/B/C。编码目标、约束、风险和互斥项，然后验证哪个方案足以达成目标。
```

不适合的 prompt：

```text
搜索 AI 新闻。
总结这篇文章。
记住所有相关事实。
```

详见 [Agent 使用策略](docs/agent-usage.md)，其中包含 Do/Don't 表、可复制 prompt、自动抽取流程和过度断言反例。

自动候选关系抽取与审核见
[Agent SDK ingestion 设计](docs/agent-sdk-ingestion.md)。该 ingestion app 规划为外部
Agent SDK 工作流；NeSy MCP 仍只负责推理与存储。

## 工具列表

| Tool | 用途 | 修改状态 |
|---|---|---:|
| `nesy.assert_relations` | 新增或更新结构化关系。 | 是 |
| `nesy.list_relations` | 列出原始关系和派生 implication edges。 | 否 |
| `nesy.clear_relations` | 清理 context、store、filter 或允许范围。 | 是 |
| `nesy.classify` | 基于图可达性判断 source/target 关系。 | 否 |
| `nesy.verify_chain` | 验证显式或搜索得到的蕴含路径。 | 否 |
| `nesy.assert_exclusive` | 声明互斥命题组。 | 是 |
| `nesy.check_contradictions` | 检查图、facts 或 combined 矛盾。 | 否 |
| `nesy.load_relations` | 从 inline、文件、安全本地 `file://` URI 加载关系集。 | 是 |
| `nesy.export_relations` | inline 或写入允许路径导出关系集。 | 可选 |
| `nesy.summarize_graph` | 返回紧凑、确定性的图摘要。 | 否 |
| `nesy.counterfactual` | 分析某命题被假设为 false 后的影响。 | 否 |
| `nesy.reason_over_relations` | 基于调用方提供的临时关系运行推理。 | 否 |

## 临时推理

当外部 memory 或 retrieval 系统返回候选关系，但不应该写入 NeSy 长期图时，使用
`nesy.reason_over_relations`：

```text
external memory retrieval -> candidate relations -> NeSy ephemeral reasoning -> answer/evidence
```

该工具接收临时 `relations`、`exclusive_groups`、`propositions`、
`independence_records`，以及 `query` mode：`classify`、`verify_chain`、
`counterfactual`、`check_contradictions` 或 `summarize_graph`。结果放在
`result` 下，并返回 `persisted=false`。

这只是下游逻辑检查层。它不抓取文档，不构建 embedding index，不做自然语言事实抽取，
也不会把临时候选关系自动写入持久记忆。

自动外部证据摄取见
[Agent SDK ingestion 设计](docs/agent-sdk-ingestion.md)。默认模式为 dry-run；只有显式开启
write mode 且通过 gate 后才允许持久写图。

## 存储和传输

存储后端：

- `memory`：适合短测试；重启即丢状态。
- `json`：适合简单单用户本地持久化。
- `sqlite`：推荐用于长期本地记忆，以及 MCP 与 hooks 共享状态。

传输方式：

- `stdio`：默认 MCP server 模式。
- `http`：带认证的本地 Streamable HTTP daemon。

HTTP 默认只绑定本机，并要求 `NESY_LOCAL_TOKEN`。

## Hooks

CLI 包含 Claude Code hook helpers：

```bash
uv run nesy-reasoning-mcp hook pretooluse
uv run nesy-reasoning-mcp hook stop
```

- **PreToolUse**：注入紧凑图摘要作为 additional context。
- **Stop**：检查当前图，或最终回答里的显式 `NESY_FACTS:` JSON array。

Hooks 是单独进程，所以应使用 SQLite、JSON 或同一个 HTTP daemon。stdio MCP 的进程内 memory 不会被 hook 进程共享。

## 安全模型

本项目是 local-first：

- HTTP mode 使用本地 bearer token。
- 文件 load/export 限制在配置的 `allowed_roots` 内。
- hidden relation paths 默认阻断，除非显式开启。
- 启用 audit logging 时，写操作会记录审计条目。
- 破坏性或写文件工具仍应由 MCP client 或 wrapper policy 要求确认。

查看 audit history：

```bash
NESY_CONFIG=/path/to/nesy-config.json uv run nesy-reasoning-mcp audit list --format json
```

详见 [docs/security.md](docs/security.md)。

## 评估

离线确定性评估：

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

Agent 模式矩阵评估：

```bash
env PYTHONPATH=src uv run nesy-reasoning-mcp eval agent \
  --fixture benchmarks/fixtures/core.json \
  --format json
```

可选 live OpenAI evaluation 只用于手动运行，CI 不要求：

```bash
uv sync --extra eval
export OPENAI_API_KEY='<set outside the repo>'
env PYTHONPATH=src uv run --extra eval nesy-reasoning-mcp eval llm \
  --fixture benchmarks/fixtures/core.json \
  --case-id classify_direct_sufficient \
  --format json
```

## 边界

- 不自动做自然语言关系抽取。
- 不提供 hosted multi-user auth。
- 暂无 Postgres/team graph backend。
- 不做远程 MCP resource fetching；`resource_uri` 只支持安全本地 `file://` 加载。
- v1.0 没有 PostToolBatch hook。
- 这是推理辅助工具，不替代法律、医疗、金融或安全关键领域专家判断。

## 开发

```bash
uv sync --locked
uv run ruff format --check .
uv run ruff check .
uv run mypy src/nesy_reasoning_mcp
uv run pytest
env PYTHONPATH=src uv run nesy-reasoning-mcp eval run --fixture benchmarks/fixtures/core.json --format json
env PYTHONPATH=src uv run nesy-reasoning-mcp eval agent --fixture benchmarks/fixtures/core.json --format json
```

## 文档

- [完整规格](docs/spec-v2.md)
- [SPEC 合规矩阵](SPEC_COMPLIANCE.md)
- [Agent 指令](AGENTS.md)
- [Agent 使用策略](docs/agent-usage.md)
- [Claude Code 指令](CLAUDE.md)
- [MCP 安装](docs/install.md)
- [内部测试 profile](docs/internal-testing.md)
- [安全](docs/security.md)
- [卸载 / 回滚](docs/uninstall.md)
- [评估](docs/evaluation.md)
- [路线图](docs/roadmap.md)
- [开发](docs/development.md)
