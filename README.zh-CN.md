# NeSy Reasoning MCP

[![CI](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/6tizer/nesy-reasoning-mcp/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11--3.14-blue)
![MCP](https://img.shields.io/badge/MCP-stdio%20%7C%20Streamable%20HTTP-green)
![Version](https://img.shields.io/badge/version-1.0.0-informational)

[English](README.md) | 简体中文

一个本地 MCP server，为 AI Agent 提供确定性的推理记忆：结构化关系存储、因果分类、链路验证、矛盾检查、图摘要和反事实分析。

它不替代 LLM。LLM 可以提出结构化事实；这个 server 用小而可测试的符号引擎检查这些事实。

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
- [Claude Code 指令](CLAUDE.md)
- [MCP 安装](docs/install.md)
- [内部测试 profile](docs/internal-testing.md)
- [安全](docs/security.md)
- [卸载 / 回滚](docs/uninstall.md)
- [评估](docs/evaluation.md)
- [路线图](docs/roadmap.md)
- [开发](docs/development.md)
