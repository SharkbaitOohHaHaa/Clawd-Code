# Claude Code Python Feature List & PR Roadmap

> 面向社区贡献者的能力清单、路线图与 PR 指南。
>
> 项目定位：**基于真实 Claude Code 源码结构的 Python 重构版**。当前已经具备可用的多 Provider 聊天 CLI、完整工具系统框架与 Agent Loop，正在分阶段完善原生 Claude Code 的关键能力。

---

## 状态说明

| 状态 | 含义 |
|------|------|
| ✅ 已实现 | 当前仓库中已有可验证实现 |
| 🟡 部分完成 | 已有骨架、镜像层或部分能力，但未形成完整闭环 |
| ⏳ 规划中 | 已明确方向，欢迎提交 PR |
| 🚫 未开始 | 当前尚无实现 |

权威运行时能力状态定义在 `src/capability_manifest.json`，并由确定性 reconciler 与实际 registry / 文件系统 / skill trust 状态比较。

<!-- CAPABILITY-MANIFEST:START -->
| Capability state | Tools | Features |
|---|---|---|
| Active / supported | AskUserQuestion, CronCreate, CronDelete, CronList, DataInspect, DataTransform, Edit, EnterPlanMode, EnterWorktree, ExitPlanMode, ExitWorktree, Glob, Grep, LSP, ListMcpResourcesTool, ListMcpToolsTool, MCP, NotebookEdit, Read, ReadMcpResourceTool, Skill, Sleep, StructuredOutput, TaskCreate, TaskGet, TaskList, TaskOutput, TaskStop, TaskUpdate, TodoWrite, ToolSearch, WebFetch, WebSearch, Write | agent_loop, authentication_recovery, capability_manifest, capability_reconciler, chinese_provider_ecosystem, context_engine, custom_commands_tools, data_engineering_runtime, developer_quality_tooling, enterprise_workflow_extensions, full_registered_tool_schema, git_worktree_runtime, lsp_runtime, mcp_resource_runtime, mcp_runtime, permission_contract, permission_policy_configuration, project_setup_advisor, provider_extensions, python_plugin_runtime, sanitized_runtime_instrumentation, sensitive_path_policy, skill_trust_runtime |
| Clawd-specific | Agent, BriefPreview, GeminiThink, Memory, QwenMediaAnalyze, SendMessage, SendUserMessage, TeamCreate, TeamDelete, YouTubeAnalyze |  |
| Intentionally disabled | Bash, Config | hook_runtime |
| Deferred / not production-ready |  | subagent_runtime |
<!-- CAPABILITY-MANIFEST:END -->

---

## 项目亮点

- **Python 重构版**：不是单纯 UI 模仿，而是按 Claude Code 的架构思路重建。
- **多模型先行**：当前已支持 Anthropic、OpenAI、DeepSeek、Qwen、GLM、MiniMax 六类 Provider。
- **CLI / REPL 可用**：已经具备基础交互能力，适合持续迭代。
- **工具系统框架完整**：已实现 30+ 工具模块、Agent Loop、权限系统框架。
- **更适合社区共建**：Python 生态更易二开，适合工具、自动化、数据工程场景扩展。
- **强调真实性**：优先补齐真正可运行的核心链路，而不是只扩大命令/工具名录。

---

## 核心系统

| 能力 | 状态 | 当前情况 |
|------|------|----------|
| CLI 启动入口 | ✅ | 已支持 `clawd`、`login`、`config`、`--version` |
| 交互式 REPL | ✅ | 支持交互式输出、历史记录、Tab 补全、多行输入 |
| Slash Commands | ✅ | canonical palette 包含 `/help`、`/clear-chat`、`/save-session`、`/load-session`、`/compact-context`、`/context-usage`、`/usage`、`/session-usage`、`/list-tools`、`/run-tool`、`/list-skills`、`/setup-project`、`/exit` 等；旧短名称保留为兼容别名 |
| 多 Provider 抽象 | ✅ | 已支持 Anthropic / OpenAI / DeepSeek / Qwen / GLM / MiniMax |
| Provider 配置管理 | ✅ | 支持默认 Provider、Base URL、默认模型配置 |
| 会话持久化 | ✅ | 支持保存/加载本地会话 |
| 会话消息管理 | ✅ | 支持会话历史维护与序列化 |
| 错误恢复 / 重新登录 | ✅ | REPL 识别 401 / provider auth failures，避免 direct fast path 重复请求；可显式重新配置 provider，并同步 session / command context；失败请求不会自动重试 |
| Token / Cost 跟踪 | ✅ | `/usage` 提供 API/model usage 与本地 skill/tool activity；`/session-usage` 提供当前 JR session 跟踪的 token usage |
| 上下文构建 | ✅ | 已支持 bounded workspace/project map、git、`CLAUDE.md`、README / 入口文件概览、持久 memory 与 compact；repo map 不跟随 symlink 且有硬预算 |
| Claude Code Agent Loop | ✅ | 已实现 agent_loop.py，支持工具调用循环 |
| `/resume` 会话恢复体验 | ✅ | 已实现最近保存会话选择器与 `/resume <session-id>` 直接恢复；保持当前 provider 路由 |
| `/compact` 对话压缩 | ✅ | 已提供 canonical `/compact-context`，并保留 `/compact` 兼容别名 |
| `/doctor` 健康诊断 | ✅ | 已实现本地只读 capability / trust / runtime health check；不自动修复，不调用 provider / network |
| Hook 系统 | 🚫 | Generic user-defined hook runtime intentionally disabled; security/lifecycle policy stays in dedicated fail-closed chokepoints |
| 权限系统 | ✅ | 已集成 fail-closed tool permission contract、敏感路径保护，以及 operator/project 权限策略；project policy 只能收紧 operator 已授予的权限 |

---

## 工具系统

> **重大进展**：当前仓库已实现完整的工具系统框架，包括 30+ 工具模块、Agent Loop、Schema 验证、权限框架等。

### 工具框架

| 能力 | 状态 | 当前情况 |
|------|------|----------|
| Tool Registry | ✅ | 已实现工具注册与发现机制 |
| Tool Protocol | ✅ | 已定义工具协议与基类 |
| Schema Validation | ✅ | 已实现参数校验系统 |
| Agent Loop | ✅ | 已实现完整的工具调用循环 |
| Tool Context | ✅ | 已实现工具上下文管理 |
| Permission Framework | ✅ | registered tools 强制显式 permission policy；已启用敏感路径保护与 operator/project policy loader，repo policy 只能减权 |
| Error Handling | ✅ | 已定义工具错误类型与处理 |
| Task Manager | ✅ | 已实现任务管理器 |

### 已实现工具模块

| 工具类别 | 工具名称 | 文件 | 状态 |
|---------|---------|------|------|
| 文件操作 | FileReadTool | `read.py` | ✅ 已实现 |
| 文件操作 | FileWriteTool | `write.py` | ✅ 已实现 |
| 文件操作 | FileEditTool | `edit.py` | ✅ 已实现 |
| 文件操作 | GlobTool | `glob.py` | ✅ 已实现 |
| 文件操作 | GrepTool | `grep.py` | ✅ 已实现 |
| 系统操作 | BashTool | `bash.py` | ✅ 已实现；安全策略明确禁止默认注册 |
| 网络工具 | WebFetchTool | `web_fetch.py` | ✅ 已实现 |
| 网络工具 | WebSearchTool | `web_search.py` | ✅ 已实现 |
| 交互工具 | AskUserQuestionTool | `ask_user_question.py` | ✅ 已实现 |
| 交互工具 | SendUserMessageTool | `send_user_message.py` | ✅ 已实现 |
| 任务管理 | TodoWriteTool | `todo_write.py` | ✅ 已实现 |
| 任务管理 | TaskStopTool | `task_stop.py` | ✅ 已实现 |
| 任务管理 | TasksV2Tool | `tasks_v2.py` | ✅ 已实现 |
| 任务管理 | TaskManager | `task_manager.py` | ✅ 已实现 |
| Agent 工具 | AgentTool | `agent.py` | ✅ Clawd-specific 本地顺序工具编排；不是隔离的 subagent runtime |
| Agent 工具 | BriefTool | `brief.py` | ✅ 已实现 |
| Agent 工具 | TeamTool | `team.py` | ✅ 已实现 |
| 配置工具 | ConfigTool | `config.py` | ✅ 类已实现；安全策略明确禁止默认注册 |
| 计划模式 | PlanModeTool | `plan_mode.py` | ✅ 已实现 |
| 定时任务 | CronTool | `cron.py` | ✅ 已实现 |
| MCP 工具 | MCPTool | `mcp.py` + `mcp_resource_runtime.py` | ✅ 本地 stdio tool execution 已启用；operator allowlist、固定 contract SHA-256、fresh `tools/list`、输入 schema 校验；写入/open-world 调用必须显式确认 |
| MCP 工具 | MCPResourcesTool | `mcp_resources.py` + `mcp_resource_runtime.py` | ✅ 只读资源 runtime 已实现；隔离 `mcp==2.2.0`、本地 stdio、operator manifest、先 list 后 read、默认注册 |
| 技能系统 | SkillTool | `skill.py` | ✅ 已实现 |
| 工具搜索 | ToolSearchTool | `tool_search.py` | ✅ 已实现 |
| LSP 集成 | LSPTool | `lsp.py` + `pyright_lsp.py` | ✅ 已实现；固定 Pyright 1.1.414，本地 stdio，workspace 限定，只读操作，默认注册 |
| Worktree | WorktreeTool | `worktree.py` | ✅ 真实 Git linked worktree；`clawd/<name>` 分支、`.git/clawd-worktrees/<name>`、显式确认进入、退出保留 worktree/branch |
| 杂项工具 | SleepTool | `sleep.py` | ✅ 已实现 |
| 杂项工具 | StructuredOutputTool | `structured_output.py` | ✅ 已实现 |
| 杂项工具 | MiscTools | `misc.py` | ✅ 已实现 |

---

## 服务与运行时

| 模块 | 状态 | 当前情况 |
|------|------|----------|
| Provider Runtime | ✅ | 已能完成基础聊天请求，Provider 层提供流式接口 |
| REPL Runtime | ✅ | 已支持基础交互、命令分流、消息记录 |
| Agent Loop Runtime | ✅ | 已实现完整的工具调用循环与结果处理 |
| Tool Execution Engine | ✅ | 已实现工具加载、执行、结果回填闭环 |
| Output Styles | ✅ | 已实现输出样式加载系统 |
| Session Persistence | ✅ | 已有会话保存/加载能力 |
| Context Engine | ✅ | Agent Loop 已接入 bounded project map + workspace/git/README/entry/`CLAUDE.md`/memory 上下文；map workspace-bound、names-only、预算受限 |
| Permission Engine | ✅ | ToolRegistry fail-closed 强制显式 permission policy；REPL 启动加载 trusted operator grants 与 repo-only restrictions，非法或扩权策略拒绝启动 |
| Compaction Engine | ✅ | 已支持手动 context compaction，并记录 compact task usage |
| Hook Runtime | 🚫 | Intentionally disabled; no settings-driven shell/HTTP/MCP/prompt/agent hook execution is exposed |
| MCP Runtime | ✅ | 本地 stdio resources + guarded tool execution 已启用；远程 HTTP、未授权/未固定 contract 的 tools 仍不开放 |

---

## 测试覆盖

| 测试类型 | 状态 | 文件 |
|---------|------|------|
| 工具系统测试 | ✅ | `test_tool_system_tools.py` (427 行) |
| Agent Loop 测试 | ✅ | `test_agent_loop.py` (134 行) |
| Claude Code 工具对等性测试 | ✅ | `test_claude_code_tool_parity.py` (137 行) |
| Provider 测试 | ✅ | `test_providers.py` (113 行) |
| 输出样式测试 | ✅ | `test_output_styles.py` (64 行) |
| 配置测试 | ✅ | `test_config.py` |


## 路线图

## Phase 0：可启动、可安装、可体验 ✅

目标：先保证项目对新用户和贡献者足够顺滑。

- [x] 解耦 CLI 启动路径，`--help`、`--version`、`config` 不应依赖 Provider SDK
- [x] Provider 改为延迟导入，缺少 SDK 时仍可浏览本地功能
- [x] 固定并验证 Python 3.11+ 开发环境
- [x] 完善安装说明与最小可运行示例
- [x] 清理 README 中与当前实现不一致的表述

## Phase 1：Claude Code 核心体验 MVP ✅

目标：先复现原生 Claude Code 最重要的第一层体验。

- [x] 统一聊天 REPL、slash commands、session store
- [x] 完成工具系统框架
- [x] 实现 Agent Loop
- [x] 统一错误处理、重试、重登流程
- [x] 完成 transcript 持久化与恢复基础设施
- [x] 整理一套稳定的用户命令集合

## Phase 2：真实工具调用闭环 ✅

目标：从"镜像工具清单"走向"真正可执行的 Python Agent"。

- [x] FileReadTool
- [x] FileWriteTool
- [x] FileEditTool
- [x] BashTool
- [x] AskUserQuestionTool
- [x] TodoWriteTool
- [x] WebFetchTool / WebSearchTool
- [x] 工具 schema、参数校验、异常处理、调用日志
- [x] 工具执行结果回填闭环

## Phase 3：上下文、权限、恢复能力 (已完成)

目标：补齐 Claude Code 的工程化能力。

- [x] 工作区上下文构建完善
- [x] git status / 文件树 / `CLAUDE.md` 注入基础版
- [x] README / 入口文件摘要注入
- [x] memory 与历史上下文管理
- [x] 权限系统完全集成
- [x] `/resume`
- [x] `/compact`
- [x] `/doctor`
- [x] Generic pre/post hook runtime intentionally excluded; tool permissions, sensitive paths, and instrumentation remain in dedicated chokepoints

## Phase 4：MCP、插件、扩展生态

目标：把项目从单体 CLI 升级为可扩展平台。

- [x] MCP 只读 resource client/runtime（本地 stdio、operator manifest、advertised-URI gate）
- [x] MCP 本地 stdio tool execution 与逐 server/tool 信任/权限契约（operator policy、contract pin、fresh discovery、explicit approval）
- [x] Python 插件系统（manifest-only discovery + exact-hash operator activation；discovery 不执行 Python）
- [x] 自定义 commands / tools（exact-hash active plugins register collision-checked commands and permission-enforced tools；generic executable hooks remain excluded）
- [x] 本地模型与第三方 provider 扩展（exact-hash trusted plugin providers；remote providers require credentials；credentialless providers are local-only loopback/localhost）
- [x] 更完善的 observability 与调试工具（sanitized ledgers + /doctor runtime snapshot + recent error diagnostics；observability failures do not break runtime）

## Phase 5：Python 版本的差异化亮点

目标：做出属于 Python 重构版的特色。

- [x] Notebook 友好工具链（Read + NotebookEdit 结构化 replace / insert / delete；read-before-write；不执行 notebook 代码）
- [x] 数据工程 / ETL 场景增强（DataInspect + DataTransform；CSV / TSV / JSON / JSONL；bounded schema/preview + exact-match filter/project/load；不覆盖现有输出）
- [x] 中国模型生态一等公民支持（DeepSeek / Qwen / GLM / MiniMax 内置 provider；独立默认模型 / endpoint / env key；CLI / REPL 一等选择）
- [x] pytest / ruff / mypy / uv 集成体验（Python 3.12 dev pin；uv sync --locked；uv run pytest / ruff / mypy；Ruff correctness baseline；Mypy gradual typed baseline）
- [x] 面向企业内自动化与工作流的扩展接口（exact-hash plugin WORKFLOWS；声明式 PromptCommand；显式非空 tool allowlist；复用现有权限/agent loop；无通用 hook 后台执行）

---

## Phase 0–5 完成后的真实待办

当前定义的 Phase 0–5 路线图已经完成，但这不等于所有可能的 Claude Code 能力都已实现。权威状态仍以 `src/capability_manifest.json` 和 `/doctor` 为准。

### P0：发布与持续验证

已落地 GitHub Actions CI：Python 3.10 / 3.11 / 3.12 矩阵运行 capability contract、完整 pytest、Ruff 与 Mypy；通过后使用 Python 3.12 构建 wheel/sdist 并执行 `twine check`。Workflow 只读 repository content、actions 使用 commit SHA 固定，不包含 secrets、publish 或 deployment 步骤。

- 在允许联网的干净环境中验证完整依赖安装和 wheel/sdist 安装矩阵；本次离线审计无法为全新 venv 下载未缓存的 `tiktoken`。
- 在未来 setuptools 强制截止日期前迁移弃用的 license metadata；当前构建可通过，但会产生 deprecation warning。
- 持续维护 CHANGELOG、release notes、安装文档与实际测试数量/能力状态。

### P1：明确 deferred / intentionally disabled 的能力

- `subagent_runtime`：**DEFERRED_NOT_PRODUCTION_READY**。只有在具备隔离 child LLM loop、独立 context/session/worker、权限与取消边界后才应提升状态。
- Generic `hook_runtime`：**INTENTIONALLY_DISABLED**，不是遗漏项。仅在出现具体、批准的 use case，并能保持 fail-closed 安全边界时重新评估。
- `Bash` / model-driven `Config`：默认 hardened registry 中继续 intentionally disabled；不要为了“功能数量”自动开启。

### P2：持续增强方向

- Provider 兼容性、模型目录和流式行为增强。
- 性能基准、内存/上下文成本测量与回归监控。
- MCP / plugin / workflow 示例、文档与兼容性测试。
- 更多测试覆盖、错误处理和安全回归案例。

---

## 建议的 PR 认领模块

| 方向 | 适合贡献内容 |
|------|--------------|
| CLI / UX | 命令设计、帮助信息、交互体验、错误提示 |
| Tools | 工具增强、新工具开发、工具测试 |
| Context | repo map、git status、项目文档注入、memory |
| Permissions | 权限集成、安全策略、命令限制 |
| Providers | 新 provider、模型选型、流式兼容 |
| MCP / Plugins | MCP / plugin / workflow 文档、兼容性测试、新扩展示例 |
| Quality | 测试、基准、文档、安装流程、CI |
| Performance | 性能优化、内存管理、并发处理 |

---

## PR 提交建议

- 优先做真实能力，不优先堆命令名录
- 每个 PR 尽量聚焦单一模块
- 新功能请附带最小测试或运行示例
- 修改 README 时请同步更新能力状态
- 对"已完成"表述保持谨慎，优先写成可验证结果

---

## 推荐的对外表述

可以这样介绍项目：

> Claude Code Python 是一个基于真实 Claude Code 源码结构的 Python 重构版。当前已经具备多 Provider 聊天 CLI、完整工具系统框架（30+ 工具）与 Agent Loop，已实现工具调用闭环、bounded 项目上下文、权限策略、会话恢复、压缩、MCP 与插件体系。欢迎围绕工具增强、runtime、permissions、context 与 Python 原生扩展能力提交 PR。

---

## 一句话总结

**当前我们已经有一个具备完整工具系统框架、Agent Loop、bounded 项目上下文、权限策略、认证恢复与会话恢复能力的 Python Agent Runtime；Phase 0–5 的当前路线图条目均已有实现与验证证据，但 capability manifest 仍明确保留 `subagent_runtime` 为 deferred、`hook_runtime` 为 intentionally disabled。**
