# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Bounded project-context map with source/test representation, workspace confinement, symlink avoidance, and sensitive-looking filename filtering
- Persistent memory, context compaction, session resume, and local `/doctor` capability/trust diagnostics
- Fail-closed operator/project permission policy and sensitive-path protections
- Exact-hash skill/plugin trust, MCP/resource runtime, provider/workflow extensions, and sanitized runtime observability
- Authentication recovery with explicit in-session provider reconfiguration, runtime/session/context rebinding, and no automatic retry of rejected requests
- Python-native data/notebook tooling and locked pytest / Ruff / Mypy / uv developer quality gates
- GitHub Actions CI across Python 3.10 / 3.11 / 3.12 with capability, full-test, Ruff, Mypy, package-build, and `twine check` gates; workflow permissions are read-only and external actions are commit-SHA pinned

### Changed
- Current scoped Phase 0–5 roadmap is complete; `subagent_runtime` remains explicitly deferred and generic `hook_runtime` intentionally disabled
- Capability status documentation now renders feature states as well as tool states
- Package description no longer claims a complete drop-in replacement experience
- Source-distribution manifest no longer references nonexistent `CLAUDE.md` or `MVP_PLAN.md`
- Skill frontmatter parsing supports inline list syntax such as `arguments: [path]`
- README and contributor docs prefer `uv`-based setup instructions
- Direct/provider streaming and agent-loop behavior are documented according to the current runtime

### Security
- Authentication failures are classified without vendor-SDK coupling and user-facing recovery output does not echo provider exception text
- Unanswered authentication-rejected turns are removed from conversation state; turns with visible/assistant/tool activity are preserved for review

## [0.1.0] - 2026-04-01

### Added

#### Core Features
- Multi-provider support for Anthropic, OpenAI, and GLM (Zhipu AI)
- Interactive REPL with prompt-toolkit integration
- Rich interactive terminal output
- Session persistence and management
- Configuration management with basic API key obfuscation

#### CLI Commands
- `clawd` - Start the interactive REPL
- `clawd login` - Interactive API key configuration
- `clawd config` - View current configuration
- `clawd --version` - Show version information

#### Provider Implementations
- **Anthropic Provider**: Claude integration with chat + streaming interfaces
- **OpenAI Provider**: GPT integration with chat + streaming interfaces
- **GLM Provider**: GLM integration with chat + streaming interfaces

#### REPL Features
- Command history with persistent storage
- Auto-suggestions from history
- Slash commands: `/help`, `/exit`, `/clear`, `/save`, `/load`, `/multiline`
- Skill slash commands backed by `SKILL.md`
- Syntax highlighting with Rich library
- Tab completion and multi-line input support

#### Configuration System
- JSON-based configuration storage
- Base64-encoded API keys for basic obfuscation
- Provider-specific settings (API key, base URL, default model)
- Session auto-save option

#### Session Management
- Unique session ID generation
- Conversation history tracking
- Session save/load functionality
- Conversation clear operation

#### Code Quality
- Type hints for all public functions
- Abstract base class for provider implementations
- Data classes for structured data (ChatMessage, ChatResponse)
- Error handling and validation

#### Testing
- Unit tests for core components
- Integration tests for providers
- End-to-end tests for REPL functionality
- Test coverage for configuration management

### Technical Details

#### Architecture
- Modular provider system with base abstraction
- Conversation management with message history
- Configuration management layer
- REPL engine with prompt-toolkit

#### Dependencies
- `anthropic>=0.18.0` - Anthropic SDK
- `openai>=1.0.0` - OpenAI SDK
- `zhipuai>=2.0.0` - Zhipu AI SDK
- `prompt-toolkit>=3.0.0` - Interactive REPL
- `rich>=13.0.0` - Terminal formatting
- `python-dotenv>=1.0.0` - Environment variables

#### File Structure
```
src/
├── providers/          # LLM provider implementations
│   ├── base.py        # Abstract base class
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   └── glm_provider.py
├── repl/              # Interactive REPL
│   └── core.py
├── agent/             # Session management
│   ├── session.py
│   └── conversation.py
├── config.py          # Configuration management
└── cli.py             # CLI commands
```

### Known Limitations

- Context building is still in early MVP form and needs deeper project summarization
- Permission enforcement exists as a framework but is not fully integrated everywhere
- `/resume`, `/compact`, and `/doctor` are not implemented yet
- The current CLI uses turn-based output even though providers expose streaming interfaces

### Migration Notes

This is the initial MVP release. No migration needed.

### Future Roadmap

- [ ] Context enrichment and project-memory improvements
- [ ] Full permission integration
- [ ] `/resume`, `/compact`, `/doctor`
- [ ] Token usage and cost tracking
- [ ] MCP and plugin-system enhancements

---

## Release Notes

### v0.1.0 - MVP Release

This is the first public release of Clawd Codex, a complete reimplementation of Claude Code. This MVP includes:

- Full multi-provider support
- Interactive REPL
- Session management
- Configuration system
- Tool system and agent loop foundations
- Type-safe implementation

The focus was on building a solid foundation with clean architecture, comprehensive testing, and good developer experience. All core features are working and tested.

**Special Thanks**: This project is inspired by Claude Code and aims to provide an open-source alternative for learning and experimentation.

---

[0.1.0]: https://github.com/GPT-AGI/Clawd-Code/releases/tag/v0.1.0
