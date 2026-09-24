"""Interactive REPL for Clawd Codex."""

from __future__ import annotations

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    from prompt_toolkit.completion import WordCompleter
    try:
        from prompt_toolkit.completion import FuzzyCompleter
    except Exception:  # pragma: no cover
        FuzzyCompleter = None  # type: ignore
    from prompt_toolkit.key_binding import KeyBindings
except ModuleNotFoundError:  # pragma: no cover
    class FileHistory:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class AutoSuggestFromHistory:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class Style:  # type: ignore
        @staticmethod
        def from_dict(*args, **kwargs):
            return None

    class WordCompleter:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass
    FuzzyCompleter = None  # type: ignore

    class KeyBindings:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

    class PromptSession:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

        def prompt(self, *args, **kwargs):
            raise EOFError()

try:
    from rich.console import Console, Group
    from rich.align import Align
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.markdown import Markdown
    from rich.columns import Columns
except ModuleNotFoundError:  # pragma: no cover
    class Console:  # type: ignore
        def print(self, *args, **kwargs):
            return None

    Group = None  # type: ignore
    Align = None  # type: ignore
    Panel = None  # type: ignore
    Table = None  # type: ignore
    Text = None  # type: ignore
    Columns = None  # type: ignore

    class Markdown:  # type: ignore
        def __init__(self, text: str):
            self.text = text
from pathlib import Path
import asyncio
import sys
import json
from typing import Any

from src.agent import Session
from src.config import get_provider_config
from src.outputStyles import resolve_output_style
from src.providers import (
    get_provider_class,
    get_provider_info,
    validate_provider_runtime_config,
)
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import ChatMessage
from src.providers.minimax_provider import MinimaxProvider
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.mcp_resource_runtime import MCPResourceConfigError, load_mcp_resource_clients
from src.tool_system.permission_policy import PermissionPolicyConfigError, load_permission_context
from src.tool_system.protocol import ToolCall
from src.tool_system.agent_loop import (
    ToolEvent,
    build_agent_preflight,
    run_agent_loop,
    summarize_tool_result,
    summarize_tool_use,
)

# New command system imports
from src.command_system import (
    CommandRegistry,
    CommandResult,
    PromptCommand,
    create_command_context,
    execute_command_async,
    execute_command_sync,
    get_command_registry,
    register_builtin_commands,
)
from src.plugins.extensions import (
    clear_registered_plugin_commands,
    load_active_plugin_extensions,
    register_plugin_extensions,
    register_plugin_provider_extensions,
)
from src.cost_tracker import CostTracker
from src.history import HistoryLog
from src.usage_ledger import append_provider_usage


_AUTH_ERROR_MESSAGE_MARKERS = (
    "401",
    "authentication error",
    "authentication failed",
    "invalid authentication",
    "unauthorized",
    "unauthenticated",
    "invalid api key",
    "incorrect api key",
    "api key is invalid",
    "api key invalid",
    "expired api key",
    "api key has expired",
    "invalid access token",
    "invalid token",
    "认证失败",
    "身份验证失败",
    "令牌无效",
)


def _is_provider_authentication_error(error: BaseException) -> bool:
    """Classify provider authentication failures without vendor-SDK coupling."""
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))

        class_name = type(current).__name__.lower()
        if "authenticationerror" in class_name or "unauthorizederror" in class_name:
            return True

        status_code = getattr(current, "status_code", None)
        response = getattr(current, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        try:
            if int(status_code) == 401:
                return True
        except (TypeError, ValueError):
            pass

        message = str(current).lower()
        if any(marker in message for marker in _AUTH_ERROR_MESSAGE_MARKERS):
            return True

        current = current.__cause__ or current.__context__
    return False


class ClawdREPL:
    """Interactive REPL for Clawd Codex."""

    def __init__(self, provider_name: str = "glm", stream: bool = False):
        self.console = Console()
        self.provider_name = provider_name
        self.stream = stream
        self.multiline_mode = False

        # Load exact-hash active plugin providers before provider resolution.
        # Reuse this same imported result later for command/tool registration.
        self._plugin_extension_load_result = load_active_plugin_extensions()
        provider_registration_issues = register_plugin_provider_extensions(
            self._plugin_extension_load_result
        )
        self._plugin_provider_issues = [
            *self._plugin_extension_load_result.issues,
            *provider_registration_issues,
        ]

        # Load configuration
        try:
            info = get_provider_info(provider_name)
            config = get_provider_config(provider_name)
        except ValueError as exc:
            self.console.print(f"[red]Error: {exc}[/red]")
            if self._plugin_provider_issues:
                self.console.print(
                    "[yellow]A trusted plugin extension is unavailable; "
                    "start with a built-in provider and run /doctor for details.[/yellow]"
                )
            sys.exit(1)
        if info.get("requires_api_key", True) and not config.get("api_key"):
            self.console.print("[red]Error: API key not configured.[/red]")
            self.console.print("Run [bold]clawd login[/bold] to configure.")
            sys.exit(1)
        try:
            validate_provider_runtime_config(provider_name, config)
        except ValueError as exc:
            self.console.print(f"[red]Error: {exc}[/red]")
            sys.exit(1)

        # Initialize provider
        provider_class = get_provider_class(provider_name)
        self.provider = provider_class(
            api_key=config["api_key"],
            base_url=config.get("base_url"),
            model=config.get("default_model")
        )

        # Create session
        self.session = Session.create(
            provider_name,
            self.provider.model
        )

        self.tool_registry = build_default_registry()
        workspace_root = Path.cwd()
        try:
            permission_context = load_permission_context(workspace_root)
        except PermissionPolicyConfigError as exc:
            self.console.print(f"[red]Error: invalid permission policy: {exc}[/red]")
            sys.exit(1)
        self.tool_context = ToolContext(
            workspace_root=workspace_root,
            permission_context=permission_context,
            instrumentation_enabled=True,
        )
        try:
            self.tool_context.mcp_clients = load_mcp_resource_clients()
        except MCPResourceConfigError as exc:
            self.tool_context.mcp_config_error = str(exc)
        self.tool_context.ask_user = self._ask_user_questions
        # Permission handler with status control for proper input handling
        self._current_status = None
        self.tool_context.permission_handler = self._handle_permission_request

        # User-facing commands: clear names, grouped by purpose.
        self._visible_command_groups = [
            ("General", [
                ("/help", "Show command help"),
                ("/exit", "Exit JR"),
            ]),
            ("Conversation & Sessions", [
                ("/clear-chat", "Clear the current conversation"),
                ("/save-session", "Save the current session"),
                ("/load-session", "Load a saved session by ID"),
                ("/resume", "Choose and resume a recent saved session"),
                ("/compact-context", "Compact the conversation to save context space"),
                ("/context-usage", "Show context-window usage and breakdown"),
            ]),
            ("Input & Display", [
                ("/multiline-input", "Toggle multiline input mode"),
                ("/stream-responses", "Control live response streaming"),
                ("/render-last-response", "Re-render the last assistant response"),
            ]),
            ("Usage", [
                ("/usage", "Show API/model usage plus skill and tool activity"),
                ("/session-usage", "Show tokens JR tracked in this session"),
            ]),
            ("Tools & Project", [
                ("/list-tools", "List available built-in tools"),
                ("/run-tool", "Run a tool directly"),
                ("/list-skills", "List available skills"),
                ("/setup-project", "Set up CLAUDE.md and optional skills"),
                ("/doctor", "Run local health and capability diagnostics"),
            ]),
        ]
        self._original_built_ins = [
            name
            for _, commands in self._visible_command_groups
            for name, _ in commands
        ]
        # Visible canonical names route to their established implementation command names.
        # Compatibility aliases remain registered separately and stay out of the palette.
        self._canonical_command_routes = {
            "save-session": "save",
            "load-session": "load",
            "multiline-input": "multiline",
            "stream-responses": "stream",
            "render-last-response": "render-last",
            "list-tools": "tools",
            "run-tool": "tool",
            "context-usage": "context",
            "compact-context": "compact",
            "setup-project": "init",
        }
        self._legacy_hidden_commands = {
            "save", "load", "multiline", "stream", "render-last", "tools", "tool"
        }
        self._built_in_commands = list(self._original_built_ins)

        # Initialize new command system
        self._init_command_system()
        self._init_plugin_extensions()
        self._update_built_in_commands_with_command_system()

        # Prompt toolkit with tab completion
        history_file = Path.home() / ".clawd" / "history"
        history_file.parent.mkdir(parents=True, exist_ok=True)

        self.completer = WordCompleter(self._get_slash_command_words(), ignore_case=True)

        # Key bindings for multiline
        self.bindings = KeyBindings()
        if hasattr(self.bindings, "add"):
            @self.bindings.add("/")  # type: ignore[attr-defined]
            def _show_slash_completions(event):  # type: ignore[no-untyped-def]
                buf = event.current_buffer
                if buf.text == "":
                    buf.insert_text("/")
                    buf.start_completion(select_first=False)

        self.prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=self.completer,
            style=Style.from_dict({
                'prompt': 'bold blue',
            }),
            key_bindings=self.bindings,
            complete_while_typing=True,
        )

    def _ask_user_questions(self, questions: list[dict]) -> dict[str, str]:
        # Stop the Rich status spinner if running, so we can get clean input
        if self._current_status is not None:
            try:
                self._current_status.stop()
            except Exception:
                pass

        answers: dict[str, str] = {}
        for q in questions:
            question_text = str(q.get("question", "")).strip()
            options = q.get("options") or []
            multi = bool(q.get("multiSelect", False))
            if not question_text or not isinstance(options, list) or len(options) < 2:
                continue

            self.console.print(f"\n[bold]{question_text}[/bold]")
            labels: list[str] = []
            for i, opt in enumerate(options, start=1):
                label = str((opt or {}).get("label", "")).strip()
                desc = str((opt or {}).get("description", "")).strip()
                labels.append(label)
                self.console.print(f"  {i}. {label}  [dim]{desc}[/dim]")
            other_idx = len(labels) + 1
            self.console.print(f"  {other_idx}. Other  [dim]Provide custom text[/dim]")

            prompt = "Select (comma-separated) > " if multi else "Select > "
            raw = input(prompt).strip()
            if not raw:
                choice_str = "1"
            else:
                choice_str = raw

            selected: list[str] = []
            parts = [p.strip() for p in choice_str.split(",") if p.strip()]
            if not parts:
                parts = ["1"]
            for part in parts:
                try:
                    idx = int(part)
                except ValueError:
                    idx = -1
                if idx == other_idx:
                    free = input("Other > ").strip()
                    if free:
                        selected.append(free)
                    continue
                if 1 <= idx <= len(labels):
                    selected.append(labels[idx - 1])
            if not selected:
                selected = [labels[0]]
            answers[question_text] = ", ".join(selected) if multi else selected[0]

        # Restart spinner after getting answers
        if self._current_status is not None:
            try:
                self._current_status.start()
            except Exception:
                pass

        return answers

    def _handle_permission_request(
        self,
        tool_name: str,
        message: str,
        suggestion: str | None,
    ) -> tuple[bool, bool]:
        """Handle interactive permission requests from tools.

        Args:
            tool_name: Name of the tool requesting permission.
            message: Message explaining what permission is needed.
            suggestion: Optional suggestion for enabling the setting.

        Returns:
            Tuple of (allowed: bool, continue_without_caching: bool).
            continue_without_caching is always False since we don't cache in REPL.
        """
        # Stop the Rich status spinner if running, so we can get clean input
        if self._current_status is not None:
            try:
                self._current_status.stop()
            except Exception:
                pass

        self.console.print("")
        self.console.print("[bold yellow]⚠ Permission Required[/bold yellow]")
        self.console.print(f"  {message}")
        self.console.print("")

        # Determine if this is a setting that can be enabled
        can_enable_setting = False
        setting_to_enable: str | None = None

        msg_lower = message.lower()
        if "allow_docs" in msg_lower or "documentation files" in msg_lower:
            pc = self.tool_context.permission_context
            if (
                hasattr(pc, "allow_docs")
                and not pc.allow_docs
                and not getattr(pc, "allow_docs_locked_off", False)
            ):
                can_enable_setting = True
                setting_to_enable = "allow_docs"

        # Build options
        options: list[tuple[str, str]] = [
            ("y", "Yes, allow this action"),
            ("n", "No, deny this action"),
        ]
        if can_enable_setting:
            options.insert(0, ("e", f"Enable {setting_to_enable} and allow"))

        self.console.print("[bold]Options:[/bold]")
        for i, (key, desc) in enumerate(options, start=1):
            self.console.print(f"  {i}. [{key}] {desc}")
        self.console.print("")

        # Get input - use standard input() which works after stopping status
        choice = input("Select option> ").strip().lower()

        # Every ask needs an explicit choice; numbers map to the options displayed above.
        displayed = {str(i): key for i, (key, _desc) in enumerate(options, start=1)}
        selected = displayed.get(choice, choice)

        if can_enable_setting and selected in ("e", "enable"):
            self._enable_permission_setting(setting_to_enable)
            return True, False
        if selected in ("y", "yes"):
            return True, False
        if selected in ("n", "no"):
            return False, False

        if not choice:
            self.console.print("[dim]No choice entered — denied.[/dim]")
        else:
            self.console.print("[dim]Invalid choice — denied.[/dim]")
        return False, False

    def _enable_permission_setting(self, setting_name: str | None) -> None:
        """Enable a permission setting in the tool context."""
        if not setting_name:
            return

        self.console.print(f"\n[dim]Enabling {setting_name}...[/dim]")

        if setting_name == "allow_docs":
            pc = self.tool_context.permission_context
            if getattr(pc, "allow_docs_locked_off", False):
                return
            if hasattr(pc, "allow_docs"):
                pc.allow_docs = True
                self.console.print(f"[green]✓ {setting_name} enabled for this session[/green]")
                return

        self.console.print(f"[dim]Could not enable {setting_name}.[/dim]")

    def _init_command_system(self):
        """Initialize the new command system."""
        # Also register to global registry so execute_command_async can find commands
        register_builtin_commands(None)  # None = use global registry

        # Create command registry and register built-ins
        self.command_registry = CommandRegistry()
        register_builtin_commands(self.command_registry)

        # Create cost tracker and history
        self.cost_tracker = CostTracker()
        self.history_log = HistoryLog()

        # Create command context
        self.command_context = create_command_context(
            workspace_root=Path.cwd(),
            conversation=self.session.conversation,
            cost_tracker=self.cost_tracker,
            history=self.history_log,
            config={
                "context_provider": self.provider,
                "context_tool_registry": self.tool_registry,
                "context_tool_context": self.tool_context,
            },
            permission_handler=self._handle_permission_request,
        )

        # Merge new commands with built-in list for completion
        self._update_built_in_commands_with_command_system()

    def _init_plugin_extensions(self) -> None:
        """Load exact-hash active plugin commands/tools into existing registries."""
        global_registry = get_command_registry()
        clear_registered_plugin_commands(global_registry)

        load_result = getattr(self, "_plugin_extension_load_result", None)
        if load_result is None:
            load_result = load_active_plugin_extensions()
            self._plugin_extension_load_result = load_result
        provider_issues = getattr(self, "_plugin_provider_issues", [])
        issues = register_plugin_extensions(
            load_result,
            tool_registry=self.tool_registry,
            command_registries=(global_registry, self.command_registry),
        )
        combined = [*provider_issues, *issues]
        deduped: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for issue in combined:
            key = (str(issue.get("code") or ""), str(issue.get("subject") or ""))
            if key not in seen:
                deduped.append(issue)
                seen.add(key)
        self.plugin_extension_issues = deduped
        self.command_context.config["plugin_runtime_issues"] = list(deduped)

    def _update_built_in_commands_with_command_system(self):
        """Update the built-in commands list with commands from the new system."""
        # Start with original built-ins
        self._built_in_commands = list(self._original_built_ins)

        # Add canonical commands only. Compatibility aliases still execute but do not clutter completion.
        try:
            for cmd in self.command_registry.list_commands():
                cmd_name = f"/{cmd.name}"
                if cmd_name not in self._built_in_commands:
                    self._built_in_commands.append(cmd_name)
        except Exception:
            pass

    def _try_execute_new_command(self, command: str, args: str) -> tuple[bool, str | None]:
        """Try to execute a command using the new command system (sync path for LocalCommand only).

        Returns:
            Tuple of (handled: bool, result_text: str | None)
        """
        try:
            success, result_text, error = execute_command_sync(
                command, args, self.command_context
            )
            if success:
                return True, result_text
            else:
                return False, error
        except Exception as e:
            return False, str(e)

    async def _try_execute_command_async(self, command: str, args: str) -> CommandResult:
        """Execute a command asynchronously, supporting both LocalCommand and PromptCommand.

        Returns:
            CommandResult with the execution result
        """
        try:
            return await execute_command_async(command, args, self.command_context)
        except Exception as e:
            return CommandResult.error(command, str(e))

    def _handle_command_result(self, result: CommandResult) -> bool:
        """Handle the result of a command execution.

        Returns True if the command was handled, False otherwise.
        """
        if not result.success:
            if result.error:
                self.console.print(f"[red]{result.error}[/red]")
            return True

        if result.result_type == "text":
            if result.text:
                self.console.print("\n" + result.text)
                self.console.print()
            return True

        elif result.result_type == "prompt":
            # For PromptCommand, extract the text content and send to LLM
            prompt_text = ""
            for item in result.prompt_content:
                if item.get("type") == "text":
                    prompt_text = item.get("text", "")
                    break

            if prompt_text:
                command = self.command_registry.get(result.command_name)
                previous_tool_allowlist = self.tool_context.tool_allowlist
                if isinstance(command, PromptCommand) and command.allowed_tools:
                    self.tool_context.restrict_tool_allowlist(command.allowed_tools)
                try:
                    if result.command_name == "init":
                        self.console.print("[dim]Initializing workspace setup...[/dim]")
                    # Send the prompt to the LLM for interactive execution.
                    self.chat(prompt_text, max_turns=100)
                finally:
                    self.tool_context.tool_allowlist = previous_tool_allowlist
            return True

        elif result.result_type == "skip":
            # Command handled silently
            return True

        return False

    def _get_slash_command_words(self) -> list[str]:
        words = list(self._built_in_commands)
        try:
            from src.skills.loader import get_all_skills

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            skills = sorted(get_all_skills(project_root=cwd), key=lambda s: s.name.lower())
            for s in skills:
                words.append(f"/{s.name}")
        except Exception:
            pass
        deduped: list[str] = []
        seen: set[str] = set()
        for w in words:
            lw = w.lower()
            if lw in seen:
                continue
            seen.add(lw)
            deduped.append(w)
        return deduped

    def _refresh_completer(self) -> None:
        try:
            words = self._get_slash_command_words()
            try:
                base = WordCompleter(words, ignore_case=True, match_middle=True)
            except TypeError:
                base = WordCompleter(words, ignore_case=True)
            self.completer = FuzzyCompleter(base) if FuzzyCompleter is not None else base
            if hasattr(self, "prompt_session") and getattr(self.prompt_session, "completer", None) is not None:
                self.prompt_session.completer = self.completer
        except Exception:
            return

    def _show_slash_palette(self, query: str | None = None) -> None:
        q = (query or "").strip().lower()
        self.console.print("\n[bold]Available commands and skills:[/bold]")

        visible_names: set[str] = set()
        for group_name, commands in self._visible_command_groups:
            matches = [
                (name, desc)
                for name, desc in commands
                if not q or q in name.lower() or q in desc.lower()
            ]
            if not matches:
                continue
            self.console.print(f"\n[cyan]{group_name}[/cyan]")
            for name, desc in matches:
                visible_names.add(name.lower())
                self.console.print(f"  {name}  [dim]- {desc}[/dim]")

        # Preserve any future canonical command-system entries without mixing in aliases.
        other_commands: list[tuple[str, str]] = []
        try:
            for cmd in self.command_registry.list_commands():
                name = f"/{cmd.name}"
                if name.lower() in visible_names:
                    continue
                desc = (cmd.description or "").strip()
                if q and q not in name.lower() and q not in desc.lower():
                    continue
                other_commands.append((name, desc))
        except Exception:
            pass

        if other_commands:
            self.console.print("\n[cyan]Other Commands[/cyan]")
            for name, desc in sorted(other_commands, key=lambda item: item[0].lower()):
                self.console.print(f"  {name}  [dim]- {desc}[/dim]")

        if self.tool_registry.get("Skill") is not None:
            try:
                from src.skills.loader import get_all_skills

                cwd = self.tool_context.cwd or self.tool_context.workspace_root
                skills = list(get_all_skills(project_root=cwd))
                skills.sort(key=lambda s: s.name.lower())
                skill_matches = []
                for skill in skills:
                    name = f"/{skill.name}"
                    desc = (skill.description or "").strip()
                    if q and q not in name.lower() and q not in desc.lower():
                        continue
                    skill_matches.append((name, desc))

                if skill_matches:
                    self.console.print("\n[magenta]Skills[/magenta]")
                    for name, desc in skill_matches:
                        self.console.print(f"  [magenta]{name}[/magenta]")
                        if desc:
                            self.console.print(f"    [dim]{desc}[/dim]")
            except Exception:
                pass

        self.console.print()

    def _shorten_path_text(self, text: str) -> str:
        root = str(self.tool_context.workspace_root)
        cwd = str(self.tool_context.cwd or self.tool_context.workspace_root)
        for base in (cwd, root):
            prefix = base.rstrip("/") + "/"
            if text.startswith(prefix):
                return "./" + text[len(prefix):]
            text = text.replace(prefix, "")
        return text

    def _display_cwd(self) -> str:
        cwd = str(Path.cwd())
        home = str(Path.home())
        if cwd.startswith(home):
            return cwd.replace(home, "~", 1)
        return cwd

    def _truncate_middle(self, text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        if limit <= 3:
            return text[:limit]
        head = max(1, (limit - 1) // 2)
        tail = max(1, limit - head - 1)
        return f"{text[:head]}…{text[-tail:]}"

    def _print_startup_header(self):
        from src import __version__

        display_path = self._display_cwd()
        provider_label = f"{self.provider_name.upper()} Provider"
        model_label = self.provider.model or "Unknown model"

        mascot_ascii = "\n".join([
            "  /\\__/\\",
            " / o  o \\",
            "(  __  )",
            " \\/__/  ",
        ])

        if Panel is None or Group is None or Align is None or Table is None or Text is None or Columns is None:
            print(mascot_ascii)
            print(f"Clawd Codex v{__version__}")
            print(f"{model_label} · {provider_label}")
            print(f"{display_path}\n")
            return

        width = getattr(self.console, "width", 80)
        content_width = max(28, min(width - 12, 72))
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bright_black", justify="right", no_wrap=True)
        table.add_column(style="white", ratio=1)
        table.add_row("Version", Text.assemble(("Clawd Codex", "bold white"), ("  ", ""), (f"v{__version__}", "bold cyan")))
        table.add_row("Model", Text(model_label, style="bold magenta"))
        table.add_row("Provider", Text(provider_label, style="bold green"))
        table.add_row("Workspace", Text(self._truncate_middle(display_path, content_width - 12), style="bold blue"))

        footer = Text("/help  •  /list-tools  •  /list-skills  •  /clear-chat  •  /save-session /resume  •  /exit", style="dim")
        mascot_block = Text(mascot_ascii, style="bold orange3", no_wrap=True)
        body = Group(
            Columns([mascot_block, table], align="center", expand=False),
            Text(""),
            Align.center(footer),
        )
        header = Panel(
            body,
            border_style="bright_black",
            title="[bold bright_cyan] CLAWD CODE [/bold bright_cyan]",
            padding=(1, 2),
        )
        self.console.print(header)
        self.console.print()

    def run(self):
        """Run the REPL."""
        self._print_startup_header()

        while True:
            try:
                self._refresh_completer()
                # Dynamic prompt based on multiline mode
                # Using '❯' for a modern feel
                prompt_text = '... ' if self.multiline_mode else '❯ '
                user_input = self.prompt_session.prompt(
                    prompt_text,
                    multiline=self.multiline_mode
                )

                if not user_input.strip():
                    self.multiline_mode = False
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    self.handle_command(user_input)
                    continue

                # Send to LLM
                self.chat(user_input)
                self.multiline_mode = False

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Interrupted. Type /exit to quit.[/yellow]")
                self.multiline_mode = False
                continue
            except EOFError:
                self.console.print("\n[blue]Goodbye![/blue]")
                break

    def handle_command(self, command: str):
        """Handle slash commands."""
        raw = command.strip()
        if raw == "/":
            self._show_slash_palette()
            return
        if raw.startswith("/") and " " not in raw:
            raw_name = raw[1:].lower()
            visible = raw.lower() in (c.lower() for c in self._built_in_commands)
            registered = self.command_registry.has(raw_name)
            legacy = raw_name in self._legacy_hidden_commands
            if not (visible or registered or legacy):
                if raw_name:
                    self._show_slash_palette(query=raw_name)
                    return

        # First, try the new command system
        if raw.startswith("/"):
            parts = raw[1:].split(maxsplit=1)
            typed_cmd_name = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            cmd_name = self._canonical_command_routes.get(typed_cmd_name, typed_cmd_name)
            if cmd_name != typed_cmd_name:
                raw = f"/{cmd_name}" + (f" {args}" if args else "")

            # Check if this command exists in the new command system
            # but skip the ones we handle specially
            # Note: /context, /compact, /skill need special handling, don't route through new system
            # /init is handled via new command system (PromptCommand) so it's NOT in special_commands
            special_commands = {
                'exit', 'quit', 'q',
                'help', 'tools', 'tool',
                'save', 'load', 'resume', 'multiline', 'stream', 'render-last',
                'skill',
                'context', 'compact',  # These need special handling
                ''
            }

            # Handle /init through the new command system (PromptCommand path)
            if cmd_name == 'init':
                # Use async path for PromptCommand
                try:
                    # Run async command execution in a new event loop
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._try_execute_command_async(cmd_name, args)
                        )
                        result = future.result()

                    if result.success:
                        self._handle_command_result(result)
                    elif result.error:
                        self.console.print(f"[red]{result.error}[/red]")
                except Exception as e:
                    self.console.print(f"[red]Error executing /setup-project: {e}[/red]")
                return

            if cmd_name not in special_commands:
                # Try to execute via new command system
                # First try sync path for LocalCommand (faster)
                try:
                    handled, result_text = self._try_execute_new_command(cmd_name, args)
                    if handled:
                        if result_text:
                            self.console.print("\n" + result_text)
                        self.console.print()
                        return
                except Exception as e:
                    # Fall through to async path
                    pass

                # Use async path for PromptCommand
                # Run in a new event loop since we're in a sync context
                try:
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run,
                            self._try_execute_command_async(cmd_name, args)
                        )
                        result = future.result()

                    if result.success:
                        if self._handle_command_result(result):
                            return
                except Exception:
                    pass

        # Fall back to original command handling
        cmd = raw.lower()

        if cmd in ['/exit', '/quit', '/q']:
            self.console.print("[blue]Goodbye![/blue]")
            sys.exit(0)

        elif cmd == '/help':
            self.show_help()

        elif cmd == '/tools':
            names = [spec.name for spec in self.tool_registry.list_specs()]
            names.sort(key=str.lower)
            self.console.print("\n[bold]Available tools:[/bold]")
            for name in names:
                self.console.print(f"  - {name}")
            self.console.print()

        elif cmd.startswith('/tool'):
            parts = command.strip().split(maxsplit=2)
            if len(parts) < 2:
                self.console.print("[red]Usage: /run-tool <name> <json-input>[/red]")
                return
            name = parts[1]
            payload = {}
            if len(parts) == 3:
                try:
                    payload = json.loads(parts[2])
                except json.JSONDecodeError as e:
                    self.console.print(f"[red]Invalid JSON input: {e}[/red]")
                    return
            self.tool_context.usage_records.clear()
            try:
                result = self.tool_registry.dispatch(ToolCall(name=name, input=payload), self.tool_context)
            except Exception as e:
                self.console.print(f"[red]Tool error: {e}[/red]")
                return
            self.console.print("\n[bold]Tool result:[/bold]")
            self.console.print(json.dumps(result.output, indent=2, ensure_ascii=False))
            self.console.print()
            if self.tool_context.usage_records:
                self._record_and_print_task_usage({})
                self.console.print()

        elif cmd == '/clear':
            # Try new command system first, fall back to original
            try:
                handled, result_text = self._try_execute_new_command('clear', '')
                if handled and result_text:
                    self.console.print("\n[green]" + result_text + "[/green]")
                    return
            except Exception:
                pass
            # Original implementation
            self.session.conversation.clear()
            self.console.print("[green]Conversation cleared.[/green]")

        elif cmd == '/save':
            self.save_session()

        elif cmd == '/multiline':
            self.multiline_mode = not self.multiline_mode
            status = "enabled" if self.multiline_mode else "disabled"
            self.console.print(f"[green]Multiline mode {status}.[/green]")
            if self.multiline_mode:
                self.console.print("[dim]Press Meta+Enter or Esc+Enter to submit.[/dim]")

        elif cmd == '/stream' or cmd.startswith('/stream '):
            parts = raw.split(maxsplit=1)
            if len(parts) == 1:
                status = "enabled" if self.stream else "disabled"
                self.console.print(f"[green]Stream mode {status}.[/green]")
                return

            action = parts[1].strip().lower()
            if action in {"on", "true", "1", "enable", "enabled"}:
                self.stream = True
            elif action in {"off", "false", "0", "disable", "disabled"}:
                self.stream = False
            elif action == "toggle":
                self.stream = not self.stream
            else:
                self.console.print("[red]Usage: /stream-responses [on|off|toggle][/red]")
                return

            status = "enabled" if self.stream else "disabled"
            self.console.print(f"[green]Stream mode {status}.[/green]")

        elif cmd == '/render-last':
            rendered = self._render_last_assistant_message()
            if not rendered:
                self.console.print("[yellow]No assistant response available to render.[/yellow]")

        elif cmd.startswith('/load'):
            parts = command.strip().split(maxsplit=1)
            if len(parts) < 2:
                self.console.print("[red]Usage: /load-session <session-id>[/red]")
            else:
                session_id = parts[1]
                self.load_session(session_id)

        elif cmd == '/resume' or cmd.startswith('/resume '):
            parts = command.strip().split(maxsplit=1)
            session_id = parts[1].strip() if len(parts) > 1 else None
            self.resume_session(session_id)

        elif cmd == '/skill':
            self._handle_skill_command()

        elif cmd == '/context':
            # /context-usage reads the live preflight through context-specific
            # references created at REPL initialization. Do not populate the
            # generic provider key used by commands that may make provider calls.
            self.command_context.config["context_provider"] = self.provider
            self.command_context.config["context_tool_registry"] = self.tool_registry
            self.command_context.config["context_tool_context"] = self.tool_context
            # Try new command system
            try:
                handled, result_text = self._try_execute_new_command('context', '')
                if handled and result_text:
                    self.console.print(Markdown(result_text))
                    return
            except Exception:
                pass
            self.console.print("[yellow]/context-usage analysis unavailable in this context.[/yellow]")

        elif cmd == '/doctor':
            try:
                handled, result_text = self._try_execute_new_command('doctor', '')
                if handled and result_text:
                    self.console.print(Markdown(result_text))
                    return
            except Exception as exc:
                self.console.print(f"[red]/doctor failed locally: {exc}[/red]")
                return
            self.console.print("[yellow]/doctor diagnostics unavailable.[/yellow]")

        elif cmd == '/compact':
            # /compact is an explicit request to use the active provider for
            # summarization. Keep that provider capability scoped to this command.
            previous_provider = self.command_context.config.get("provider")
            previous_model = self.command_context.config.get("model")
            had_provider = "provider" in self.command_context.config
            had_model = "model" in self.command_context.config
            self.command_context.config["provider"] = self.provider
            self.command_context.config["model"] = self.provider.model
            try:
                handled, result_text = self._try_execute_new_command('compact', '')
                if handled:
                    if result_text:
                        self.console.print("\n[green]" + result_text + "[/green]")
                    return
                self.console.print("[yellow]/compact is unavailable; conversation preserved.[/yellow]")
            except Exception as exc:
                self.console.print(
                    f"[red]/compact failed safely; conversation preserved: {exc}[/red]"
                )
            finally:
                if had_provider:
                    self.command_context.config["provider"] = previous_provider
                else:
                    self.command_context.config.pop("provider", None)
                if had_model:
                    self.command_context.config["model"] = previous_model
                else:
                    self.command_context.config.pop("model", None)

        else:
            if raw.startswith("/"):
                if self._try_run_skill_slash(raw):
                    return
            self.console.print(f"[red]Unknown command: {command}[/red]")

    def _try_run_skill_slash(self, raw: str) -> bool:
        if self.tool_registry.get("Skill") is None:
            return False

        text = raw.strip()
        if not text.startswith("/"):
            return False
        body = text[1:]
        if not body:
            return False
        if body.split(maxsplit=1)[0].lower() in {c.lstrip("/").lower() for c in self._built_in_commands if c != "/"}:
            return False

        parts = body.split(maxsplit=1)
        skill_name = parts[0].strip()
        args = parts[1] if len(parts) > 1 else ""
        if not skill_name:
            return False

        previous_tool_allowlist = self.tool_context.tool_allowlist
        try:
            result = self.tool_registry.dispatch(
                ToolCall(name="Skill", input={"skill": skill_name, "args": args}),
                self.tool_context,
            )
        except Exception as e:
            self.tool_context.tool_allowlist = previous_tool_allowlist
            self.console.print(f"[red]Skill error: {e}[/red]")
            return True

        payload = result.output if isinstance(result.output, dict) else {}
        if result.is_error or not payload.get("success"):
            self.tool_context.tool_allowlist = previous_tool_allowlist
            err = payload.get("error") if isinstance(payload.get("error"), str) else "Unknown skill error"
            self.console.print(f"[red]{err}[/red]")
            return True

        self.console.print(f"[dim]Launching skill: {payload.get('commandName', skill_name)}[/dim]")
        meta_parts: list[str] = []
        loaded = payload.get("loadedFrom")
        if isinstance(loaded, str) and loaded:
            meta_parts.append(f"source={loaded}")
        model = payload.get("model")
        if isinstance(model, str) and model:
            meta_parts.append(f"model={model}")
        tools = payload.get("allowedTools")
        if isinstance(tools, list) and tools:
            shown = ", ".join(str(t) for t in tools[:6])
            more = f" (+{len(tools) - 6})" if len(tools) > 6 else ""
            meta_parts.append(f"tools={shown}{more}")
        if meta_parts:
            self.console.print(f"[dim]{' · '.join(meta_parts)}[/dim]")

        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            self.tool_context.tool_allowlist = previous_tool_allowlist
            self.console.print("[red]Skill produced empty prompt[/red]")
            return True

        try:
            self.chat(prompt)
        finally:
            self.tool_context.tool_allowlist = previous_tool_allowlist
        return True

    def show_help(self):
        """Show organized command help."""
        self._show_slash_palette()
        help_text = """
**Usage**
- Type a message and press Enter to chat
- Type `/` or use Tab for command completion
- Press Ctrl+C to interrupt the current operation
- Press Ctrl+D to exit
- Use `/multiline-input` for multi-paragraph input
"""
        self.console.print(Markdown(help_text))

    def _handle_skill_command(self) -> None:
        """Handle /skill command - list all available skills."""
        try:
            from src.skills.loader import get_all_skills

            cwd = self.tool_context.cwd or self.tool_context.workspace_root
            skills = list(get_all_skills(project_root=cwd))
            skills.sort(key=lambda s: s.name.lower())

            if not skills:
                self.console.print("\n[bold]Available Skills:[/bold]")
                self.console.print("[dim]No skills found.[/dim]")
                self.console.print("[dim]Create skills in ~/.clawd/skills/ or ~/.claude/skills/ or .clawd/skills/ in your project.[/dim]")
                return

            # Group skills by source
            from collections import defaultdict
            by_source: dict[str, list] = defaultdict(list)
            for s in skills:
                loaded = getattr(s, "loaded_from", "") or "unknown"
                by_source[loaded].append(s)

            self.console.print(f"\n[bold]Available Skills ({len(skills)}):[/bold]")
            for source in sorted(by_source.keys()):
                source_skills = by_source[source]
                self.console.print(f"\n[cyan]{source.title()} Skills:[/cyan]")
                for s in source_skills:
                    desc = (getattr(s, "description", None) or "").strip()
                    user_invocable = getattr(s, "user_invocable", True)
                    inv_str = "" if user_invocable else " [dim](not user-invocable)[/dim]"
                    self.console.print(f"  [green]/{s.name}[/green]{inv_str}")
                    if desc:
                        self.console.print(f"    [dim]{desc}[/dim]")
            self.console.print()
        except Exception as e:
            self.console.print(f"[red]Error loading skills: {e}[/red]")

    def _is_recoverable_tool_error(self, tool_name: str, tool_output) -> bool:
        if not isinstance(tool_name, str):
            return False
        if not isinstance(tool_output, dict):
            return False
        name = tool_name.strip().lower()
        err = tool_output.get("error")
        if not isinstance(err, str):
            return False
        e = err.lower()
        if name == "read" and e.startswith("file not found:"):
            p = err.split(":", 2)[-1].strip()
            if "/.clawd/skills/" in p or "\\.clawd\\skills\\" in p or "/.claude/skills/" in p or "\\.claude\\skills\\" in p:
                return True
        return False

    def _provider_uses_system_kwarg(self) -> bool:
        return isinstance(self.provider, (AnthropicProvider, MinimaxProvider))

    def _build_direct_stream_payload(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        style_name = getattr(self.tool_context, "output_style_name", None)
        style_dir = getattr(self.tool_context, "output_style_dir", None)
        style_prompt = resolve_output_style(style_name, style_dir).prompt

        if self._provider_uses_system_kwarg():
            return self.session.conversation.get_messages(), (
                {"system": style_prompt} if style_prompt.strip() else {}
            )

        messages: list[dict[str, Any]] = []
        for msg in self.session.conversation.messages:
            if isinstance(msg.content, str):
                messages.append({"role": msg.role, "content": msg.content})
        if style_prompt.strip():
            messages = [{"role": "system", "content": style_prompt}, *messages]
        return messages, {}

    def _should_try_direct_response(self, user_input: str) -> bool:
        text = user_input.strip().lower()
        if not text or text.startswith("/"):
            return False
        if len(text) > 240:
            return False

        code_task_markers = (
            "/", "\\", "src/", "tests/", ".py", ".ts", ".md",
            "file", "files", "read", "write", "edit", "modify", "change",
            "search", "grep", "glob", "bash", "shell", "command", "run",
            "test", "fix", "bug", "refactor", "repo", "repository",
            "project", "workspace", "folder", "directory", "function",
            "class", "module", "code", "implementation", "readme",
            "pyproject", "package.json", "git", "commit", "diff", "tool",
            "geminithink", "youtubeanalyze",
            "文件", "代码", "仓库", "项目", "目录", "读取", "写入", "修改",
            "搜索", "运行", "测试", "修复", "命令", "工具", "函数", "类",
        )
        return not any(marker in text for marker in code_task_markers)

    def _direct_response(self, on_text_chunk=None):
        # A provider failure here surfaces to the user; it never falls through to
        # the agent route, which would send a second request. Only a local payload
        # failure (nothing sent yet) still hands over to the agent route.
        try:
            api_messages, call_kwargs = self._build_direct_stream_payload()
        except Exception:
            return None

        if not self.stream:
            response = self.provider.chat(api_messages, tools=None, **call_kwargs)
            full_response = getattr(response, "content", "") or ""
            if not full_response:
                return None
            self.session.conversation.add_assistant_message(full_response)
            return response

        streamed_chunks: list[str] = []

        def capture_chunk(chunk: str) -> None:
            if not chunk:
                return
            # Recorded before display so a display error is never mistaken for "unsupported".
            streamed_chunks.append(chunk)
            if on_text_chunk is not None:
                on_text_chunk(chunk)

        try:
            response = self.provider.chat_stream_response(
                api_messages,
                tools=None,
                on_text_chunk=capture_chunk,
                **call_kwargs,
            )
        except NotImplementedError:
            if streamed_chunks:
                raise
            # Provider has no structured stream result. Preserve the old stream path,
            # but usage will be unavailable for this direct response.
            api_messages, call_kwargs = self._build_direct_stream_payload()
            for chunk in self.provider.chat_stream(api_messages, tools=None, **call_kwargs):
                capture_chunk(chunk)
            if not streamed_chunks:
                return None
            full_response = "".join(streamed_chunks)
            self.session.conversation.add_assistant_message(full_response)
            return {"content": full_response, "usage": {}}

        full_response = getattr(response, "content", "") or "".join(streamed_chunks)
        if not full_response:
            return None
        self.session.conversation.add_assistant_message(full_response)
        return response

    def _confirm_high_token_agent_request(self, estimated_input_tokens: int) -> bool:
        """Ask once before a high-token first agent request is sent."""
        self.console.print("")
        self.console.print("[bold yellow]High token estimate[/bold yellow]")
        self.console.print(
            "  This task is estimated to send about "
            f"[bold]{estimated_input_tokens:,} first-request input tokens[/bold] "
            "to the provider."
        )
        self.console.print(
            "  Actual usage may differ, and additional tool turns can use more tokens."
        )
        choice = input("Continue? [y/n]> ").strip().lower()
        return choice in ("y", "yes")

    def _get_last_assistant_text(self) -> str | None:
        for message in reversed(self.session.conversation.messages):
            if message.role != "assistant":
                continue
            content = message.content
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    block_type = getattr(block, "type", None)
                    if block_type == "text":
                        text = getattr(block, "text", "")
                        if isinstance(text, str) and text:
                            parts.append(text)
                joined = "".join(parts).strip()
                if joined:
                    return joined
        return None

    def _render_last_assistant_message(self) -> bool:
        text = self._get_last_assistant_text()
        if not text:
            return False
        self.console.print("\n[bold]Last Assistant Response[/bold]")
        self.console.print(Markdown(text))
        self.console.print()
        return True

    def _primary_usage_label(self) -> str:
        provider_labels = {
            "anthropic": "Claude",
            "openai": "OpenAI",
            "deepseek": "DeepSeek",
            "qwen": "Qwen",
            "glm": "GLM",
            "minimax": "MiniMax",
        }
        provider_key = str(getattr(self, "provider_name", "") or "").strip().lower()
        provider_label = provider_labels.get(provider_key, provider_key.title() or "Primary provider")
        model = str(getattr(self.provider, "model", "") or "").strip()
        return f"{provider_label} ({model})" if model else provider_label

    def _record_and_print_task_usage(self, usage, skills_used: set[str] | None = None) -> None:
        usage = usage if isinstance(usage, dict) else {}
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        thought_tokens = int(usage.get("thought_tokens", 0) or 0)
        tool_use_tokens = int(usage.get("tool_use_tokens", 0) or 0)
        cached_tokens = int(usage.get("cached_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0) or (input_tokens + output_tokens)
        task_usage: dict[str, dict[str, int]] = {}

        def add_task_usage(label: str, values: dict[str, int]) -> None:
            bucket = task_usage.setdefault(
                label,
                {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "thought_tokens": 0,
                    "tool_use_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                },
            )
            for key in bucket:
                bucket[key] += max(0, int(values.get(key, 0) or 0))

        if total_tokens > 0:
            primary_label = self._primary_usage_label()
            primary = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "thought_tokens": thought_tokens,
                "tool_use_tokens": tool_use_tokens,
                "cached_tokens": cached_tokens,
                "total_tokens": total_tokens,
            }
            self.cost_tracker.record_usage(
                primary_label,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                thought_tokens=thought_tokens,
                tool_use_tokens=tool_use_tokens,
                cached_tokens=cached_tokens,
            )
            append_provider_usage({"label": primary_label, **primary})
            add_task_usage(primary_label, primary)

        records = self.tool_context.consume_usage_records()
        for record in records:
            label = str(record.get("label") or "External provider")
            values = {
                "input_tokens": int(record.get("input_tokens", 0) or 0),
                "output_tokens": int(record.get("output_tokens", 0) or 0),
                "thought_tokens": int(record.get("thought_tokens", 0) or 0),
                "tool_use_tokens": int(record.get("tool_use_tokens", 0) or 0),
                "cached_tokens": int(record.get("cached_tokens", 0) or 0),
                "total_tokens": int(record.get("total_tokens", 0) or 0),
            }
            self.cost_tracker.record_usage(
                label,
                input_tokens=values["input_tokens"],
                output_tokens=values["output_tokens"],
                total_tokens=values["total_tokens"],
                thought_tokens=values["thought_tokens"],
                tool_use_tokens=values["tool_use_tokens"],
                cached_tokens=values["cached_tokens"],
            )
            add_task_usage(label, values)

        if hasattr(self, "command_context") and self.command_context:
            self.command_context.cost_tracker = self.cost_tracker

        extra_usage: list[str] = []
        used = sorted(skills_used or set())
        if used:
            try:
                from src.skills.trust_registry import SkillTrustRegistry

                trust = SkillTrustRegistry()
                for name in used:
                    record = trust.get(name) or {}
                    declared = record.get("additional_claude_usage") or {}
                    if bool(declared.get("expected", False)):
                        level = str(declared.get("level") or "unknown")
                        reason = str(declared.get("reason") or "").strip()
                        detail = f"{name} ({level})"
                        if reason:
                            detail += f": {reason}"
                        extra_usage.append(detail)
            except Exception:
                extra_usage = [f"{name} (usage metadata unavailable)" for name in used]

        if task_usage:
            self.console.print("[dim]Usage this task:[/dim]")
            for label, values in task_usage.items():
                details = (
                    f"{values['input_tokens']:,} input, "
                    f"{values['output_tokens']:,} output"
                )
                if values["thought_tokens"]:
                    details += f", {values['thought_tokens']:,} thought"
                if values["tool_use_tokens"]:
                    details += f", {values['tool_use_tokens']:,} tool-use"
                if values["cached_tokens"]:
                    details += f", {values['cached_tokens']:,} cached"
                self.console.print(
                    f"[dim]  {label}: {details} = "
                    f"{values['total_tokens']:,} total tokens[/dim]"
                )
            if len(task_usage) > 1:
                combined = sum(v["total_tokens"] for v in task_usage.values())
                self.console.print(f"[dim]  Combined tracked total: {combined:,} tokens[/dim]")
        else:
            self.console.print("[dim]Usage this task: providers did not return token counts.[/dim]")

        if extra_usage:
            self.console.print(
                "[yellow]Additional Claude usage declared by skill(s): "
                + "; ".join(extra_usage)
                + "[/yellow]"
            )
        elif used:
            self.console.print(
                "[dim]Additional Claude usage declared by used skills: none.[/dim]"
            )

        self.console.print(
            "[dim]Use /session-usage for JR's tracked session tokens "
            "or /usage for API/model usage plus skill and tool activity.[/dim]"
        )

    def chat(self, user_input: str, max_turns: int = 20):
        """Send message to LLM and display response.

        Args:
            user_input: The user message to send.
            max_turns: Maximum number of tool call turns (default 20, higher for complex commands).
        """
        # Start each user task with a fresh external-provider usage bucket.
        self.tool_context.usage_records.clear()

        # Preserve the exact pre-turn state so a provider authentication rejection
        # can roll back an unanswered user message without erasing any later
        # assistant/tool activity that may already have happened.
        pre_task_messages = list(self.session.conversation.messages)
        self.session.conversation.add_user_message(user_input)
        added_user_message = (
            self.session.conversation.messages[-1]
            if self.session.conversation.messages
            and self.session.conversation.messages[-1].role == "user"
            and self.session.conversation.messages[-1].content == user_input
            else None
        )

        try:
            self.console.print("\n[bold]Assistant[/bold]")

            stream_started = False
            skills_used: set[str] = set()

            def _stop_status_once() -> None:
                nonlocal stream_started
                if self._current_status is not None and not stream_started:
                    try:
                        self._current_status.stop()
                    except Exception:
                        pass
                stream_started = True

            def on_event(ev: ToolEvent) -> None:
                if ev.kind == "tool_use":
                    if ev.tool_name == "Skill" and isinstance(ev.tool_input, dict):
                        skill_name = ev.tool_input.get("skill")
                        if isinstance(skill_name, str) and skill_name.strip():
                            skills_used.add(skill_name.strip().lstrip("/"))
                    summary = summarize_tool_use(ev.tool_name, ev.tool_input or {})
                    if isinstance(summary, str) and summary:
                        summary = self._shorten_path_text(summary)
                    suffix = f" [dim]({summary})[/dim]" if summary else ""
                    self.console.print(f"[dim]•[/dim] [cyan]{ev.tool_name}[/cyan]{suffix} [dim]running...[/dim]")
                    return
                if ev.kind == "tool_result":
                    if ev.is_error:
                        if self._is_recoverable_tool_error(ev.tool_name, ev.tool_output):
                            return
                        msg = ""
                        if isinstance(ev.tool_output, dict) and isinstance(ev.tool_output.get("error"), str):
                            msg = ev.tool_output["error"]
                        self.console.print(f"[red]  ↳ {msg or 'Error'}[/red]")
                        return
                    msg = summarize_tool_result(ev.tool_name, ev.tool_output)
                    if isinstance(msg, str):
                        prefix = f"{ev.tool_name} · "
                        if msg.startswith(prefix):
                            msg = msg[len(prefix):]
                        msg = self._shorten_path_text(msg)
                    self.console.print(f"[dim]  ↳ {msg}[/dim]")
                    return
                if ev.kind == "tool_error":
                    msg = ev.error or "Error"
                    self.console.print(f"[red]  ↳ {msg}[/red]")

            def on_text_chunk(chunk: str) -> None:
                if not chunk:
                    return
                _stop_status_once()
                self.console.print(chunk, end="", markup=False, highlight=False, soft_wrap=True)

            if self._should_try_direct_response(user_input):
                self._current_status = self.console.status("[dim]Thinking...[/dim]", spinner="dots")
                with self._current_status:
                    direct_response = self._direct_response(
                        on_text_chunk=on_text_chunk if self.stream else None
                    )
                self._current_status = None
                if direct_response is not None:
                    direct_usage = (
                        direct_response.get("usage", {})
                        if isinstance(direct_response, dict)
                        else getattr(direct_response, "usage", {})
                    )
                    if self.stream:
                        self.console.print("\n")
                    else:
                        direct_text = (
                            direct_response.get("content", "")
                            if isinstance(direct_response, dict)
                            else getattr(direct_response, "content", "")
                        )
                        self.console.print(Markdown(direct_text))
                        self.console.print("\n")
                    self._record_and_print_task_usage(direct_usage, skills_used)
                    self.console.print()
                    return

            # The direct/no-tools route above stays warning-free. Only now, once
            # the agent route is selected, assemble the exact first request locally.
            preflight = build_agent_preflight(
                self.session.conversation,
                self.provider,
                self.tool_registry,
                self.tool_context,
            )
            if (
                preflight.estimated_input_tokens >= 10_000
                and not self._confirm_high_token_agent_request(preflight.estimated_input_tokens)
            ):
                self.console.print("[dim]Request cancelled before contacting the provider.[/dim]")
                self.console.print()
                return

            # Use agent loop with tools for any provider that supports it
            self._current_status = self.console.status("[dim]Thinking...[/dim]", spinner="dots")
            with self._current_status:
                result = run_agent_loop(
                    conversation=self.session.conversation,
                    provider=self.provider,
                    tool_registry=self.tool_registry,
                    tool_context=self.tool_context,
                    max_turns=max_turns,
                    stream=self.stream,
                    verbose=False,
                    on_event=on_event,
                    on_text_chunk=on_text_chunk if self.stream else None,
                    preflight=preflight,
                )
            self._current_status = None

            if self.stream and stream_started:
                self.console.print()
                self.console.print()
            else:
                self.console.print(Markdown(result.response_text))
                self.console.print("\n")

            self._record_and_print_task_usage(result.usage, skills_used)
            self.console.print()

        except Exception as e:
            # The status context manager has already stopped; clear the stale
            # reference so later permission/user prompts do not try to restart it.
            self._current_status = None

            if _is_provider_authentication_error(e):
                current_messages = self.session.conversation.messages
                clean_unanswered_turn = current_messages == pre_task_messages
                if (
                    added_user_message is not None
                    and current_messages
                    and current_messages[-1] is added_user_message
                    and not stream_started
                ):
                    current_messages[:] = pre_task_messages
                    clean_unanswered_turn = True

                self.console.print("\n[red]❌ Authentication Error[/red]")
                self.console.print(
                    "\n[yellow]The active provider rejected authentication. "
                    "Your API key may be invalid or expired.[/yellow]"
                )

                # Ask if user wants to reconfigure. Do not retry the failed provider
                # request automatically; a retry can incur provider usage/cost.
                from rich.prompt import Prompt
                choice = Prompt.ask(
                    "\nWould you like to reconfigure your provider now?",
                    choices=["y", "n"],
                    default="y",
                )

                if choice == "y":
                    if self._handle_relogin():
                        if clean_unanswered_turn:
                            self.console.print(
                                "[dim]The rejected user turn was removed from conversation "
                                "and was not retried automatically. Re-send it when ready.[/dim]"
                            )
                        else:
                            self.console.print(
                                "[yellow]The incomplete turn already contains assistant/tool "
                                "activity, so it was preserved and was not retried automatically. "
                                "Review it before continuing.[/yellow]"
                            )
                else:
                    self.console.print(
                        "\n[dim]You can run [bold]clawd login[/bold] later "
                        "to update provider configuration.[/dim]"
                    )
                    if clean_unanswered_turn:
                        self.console.print(
                            "[dim]The rejected user turn was removed from conversation; "
                            "re-send it after reconfiguration.[/dim]"
                        )
            else:
                # Generic error handling
                self.console.print(f"\n[red]Error: {e}[/red]")
                self.console.print("[dim]The request failed. Clawd did not issue a fallback retry.[/dim]")
                import traceback
                traceback.print_exc()

    def _handle_relogin(self) -> bool:
        """Reconfigure and activate a provider after an authentication failure."""
        from rich.prompt import Prompt
        from src.config import get_provider_config, set_api_key, set_default_provider
        from src.providers import (
            PROVIDER_INFO,
            get_provider_class,
            validate_provider_runtime_config,
        )

        self.console.print("\n[bold blue]🔑 Reconfigure Provider[/bold blue]\n")

        provider_names = list(PROVIDER_INFO.keys())
        self.console.print("[bold]Available providers:[/bold]")
        for name, info in PROVIDER_INFO.items():
            self.console.print(
                f"  [cyan]{name}[/cyan] - {info['label']} "
                f"(default model: {info['default_model']})"
            )
        self.console.print()

        provider = Prompt.ask(
            "Select LLM provider",
            choices=provider_names,
            default=self.provider_name if self.provider_name in provider_names else "anthropic",
        )
        info = PROVIDER_INFO[provider]
        configured = get_provider_config(provider)

        # Re-authentication always requires a fresh key for credentialed
        # providers; do not echo or silently reuse the rejected credential.
        requires_api_key = info.get("requires_api_key", True)
        api_key = ""
        if requires_api_key:
            api_key = Prompt.ask(
                f"Enter {provider.upper()} API Key",
                password=True,
            )
            if not api_key:
                self.console.print("\n[red]Error: API Key cannot be empty[/red]")
                return False
        else:
            self.console.print(
                "\n[dim]This local-only provider does not require an API key.[/dim]"
            )

        base_url_default = configured.get("base_url") or info["default_base_url"]
        model_default = configured.get("default_model") or info["default_model"]

        self.console.print(f"\n[dim]Current/default:[/dim] {base_url_default}")
        base_url = Prompt.ask(
            f"{provider.upper()} Base URL",
            default=base_url_default,
        )

        self.console.print(
            f"\n[dim]Available models:[/dim] {', '.join(info['available_models'])}"
        )
        self.console.print(f"[dim]Current/default:[/dim] [bold]{model_default}[/bold]")
        default_model = Prompt.ask(
            f"{provider.upper()} Default Model",
            default=model_default,
        )

        candidate_config = {
            "api_key": api_key,
            "base_url": base_url,
            "default_model": default_model,
        }
        try:
            validate_provider_runtime_config(provider, candidate_config)
        except ValueError as exc:
            self.console.print(f"\n[red]Error: {exc}[/red]")
            return False

        # Construct the replacement before persisting or switching runtime state.
        # Provider constructors are expected to be local/lazy; no request is sent.
        try:
            provider_class = get_provider_class(provider)
            candidate_provider = provider_class(
                api_key=api_key,
                base_url=base_url,
                model=default_model,
            )
        except Exception as exc:
            self.console.print(
                "\n[red]Unable to initialize the selected provider "
                f"({type(exc).__name__}). Configuration was not changed.[/red]"
            )
            return False

        try:
            set_api_key(
                provider,
                api_key=api_key,
                base_url=base_url,
                default_model=default_model,
            )
            set_default_provider(provider)
        except Exception as exc:
            self.console.print(
                "\n[red]Unable to save provider configuration "
                f"({type(exc).__name__}). Runtime provider was not changed.[/red]"
            )
            return False

        self.provider = candidate_provider
        self.provider_name = provider

        # Keep every live consumer aligned with the provider that is now active.
        self.session.provider = provider
        self.session.model = candidate_provider.model
        self.command_context.config["context_provider"] = candidate_provider

        if requires_api_key:
            self.console.print(
                f"\n[green]✓ {provider.upper()} API key updated successfully![/green]"
            )
        else:
            self.console.print(
                f"\n[green]✓ {provider.upper()} local provider configuration updated.[/green]"
            )
        self.console.print(
            "[green]✓ Provider reinitialized. Future requests use the new provider.[/green]\n"
        )
        return True

    def save_session(self):
        """Save current session using the provider/model actually active now."""
        self.session.provider = self.provider_name
        self.session.model = self.provider.model
        self.session.save()
        self.console.print(f"[green]Session saved: {self.session.session_id}[/green]")

    def load_session(self, session_id: str):
        """Load a previous conversation without silently changing provider routing.

        Args:
            session_id: Session ID to load
        """
        from src.agent import Session

        try:
            loaded_session = Session.load(session_id)
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            self.console.print(f"[red]Unable to load session {session_id}: {exc}[/red]")
            return
        if loaded_session is None:
            self.console.print(f"[red]Session not found: {session_id}[/red]")
            return

        saved_provider = loaded_session.provider
        saved_model = loaded_session.model

        # Replace the active conversation and rebind command-system consumers.
        # Provider routing remains the explicitly active REPL provider.
        self.session = loaded_session
        self.command_context.conversation = loaded_session.conversation
        self.command_context.config["context_provider"] = self.provider
        self.command_context.config["context_tool_registry"] = self.tool_registry
        self.command_context.config["context_tool_context"] = self.tool_context

        # Keep in-memory session metadata truthful for any subsequent save.
        self.session.provider = self.provider_name
        self.session.model = self.provider.model

        self.console.print(f"[green]Session loaded: {session_id}[/green]")
        if saved_provider != self.provider_name or saved_model != self.provider.model:
            self.console.print(
                f"[yellow]Saved provider/model: {saved_provider} / {saved_model}. "
                f"Active runtime remains: {self.provider_name} / {self.provider.model}.[/yellow]"
            )
        else:
            self.console.print(
                f"[dim]Provider: {self.provider_name}, Model: {self.provider.model}[/dim]"
            )
        self.console.print(f"[dim]Messages: {len(loaded_session.conversation.messages)}[/dim]")

        # Show conversation history
        if loaded_session.conversation.messages:
            self.console.print("\n[bold]Conversation History:[/bold]")
            for msg in loaded_session.conversation.messages[-5:]:  # Show last 5 messages
                role_color = "blue" if msg.role == "user" else "green"
                self.console.print(f"[{role_color}]{msg.role}[/{role_color}]: {msg.content[:100]}...")

    def resume_session(self, session_id: str | None = None):
        """Resume a saved session by ID or from a recent-session picker."""
        if session_id:
            self.load_session(session_id)
            return

        sessions = [
            session
            for session in Session.list_saved(limit=20)
            if session.session_id != self.session.session_id
        ]
        if not sessions:
            self.console.print("[yellow]No other saved sessions available to resume.[/yellow]")
            return

        table = Table(title="Saved Sessions")
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Session ID", style="white", no_wrap=True)
        table.add_column("Updated", style="dim", no_wrap=True)
        table.add_column("Messages", justify="right", no_wrap=True)
        table.add_column("Provider / Model", style="dim")
        for index, saved in enumerate(sessions, start=1):
            updated = str(saved.updated_at or saved.created_at or "unknown")
            if "T" in updated:
                updated = updated.replace("T", " ", 1)
            table.add_row(
                str(index),
                saved.session_id,
                updated[:19],
                str(len(saved.conversation.messages)),
                f"{saved.provider} / {saved.model}",
            )
        self.console.print(table)

        try:
            from rich.prompt import Prompt
            choice = Prompt.ask(
                "Resume which session?",
                choices=[*(str(index) for index in range(1, len(sessions) + 1)), "cancel"],
                default="cancel",
            )
        except (EOFError, KeyboardInterrupt):
            self.console.print("[yellow]Resume cancelled.[/yellow]")
            return

        if choice == "cancel":
            self.console.print("[yellow]Resume cancelled.[/yellow]")
            return

        self.load_session(sessions[int(choice) - 1].session_id)
