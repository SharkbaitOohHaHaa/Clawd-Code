"""Session management with persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field

from .conversation import Conversation


_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _validate_session_id(session_id: str) -> str:
    if session_id in {".", ".."} or not _SESSION_ID_RE.fullmatch(session_id):
        raise ValueError("invalid session id")
    return session_id


@dataclass
class Session:
    """Session manager with persistence."""
    session_id: str
    provider: str
    model: str
    conversation: Conversation = field(default_factory=Conversation)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self):
        """Save session to disk."""
        session_dir = Path.home() / ".clawd" / "sessions"
        session_dir.mkdir(parents=True, exist_ok=True)

        session_file = session_dir / f"{self.session_id}.json"

        session_data = {
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "conversation": self.conversation.to_dict(),
            "created_at": self.created_at,
            "updated_at": datetime.now().isoformat()
        }

        with open(session_file, 'w') as f:
            json.dump(session_data, f, indent=2)

        self.updated_at = datetime.now().isoformat()

    @classmethod
    def load(cls, session_id: str) -> Optional['Session']:
        """Load session from disk."""
        session_id = _validate_session_id(session_id)
        session_file = Path.home() / ".clawd" / "sessions" / f"{session_id}.json"

        if not session_file.exists():
            return None

        with open(session_file, 'r') as f:
            data = json.load(f)

        return cls(
            session_id=data["session_id"],
            provider=data["provider"],
            model=data["model"],
            conversation=Conversation.from_dict(data["conversation"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"]
        )

    @classmethod
    def list_saved(cls, limit: int = 20) -> list['Session']:
        """List recent valid saved sessions, newest first."""
        if limit <= 0:
            return []

        session_dir = Path.home() / ".clawd" / "sessions"
        if not session_dir.exists() or not session_dir.is_dir():
            return []

        sessions: list[Session] = []
        for session_file in session_dir.glob("*.json"):
            if session_file.is_symlink() or not session_file.is_file():
                continue
            try:
                session_id = _validate_session_id(session_file.stem)
                session = cls.load(session_id)
            except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
                continue
            if session is not None:
                sessions.append(session)

        def sort_key(session: Session) -> datetime:
            try:
                return datetime.fromisoformat(session.updated_at)
            except (TypeError, ValueError):
                try:
                    return datetime.fromisoformat(session.created_at)
                except (TypeError, ValueError):
                    return datetime.min

        sessions.sort(key=sort_key, reverse=True)
        return sessions[:limit]

    @classmethod
    def create(cls, provider: str, model: str) -> 'Session':
        """Create a new session."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls(
            session_id=session_id,
            provider=provider,
            model=model
        )
