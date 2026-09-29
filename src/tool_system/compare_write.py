"""Compare-and-write: change a file only while it still holds exactly the bytes the
change was built from, or create it only where nothing exists yet.

An existing file is opened without creating or truncating it, compared through that
same handle, and then written in place from the start, with the new bytes written before
the file is shortened, so it never passes through an empty state. Writing in place keeps
alternate data streams (Windows Mark-of-the-Web), which the truncating open dropped.

Two refusals of today's truncating open(path, "wb") are kept, before anything is
written, because the in-place open would otherwise not hit them: Windows files with the
hidden or system attribute, and on Linux fs.protected_regular (another user's file in a
sticky world- or group-writable directory; the in-place open carries no O_CREAT).

A failure part-way through the write is raised and not repaired: nothing further is
written, truncated, read or re-opened (without a lock, writing the old bytes back could
itself overwrite another writer's change). The caller marks nothing, so the read-state
check reports the file as changed unless its bytes are still exactly the verified ones.

No lock is taken: another process can still change the file in place while the final
comparison reads it or before the write lands (tests/test_read_fingerprint.py,
TestCompareAndWrite). On Linux the file is opened relative to a handle on its directory
without following a symlink swapped in for the file or that directory; higher path
components, and every component elsewhere (Windows follows a swapped-in link or
junction), are followed as by open(path).
"""
from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path
from typing import Any, Callable

from .context import (
    _NON_BLOCKING_OPEN_FLAGS,
    _fingerprint_of_bytes,
    _open_without_blocking,
    _read_matching,
)

_ON_LINUX = sys.platform.startswith("linux")
# Linux-only open flags; 0 where absent (the code using them runs only on Linux).
_O_PATH = getattr(os, "O_PATH", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_PROTECTED_REGULAR = "/proc/sys/fs/protected_regular"
_OVERFLOW_UID = "/proc/sys/kernel/overflowuid"
_THREAD_STATUS = "/proc/thread-self/status"
_DENIED = "denied"
_HIDDEN_OR_SYSTEM = 0x2 | 0x4  # FILE_ATTRIBUTE_HIDDEN | FILE_ATTRIBUTE_SYSTEM


class _NotTheVerifiedFile(Exception):
    """The path no longer leads to the regular file that was verified: it vanished, or
    it (or its directory) was replaced by a symlink or by something else."""


def _write_all(handle: Any, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = handle.write(view)
        if not written:
            raise OSError("short write: the file was not fully written")
        view = view[written:]


def _permission_denied(path: Path, detail: str = "") -> PermissionError:
    """The error today's open(path, "wb") raises when it is refused (EACCES, naming the
    target as a str). ``detail`` is added only to a fail-closed refusal."""
    message = os.strerror(errno.EACCES) + (f" ({detail})" if detail else "")
    return PermissionError(errno.EACCES, message, str(path))


def _read_proc_int(path: str) -> int:
    with open(path, "rb") as handle:
        return int(handle.read(64).decode("ascii").strip())


def _read_fsuid() -> int:
    """The calling thread's filesystem uid (the 4th ``Uid:`` field): the uid the kernel
    compares with the file owner, which can differ from the effective uid."""
    with open(_THREAD_STATUS, "rb") as handle:
        for line in handle.read(65536).decode("ascii", "replace").splitlines():
            fields = line.split()
            if fields[:1] == ["Uid:"] and len(fields) == 5:
                return int(fields[4])
    raise ValueError(f"no Uid: line in {_THREAD_STATUS}")


def _protected_regular_refusal(dir_mode: int, dir_uid: int, file_uid: int) -> str | None:
    """Linux fs.protected_regular (fs/namei.c may_create_in_sticky) for an EXISTING regular
    file opened with O_CREAT, as today's open(path, "wb") opens it. None: that open would
    pass this check. _DENIED: the rule refuses it. Anything else says why it cannot be
    classified here; the caller then refuses (fails closed). /proc is read only for a
    sticky directory that is group- or world-writable.

    The kernel's ownership exemptions come first because they hold at every level: a file
    owned by the directory's owner or by the caller's filesystem uid is never refused, so
    fs.protected_regular is not read for it. A uid shown as the overflow uid may be an
    unmapped owner, so it never proves a match, and without the overflow uid no match is
    trusted. Callers pass the owners of the directory handle and of the file actually
    opened (or, before the open, of the file found there: a check that can only refuse)."""
    if not dir_mode & stat.S_ISVTX or not dir_mode & (stat.S_IWOTH | stat.S_IWGRP):
        return None
    try:
        overflow_uid: int | None = _read_proc_int(_OVERFLOW_UID)
    except (OSError, ValueError):
        overflow_uid = None
    try:
        fsuid: int | None = _read_fsuid()
    except (OSError, ValueError):
        fsuid = None
    if overflow_uid is not None and file_uid != overflow_uid and file_uid in (dir_uid, fsuid):
        return None
    try:
        level = _read_proc_int(_PROTECTED_REGULAR)
    except (OSError, ValueError) as exc:
        return f"fs.protected_regular could not be read: {type(exc).__name__}"
    if level not in (0, 1, 2):
        return f"fs.protected_regular={level} is not understood"
    if level == 0 or not (dir_mode & stat.S_IWOTH or level == 2):
        return None
    if overflow_uid is None or fsuid is None:
        return "the owners could not be classified (overflow uid or filesystem uid unreadable)"
    if file_uid == overflow_uid and overflow_uid in (dir_uid, fsuid):
        return "an owner is unmapped (overflow uid)"
    return _DENIED


def _enforce_protected_regular(path: Path, directory: os.stat_result, file_uid: int) -> None:
    refusal = _protected_regular_refusal(directory.st_mode, directory.st_uid, file_uid)
    if refusal == _DENIED:
        raise _permission_denied(path)
    if refusal is not None:
        raise _permission_denied(path, f"fs.protected_regular not verified: {refusal}")


def _as_opened_by_path(exc: OSError, path: Path) -> Exception:
    """An error of the directory-relative open as today's open(path, ...) reports it:
    vanished or replaced (a symlink, a directory, a socket) means not the verified file;
    anything else is the same error naming the target path."""
    if exc.errno in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EISDIR, errno.ENXIO):
        return _NotTheVerifiedFile()
    return type(exc)(exc.errno, exc.strerror, str(path))


def _open_in(parent: int, path: Path, flags: int) -> int:
    failure: Exception | None = None
    try:
        found = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    except OSError as exc:
        failure = _as_opened_by_path(exc, path)
    if failure is not None:
        raise failure  # raised outside the handler: no implicit exception chaining
    if not stat.S_ISREG(found.st_mode):
        raise _NotTheVerifiedFile()
    # Before the open, only to keep today's error precedence (for an existing regular file
    # the kernel checks this rule before the other open errors): it can only refuse. The
    # decision to allow is taken below, on the file actually opened.
    _enforce_protected_regular(path, os.fstat(parent), found.st_uid)
    try:
        fd = os.open(path.name, flags | _NON_BLOCKING_OPEN_FLAGS | _O_NOFOLLOW, dir_fd=parent)
    except OSError as exc:
        failure = _as_opened_by_path(exc, path)
    if failure is not None:
        raise failure
    try:
        # On the file actually opened and the directory handle: this decides, so a
        # replacement cannot slip past the check.
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise _NotTheVerifiedFile()
        _enforce_protected_regular(path, os.fstat(parent), opened.st_uid)
    except BaseException:
        os.close(fd)
        raise
    return fd


def _anchored_opener(path: Path) -> Callable[[str, int], int]:
    """Linux: open the existing file relative to a handle on its directory, following no
    symlink swapped in for the file or that directory and never creating anything, and
    refuse exactly where today's O_CREAT open is refused by fs.protected_regular (fail
    closed when the rule cannot be verified)."""

    def opener(_name: str, flags: int) -> int:
        failure: Exception | None = None
        try:
            parent = os.open(str(path.parent), _O_PATH | _O_DIRECTORY | _O_NOFOLLOW | _O_CLOEXEC)
        except OSError as exc:
            failure = _as_opened_by_path(exc, path)
        if failure is not None:
            raise failure
        try:
            return _open_in(parent, path, flags)
        finally:
            os.close(parent)

    return opener


def write_if_unchanged(path: Path, data: bytes, *, expected: bytes | None) -> bool:
    """Write ``data`` to ``path`` only if the file still holds exactly ``expected``;
    with ``expected=None``, only create it, never replacing anything that exists.

    Returns False, having written nothing, when the file no longer holds ``expected``
    (changed, vanished, replaced, or not a regular file) or, for a create, when a file
    already exists at ``path``. Where opening a directory fails first (Windows, and a
    create over a directory), that OS error propagates as it did before. An existing file
    that today's truncating open would refuse (Windows hidden/system attribute; Linux
    fs.protected_regular) raises the same PermissionError before anything is written; on
    Linux, when that rule cannot be verified, a PermissionError whose message says so.
    Other OSErrors propagate as open(path) reports them. A failure after the in-place
    write began is not repaired: the file may hold the new bytes up to the failure
    followed by the old tail.
    """
    if expected is None:
        try:
            with open(path, "xb", buffering=0) as created:
                _write_all(created, data)
        except FileExistsError:
            return False
        return True
    recorded = _fingerprint_of_bytes(expected)
    opener = _anchored_opener(path) if _ON_LINUX else _open_without_blocking
    try:
        handle = open(path, "r+b", buffering=0, opener=opener)
    except (FileNotFoundError, _NotTheVerifiedFile):
        return False
    with handle:
        if _read_matching(handle, recorded, keep=False) is None:
            return False
        if getattr(os.fstat(handle.fileno()), "st_file_attributes", 0) & _HIDDEN_OR_SYSTEM:
            raise _permission_denied(path)
        handle.seek(0)
        _write_all(handle, data)
        if len(data) < len(expected):
            handle.truncate(len(data))
    return True


def still_holds(path: Path, expected: bytes) -> bool:
    """True only if ``path`` is a regular file whose bytes are exactly ``expected``.
    Fails closed (False) on any OSError or invalid path."""
    try:
        with open(path, "rb", buffering=0, opener=_open_without_blocking) as handle:
            return _read_matching(handle, _fingerprint_of_bytes(expected), keep=False) is not None
    except (OSError, ValueError):
        return False
