"""Read-state fingerprints bind the exact bytes each tool read or wrote.

A path counts as "read and unchanged" only when it is a regular file whose
current bytes are exactly the recorded bytes (byte length plus full SHA-256).
Metadata may prove a change (not a regular file, different size) but never
equality. Content-change cases restore the previous mtime with ``os.utime``, and
the same-size ones force the exact old ``(st_mtime_ns, st_size)`` collision, so
these safety proofs never depend on timestamp granularity or scheduling.

Compare -> mutation: each mutating tool builds its change from the exact bytes
the check verified (never a second read); Write, Edit and NotebookEdit compare again
immediately before they write, and DataTransform uses only the verified bytes
(TestCompareAndWrite). No lock is taken: a writer landing while that final
comparison reads the file, or between it and the write, is not excluded. A failed
write is raised and never repaired. The in-place write keeps today's refusals of a
truncating open: Windows hidden/system files, and Linux fs.protected_regular
(TestProtectedRegularPolicy and the Linux classes at the end). Not covered here, and
not solved by the fingerprint: partial views (offset/limit, previews), a read and a
mutation in the same assistant batch, and fingerprint lifetime across context loss.
"""
from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import errno
import hashlib
import io
import json
import os
import pathlib
import random
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from unittest.mock import patch

import src.tool_system.compare_write as compare_write_module
import src.tool_system.context as context_module
import src.tool_system.tools.read as read_module
from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolExecutionError, ToolInputError
from src.tool_system.tools import (
    DataInspectTool,
    DataTransformTool,
    FileEditTool,
    FileReadTool,
    FileWriteTool,
    NotebookEditTool,
)
from src.tool_system.tools.qwen_media import _prepare_media

SRC = Path(__file__).resolve().parents[1] / "src"
_PNG = b"\x89PNG\r\n\x1a\n" + b"safe-test-image"
_PDF = b"%PDF-1.4\n1 0 obj\n"


def _notebook(source: str) -> bytes:
    return (
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "id": "md-1", "metadata": {}, "source": [source]}
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            indent=1,
        )
        + "\n"
    ).encode("utf-8")


def rewrite_keeping_mtime(testcase: unittest.TestCase, path: Path, new: bytes) -> None:
    """Replace the bytes (any size) in place and restore the previous mtime_ns."""
    before = path.stat()
    testcase.assertNotEqual(path.read_bytes(), new)
    with open(path, "r+b") as handle:
        handle.write(new)
        handle.truncate()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    testcase.assertEqual(path.stat().st_mtime_ns, before.st_mtime_ns)


def collide(testcase: unittest.TestCase, path: Path, new: bytes) -> None:
    """Different bytes, same size, same mtime_ns: the old fingerprint's blind spot."""
    before = path.stat()
    testcase.assertEqual(len(new), before.st_size, "a collision needs the same size")
    rewrite_keeping_mtime(testcase, path, new)
    after = path.stat()
    testcase.assertEqual(
        (after.st_mtime_ns, after.st_size),
        (before.st_mtime_ns, before.st_size),
        "precondition: (st_mtime_ns, st_size) collides",
    )


def flip_byte(data: bytes, index: int = -1) -> bytes:
    index %= len(data)
    return data[:index] + bytes([data[index] ^ 0x01]) + data[index + 1 :]


def text_mode_bytes(text: str, linesep: str) -> bytes:
    """What a text-mode UTF-8 write produces where the platform line separator is ``linesep``."""
    buffer = io.BytesIO()
    handle = io.TextIOWrapper(buffer, encoding="utf-8", newline=linesep)
    handle.write(text)
    handle.flush()
    data = buffer.getvalue()
    handle.detach()
    return data


class _FireOnClose:
    """File-object proxy that runs ``fire`` once, right after the real file is closed."""

    def __init__(self, raw: Any, fire: Callable[[], None]) -> None:
        self._raw = raw
        self._fire = fire
        self._done = False

    def _finish(self) -> None:
        if not self._done:
            self._done = True
            self._fire()

    def __enter__(self) -> "_FireOnClose":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._raw.__exit__(*exc)
        self._finish()

    def close(self) -> None:
        self._raw.close()
        self._finish()

    def __iter__(self) -> Iterator[Any]:
        return iter(self._raw)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def _is_write_mode(mode: str) -> bool:
    return bool(set(mode) & set("wax+"))


@contextlib.contextmanager
def change_after_content_read(
    testcase: unittest.TestCase, target: Path, new: bytes
) -> Iterator[dict[str, bool]]:
    """As soon as the tool's first read-mode handle on ``target`` is closed (whatever
    route opened it: read_bytes, read_text or Path.open), the file is changed on disk
    (same size, same mtime) before the tool continues. Models a writer landing between
    a tool's content read and its mark_file_read; a recorder that reads the file again
    for its mark would record the new bytes."""
    target = target.resolve()
    state = {"fired": False}
    orig_open = pathlib.Path.open

    def fire() -> None:
        if not state["fired"]:
            state["fired"] = True
            collide(testcase, target, new)

    def fake_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = orig_open(self, mode, *args, **kwargs)
        if not state["fired"] and not _is_write_mode(mode) and self.resolve() == target:
            return _FireOnClose(handle, fire)
        return handle

    with patch.object(pathlib.Path, "open", fake_open):
        yield state
    testcase.assertTrue(state["fired"], "injection did not fire")


@contextlib.contextmanager
def change_after_own_write(
    testcase: unittest.TestCase, target: Path, new_for: Callable[[bytes], bytes]
) -> Iterator[dict[str, bool]]:
    """Right after the tool's own write of ``target`` completes (a write-mode handle
    is closed, or a temp file is os.replace()d onto it), the file is changed on disk
    (same size, same mtime). A recorder that reads the file back for its mark would
    record the new bytes instead of the ones it wrote."""
    target = target.resolve()
    state = {"fired": False}
    orig_open = pathlib.Path.open
    orig_replace = os.replace

    def fire() -> None:
        if not state["fired"]:
            state["fired"] = True
            collide(testcase, target, new_for(target.read_bytes()))

    def fake_open(self: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = orig_open(self, mode, *args, **kwargs)
        if not state["fired"] and _is_write_mode(mode) and self.resolve() == target:
            return _FireOnClose(handle, fire)
        return handle

    def fake_replace(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        orig_replace(src, dst, *args, **kwargs)
        if Path(dst).resolve() == target:
            fire()

    real_builtin_open = builtins.open

    def fake_builtin_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        handle = real_builtin_open(file, mode, *args, **kwargs)
        if not state["fired"] and _is_write_mode(mode) and Path(file).resolve() == target:
            return _FireOnClose(handle, fire)
        return handle

    with patch.object(pathlib.Path, "open", fake_open), patch.object(
        os, "replace", fake_replace
    ), patch.object(compare_write_module, "open", fake_builtin_open, create=True):
        yield state
    testcase.assertTrue(state["fired"], "injection did not fire")


@contextlib.contextmanager
def change_before_mark(
    testcase: unittest.TestCase,
    ctx: ToolContext,
    target: Path,
    new_for: Callable[[bytes], bytes],
) -> Iterator[dict[str, bool]]:
    """Just before mark_file_read runs for ``target``, change the file on disk (same
    size, same mtime). Models a writer landing between a tool's write and its mark."""
    target = target.resolve()
    original = ctx.mark_file_read
    state = {"fired": False}

    def wrapped(path: Path, *args: Any, **kwargs: Any) -> None:
        if not state["fired"] and Path(path).resolve() == target:
            state["fired"] = True
            collide(testcase, target, new_for(target.read_bytes()))
        original(path, *args, **kwargs)

    ctx.mark_file_read = wrapped  # type: ignore[method-assign]
    try:
        yield state
    finally:
        del ctx.mark_file_read
    testcase.assertTrue(state["fired"], "injection did not fire")


class _ReadSpy:
    """Wraps the gate's file object: records each read's length; can fail reads, or
    answer None (what a non-blocking raw read returns when it would block)."""

    def __init__(self, raw: Any, log: list[int], fail_reads: bool, none_after: int | None) -> None:
        self._raw = raw
        self._log = log
        self._fail_reads = fail_reads
        self._none_after = none_after
        self._reads = 0

    def __enter__(self) -> "_ReadSpy":
        return self

    def __exit__(self, *exc: object) -> None:
        self._raw.close()

    def fileno(self) -> int:
        return int(self._raw.fileno())

    def read(self, size: int = -1) -> Any:
        if self._fail_reads:
            raise OSError(5, "simulated I/O error")
        self._reads += 1
        if self._none_after is not None and self._reads > self._none_after:
            return None
        chunk = self._raw.read(size)
        self._log.append(len(chunk or b""))
        return chunk

    def close(self) -> None:
        self._raw.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def spy_open(
    log: list[int], *, fail_reads: bool = False, none_after: int | None = None
) -> Callable[..., _ReadSpy]:
    real_open = builtins.open

    def fake_open(*args: Any, **kwargs: Any) -> _ReadSpy:
        return _ReadSpy(real_open(*args, **kwargs), log, fail_reads, none_after)

    return fake_open


class _Base(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def put(self, name: str, data: bytes) -> Path:
        path = self.root / name
        path.write_bytes(data)
        return path

    def read(self, path: Path, **extra: Any) -> dict[str, Any]:
        result = FileReadTool().run({"file_path": str(path), **extra}, self.ctx)
        self.assertFalse(result.is_error, result.output)
        return result.output

    def gate(self, path: Path) -> bool:
        return self.ctx.was_file_read_and_unchanged(path)


class TestFingerprintPrimitive(_Base):
    def test_never_read_is_not_unchanged(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.assertFalse(self.gate(path))

    def test_unchanged_after_read_is_unchanged(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.read(path)
        self.assertTrue(self.gate(path))

    def test_unchanged_after_mark_is_unchanged(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.ctx.mark_file_read(path, data=b"AAAA\n")
        self.assertTrue(self.gate(path))

    def test_same_size_same_mtime_different_bytes_after_read_is_changed(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.read(path)
        collide(self, path, b"BBBB\n")
        self.assertFalse(self.gate(path))

    def test_same_size_same_mtime_different_bytes_after_mark_is_changed(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.ctx.mark_file_read(path, data=b"AAAA\n")
        collide(self, path, b"BBBB\n")
        self.assertFalse(self.gate(path))

    def test_size_change_is_changed(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.read(path)
        rewrite_keeping_mtime(self, path, b"AAAA\nB")
        self.assertFalse(self.gate(path))

    def test_exact_byte_differences_are_changed(self) -> None:
        # Byte equality, not text equality: none of these is normalized away.
        pairs = {
            "crlf_lf_swap_same_size": (b"a\r\nb\n", b"a\nb\r\n"),
            "lf_to_crlf": (b"a\nb\n", b"a\r\nb\r\n"),
            "trailing_cr_vs_lf": (b"x\r", b"x\n"),
            "invalid_bytes_decoding_alike": (b"caf\xe9\n", b"caf\xff\n"),
            "bom_added": (b"x\n", b"\xef\xbb\xbfx\n"),
            "bom_removed": (b"\xef\xbb\xbfx\n", b"x\n"),
            "nfc_vs_nfd": ("é\n".encode("utf-8"), "é\n".encode("utf-8")),
            "utf8_vs_latin1": ("é\n".encode("utf-8"), "é\n".encode("latin-1")),
            "trailing_newline_removed": (b"x\n", b"x"),
        }
        for name, (old, new) in pairs.items():
            with self.subTest(case=name):
                path = self.put(f"{name}.txt", old)
                self.read(path)
                self.assertTrue(self.gate(path))
                rewrite_keeping_mtime(self, path, new)
                self.assertFalse(self.gate(path))

    def test_touch_with_identical_bytes_is_unchanged(self) -> None:
        # Exact-bytes semantics: a timestamp alone never decides either way.
        path = self.put("a.txt", b"AAAA\n")
        self.read(path)
        before = path.stat()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 5_000_000_000))
        self.assertTrue(self.gate(path))

    def test_empty_file_equality_and_change(self) -> None:
        path = self.put("empty.txt", b"")
        self.ctx.mark_file_read(path, data=b"")
        self.assertTrue(self.gate(path))
        rewrite_keeping_mtime(self, path, b"X")
        self.assertFalse(self.gate(path))
        other = self.put("was-x.txt", b"X")
        self.ctx.mark_file_read(other, data=b"X")
        rewrite_keeping_mtime(self, other, b"")
        self.assertFalse(self.gate(other))

    def test_record_is_full_sha256_of_exact_bytes(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.ctx.mark_file_read(path, data=b"AAAA\n")
        (record,) = self.ctx.read_file_fingerprints.values()
        self.assertEqual(tuple(record), (5, hashlib.sha256(b"AAAA\n").digest()))
        self.assertEqual(len(record[1]), 32)

    def test_mark_requires_exact_bytes(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        with self.assertRaises(TypeError):
            self.ctx.mark_file_read(path)  # type: ignore[call-arg]
        for wrong in ("AAAA\n", bytearray(b"AAAA\n"), memoryview(b"AAAA\n"), None):
            with self.subTest(data_type=type(wrong).__name__):
                with self.assertRaises(TypeError):
                    self.ctx.mark_file_read(path, data=wrong)  # type: ignore[arg-type]
        self.assertEqual(self.ctx.read_file_fingerprints, {})
        self.assertFalse(self.gate(path))

    def test_supplied_bytes_not_current_disk_bytes_determine_the_record(self) -> None:
        path = self.put("a.txt", b"BBBB\n")
        self.ctx.mark_file_read(path, data=b"AAAA\n")
        self.assertFalse(self.gate(path))
        self.ctx.mark_file_read(path, data=b"BBBB\n")
        self.assertTrue(self.gate(path))

    def test_mark_never_reads_the_file(self) -> None:
        missing = self.root / "not-there.txt"
        self.ctx.mark_file_read(missing, data=b"AAAA\n")
        self.assertIn(missing.resolve(), self.ctx.read_file_fingerprints)
        self.assertFalse(self.gate(missing))

    def test_binary_mode_crlf_and_ctrl_z(self) -> None:
        # A text-mode read would translate newlines; a raw read on a C-runtime text-mode
        # descriptor (os.open without O_BINARY on Windows) would also stop at Ctrl-Z (0x1A).
        for name, data in {
            "crlf": b"a\r\nb\r\n",
            "ctrl_z": b"ab\x1acd\r\n",
            "ctrl_z_first": b"\x1aEVIL\n",
        }.items():
            with self.subTest(case=name):
                path = self.put(f"{name}.bin", data)
                self.ctx.mark_file_read(path, data=data)
                self.assertTrue(self.gate(path))
                collide(self, path, flip_byte(data))
                self.assertFalse(self.gate(path))

    def test_multi_chunk_equality_and_changes(self) -> None:
        with patch.object(context_module, "_FINGERPRINT_CHUNK_BYTES", 8):
            for size in (7, 8, 9, 16, 17, 40):
                data = bytes((index * 7 + 3) % 251 for index in range(size))
                for index in (0, size // 2, size - 1):
                    with self.subTest(size=size, flipped=index):
                        path = self.put(f"chunks-{size}-{index}.bin", data)
                        self.ctx.mark_file_read(path, data=data)
                        log: list[int] = []
                        with patch.object(context_module, "open", spy_open(log), create=True):
                            self.assertTrue(self.gate(path))
                        self.assertTrue(log, "the check must read the file")
                        self.assertLessEqual(max(log), 8, "reads are chunked")
                        self.assertEqual(sum(log), size)
                        if size > 8:
                            self.assertGreater(sum(1 for n in log if n), 1)
                        collide(self, path, flip_byte(data, index))
                        self.assertFalse(self.gate(path))

    def test_real_multi_mebibyte_file(self) -> None:
        size = 2 * 1024 * 1024 + 512 * 1024
        data = random.Random(20260927).randbytes(size)
        for index in (0, 1024 * 1024 - 1, 1024 * 1024, 2 * 1024 * 1024, size - 1):
            with self.subTest(flipped=index):
                path = self.put(f"big-{index}.bin", data)
                self.ctx.mark_file_read(path, data=data)
                self.assertTrue(self.gate(path))
                collide(self, path, flip_byte(data, index))
                self.assertFalse(self.gate(path))

    def test_grown_file_is_rejected_by_size_without_hashing(self) -> None:
        path = self.put("log.txt", b"AAAA\n")
        self.ctx.mark_file_read(path, data=b"AAAA\n")
        rewrite_keeping_mtime(self, path, b"AAAA\n" + b"X" * (2 * 1024 * 1024))
        log: list[int] = []
        with patch.object(context_module, "open", spy_open(log), create=True):
            self.assertFalse(self.gate(path))
        self.assertEqual(sum(log), 0, "a size mismatch must reject before reading content")

    def test_growth_after_the_size_check_is_detected_with_a_bounded_read(self) -> None:
        path = self.put("grow.txt", b"AAAA\n")
        self.ctx.mark_file_read(path, data=b"AAAA\n")
        real_fstat = os.fstat
        grown = {"done": False}

        def fstat_then_grow(fd: int) -> os.stat_result:
            info = real_fstat(fd)
            if not grown["done"]:
                grown["done"] = True
                with open(path, "ab") as handle:
                    handle.write(b"X" * (3 * 1024 * 1024))
            return info

        log: list[int] = []
        with patch.object(os, "fstat", fstat_then_grow), patch.object(
            context_module, "open", spy_open(log), create=True
        ):
            self.assertFalse(self.gate(path))
        self.assertTrue(grown["done"], "the check must use fstat on the opened file")
        self.assertLessEqual(sum(log), 6, "read at most recorded size + 1 bytes")


class TestFailClosed(_Base):
    def _recorded(self, name: str = "a.txt", data: bytes = b"AAAA\n") -> Path:
        path = self.put(name, data)
        self.read(path)
        return path

    def test_missing_file_is_changed_without_raising(self) -> None:
        path = self._recorded()
        path.unlink()
        self.assertFalse(self.gate(path))
        self.assertIn(path.resolve(), self.ctx.read_file_fingerprints)  # no eviction

    def test_directory_in_place_of_file_is_changed(self) -> None:
        path = self._recorded("a.txt", b"")
        before = path.stat()
        path.unlink()
        path.mkdir()
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertFalse(self.gate(path))

    def test_open_errors_are_changed(self) -> None:
        path = self._recorded()
        for error in (
            PermissionError(13, "denied"),
            OSError(5, "I/O error"),
            ValueError("bad path"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch.object(context_module, "open", side_effect=error, create=True):
                    self.assertFalse(self.gate(path))

    def test_mid_read_error_is_changed(self) -> None:
        path = self._recorded()
        with patch.object(context_module, "open", spy_open([], fail_reads=True), create=True):
            self.assertFalse(self.gate(path))

    def test_would_block_read_is_not_a_proven_eof(self) -> None:
        # With O_NONBLOCK (POSIX) a raw read may return None instead of bytes or b"".
        cases = {
            "none_at_the_eof_probe": (b"AAAA\n", 1, None),
            "empty_file": (b"", 0, None),
            "mid_file": (b"AAAA\n", 1, 2),
        }
        for name, (data, none_after, chunk_bytes) in cases.items():
            with self.subTest(case=name):
                path = self.put(f"{name}.txt", data)
                self.ctx.mark_file_read(path, data=data)
                self.assertTrue(self.gate(path))
                with contextlib.ExitStack() as stack:
                    stack.enter_context(
                        patch.object(context_module, "open", spy_open([], none_after=none_after), create=True)
                    )
                    if chunk_bytes is not None:
                        stack.enter_context(patch.object(context_module, "_FINGERPRINT_CHUNK_BYTES", chunk_bytes))
                    self.assertFalse(self.gate(path))

    def test_non_regular_object_is_changed(self) -> None:
        path = self._recorded()
        size = path.stat().st_size
        for mode in (stat.S_IFIFO, stat.S_IFCHR, stat.S_IFDIR, stat.S_IFSOCK):
            with self.subTest(mode=oct(mode)):
                fake = os.stat_result((mode | 0o644, 0, 0, 1, 0, 0, size, 0, 0, 0))
                with patch.object(os, "fstat", return_value=fake):
                    self.assertFalse(self.gate(path))

    def test_path_resolution_errors_are_changed(self) -> None:
        path = self._recorded()
        for error in (RuntimeError("Symlink loop"), ValueError("embedded null"), OSError(5, "x")):
            with self.subTest(error=type(error).__name__):
                with patch.object(pathlib.Path, "resolve", side_effect=error):
                    self.assertFalse(self.gate(path))

    def test_embedded_nul_path_is_changed(self) -> None:
        self._recorded()
        self.assertFalse(self.gate(self.root / "a\x00.txt"))

    def test_interpreter_errors_propagate_and_never_permit(self) -> None:
        path = self._recorded()
        with patch.object(context_module, "open", side_effect=MemoryError, create=True):
            with self.assertRaises(MemoryError):
                self.gate(path)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "needs os.mkfifo (POSIX)")
    def test_fifo_in_place_of_file_is_changed_without_blocking(self) -> None:
        path = self._recorded()
        path.unlink()
        os.mkfifo(path)
        result: list[bool] = []
        worker = threading.Thread(
            target=lambda: result.append(self.gate(path)), daemon=True
        )
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive(), "the check blocked on a FIFO")
        self.assertEqual(result, [False])

    @unittest.skipUnless(sys.platform == "win32", "Windows share-mode lock")
    def test_exclusively_locked_file_is_changed_and_never_stubbed(self) -> None:
        import _winapi

        path = self._recorded("locked.txt", b"key=old\n")
        generic_read, open_existing, normal = 0x80000000, 3, 0x80
        handle = _winapi.CreateFile(str(path), generic_read, 0, 0, open_existing, normal, 0)
        try:
            self.assertFalse(self.gate(path))
            with self.assertRaises(ToolExecutionError):
                FileReadTool().run({"file_path": str(path)}, self.ctx)
            with self.assertRaises(ToolInputError):
                FileEditTool().run(
                    {"file_path": str(path), "old_string": "old", "new_string": "new"},
                    self.ctx,
                )
        finally:
            _winapi.CloseHandle(handle)
        self.assertEqual(path.read_bytes(), b"key=old\n")
        self.assertTrue(self.gate(path))


class TestRecordingSites(_Base):
    """Each recorder records the bytes it used, a later same-size/same-mtime change is
    detected, and a change landing right after the recorder's own read or write (before
    its mark) is detected, so the recorder did not read the file again for its mark."""

    def _read_sites(
        self,
    ) -> list[tuple[str, str, bytes, Callable[[Path], Any], Callable[[Any], bool]]]:
        """(site, file name, bytes, run(path), saw_the_original(result))."""
        return [
            ("read_text", "a.txt", b"AAAA\nline2\n", lambda p: self.read(p),
             lambda r: r["file"]["content"] == "1\tAAAA\n2\tline2"),
            ("read_image", "a.png", _PNG, lambda p: self.read(p),
             lambda r: base64.b64decode(r["file"]["base64"]) == _PNG),
            ("read_pdf", "a.pdf", _PDF, lambda p: self.read(p),
             lambda r: base64.b64decode(r["file"]["base64"]) == _PDF),
            ("read_notebook", "a.ipynb", _notebook("aaaa"), lambda p: self.read(p),
             lambda r: r["file"]["cells"][0]["source"] == ["aaaa"]),
            ("data_inspect", "a.csv", b"id,v\n1,a\n",
             lambda p: DataInspectTool().run({"file_path": str(p)}, self.ctx),
             lambda r: r.output["rows"] == [{"id": "1", "v": "a"}]),
            ("qwen_prepare_media", "q.png", _PNG, lambda p: _prepare_media(str(p), None, self.ctx),
             lambda r: base64.b64decode(r[1].split(",", 1)[1]) == _PNG),
        ]

    def test_read_sites_record_and_detect_change(self) -> None:
        for name, fname, data, run, _saw in self._read_sites():
            with self.subTest(site=name):
                path = self.put(fname, data)
                run(path)
                self.assertTrue(self.gate(path))
                collide(self, path, flip_byte(data))
                self.assertFalse(self.gate(path))

    def test_read_sites_bind_the_bytes_they_read(self) -> None:
        for name, fname, data, run, saw_the_original in self._read_sites():
            with self.subTest(site=name):
                path = self.put("race-" + fname, data)
                with change_after_content_read(self, path, flip_byte(data)):
                    result = run(path)
                self.assertTrue(saw_the_original(result), "the tool must show what it read")
                self.assertEqual(path.read_bytes(), flip_byte(data))
                self.assertFalse(self.gate(path))

    def _write_sites(
        self,
    ) -> list[tuple[str, str, Callable[[Path], None], Callable[[Path], None]]]:
        def nothing(p: Path) -> None:
            return None

        def write(p: Path) -> None:
            FileWriteTool().run({"file_path": str(p), "content": "v=1\nw=2\n"}, self.ctx)

        def prepare_text(p: Path) -> None:
            p.write_bytes(b"v=1\nw=2\n")
            self.read(p)

        def edit(p: Path) -> None:
            FileEditTool().run(
                {"file_path": str(p), "old_string": "v=1", "new_string": "v=9"}, self.ctx
            )

        def prepare_notebook(p: Path) -> None:
            p.write_bytes(_notebook("aaaa"))
            self.read(p)

        def notebook_edit(p: Path) -> None:
            NotebookEditTool().run(
                {"notebook_path": str(p), "cell_id": "md-1", "new_source": "bbbb"}, self.ctx
            )

        def prepare_source(p: Path) -> None:
            source = self.put("src-" + p.stem + ".csv", b"id,v\n1,a\n2,b\n")
            DataInspectTool().run({"file_path": str(source)}, self.ctx)

        def transform_target(p: Path) -> None:
            source = self.root / ("src-" + p.stem + ".csv")
            DataTransformTool().run({"input_path": str(source), "output_path": str(p)}, self.ctx)

        return [
            ("write_create", "w.txt", nothing, write),
            ("write_overwrite", "wo.txt", prepare_text, write),
            ("edit", "e.txt", prepare_text, edit),
            ("notebook_edit", "n.ipynb", prepare_notebook, notebook_edit),
            ("data_transform_target", "t.csv", prepare_source, transform_target),
        ]

    def test_write_sites_record_written_bytes_and_detect_change(self) -> None:
        for name, fname, prepare, act in self._write_sites():
            with self.subTest(site=name):
                path = self.root / fname
                prepare(path)
                act(path)
                self.assertTrue(self.gate(path))
                collide(self, path, flip_byte(path.read_bytes()))
                self.assertFalse(self.gate(path))

    def test_write_sites_bind_the_bytes_they_wrote(self) -> None:
        for name, fname, prepare, act in self._write_sites():
            with self.subTest(site=name):
                path = self.root / ("race-" + fname)
                prepare(path)
                with change_before_mark(self, self.ctx, path, flip_byte):
                    act(path)
                self.assertFalse(self.gate(path))

    def test_write_sites_never_read_back_what_they_wrote(self) -> None:
        # The change lands right after the tool's own OS write, before any read-back.
        for name, fname, prepare, act in self._write_sites():
            with self.subTest(site=name):
                path = self.root / ("readback-" + fname)
                prepare(path)
                with change_after_own_write(self, path, flip_byte):
                    act(path)
                self.assertFalse(self.gate(path))

    def test_rejected_reads_are_not_recorded(self) -> None:
        big = 5 * 1024 * 1024 + 1
        cases: list[tuple[str, str, bytes, Callable[[Path], object]]] = [
            ("oversize_image", "big.png", _PNG + b"\0" * big, lambda p: FileReadTool().run({"file_path": str(p)}, self.ctx)),
            ("oversize_pdf", "big.pdf", _PDF + b"\0" * big, lambda p: FileReadTool().run({"file_path": str(p)}, self.ctx)),
            ("pdf_pages", "p.pdf", _PDF, lambda p: FileReadTool().run({"file_path": str(p), "pages": "1"}, self.ctx)),
            ("notebook_invalid_json", "bad.ipynb", b"{not json", lambda p: FileReadTool().run({"file_path": str(p)}, self.ctx)),
            ("notebook_without_cells", "nocells.ipynb", b'{"metadata": {}}', lambda p: FileReadTool().run({"file_path": str(p)}, self.ctx)),
        ]
        for name, fname, data, run in cases:
            with self.subTest(case=name):
                path = self.put(fname, data)
                result = run(path)
                self.assertTrue(getattr(result, "is_error", False), result)
                self.assertFalse(self.gate(path))
        raising: list[tuple[str, str, bytes, Callable[[Path], object]]] = [
            ("data_invalid_json", "bad.json", b"[1,", lambda p: DataInspectTool().run({"file_path": str(p)}, self.ctx)),
            ("data_invalid_utf8", "bad.csv", b"id\n\xff\n", lambda p: DataInspectTool().run({"file_path": str(p)}, self.ctx)),
            ("qwen_not_an_image", "fake.png", b"not an image at all", lambda p: _prepare_media(str(p), None, self.ctx)),
        ]
        for name, fname, data, run in raising:
            with self.subTest(case=name):
                path = self.put(fname, data)
                with self.assertRaises(ToolInputError):
                    run(path)
                self.assertFalse(self.gate(path))


class TestGateConsumers(_Base):
    def test_read_returns_new_content_not_stub_after_collision(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.assertEqual(self.read(path)["type"], "text")
        self.assertEqual(self.read(path)["type"], "file_unchanged")
        collide(self, path, b"BBBB\n")
        out = self.read(path)
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["file"]["content"], "1\tBBBB")

    def test_write_refuses_after_collision(self) -> None:
        path = self.put("a.txt", b"AAAA\n")
        self.read(path)
        collide(self, path, b"BBBB\n")
        with self.assertRaises(ToolInputError):
            FileWriteTool().run({"file_path": str(path), "content": "CCCC\n"}, self.ctx)
        self.assertEqual(path.read_bytes(), b"BBBB\n")

    def test_edit_refuses_after_collision(self) -> None:
        path = self.put("a.txt", b"key=old\nAAAA\n")
        self.read(path)
        collide(self, path, b"key=old\nBBBB\n")
        with self.assertRaises(ToolInputError):
            FileEditTool().run(
                {"file_path": str(path), "old_string": "key=old", "new_string": "key=new"},
                self.ctx,
            )
        self.assertEqual(path.read_bytes(), b"key=old\nBBBB\n")

    def test_notebook_edit_refuses_after_collision(self) -> None:
        path = self.put("a.ipynb", _notebook("aaaa"))
        self.read(path)
        collide(self, path, _notebook("bbbb"))
        with self.assertRaises(ToolInputError):
            NotebookEditTool().run(
                {"notebook_path": str(path), "cell_id": "md-1", "new_source": "x"}, self.ctx
            )
        self.assertEqual(path.read_bytes(), _notebook("bbbb"))

    def test_data_transform_refuses_after_collision(self) -> None:
        for recorder in ("DataInspect", "Read"):
            with self.subTest(recorded_by=recorder):
                source = self.put(f"s-{recorder}.csv", b"id,v\n1,a\n")
                target = self.root / f"o-{recorder}.json"
                if recorder == "Read":
                    self.read(source)
                else:
                    DataInspectTool().run({"file_path": str(source)}, self.ctx)
                collide(self, source, b"id,v\n1,b\n")
                with self.assertRaises(ToolInputError):
                    DataTransformTool().run(
                        {"input_path": str(source), "output_path": str(target)}, self.ctx
                    )
                self.assertFalse(target.exists())
                self.assertEqual(source.read_bytes(), b"id,v\n1,b\n")

    def test_normal_chains_still_work(self) -> None:
        path = self.put("a.txt", b"key=old\n")
        self.read(path)
        for old, new in (("key=old", "key=mid"), ("key=mid", "key=new")):
            FileEditTool().run(
                {"file_path": str(path), "old_string": old, "new_string": new}, self.ctx
            )
        FileWriteTool().run({"file_path": str(path), "content": "w\n"}, self.ctx)
        FileEditTool().run({"file_path": str(path), "old_string": "w", "new_string": "z"}, self.ctx)
        self.assertEqual(self.read(path)["type"], "file_unchanged")
        self.assertEqual(path.read_text(encoding="utf-8"), "z\n")
        empty = self.put("empty.txt", b"")
        self.read(empty)
        FileWriteTool().run({"file_path": str(empty), "content": "now\n"}, self.ctx)
        notebook = self.put("n.ipynb", _notebook("aaaa"))
        self.read(notebook)
        for text in ("b1", "b2"):
            NotebookEditTool().run(
                {"notebook_path": str(notebook), "cell_id": "md-1", "new_source": text}, self.ctx
            )
        self.assertEqual(json.loads(notebook.read_bytes())["cells"][0]["source"], ["b2"])
        source = self.put("s.csv", b"id,v\n1,a\n")
        DataInspectTool().run({"file_path": str(source)}, self.ctx)
        target = self.root / "o.json"
        DataTransformTool().run({"input_path": str(source), "output_path": str(target)}, self.ctx)
        self.assertEqual(self.read(target)["type"], "file_unchanged")

    def test_ranged_read_after_full_read_returns_the_requested_lines(self) -> None:
        path = self.put("a.txt", b"one\ntwo\nthree\n")
        self.read(path)
        out = self.read(path, offset=2, limit=1)
        self.assertEqual(out["type"], "text")
        self.assertEqual(out["file"]["content"], "2\ttwo")
        # Explicitly passing the default values is still a ranged request, never stubbed.
        for extra in ({"offset": 1}, {"limit": 2000}, {"offset": 1, "limit": 2000}):
            with self.subTest(**{k: str(v) for k, v in extra.items()}):
                self.assertEqual(self.read(path, **extra)["type"], "text")

    def test_repeated_full_reads_of_media_and_notebooks_are_never_stubbed(self) -> None:
        for fname, data, kind in (
            ("a.png", _PNG, "image"),
            ("a.pdf", _PDF, "pdf"),
            ("a.ipynb", _notebook("aaaa"), "notebook"),
        ):
            with self.subTest(kind=kind):
                path = self.put(fname, data)
                self.assertEqual(self.read(path)["type"], kind)
                self.assertTrue(self.gate(path))
                self.assertEqual(self.read(path)["type"], kind)

    def test_data_transform_never_overwrites_a_target_created_after_its_checks(self) -> None:
        import src.tool_system.tools.data as data_module

        source = self.put("s.csv", b"id,v\n1,a\n")
        target = self.root / "late.json"
        DataInspectTool().run({"file_path": str(source)}, self.ctx)
        real_render = data_module._render_rows

        def render_then_race(*args: Any, **kwargs: Any) -> str:
            target.write_bytes(b"created by another process\n")
            return real_render(*args, **kwargs)

        with patch.object(data_module, "_render_rows", render_then_race):
            with self.assertRaises(ToolInputError):
                DataTransformTool().run(
                    {"input_path": str(source), "output_path": str(target)}, self.ctx
                )
        self.assertEqual(target.read_bytes(), b"created by another process\n")
        self.assertFalse(self.gate(target))

    def test_ranged_reads_never_evaluate_the_unchanged_gate(self) -> None:
        path = self.put("a.txt", b"one\ntwo\nthree\n")
        calls: list[Path] = []
        original = self.ctx.was_file_read_and_unchanged

        def counting(p: Path) -> bool:
            calls.append(p)
            return original(p)

        self.ctx.was_file_read_and_unchanged = counting  # type: ignore[method-assign]
        try:
            for extra in (
                {"offset": 2}, {"limit": 1}, {"offset": 1, "limit": 5}, {"pages": "1"},
                {"offset": 1}, {"limit": 2000}, {"offset": 1, "limit": 2000},
            ):
                with self.subTest(**{k: str(v) for k, v in extra.items()}):
                    calls.clear()
                    self.assertEqual(self.read(path, **extra)["type"], "text")
                    self.assertEqual(calls, [])
            calls.clear()
            self.read(path)
            self.assertEqual(len(calls), 1)
        finally:
            del self.ctx.was_file_read_and_unchanged

    def test_natural_immediate_same_size_rewrites_are_changed(self) -> None:
        # Supplementary: no forced timestamps. The deterministic proofs are above.
        for index in range(50):
            path = self.put(f"n{index}.txt", b"AAAA\n")
            self.read(path)
            path.write_bytes(b"BBBB\n")
            self.assertFalse(self.gate(path), f"iteration {index}")


class TestByteEquivalence(_Base):
    """Write/Edit/NotebookEdit/DataTransform put the same bytes on disk as before the
    fingerprint change, and Read shows the model the same text."""

    WRITE_CONTENTS = {
        "lf": "a\nb\n",
        "crlf_in_content": "a\r\nb\r\n",
        "mixed": "a\r\nb\nc\rd\n",
        "no_final_newline": "no final newline",
        "empty": "",
        "unicode": "ünïcødé ✓\n",
    }
    EDIT_INPUTS = {
        "lf": b"x=1\ny=2\n",
        "crlf": b"x=1\r\ny=2\r\n",
        "mixed": b"x=1\r\ny=2\nz=3\r",
        "lone_cr": b"x=1\ry=2",
        "no_final_newline": b"x=1",
        "unicode": "x=1 ünïcødé\n".encode("utf-8"),
        "invalid_utf8": b"x=1\xe9\n",
        "bom": b"\xef\xbb\xbfx=1\n",
    }

    def test_write_matches_text_mode_output_for_both_line_separators(self) -> None:
        for linesep in ("\r\n", "\n"):
            for name, content in self.WRITE_CONTENTS.items():
                with self.subTest(linesep=repr(linesep), content=name), patch.object(
                    os, "linesep", linesep
                ):
                    path = self.root / f"w-{len(linesep)}-{name}.txt"
                    FileWriteTool().run({"file_path": str(path), "content": content}, self.ctx)
                    self.assertEqual(path.read_bytes(), text_mode_bytes(content, linesep))
                    self.assertTrue(self.gate(path))

    def test_write_matches_real_write_text_on_this_host(self) -> None:
        for name, content in self.WRITE_CONTENTS.items():
            with self.subTest(content=name):
                reference = self.root / f"ref-{name}.txt"
                reference.write_text(content, encoding="utf-8")
                path = self.root / f"w-{name}.txt"
                FileWriteTool().run({"file_path": str(path), "content": content}, self.ctx)
                self.assertEqual(path.read_bytes(), reference.read_bytes())

    # new_string values; the CR ones keep Edit's newline quirk (\r\n -> \r\r\n on Windows).
    EDIT_NEW_STRINGS = {"plain": "x=9", "crlf": "k=1\r\nm=2", "lone_cr": "k=1\rm=2"}

    def test_edit_matches_vector14_output_for_both_line_separators(self) -> None:
        for linesep in ("\r\n", "\n"):
            for name, raw in self.EDIT_INPUTS.items():
                for new_name, new in self.EDIT_NEW_STRINGS.items():
                    with self.subTest(linesep=repr(linesep), content=name, new=new_name), patch.object(
                        os, "linesep", linesep
                    ):
                        path = self.put(f"e-{len(linesep)}-{name}-{new_name}.txt", raw)
                        before = path.read_text(encoding="utf-8", errors="replace")
                        self.read(path)
                        FileEditTool().run(
                            {"file_path": str(path), "old_string": "x=1", "new_string": new},
                            self.ctx,
                        )
                        expected = text_mode_bytes(before.replace("x=1", new, 1), linesep)
                        self.assertEqual(path.read_bytes(), expected)
                        self.assertTrue(self.gate(path))

    def test_edit_matches_real_write_text_on_this_host(self) -> None:
        for name, raw in self.EDIT_INPUTS.items():
            for new_name, new in self.EDIT_NEW_STRINGS.items():
                with self.subTest(content=name, new=new_name):
                    path = self.put(f"e-host-{name}-{new_name}.txt", raw)
                    reference = self.root / f"e-ref-{name}-{new_name}.txt"
                    reference.write_text(
                        path.read_text(encoding="utf-8", errors="replace").replace("x=1", new, 1),
                        encoding="utf-8",
                    )
                    self.read(path)
                    FileEditTool().run(
                        {"file_path": str(path), "old_string": "x=1", "new_string": new}, self.ctx
                    )
                    self.assertEqual(path.read_bytes(), reference.read_bytes())

    def test_unencodable_text_fails_before_anything_is_touched(self) -> None:
        for surrogate in ("\ud800", "\udc80", "\udcff"):
            with self.subTest(surrogate=hex(ord(surrogate))):
                path = self.put(f"precious-{ord(surrogate):x}.txt", b"precious\n")
                self.read(path)
                with self.assertRaises(UnicodeEncodeError):
                    FileWriteTool().run({"file_path": str(path), "content": f"x{surrogate}\n"}, self.ctx)
                self.assertEqual(path.read_bytes(), b"precious\n")
                with self.assertRaises(UnicodeEncodeError):
                    FileEditTool().run(
                        {"file_path": str(path), "old_string": "precious", "new_string": surrogate},
                        self.ctx,
                    )
                self.assertEqual(path.read_bytes(), b"precious\n")
                self.assertTrue(self.gate(path))
                created = self.root / f"new-dir-{ord(surrogate):x}" / "f.txt"
                with self.assertRaises(UnicodeEncodeError):
                    FileWriteTool().run({"file_path": str(created), "content": surrogate}, self.ctx)
                self.assertFalse(created.parent.exists())
                folder = self.root / f"nb-{ord(surrogate):x}"
                folder.mkdir()
                notebook = folder / "n.ipynb"
                notebook.write_bytes(_notebook("aaaa"))
                self.read(notebook)
                with self.assertRaises(UnicodeEncodeError):
                    NotebookEditTool().run(
                        {"notebook_path": str(notebook), "cell_id": "md-1", "new_source": surrogate},
                        self.ctx,
                    )
                self.assertEqual(notebook.read_bytes(), _notebook("aaaa"))
                self.assertEqual(sorted(p.name for p in folder.iterdir()), ["n.ipynb"])

    def test_notebook_edit_keeps_the_vector14_serialization(self) -> None:
        original = {
            "cells": [
                {"cell_type": "markdown", "id": "md-1", "metadata": {}, "source": ["héllo"]},
                {
                    "cell_type": "code",
                    "execution_count": 3,
                    "id": "c-1",
                    "metadata": {},
                    "outputs": [],
                    "source": ["print('✓')\n", "x = 1"],
                },
            ],
            "metadata": {"kernelspec": {"name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        raw = json.dumps(original, ensure_ascii=False, indent=4).replace("\n", "\r\n").encode("utf-8")
        path = self.put("nb.ipynb", raw)
        self.read(path)
        NotebookEditTool().run(
            {"notebook_path": str(path), "cell_id": "md-1", "new_source": "wörld\nline 2"},
            self.ctx,
        )
        expected = json.loads(raw)
        expected["cells"][0]["source"] = ["wörld\n", "line 2"]
        want = (json.dumps(expected, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        self.assertEqual(path.read_bytes(), want)
        self.assertNotIn(b"\r", path.read_bytes())
        self.assertTrue(self.gate(path))

    def test_data_transform_output_bytes_are_unchanged(self) -> None:
        source = self.put("s.csv", "id,v\r\n1,é\r\n2,b\r\n".encode("utf-8"))
        DataInspectTool().run({"file_path": str(source)}, self.ctx)
        rows = [{"id": "1", "v": "é"}, {"id": "2", "v": "b"}]
        expected = {
            "csv": "id,v\n1,é\n2,b\n".encode("utf-8"),
            "tsv": "id\tv\n1\té\n2\tb\n".encode("utf-8"),
            "json": (json.dumps(rows, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
            "jsonl": "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
            ).encode("utf-8"),
        }
        for fmt, want in expected.items():
            with self.subTest(format=fmt):
                target = self.root / f"out.{fmt}"
                DataTransformTool().run(
                    {"input_path": str(source), "output_path": str(target)}, self.ctx
                )
                self.assertEqual(target.read_bytes(), want)
                self.assertTrue(self.gate(target))

    def test_data_inspect_output_is_unchanged(self) -> None:
        raw = "﻿id,v\r\n1,\"a\r\nb\"\r\n2,é\r\n".encode("utf-8")
        path = self.put("d.csv", raw)
        out = DataInspectTool().run({"file_path": str(path)}, self.ctx).output
        self.assertEqual(out["sizeBytes"], len(raw))
        self.assertEqual(out["rows"], [{"id": "1", "v": "a\r\nb"}, {"id": "2", "v": "é"}])
        self.assertEqual([column["name"] for column in out["columns"]], ["id", "v"])
        self.assertEqual(out["rowCount"], 2)

    def test_data_file_grown_after_its_size_check_is_rejected_without_a_full_read(self) -> None:
        import src.tool_system.tools.data as data_module

        path = self.put("grown.csv", b"id\n" + b"1\n" * 50)
        lengths: list[int] = []
        orig_open = pathlib.Path.open

        def spying_open(self_: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = orig_open(self_, mode, *args, **kwargs)
            if "b" in mode and self_.resolve() == path.resolve():
                return _ReadSpy(handle, lengths, False, None)
            return handle

        with patch.object(data_module, "_MAX_INPUT_BYTES", 16), patch.object(
            data_module, "_validate_file_size", lambda p: 0
        ), patch.object(pathlib.Path, "open", spying_open):
            with self.assertRaises(ToolInputError):
                DataInspectTool().run({"file_path": str(path)}, self.ctx)
        self.assertEqual(lengths, [17], "read at most the limit + 1 bytes")
        self.assertFalse(self.gate(path))

    def test_data_loader_decoding_and_error_precedence_are_unchanged(self) -> None:
        # The loaders now parse the one in-memory buffer; results and error texts
        # (including which error wins when two sit in different 8 KiB chunks) are as before.
        rows_ok = {
            "tsv": ("d.tsv", b"id\tv\n1\ta\n", [{"id": "1", "v": "a"}]),
            "jsonl_u2028_nel": (
                "d.jsonl",
                '{"a":"x y"}\n{"a":"p\x85q"}\n'.encode("utf-8"),
                [{"a": "x y"}, {"a": "p\x85q"}],
            ),
        }
        for name, (fname, raw, rows) in rows_ok.items():
            with self.subTest(case=name):
                out = DataInspectTool().run({"file_path": str(self.put(fname, raw))}, self.ctx).output
                self.assertEqual(out["rows"], rows)
        csv_error_first = (
            b"a,b\n" + b"1,2\n" * 1000 + b"1,2,3\n" + b"3,4\n" * 2000 + b"\xff,2\n"
        )
        utf8_error_first = b"a,b\n\xff,2\n" + b"1,2\n" * 3000 + b"1,2,3\n"
        jsonl_error_first = (
            b'{"a":1}\n' * 100 + b"{oops\n" + b'{"a":2}\n' * 1200 + b'{"a":"\xff"}\n'
        )
        failing = {
            "json_bom": ("b.json", b'\xef\xbb\xbf[{"a":1}]', "invalid JSON: Unexpected UTF-8 BOM"),
            "json_utf16": ("u.json", '[{"a":1}]'.encode("utf-16"), "data file is not valid UTF-8 text"),
            "json_crlf_offsets": ("c.json", b'[\r\n {"id": 1,\r\n', "invalid JSON: Expecting property name"),
            "csv_structure_then_utf8": ("s.csv", csv_error_first, "row 1002 has 3 field(s); expected 2"),
            "csv_utf8_then_structure": ("t.csv", utf8_error_first, "data file is not valid UTF-8 text"),
            "jsonl_line_then_utf8": ("l.jsonl", jsonl_error_first, "invalid JSONL at line 101:"),
        }
        for name, (fname, raw, message) in failing.items():
            with self.subTest(case=name):
                path = self.put(fname, raw)
                with self.assertRaises(ToolInputError) as caught:
                    DataInspectTool().run({"file_path": str(path)}, self.ctx)
                self.assertTrue(str(caught.exception).startswith(message), str(caught.exception))
                if name == "json_crlf_offsets":  # offsets count translated newlines
                    self.assertTrue(str(caught.exception).endswith("(char 13)"), str(caught.exception))
                self.assertFalse(self.gate(path))

    def test_text_read_view_matches_vector14(self) -> None:
        samples = {
            "lf": b"a\nb\n",
            "crlf": b"a\r\nb\r\n",
            "lone_cr": b"a\rb\rc",
            "mixed": b"a\r\nb\rc\nd",
            "bom": b"\xef\xbb\xbfbom\n",
            "invalid_utf8": b"bad\xff\xfe\xc3\n",
            "cr_before_buffer_boundary": b"x" * 8191 + b"\r\ny\r\n",
            "cr_at_buffer_boundary": b"x" * 8192 + b"\r\ny",
            "split_utf8_at_boundary": b"x" * 8191 + "é\n".encode("utf-8"),
            "trailing_cr": b"tail\r",
            "form_feed_and_line_separator": b"\x0cff\xe2\x80\xa8ls\n",
            "empty": b"",
        }
        for name, raw in samples.items():
            with self.subTest(case=name):
                path = self.put(f"v-{name}.txt", raw)
                out = self.read(path, limit=2000)
                text = path.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
                self.assertEqual(out["file"]["totalLines"], len(lines))
                self.assertEqual(out["file"]["numLines"], len(lines))
                self.assertEqual(
                    out["file"]["content"],
                    "\n".join(f"{number + 1}\t{line}" for number, line in enumerate(lines)),
                )

    def test_decoding_matches_path_read_text(self) -> None:
        decode = read_module._decode_like_read_text
        pieces = [b"a", b"\r", b"\n", b"\r\n", b"\xff", b"\xc3", b"\xa9", b"\xef\xbb\xbf",
                  b"\xe2\x80\xa8", b"\x00", b"\x1a", b"\x0c", "é".encode("utf-8")]
        rng = random.Random(8192)
        for index in range(200):
            length = rng.choice((0, 1, 2, 5, 50, 8190, 8193, 20000))
            raw = b"".join(rng.choice(pieces) for _ in range(length))
            with self.subTest(sample=index, length=len(raw)):
                path = self.put("fuzz.bin", raw)
                self.assertEqual(decode(raw), path.read_text(encoding="utf-8", errors="replace"))

    def test_notebook_read_view_matches_vector14(self) -> None:
        raw = (
            json.dumps(
                {"cells": [{"cell_type": "markdown", "id": "m", "metadata": {}, "source": ["hé\r\n", "x"]}]},
                ensure_ascii=False,
                indent=2,
            )
            .replace("\n", "\r\n")
            .encode("utf-8")
        )
        path = self.put("view.ipynb", raw)
        out = self.read(path)
        self.assertEqual(
            out["file"]["cells"], json.loads(path.read_text(encoding="utf-8", errors="replace"))["cells"]
        )


REFUSE_EDIT = "refusing to edit: file must be read first and unchanged since last read"
REFUSE_OVERWRITE = "refusing to overwrite: file must be read first and unchanged since last read"
REFUSE_NOTEBOOK = "refusing to edit notebook: file must be read first and unchanged since last read"


@contextlib.contextmanager
def after_verified_read(
    testcase: unittest.TestCase, ctx: ToolContext, action: Callable[[], None]
) -> Iterator[dict[str, bool]]:
    """Right after the gate has read and verified the file (read_if_unchanged returned
    its bytes), run ``action`` once: a change landing before the tool builds its edit."""
    original = ctx.read_if_unchanged
    state = {"fired": False}

    def wrapped(path: Path) -> bytes | None:
        result = original(path)
        if result is not None and not state["fired"]:
            state["fired"] = True
            action()
        return result

    ctx.read_if_unchanged = wrapped  # type: ignore[method-assign]
    try:
        yield state
    finally:
        del ctx.read_if_unchanged
    testcase.assertTrue(state["fired"], "injection did not fire")


@contextlib.contextmanager
def around_final_open(
    testcase: unittest.TestCase,
    target: Path,
    modes: tuple[str, ...],
    *,
    before: Callable[[], None] | None = None,
    wrap: Callable[[Any], Any] | None = None,
) -> Iterator[dict[str, bool]]:
    """Hook compare_write's own open() of ``target`` (after the tool verified the file
    and built its change): run ``before`` right before that open, and/or wrap the
    handle it returns."""
    resolved = target.resolve()
    real_open = builtins.open
    state = {"fired": False}

    def fake_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        hit = not state["fired"] and mode in modes and Path(file).resolve() == resolved
        if hit:
            state["fired"] = True
            if before is not None:
                before()
        handle = real_open(file, mode, *args, **kwargs)
        return wrap(handle) if hit and wrap is not None else handle

    with patch.object(compare_write_module, "open", fake_open, create=True):
        yield state
    testcase.assertTrue(state["fired"], "injection did not fire")


@contextlib.contextmanager
def after_final_compare(testcase: unittest.TestCase, action: Callable[[], None]) -> Iterator[dict[str, bool]]:
    """Right after the final comparison succeeded, before the write or replace: the
    residual window no lock-free design closes."""
    real = compare_write_module._read_matching
    state = {"fired": False}

    def wrapped(handle: Any, recorded: Any, *, keep: bool) -> bytes | None:
        result = real(handle, recorded, keep=keep)
        if result is not None and not state["fired"]:
            state["fired"] = True
            action()
        return result

    with patch.object(compare_write_module, "_read_matching", wrapped):
        yield state
    testcase.assertTrue(state["fired"], "injection did not fire")


class _HandleProxy:
    """Wraps compare_write's handle; subclasses override write/truncate."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def __enter__(self) -> "_HandleProxy":
        return self

    def __exit__(self, *exc: Any) -> None:
        self._raw.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._raw, name)


def external_rewrite(path: Path, new: bytes) -> None:
    """Another writer changes the file in place (its own handle, default sharing)."""
    with open(path, "r+b") as handle:
        handle.write(new)
        handle.truncate()


# Every open() of any path (builtins.open, io.open, os.open) while armed: proves that a
# failed write is followed by no re-open of the target (no repair through another handle).
_ARMED_OPENS: list[str] | None = None


def _audit_opens(event: str, args: tuple[Any, ...]) -> None:
    if _ARMED_OPENS is not None and event == "open" and args and not isinstance(args[0], int):
        try:
            _ARMED_OPENS.append(os.fsdecode(args[0]))
        except (TypeError, ValueError):  # an audit hook must never break the audited open
            _ARMED_OPENS.append(repr(args[0]))


sys.addaudithook(_audit_opens)


def _arm_open_audit() -> list[str]:
    global _ARMED_OPENS
    _ARMED_OPENS = []
    return _ARMED_OPENS


def _disarm_open_audit() -> None:
    global _ARMED_OPENS
    _ARMED_OPENS = None


class _Faulty(_HandleProxy):
    """compare_write's handle with one injected failure. ``shape`` "counted": the first
    write lands ``k`` bytes and reports them, the next write raises ``error``; "leak": the
    first write lands ``k`` bytes and then raises; "truncate": truncate raises; "zero":
    after ``k`` bytes (0 = at once) writes report 0, and a retry of that is an error of its
    own. From the failure on, every write/truncate/seek/read/readinto/readall/readline/
    flush/tell/fileno call on the handle is recorded (``after``), and so is every open()
    in the process (``opens_after``, filtered to the test directory by the caller)."""

    def __init__(
        self,
        raw: Any,
        *,
        shape: str,
        k: int,
        error: BaseException,
        on_failure: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(raw)
        self.shape, self.k, self.error, self.on_failure = shape, k, error, on_failure
        self.writes = 0
        self.zero_returns = 0
        self.failed = False
        self.after: list[str] = []
        self.opens_after: list[str] | None = None

    def _fail(self) -> BaseException:
        self.failed = True
        if self.on_failure is not None:
            self.on_failure()
        self.opens_after = _arm_open_audit()
        return self.error

    def _note(self, name: str) -> None:
        if self.failed:
            self.after.append(name)

    def write(self, data: Any) -> Any:
        self._note("write")
        self.writes += 1
        view = bytes(data)
        if self.shape == "zero":
            if self.writes == 1 and self.k:
                return self._raw.write(view[: self.k])
            self.zero_returns += 1
            if self.zero_returns > 3:
                raise RuntimeError("a 0-byte write was retried")
            if self.zero_returns == 1:
                self.failed = True
                self.opens_after = _arm_open_audit()
            return 0
        if self.writes == 1 and self.shape in ("counted", "leak"):
            written = self._raw.write(view[: self.k])
            if self.shape == "leak":
                raise self._fail()
            return written
        if self.writes == 2 and self.shape == "counted":
            raise self._fail()
        return self._raw.write(data)

    def truncate(self, *args: Any) -> Any:
        self._note("truncate")
        if self.shape == "truncate" and not self.failed:
            raise self._fail()
        return self._raw.truncate(*args)

    def seek(self, *args: Any) -> Any:
        self._note("seek")
        return self._raw.seek(*args)

    def read(self, *args: Any) -> Any:
        self._note("read")
        return self._raw.read(*args)

    def readinto(self, *args: Any) -> Any:
        self._note("readinto")
        return self._raw.readinto(*args)

    def readall(self) -> Any:
        self._note("readall")
        return self._raw.readall()

    def readline(self, *args: Any) -> Any:
        self._note("readline")
        return self._raw.readline(*args)

    def flush(self) -> Any:
        self._note("flush")
        return self._raw.flush()

    def tell(self) -> Any:
        self._note("tell")
        return self._raw.tell()

    def fileno(self) -> Any:
        self._note("fileno")
        return self._raw.fileno()


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _K32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_void_p),
            ("InternalHigh", ctypes.c_void_p),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _K32.CreateFileW.restype = wintypes.HANDLE
    _K32.CreateFileW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    _K32.LockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
        ctypes.POINTER(_OVERLAPPED),
    ]
    _K32.UnlockFileEx.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(_OVERLAPPED),
    ]
    _K32.CloseHandle.argtypes = [wintypes.HANDLE]
    _K32.SetFileAttributesW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD]
    _INVALID_HANDLE = wintypes.HANDLE(-1).value


@contextlib.contextmanager
def _win_handle(path: Path, access: int, share: int) -> Iterator[Any]:
    """Another process's handle on ``path`` (GENERIC_READ = 0x80000000; share 0 = none)."""
    handle = _K32.CreateFileW(str(path), access, share, None, 3, 0x80, None)  # OPEN_EXISTING
    if handle in (None, _INVALID_HANDLE):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield handle
    finally:
        _K32.CloseHandle(handle)


@contextlib.contextmanager
def _win_shared_lock(path: Path, offset: int) -> Iterator[None]:
    """A shared byte-range lock from ``offset`` to the end, held by another handle."""
    with _win_handle(path, 0x80000000, 0x1 | 0x2) as handle:
        overlapped = _OVERLAPPED()
        overlapped.Offset = offset
        if not _K32.LockFileEx(handle, 0, 0, 0xFFFFFFFF, 0x7FFFFFFF, ctypes.byref(overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            yield
        finally:
            _K32.UnlockFileEx(handle, 0, 0xFFFFFFFF, 0x7FFFFFFF, ctypes.byref(overlapped))


def _win_set_attributes(path: Path, attributes: int) -> None:
    if not _K32.SetFileAttributesW(str(path), attributes):
        raise ctypes.WinError(ctypes.get_last_error())


class TestCompareAndWrite(_Base):
    """The mutation is built from the exact verified bytes and lands only while the
    file still holds them; a file is created only where nothing exists."""

    ORIGINAL = b"head\nkey=old\ntail\n"

    def _mutators(self) -> list[tuple[str, str, Callable[[Path], bytes], Callable[[Path], Any], str]]:
        def text(p: Path) -> bytes:
            p.write_bytes(self.ORIGINAL)
            self.read(p)
            return self.ORIGINAL + b"EXTERNAL\n"

        def notebook(p: Path) -> bytes:
            p.write_bytes(_notebook("aaaa"))
            self.read(p)
            return _notebook("EXTERNAL")

        def edit(p: Path) -> Any:
            return FileEditTool().run(
                {"file_path": str(p), "old_string": "key=old", "new_string": "key=new"}, self.ctx
            )

        def write(p: Path) -> Any:
            return FileWriteTool().run({"file_path": str(p), "content": "written\n"}, self.ctx)

        def notebook_edit(p: Path) -> Any:
            return NotebookEditTool().run(
                {"notebook_path": str(p), "cell_id": "md-1", "new_source": "bbbb"}, self.ctx
            )

        return [
            ("edit", "e.txt", text, edit, REFUSE_EDIT),
            ("write_overwrite", "w.txt", text, write, REFUSE_OVERWRITE),
            ("notebook_edit", "n.ipynb", notebook, notebook_edit, REFUSE_NOTEBOOK),
        ]

    def _no_leftovers(self, folder: Path) -> None:
        self.assertEqual([p.name for p in folder.iterdir() if p.name.endswith(".tmp")], [])

    def _refused(self, path: Path, act: Callable[[Path], Any], refusal: str) -> None:
        with self.assertRaises(ToolInputError) as caught:
            act(path)
        self.assertEqual(str(caught.exception), refusal)

    def test_change_after_the_verified_read_is_refused_not_absorbed(self) -> None:
        # W1: the edit is built from the verified bytes, never from a second read.
        for name, fname, prepare, act, refusal in self._mutators():
            with self.subTest(site=name):
                folder = self.root / f"w1-{name}"
                folder.mkdir()
                path = folder / fname
                external = prepare(path)
                with after_verified_read(self, self.ctx, lambda: path.write_bytes(external)):
                    self._refused(path, act, refusal)
                self.assertEqual(path.read_bytes(), external)
                self.assertFalse(self.gate(path))
                self._no_leftovers(folder)

    def test_change_after_the_edit_was_built_is_refused(self) -> None:
        # W2: re-verified immediately before the write (Write/Edit: through the write
        # handle; NotebookEdit: after its temp file is written, before os.replace).
        for name, fname, prepare, act, refusal in self._mutators():
            with self.subTest(site=name):
                folder = self.root / f"w2-{name}"
                folder.mkdir()
                path = folder / fname
                external = prepare(path)
                with around_final_open(self, path, ("r+b", "rb"), before=lambda: path.write_bytes(external)):
                    self._refused(path, act, refusal)
                self.assertEqual(path.read_bytes(), external)
                self.assertFalse(self.gate(path))
                self._no_leftovers(folder)

    def test_notebook_change_while_its_temp_file_is_written_is_refused(self) -> None:
        folder = self.root / "nb-temp"
        folder.mkdir()
        path = folder / "n.ipynb"
        path.write_bytes(_notebook("aaaa"))
        self.read(path)
        real = tempfile.NamedTemporaryFile
        state = {"fired": False}

        @contextlib.contextmanager
        def temp_then_change(*args: Any, **kwargs: Any) -> Iterator[Any]:
            with real(*args, **kwargs) as handle:
                yield handle
            state["fired"] = True
            path.write_bytes(_notebook("EXTERNAL"))

        with patch.object(tempfile, "NamedTemporaryFile", temp_then_change):
            self._refused(
                path,
                lambda p: NotebookEditTool().run(
                    {"notebook_path": str(p), "cell_id": "md-1", "new_source": "bbbb"}, self.ctx
                ),
                REFUSE_NOTEBOOK,
            )
        self.assertTrue(state["fired"])
        self.assertEqual(path.read_bytes(), _notebook("EXTERNAL"))
        self._no_leftovers(folder)

    def test_deleted_after_the_verified_read_is_refused_and_never_recreated(self) -> None:
        for name, fname, prepare, act, refusal in self._mutators():
            for when in ("after_gate", "before_final_compare"):
                with self.subTest(site=name, when=when):
                    folder = self.root / f"del-{name}-{when}"
                    folder.mkdir()
                    path = folder / fname
                    prepare(path)
                    if when == "after_gate":
                        hook = after_verified_read(self, self.ctx, path.unlink)
                    else:
                        hook = around_final_open(self, path, ("r+b", "rb"), before=path.unlink)
                    with hook:
                        self._refused(path, act, refusal)
                    self.assertFalse(path.exists())
                    self._no_leftovers(folder)

    def _capture_marks(self) -> list[bytes]:
        marked: list[bytes] = []
        original_mark = self.ctx.mark_file_read

        def capture(p: Path, *, data: bytes) -> None:
            marked.append(data)
            original_mark(p, data=data)

        self.ctx.mark_file_read = capture  # type: ignore[method-assign]
        self.addCleanup(lambda: self.ctx.__dict__.pop("mark_file_read", None))
        return marked

    def test_residual_after_the_final_compare_never_yields_a_false_unchanged(self) -> None:
        # Documents the window no lock-free design closes: a writer landing between the
        # final comparison and the write/replace. A change inside the tool's bytes is lost;
        # one reaching past them survives as a torn tail unless the tool shortens or
        # replaces the file. Either way the gate says "unchanged" only if the disk holds
        # exactly the bytes the tool wrote and recorded.
        outcomes = {}
        for name, fname, prepare, act, _refusal in self._mutators():
            for label in ("inside", "past_the_end"):
                with self.subTest(site=name, external=label):
                    path = self.root / f"residual-{label}-{fname}"
                    grown = prepare(path)
                    original = path.read_bytes()
                    # "inside": same length, one byte changed within what the tool rewrites.
                    external = flip_byte(original, 1) if label == "inside" else grown + b"X" * 64
                    marked = self._capture_marks()
                    with after_final_compare(self, lambda: external_rewrite(path, external)):
                        act(path)
                    disk = path.read_bytes()
                    self.assertNotEqual(disk, external)
                    self.assertEqual(self.gate(path), disk == marked[-1])
                    outcomes[(name, label)] = "lost" if disk == marked[-1] else "torn"
                    self.ctx.__dict__.pop("mark_file_read", None)
        for name in ("edit", "write_overwrite", "notebook_edit"):
            self.assertEqual(outcomes[(name, "inside")], "lost")
        # Write here shortens the file and NotebookEdit replaces it: a late extension is
        # lost too. Edit's bytes are not shorter than the verified ones, so nothing is
        # truncated and the late extension survives as a torn tail.
        self.assertEqual(outcomes[("write_overwrite", "past_the_end")], "lost")
        self.assertEqual(outcomes[("notebook_edit", "past_the_end")], "lost")
        self.assertEqual(outcomes[("edit", "past_the_end")], "torn")

    def test_equal_length_write_keeps_a_late_extension_as_a_torn_tail(self) -> None:
        # Truncation happens only when the new bytes are shorter: an equal-length Write
        # (no newline, so the same bytes on every platform) leaves a late extension alone.
        path = self.put("equal.txt", self.ORIGINAL)
        self.read(path)
        content = "E" * len(self.ORIGINAL)
        external = self.ORIGINAL + b"LATE-EXTENSION\n"
        with after_final_compare(self, lambda: external_rewrite(path, external)):
            FileWriteTool().run({"file_path": str(path), "content": content}, self.ctx)
        self.assertEqual(path.read_bytes(), content.encode() + b"LATE-EXTENSION\n")
        self.assertFalse(self.gate(path))

    def test_longer_edit_keeps_a_late_extension_past_its_bytes_as_a_torn_tail(self) -> None:
        # Longer new bytes on every platform: nothing is truncated, so a late extension
        # reaching past them survives behind them (torn), and the gate says changed.
        path = self.put("longer-torn.txt", self.ORIGINAL)
        self.read(path)
        data = self._edit_bytes("key=newer")
        external = self.ORIGINAL + b"X" * 64
        with after_final_compare(self, lambda: external_rewrite(path, external)):
            FileEditTool().run({"file_path": str(path), "old_string": "key=old", "new_string": "key=newer"}, self.ctx)
        self.assertEqual(path.read_bytes(), data + external[len(data):])
        self.assertFalse(self.gate(path))

    def test_change_while_the_final_comparison_reads_is_the_documented_residual(self) -> None:
        # The residual starts while the final comparison reads the file: a change to a
        # range it has already read is not seen. With small chunks, bytes 0-3 change right
        # after the first chunk was read. It is lost (the tool rewrites or replaces them),
        # and the gate still says "unchanged" only for exactly the recorded bytes.
        for name, fname, prepare, act, _refusal in self._mutators():
            with self.subTest(site=name):
                path = self.root / f"during-{fname}"
                prepare(path)
                original = path.read_bytes()
                external = b"ZZZZ" + original[4:]
                reads = {"n": 0}

                class ChangeAfterFirstChunk(_HandleProxy):
                    def read(self, *args: Any) -> Any:
                        chunk = self._raw.read(*args)
                        reads["n"] += 1
                        if reads["n"] == 1:
                            with open(path, "r+b") as other:
                                other.write(b"ZZZZ")
                        return chunk

                marked = self._capture_marks()
                with patch.object(context_module, "_FINGERPRINT_CHUNK_BYTES", 4):
                    with around_final_open(self, path, ("r+b", "rb"), wrap=ChangeAfterFirstChunk):
                        act(path)
                self.assertGreater(reads["n"], 1)
                disk = path.read_bytes()
                self.assertNotEqual(disk, external)
                self.assertEqual(disk, marked[-1])
                self.assertTrue(self.gate(path))
                self.ctx.__dict__.pop("mark_file_read", None)

    def test_rename_over_the_file_during_compare_and_write(self) -> None:
        # Windows: the compare-and-write handle blocks rename/replace/delete of the file
        # until the write is done. POSIX: the rename wins; the tool's bytes land in the
        # replaced inode, never at the path, and the gate reports the change.
        path = self.put("swap.txt", self.ORIGINAL)
        self.read(path)
        other = self.put("other.txt", b"renamed over by another process\n")
        outcome: list[str] = []

        def swap() -> None:
            try:
                os.replace(other, path)
                outcome.append("replaced")
            except PermissionError:
                outcome.append("blocked")

        with after_final_compare(self, swap):
            FileEditTool().run(
                {"file_path": str(path), "old_string": "key=old", "new_string": "key=new"}, self.ctx
            )
        if sys.platform == "win32":
            self.assertEqual(outcome, ["blocked"])
            self.assertTrue(other.exists())
            self.assertTrue(self.gate(path))
        else:
            self.assertEqual(outcome, ["replaced"])
            self.assertEqual(path.read_bytes(), b"renamed over by another process\n")
            self.assertFalse(self.gate(path))

    def test_write_never_overwrites_a_file_created_after_its_exists_check(self) -> None:
        # W5: exclusive create; the late file is kept, refused like an unread file.
        path = self.root / "late.txt"
        with around_final_open(self, path, ("xb",), before=lambda: path.write_bytes(b"created by another process\n")):
            self._refused(path, lambda p: FileWriteTool().run({"file_path": str(p), "content": "mine\n"}, self.ctx), REFUSE_OVERWRITE)
        self.assertEqual(path.read_bytes(), b"created by another process\n")
        self.assertNotIn(path.resolve(), self.ctx.read_file_fingerprints)
        self.assertFalse(self.gate(path))

    def test_two_sessions_creating_the_same_file(self) -> None:
        path = self.root / "both.txt"
        other_session = ToolContext(workspace_root=self.root)

        def other_creates() -> None:
            FileWriteTool().run({"file_path": str(path), "content": "theirs\n"}, other_session)

        with around_final_open(self, path, ("xb",), before=other_creates):
            self._refused(path, lambda p: FileWriteTool().run({"file_path": str(p), "content": "mine\n"}, self.ctx), REFUSE_OVERWRITE)
        self.assertEqual(path.read_bytes(), text_mode_bytes("theirs\n", os.linesep))
        self.assertTrue(other_session.was_file_read_and_unchanged(path))
        self.assertFalse(self.gate(path))

    def test_directory_created_after_the_exists_check_is_never_written(self) -> None:
        path = self.root / "late-dir"
        with around_final_open(self, path, ("xb",), before=path.mkdir):
            with self.assertRaises((ToolInputError, PermissionError, IsADirectoryError)):
                FileWriteTool().run({"file_path": str(path), "content": "mine\n"}, self.ctx)
        self.assertTrue(path.is_dir())
        self.assertFalse(self.gate(path))

    def test_in_place_write_never_passes_through_an_empty_file(self) -> None:
        # Write before shortening: another reader never sees b"" (truncate-first 'wb' did).
        for label, new in (("longer", "key=newer"), ("same", "key=new"), ("shorter", "k")):
            with self.subTest(case=label):
                path = self.put(f"view-{label}.txt", self.ORIGINAL)
                self.read(path)
                seen: list[bytes] = []

                class Spy(_HandleProxy):
                    def write(self, data: Any) -> Any:
                        written = self._raw.write(data)
                        seen.append(path.read_bytes())
                        return written

                with around_final_open(self, path, ("r+b",), wrap=Spy):
                    FileEditTool().run(
                        {"file_path": str(path), "old_string": "key=old", "new_string": new}, self.ctx
                    )
                final = path.read_bytes()
                self.assertTrue(seen and all(view for view in seen))
                self.assertEqual(seen[0][: len(final)], final)
                self.assertEqual(len(seen[0]), max(len(final), len(self.ORIGINAL)))
                self.assertTrue(self.gate(path))

    def _edit_bytes(self, new: str) -> bytes:
        text = self.ORIGINAL.decode().replace("key=old", new)
        return text.replace("\n", os.linesep).encode()

    def _fail_during_write(
        self, path: Path, act: Callable[[], Any], proxy_args: dict[str, Any], *, mode: str = "r+b"
    ) -> tuple[BaseException, _Faulty, list[str], list[bytes]]:
        marked = self._capture_marks()
        proxies: list[_Faulty] = []

        def make(raw: Any) -> _Faulty:
            proxies.append(_Faulty(raw, **proxy_args))
            return proxies[-1]

        caught: BaseException | None = None
        try:
            with around_final_open(self, path, (mode,), wrap=make):
                act()
        except BaseException as exc:  # compared by identity below
            caught = exc
        finally:
            _disarm_open_audit()
        self.ctx.__dict__.pop("mark_file_read", None)
        self.assertIsNotNone(caught, "the injected failure did not propagate")
        assert caught is not None
        (proxy,) = proxies
        root = str(self.root).lower()
        opens = [o for o in proxy.opens_after or [] if os.path.abspath(o).lower().startswith(root)]
        return caught, proxy, opens, marked

    def test_failure_after_the_write_began_is_raised_and_never_repaired(self) -> None:
        # No restorative write, retry, truncate, read-back or re-open after a failed
        # in-place write: the original error propagates, nothing is marked, and the file
        # keeps whatever the failed write left. The gate then says "unchanged" only if
        # those bytes are still exactly the recorded ones (here they never are: every
        # failure point lies past the first byte the new text changes, on CRLF and LF).
        shorter, longer = "k", "key=" + "n" * 40
        past = len(self.ORIGINAL) + 5
        enospc = lambda: OSError(errno.ENOSPC, "simulated: no space left on device")  # noqa: E731
        cases = (
            ("counted_shortening", shorter, "counted", 8),
            ("leak_shortening", shorter, "leak", 8),
            ("counted_extending", longer, "counted", 12),
            ("counted_past_old_length", longer, "counted", past),
            ("leak_past_old_length", longer, "leak", past),
            ("truncate_refused", shorter, "truncate", 0),
            ("external_writer_at_failure", shorter, "counted", 8),
        )
        for label, new, shape, k in cases:
            with self.subTest(case=label):
                path = self.put(f"fail-{label}.txt", self.ORIGINAL)
                self.read(path)
                data = self._edit_bytes(new)
                if k:  # the failure point lies past the first changed byte
                    self.assertNotEqual(data[:k], self.ORIGINAL[:k])
                if shape == "truncate":
                    error: BaseException = PermissionError(13, "simulated: user-mapped section open")
                    predicted = data + self.ORIGINAL[len(data):]
                else:
                    error = enospc()
                    predicted = data[:k] + self.ORIGINAL[k:] if k < len(self.ORIGINAL) else data[:k]
                external = b"EXTERNAL WRITER AT THE FAILURE POINT\n"
                on_failure = (
                    (lambda: external_rewrite(path, external))
                    if label == "external_writer_at_failure"
                    else None
                )
                caught, proxy, opens_after, marked = self._fail_during_write(
                    path,
                    lambda: FileEditTool().run(
                        {"file_path": str(path), "old_string": "key=old", "new_string": new}, self.ctx
                    ),
                    {"shape": shape, "k": k, "error": error, "on_failure": on_failure},
                )
                self.assertIs(caught, error)
                self.assertEqual(proxy.after, [])
                self.assertEqual(opens_after, [])
                self.assertEqual(marked, [])
                disk = path.read_bytes()
                self.assertEqual(disk, external if on_failure else predicted)
                self.assertFalse(self.gate(path))

    def test_a_zero_byte_write_is_an_error_not_a_retry(self) -> None:
        # A write that reports 0 bytes ends the write with an OSError (never a spin);
        # nothing is repaired afterwards.
        content = "ZZZZZZZZ\n"  # differs from the original at byte 0 on every platform
        data = text_mode_bytes(content, os.linesep)
        for label, k in (("first_call", 0), ("after_5_bytes", 5)):
            with self.subTest(case=label):
                path = self.put(f"zero-{label}.txt", self.ORIGINAL)
                self.read(path)
                caught, proxy, opens_after, marked = self._fail_during_write(
                    path,
                    lambda: FileWriteTool().run({"file_path": str(path), "content": content}, self.ctx),
                    {"shape": "zero", "k": k, "error": OSError("unused")},
                )
                self.assertIs(type(caught), OSError)
                self.assertIn("short write", str(caught))
                self.assertEqual(proxy.zero_returns, 1)
                self.assertEqual(proxy.after, [])
                self.assertEqual(opens_after, [])
                self.assertEqual(marked, [])
                expected = self.ORIGINAL if k == 0 else data[:k] + self.ORIGINAL[k:]
                self.assertEqual(path.read_bytes(), expected)
                self.assertEqual(self.gate(path), k == 0)

    def test_a_failure_inside_the_unchanged_prefix_leaves_the_file_unchanged(self) -> None:
        # The first 3 bytes of the new text equal the old ones (CRLF and LF): a failure
        # there changes nothing, so the gate correctly still says "unchanged".
        path = self.put("prefix.txt", self.ORIGINAL)
        self.read(path)
        error = OSError(errno.ENOSPC, "simulated: no space left on device")
        caught, proxy, opens_after, marked = self._fail_during_write(
            path,
            lambda: FileEditTool().run(
                {"file_path": str(path), "old_string": "key=old", "new_string": "key=new"}, self.ctx
            ),
            {"shape": "counted", "k": 3, "error": error},
        )
        self.assertIs(caught, error)
        self.assertEqual((proxy.after, opens_after, marked), ([], [], []))
        self.assertEqual(path.read_bytes(), self.ORIGINAL)
        self.assertTrue(self.gate(path))

    def test_deleted_parent_directory_is_refused_and_never_recreated(self) -> None:
        for name, fname, prepare, act, refusal in self._mutators():
            with self.subTest(site=name):
                folder = self.root / f"parent-{name}"
                folder.mkdir()
                path = folder / fname
                prepare(path)
                with after_verified_read(self, self.ctx, lambda: shutil.rmtree(folder)):
                    if name == "notebook_edit":  # its temporary file cannot be created
                        with self.assertRaises(ToolExecutionError):
                            act(path)
                    else:
                        self._refused(path, act, refusal)
                self.assertFalse(folder.exists())

    def test_past_errors_file_appearing_before_the_create_is_never_overwritten(self) -> None:
        # Permission for a create is decided while the path is absent (no confirmation
        # for new PAST-ERRORS history); the exclusive create enforces exactly that.
        from src.tool_system.defaults import build_default_registry
        from src.tool_system.protocol import ToolCall

        path = self.root / "PAST-ERRORS" / "log.txt"
        path.parent.mkdir()
        asks: list[str] = []
        self.ctx.permission_handler = lambda tool, message, suggestion: (asks.append(message), (False, False))[1]
        registry = build_default_registry(include_user_tools=False)
        with around_final_open(self, path, ("xb",), before=lambda: path.write_bytes(b"precious history\n")):
            with self.assertRaises(ToolInputError) as caught:
                registry.dispatch(
                    ToolCall(name="Write", input={"file_path": str(path), "content": "overwritten\n"}), self.ctx
                )
        self.assertEqual(str(caught.exception), REFUSE_OVERWRITE)
        self.assertEqual(asks, [])
        self.assertEqual(path.read_bytes(), b"precious history\n")
        self.assertNotIn(path.resolve(), self.ctx.read_file_fingerprints)

    def test_failed_create_is_raised_left_as_written_and_never_marked(self) -> None:
        # The create path has no repair either (as today): a failure leaves the partial new
        # file as written, raises, and records nothing.
        content = "created content\n"
        data = text_mode_bytes(content, os.linesep)
        for shape, k in (("counted", 5), ("leak", 5), ("zero", 0), ("zero", 5)):
            with self.subTest(shape=shape, k=k):
                path = self.root / f"create-fail-{shape}-{k}.txt"
                error = OSError(errno.ENOSPC, "simulated: no space left on device")
                caught, proxy, opens_after, marked = self._fail_during_write(
                    path,
                    lambda p=path: FileWriteTool().run({"file_path": str(p), "content": content}, self.ctx),
                    {"shape": shape, "k": k, "error": error},
                    mode="xb",
                )
                if shape == "zero":
                    self.assertIs(type(caught), OSError)
                    self.assertIn("short write", str(caught))
                else:
                    self.assertIs(caught, error)
                self.assertEqual((proxy.after, opens_after, marked), ([], [], []))
                self.assertEqual(path.read_bytes(), data[:k])
                self.assertNotIn(path.resolve(), self.ctx.read_file_fingerprints)
                self.assertFalse(self.gate(path))

    def _change_at_end_of_file(self, path: Path, changed: bytes) -> tuple[Callable[..., Any], dict[str, bool]]:
        """A context_module.open stand-in: once the verifying read of ``path`` reaches end of
        file, another handle rewrites its first byte (same size, different bytes)."""
        real_open = builtins.open
        state = {"fired": False}

        class ChangeAtEndOfFile(_HandleProxy):
            def read(self, *args: Any) -> Any:
                chunk = self._raw.read(*args)
                if chunk == b"" and not state["fired"]:
                    state["fired"] = True
                    with real_open(path, "r+b") as other:
                        other.write(changed[:1])
                return chunk

        def fake_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            handle = real_open(file, mode, *args, **kwargs)
            if mode == "rb" and not state["fired"] and Path(file).resolve() == path.resolve():
                return ChangeAtEndOfFile(handle)
            return handle

        return fake_open, state

    def test_read_if_unchanged_returns_the_bytes_it_verified_never_a_second_read(self) -> None:
        # The bytes come from the very pass that hashed them. A same-size change landing
        # right after that pass reached end of file is never returned, and Edit never
        # builds on it (the final comparison refuses it instead).
        for label, data, chunk in (
            ("one_chunk", self.ORIGINAL, context_module._FINGERPRINT_CHUNK_BYTES),
            ("multi_chunk", self.ORIGINAL + bytes(range(256)) * 3, 64),
        ):
            with self.subTest(case=label):
                path = self.put(f"same-pass-{label}.bin", data)
                self.ctx.mark_file_read(path, data=data)
                changed = flip_byte(data, 0)
                fake_open, state = self._change_at_end_of_file(path, changed)
                with patch.object(context_module, "_FINGERPRINT_CHUNK_BYTES", chunk), patch.object(
                    context_module, "open", fake_open, create=True
                ):
                    self.assertEqual(self.ctx.read_if_unchanged(path), data)
                self.assertTrue(state["fired"])
                self.assertEqual(path.read_bytes(), changed)
        path = self.put("same-pass-edit.txt", self.ORIGINAL)
        self.read(path)
        changed = flip_byte(self.ORIGINAL, 0)
        fake_open, state = self._change_at_end_of_file(path, changed)
        with patch.object(context_module, "open", fake_open, create=True):
            self._refused(
                path,
                lambda p: FileEditTool().run(
                    {"file_path": str(p), "old_string": "key=old", "new_string": "key=new"}, self.ctx
                ),
                REFUSE_EDIT,
            )
        self.assertTrue(state["fired"])
        self.assertEqual(path.read_bytes(), changed)

    def test_a_change_undone_before_the_final_compare_is_never_absorbed(self) -> None:
        # The change is built from the verified bytes: a change landing after the check and
        # undone before the final comparison leaves no trace in the result (a consumer that
        # read the file again would carry it into its edit).
        nb_original = _notebook("aaaa")
        nb_external = nb_original.replace(b'"metadata": {}', b'"metadata": {"external": 1}', 1)
        edited = json.loads(nb_original)
        edited["cells"][0]["source"] = ["bbbb"]
        cases: dict[str, tuple[Callable[[Path], Any], bytes, bytes, bytes, str]] = {
            "edit": (
                lambda p: FileEditTool().run(
                    {"file_path": str(p), "old_string": "key=old", "new_string": "key=new"}, self.ctx
                ),
                self.ORIGINAL, self.ORIGINAL + b"EXTERNAL\n", self._edit_bytes("key=new"), "txt",
            ),
            "write": (
                lambda p: FileWriteTool().run({"file_path": str(p), "content": "written\n"}, self.ctx),
                self.ORIGINAL, self.ORIGINAL + b"EXTERNAL\n", text_mode_bytes("written\n", os.linesep), "txt",
            ),
            "notebook_edit": (
                lambda p: NotebookEditTool().run(
                    {"notebook_path": str(p), "cell_id": "md-1", "new_source": "bbbb"}, self.ctx
                ),
                nb_original, nb_external,
                (json.dumps(edited, ensure_ascii=False, indent=1) + "\n").encode("utf-8"), "ipynb",
            ),
        }
        for name, (act, original, external, expected, suffix) in cases.items():
            with self.subTest(site=name):
                path = self.put(f"undone-{name}.{suffix}", original)
                self.read(path)
                with after_verified_read(self, self.ctx, lambda p=path, e=external: p.write_bytes(e)):
                    with around_final_open(
                        self, path, ("r+b", "rb"), before=lambda p=path, o=original: p.write_bytes(o)
                    ):
                        result = act(path)
                self.assertEqual(path.read_bytes(), expected)
                if name == "write":
                    self.assertEqual(result.output["originalFile"], original.decode())
                self.assertTrue(self.gate(path))
    def test_short_writes_are_completed(self) -> None:
        class Trickle(_HandleProxy):
            def write(self, data: Any) -> Any:
                return self._raw.write(bytes(data)[:3])

        for mode, prepare in (("r+b", True), ("xb", False)):
            with self.subTest(mode=mode):
                path = self.root / f"trickle-{mode.strip('+')}.txt"
                if prepare:
                    path.write_bytes(b"old content that is long\n")
                    self.read(path)
                with around_final_open(self, path, (mode,), wrap=Trickle):
                    FileWriteTool().run({"file_path": str(path), "content": "new\ncontent\n"}, self.ctx)
                self.assertEqual(path.read_bytes(), text_mode_bytes("new\ncontent\n", os.linesep))
                self.assertTrue(self.gate(path))

    def test_os_errors_other_than_a_change_propagate_unchanged(self) -> None:
        # Read-only file: PermissionError as before this change; nothing written.
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root ignores file permission bits")
        path = self.put("ro.txt", self.ORIGINAL)
        self.read(path)
        os.chmod(path, stat.S_IREAD)
        try:
            with self.assertRaises(PermissionError):
                FileEditTool().run(
                    {"file_path": str(path), "old_string": "key=old", "new_string": "key=new"}, self.ctx
                )
            with self.assertRaises(PermissionError):
                FileWriteTool().run({"file_path": str(path), "content": "x\n"}, self.ctx)
        finally:
            os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
        self.assertEqual(path.read_bytes(), self.ORIGINAL)
        self.assertTrue(self.gate(path))

    @unittest.skipUnless(sys.platform == "win32", "Windows user-mapped section semantics")
    def test_memory_mapped_file_edits_and_the_unrepaired_shortening_failure(self) -> None:
        # Same-length and longer edits of a file another handle has memory-mapped succeed.
        # A shortening edit writes, then the truncate is refused: PermissionError, the file
        # keeps the new bytes followed by the old tail, nothing is marked, gate False.
        import mmap

        original = b"key=old;" + b"." * 4088  # no newline: identical bytes on every platform
        for label, new in (("same", "key=new"), ("longer", "key=newer"), ("shorter", "k")):
            with self.subTest(case=label):
                path = self.put(f"mapped-{label}.txt", original)
                self.read(path)
                data = original.replace(b"key=old", new.encode())
                marked = self._capture_marks()
                holder: Any = None
                mapping: Any = None
                try:
                    holder = open(path, "r+b")
                    mapping = mmap.mmap(holder.fileno(), 0)
                    edit = lambda: FileEditTool().run(  # noqa: E731
                        {"file_path": str(path), "old_string": "key=old", "new_string": new}, self.ctx
                    )
                    if label == "shorter":
                        with self.assertRaises(PermissionError) as caught:
                            edit()
                        self.assertEqual(caught.exception.errno, errno.EACCES)
                        self.assertEqual(path.read_bytes(), data + original[len(data):])
                        self.assertFalse(self.gate(path))
                        self.assertEqual(marked, [])
                    else:
                        edit()
                        self.assertEqual(path.read_bytes(), data)
                        self.assertTrue(self.gate(path))
                        self.assertEqual(marked, [data])
                finally:
                    if mapping is not None:
                        mapping.close()
                    if holder is not None:
                        holder.close()
                    self.ctx.__dict__.pop("mark_file_read", None)

    @unittest.skipUnless(sys.platform == "win32", "Windows file attributes")
    def test_hidden_and_system_files_keep_todays_refusal(self) -> None:
        # open(path, "wb") refuses a hidden or system file (CREATE_ALWAYS); the in-place open
        # would not, so compare_write refuses with the identical error before any write.
        for label, attributes in (("hidden", 0x2), ("system", 0x4), ("hidden_system", 0x6)):
            for tool in ("Write", "Edit"):
                with self.subTest(attributes=label, tool=tool):
                    folder = self.root / f"attr-{label}-{tool}"
                    folder.mkdir()
                    target, reference = folder / "t.txt", folder / "r.txt"
                    for p in (target, reference):
                        p.write_bytes(self.ORIGINAL)
                    self.read(target)
                    try:
                        for p in (target, reference):
                            _win_set_attributes(p, attributes)
                        want = _pr_open_wb_error(reference)
                        assert want is not None
                        marked = self._capture_marks()
                        with self.assertRaises(PermissionError) as caught:
                            if tool == "Write":
                                FileWriteTool().run({"file_path": str(target), "content": "written\n"}, self.ctx)
                            else:
                                FileEditTool().run(
                                    {"file_path": str(target), "old_string": "key=old", "new_string": "key=new"},
                                    self.ctx,
                                )
                        self.assertEqual(
                            _pr_error_fields(caught.exception), _pr_error_fields(_pr_rebuilt(want, target))
                        )
                        self.assertEqual(target.read_bytes(), self.ORIGINAL)
                        self.assertTrue(self.gate(target))
                        self.assertEqual(marked, [])
                    finally:
                        for p in (target, reference):
                            _win_set_attributes(p, 0x80)
                        self.ctx.__dict__.pop("mark_file_read", None)
        # Precedence as today: a changed hidden file gets the refusal text, not the error.
        path = self.put("hidden-changed.txt", self.ORIGINAL)
        self.read(path)
        path.write_bytes(b"changed underneath\n")
        _win_set_attributes(path, 0x2)
        try:
            self._refused(
                path,
                lambda p: FileWriteTool().run({"file_path": str(p), "content": "x\n"}, self.ctx),
                REFUSE_OVERWRITE,
            )
        finally:
            _win_set_attributes(path, 0x80)

    @unittest.skipUnless(sys.platform == "win32", "Windows file attributes")
    def test_hidden_file_changed_before_the_final_compare_gets_the_refusal_text(self) -> None:
        # The comparison comes first: a hidden file that changed after the check is
        # refused as changed, not with the hidden-file PermissionError.
        path = self.put("hidden-late.txt", self.ORIGINAL)
        self.read(path)
        _win_set_attributes(path, 0x2)
        external = b"changed underneath, same handle rules\n"
        try:
            with around_final_open(self, path, ("r+b",), before=lambda: external_rewrite(path, external)):
                self._refused(
                    path,
                    lambda p: FileWriteTool().run({"file_path": str(p), "content": "x\n"}, self.ctx),
                    REFUSE_OVERWRITE,
                )
            self.assertEqual(path.read_bytes(), external)
        finally:
            _win_set_attributes(path, 0x80)

    @unittest.skipUnless(sys.platform == "win32", "NTFS alternate data streams")
    def test_mark_of_the_web_is_kept_in_place_and_dropped_by_the_notebook_replace(self) -> None:
        zone = b"[ZoneTransfer]\r\nZoneId=3\r\n"
        for name, fname, prepare, act, _refusal in self._mutators():
            with self.subTest(site=name):
                folder = self.root / f"motw-{name}"
                folder.mkdir()
                path = folder / fname
                prepare(path)
                with open(f"{path}:Zone.Identifier", "wb") as stream:
                    stream.write(zone)
                self.assertTrue(self.gate(path))  # another stream is not the file's bytes
                act(path)
                if name == "notebook_edit":
                    with self.assertRaises(FileNotFoundError):
                        open(f"{path}:Zone.Identifier", "rb").close()
                else:
                    with open(f"{path}:Zone.Identifier", "rb") as stream:
                        self.assertEqual(stream.read(), zone)

    @unittest.skipUnless(sys.platform == "win32", "Windows byte-range locks")
    def test_pre_held_shared_lock_refuses_before_any_byte_lands(self) -> None:
        # Today's truncating open emptied a file another handle had share-locked (b"")
        # before its write failed. The in-place write fails before any byte lands.
        cases = {
            "Write": (("shorter", "shorter-content\n"), ("same", "E" * 18), ("longer", "E" * 30)),
            "Edit": (("shorter", "k"), ("same", "key=new"), ("longer", "key=newer")),
        }
        for offset in (0, 8):
            for tool, sizes in cases.items():
                for label, new in sizes:
                    with self.subTest(lock_from=offset, tool=tool, size=label):
                        path = self.put(f"locked-{offset}-{tool}-{label}.txt", self.ORIGINAL)
                        calls: list[str] = []

                        class Spy(_HandleProxy):
                            def seek(self, *args: Any) -> Any:
                                calls.append("seek")
                                return self._raw.seek(*args)

                            def write(self, data: Any) -> Any:
                                calls.append("write")
                                return self._raw.write(data)

                            def truncate(self, *args: Any) -> Any:
                                calls.append("truncate")
                                return self._raw.truncate(*args)

                        with _win_shared_lock(path, offset):
                            self.read(path)
                            marked = self._capture_marks()
                            with around_final_open(self, path, ("r+b",), wrap=Spy):
                                with self.assertRaises(PermissionError) as caught:
                                    if tool == "Write":
                                        FileWriteTool().run({"file_path": str(path), "content": new}, self.ctx)
                                    else:
                                        FileEditTool().run(
                                            {"file_path": str(path), "old_string": "key=old", "new_string": new},
                                            self.ctx,
                                        )
                        self.assertEqual(caught.exception.errno, errno.EACCES)
                        self.assertEqual(calls, ["seek", "write"])
                        self.assertEqual(path.read_bytes(), self.ORIGINAL)
                        self.assertTrue(self.gate(path))
                        self.assertEqual(marked, [])
                        self.ctx.__dict__.pop("mark_file_read", None)

    @unittest.skipUnless(sys.platform == "win32", "Windows share modes")
    def test_notebook_holder_opened_after_the_temp_write_gets_the_refusal(self) -> None:
        folder = self.root / "nb-holder"
        folder.mkdir()
        path = folder / "n.ipynb"
        path.write_bytes(_notebook("aaaa"))
        self.read(path)
        real = tempfile.NamedTemporaryFile
        with contextlib.ExitStack() as stack:

            @contextlib.contextmanager
            def temp_then_exclusive_holder(*args: Any, **kwargs: Any) -> Iterator[Any]:
                with real(*args, **kwargs) as handle:
                    yield handle
                stack.enter_context(_win_handle(path, 0x80000000, 0))  # GENERIC_READ, share none

            with patch.object(tempfile, "NamedTemporaryFile", temp_then_exclusive_holder):
                self._refused(
                    path,
                    lambda p: NotebookEditTool().run(
                        {"notebook_path": str(p), "cell_id": "md-1", "new_source": "bbbb"}, self.ctx
                    ),
                    REFUSE_NOTEBOOK,
                )
        self.assertEqual(path.read_bytes(), _notebook("aaaa"))
        self._no_leftovers(folder)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFOs")
    def test_notebook_final_compare_on_a_fifo_swapped_in_neither_hangs_nor_writes(self) -> None:
        folder = self.root / "nb-fifo"
        folder.mkdir()
        path = folder / "n.ipynb"
        path.write_bytes(_notebook("aaaa"))
        self.read(path)
        real = tempfile.NamedTemporaryFile

        @contextlib.contextmanager
        def temp_then_fifo(*args: Any, **kwargs: Any) -> Iterator[Any]:
            with real(*args, **kwargs) as handle:
                yield handle
            path.unlink()
            os.mkfifo(path)

        outcome: dict[str, Any] = {}

        def run() -> None:
            try:
                NotebookEditTool().run(
                    {"notebook_path": str(path), "cell_id": "md-1", "new_source": "bbbb"}, self.ctx
                )
                outcome["raised"] = None
            except BaseException as exc:  # inspected below
                outcome["raised"] = exc

        with patch.object(tempfile, "NamedTemporaryFile", temp_then_fifo):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            worker.join(15)
        self.assertFalse(worker.is_alive(), "the final comparison blocked on a FIFO")
        self.assertIsInstance(outcome["raised"], ToolInputError)
        self.assertEqual(str(outcome["raised"]), REFUSE_NOTEBOOK)
        self.assertTrue(stat.S_ISFIFO(os.lstat(path).st_mode))
        self._no_leftovers(folder)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "POSIX FIFOs")
    def test_fifo_swapped_in_after_the_gate_is_refused_without_blocking(self) -> None:
        for name, act, refusal in (
            ("edit", lambda p: FileEditTool().run(
                {"file_path": str(p), "old_string": "key=old", "new_string": "key=new"}, self.ctx
            ), REFUSE_EDIT),
            ("write", lambda p: FileWriteTool().run({"file_path": str(p), "content": "w\n"}, self.ctx), REFUSE_OVERWRITE),
        ):
            with self.subTest(tool=name):
                path = self.put(f"fifo-{name}.txt", self.ORIGINAL)
                self.read(path)

                def swap(path: Path = path) -> None:
                    path.unlink()
                    os.mkfifo(path)

                outcome: dict[str, Any] = {}

                def run(path: Path = path, act: Callable[[Path], Any] = act) -> None:
                    try:
                        act(path)
                        outcome["raised"] = None
                    except BaseException as exc:  # inspected below
                        outcome["raised"] = exc

                with after_verified_read(self, self.ctx, swap):
                    worker = threading.Thread(target=run, daemon=True)
                    worker.start()
                    worker.join(15)
                self.assertFalse(worker.is_alive(), "the in-place open blocked on a FIFO")
                self.assertIsInstance(outcome["raised"], ToolInputError)
                self.assertEqual(str(outcome["raised"]), refusal)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux directory-anchored open")
    def test_symlink_swapped_in_after_the_gate_is_refused_never_followed(self) -> None:
        # The replacement leads to identical bytes elsewhere, so only the anchored,
        # no-follow open refuses it: the file behind the symlink is never written.
        for swapped in ("file", "parent_directory"):
            with self.subTest(swapped=swapped):
                folder = self.root / f"anchored-{swapped}"
                folder.mkdir()
                path = folder / "e.txt"
                path.write_bytes(self.ORIGINAL)
                self.read(path)
                elsewhere = self.root / f"elsewhere-{swapped}"
                elsewhere.mkdir()
                copy = elsewhere / "e.txt"
                copy.write_bytes(self.ORIGINAL)

                def swap(folder: Path = folder, path: Path = path, elsewhere: Path = elsewhere,
                         copy: Path = copy, swapped: str = swapped) -> None:
                    if swapped == "file":
                        path.unlink()
                        path.symlink_to(copy)
                    else:
                        shutil.rmtree(folder)
                        folder.symlink_to(elsewhere, target_is_directory=True)

                with after_verified_read(self, self.ctx, swap):
                    self._refused(
                        path,
                        lambda p: FileEditTool().run(
                            {"file_path": str(p), "old_string": "key=old", "new_string": "key=new"}, self.ctx
                        ),
                        REFUSE_EDIT,
                    )
                self.assertEqual(copy.read_bytes(), self.ORIGINAL)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux directory-anchored open")
    def test_directory_swapped_in_after_the_gate_is_refused(self) -> None:
        # Not the verified regular file: the refusal text, never a raw IsADirectoryError.
        for name, act, refusal in (
            ("edit", lambda p: FileEditTool().run(
                {"file_path": str(p), "old_string": "key=old", "new_string": "key=new"}, self.ctx
            ), REFUSE_EDIT),
            ("write", lambda p: FileWriteTool().run({"file_path": str(p), "content": "w\n"}, self.ctx), REFUSE_OVERWRITE),
        ):
            with self.subTest(tool=name):
                path = self.put(f"now-a-dir-{name}.txt", self.ORIGINAL)
                self.read(path)

                def swap(path: Path = path) -> None:
                    path.unlink()
                    path.mkdir()

                with after_verified_read(self, self.ctx, swap):
                    self._refused(path, act, refusal)
                self.assertTrue(path.is_dir())

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux directory-anchored open")
    def test_replacement_between_the_lookup_and_the_open_is_refused(self) -> None:
        # Swapped in after the pre-open lookup and before the open itself: a symlink to
        # identical bytes is not followed (O_NOFOLLOW), a FIFO is opened without blocking and
        # then refused as not regular, and a directory (EISDIR) is refused, not raised.
        for kind in ("symlink", "fifo", "directory"):
            with self.subTest(kind=kind):
                path = self.put(f"window-{kind}.txt", self.ORIGINAL)
                self.read(path)
                copy = self.put(f"window-{kind}-elsewhere.txt", self.ORIGINAL)
                real_open = os.open
                swapped: list[str] = []

                def swap_then_open(name: Any, flags: int, *args: Any, _p: Path = path, _k: str = kind,
                                   _c: Path = copy, **kwargs: Any) -> int:
                    if kwargs.get("dir_fd") is not None and name == _p.name and not swapped:
                        _p.unlink()
                        if _k == "symlink":
                            _p.symlink_to(_c)
                        elif _k == "fifo":
                            os.mkfifo(_p)
                        else:
                            _p.mkdir()
                        swapped.append(_k)
                    return real_open(name, flags, *args, **kwargs)

                with patch.object(os, "open", swap_then_open):
                    self._refused(
                        path,
                        lambda p: FileEditTool().run(
                            {"file_path": str(p), "old_string": "key=old", "new_string": "key=new"}, self.ctx
                        ),
                        REFUSE_EDIT,
                    )
                self.assertEqual(swapped, [kind])
                self.assertEqual(copy.read_bytes(), self.ORIGINAL)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux directory-anchored open")
    def test_anchored_open_errors_name_the_target_exactly_like_open_wb(self) -> None:
        # A directory that loses its search permission after the gate: the in-place open
        # fails as today's open(path, "wb") fails (same error, same filename).
        if os.geteuid() == 0:
            self.skipTest("root ignores file permission bits")
        folder = self.root / "no-search"
        folder.mkdir()
        target, reference = folder / "t.txt", folder / "r.txt"
        for p in (target, reference):
            p.write_bytes(self.ORIGINAL)
        self.read(target)
        try:  # restored before tearDown removes the folder
            with after_verified_read(self, self.ctx, lambda: os.chmod(folder, 0o600)):
                with self.assertRaises(PermissionError) as caught:
                    FileEditTool().run(
                        {"file_path": str(target), "old_string": "key=old", "new_string": "key=new"}, self.ctx
                    )
            want = _pr_open_wb_error(reference)
        finally:
            os.chmod(folder, 0o755)
        assert want is not None
        self.assertEqual(_pr_error_fields(caught.exception), _pr_error_fields(_pr_rebuilt(want, target)))
        self.assertEqual(target.read_bytes(), self.ORIGINAL)

    def test_consumers_keep_todays_results_for_empty_and_undecodable_files(self) -> None:
        empty = self.put("empty.txt", b"")
        self.read(empty)
        with self.assertRaises(ToolInputError) as caught:
            FileEditTool().run({"file_path": str(empty), "old_string": "x", "new_string": "y"}, self.ctx)
        self.assertEqual(str(caught.exception), "old_string not found in file")
        out = FileWriteTool().run({"file_path": str(empty), "content": "now\n"}, self.ctx).output
        self.assertEqual((out["type"], out["originalFile"]), ("update", ""))
        cases = {
            "empty": b"",
            "bom": b"\xef\xbb\xbf" + _notebook("aaaa"),
            "crlf_invalid": b'{\r\n "cells": [\r\n  x\r\n',
            "utf16": _notebook("aaaa").decode("utf-8").encode("utf-16"),
            "invalid_utf8": b'{"cells": [{"cell_type": "markdown", "id": "md-1", "metadata": {}, "source": ["caf\xff"]}]}',
        }
        for name, raw in cases.items():
            with self.subTest(notebook=name):
                path = self.put(f"nb-{name}.ipynb", raw)
                self.ctx.mark_file_read(path, data=raw)
                try:
                    expected: BaseException | None = None
                    json.loads(path.read_text(encoding="utf-8"))
                except (ValueError, UnicodeDecodeError) as exc:
                    expected = exc
                assert expected is not None
                with self.assertRaises(type(expected) if isinstance(expected, UnicodeDecodeError) else ToolInputError) as caught:
                    NotebookEditTool().run(
                        {"notebook_path": str(path), "cell_id": "md-1", "new_source": "x"}, self.ctx
                    )
                want = str(expected) if isinstance(expected, UnicodeDecodeError) else f"failed to parse notebook: {expected}"
                self.assertEqual(str(caught.exception), want)
                self.assertEqual(path.read_bytes(), raw)

    def test_data_transform_uses_only_the_verified_source_bytes(self) -> None:
        import src.tool_system.tools.data as data_module

        for change in ("append_row", "delete"):
            with self.subTest(change=change):
                source = self.put(f"s-{change}.csv", b"id,v\n1,a\n")
                target = self.root / f"o-{change}.json"
                DataInspectTool().run({"file_path": str(source)}, self.ctx)
                action = (lambda: source.write_bytes(b"id,v\n1,a\n9,EXTERNAL\n")) if change == "append_row" else source.unlink
                with after_verified_read(self, self.ctx, action):
                    DataTransformTool().run({"input_path": str(source), "output_path": str(target)}, self.ctx)
                self.assertEqual(json.loads(target.read_bytes()), [{"id": "1", "v": "a"}])
                self.assertTrue(self.gate(target))
                self.assertFalse(self.gate(source))
        empty = self.put("empty.csv", b"")
        self.read(empty)
        DataTransformTool().run({"input_path": str(empty), "output_path": str(self.root / "e.json")}, self.ctx)
        self.assertEqual((self.root / "e.json").read_bytes(), b"[]\n")
        big = self.put("big.csv", b"id\n" + b"1\n" * 20)
        self.read(big)
        with patch.object(data_module, "_MAX_INPUT_BYTES", 16):
            with self.assertRaises(ToolInputError) as caught:
                DataTransformTool().run({"input_path": str(big), "output_path": str(self.root / "b.json")}, self.ctx)
        self.assertEqual(str(caught.exception), "data file exceeds 0 MiB limit")

    def test_read_if_unchanged_returns_the_exact_bytes_and_fails_closed_like_the_gate(self) -> None:
        for data in (b"", b"a\r\nb\x1a\n", bytes(range(256)) * 9):
            with self.subTest(size=len(data)):
                path = self.put(f"exact-{len(data)}.bin", data)
                self.ctx.mark_file_read(path, data=data)
                self.assertEqual(self.ctx.read_if_unchanged(path), data)
                collide(self, path, flip_byte(data)) if data else rewrite_keeping_mtime(self, path, b"X")
                self.assertIsNone(self.ctx.read_if_unchanged(path))
        path = self.put("errors.txt", b"AAAA\n")
        self.assertIsNone(self.ctx.read_if_unchanged(path))  # never recorded
        self.read(path)
        for error in (PermissionError(13, "denied"), OSError(5, "I/O"), ValueError("bad")):
            with self.subTest(error=type(error).__name__):
                with patch.object(context_module, "open", side_effect=error, create=True):
                    self.assertIsNone(self.ctx.read_if_unchanged(path))
        with patch.object(context_module, "open", spy_open([], fail_reads=True), create=True):
            self.assertIsNone(self.ctx.read_if_unchanged(path))
        with patch.object(context_module, "open", spy_open([], none_after=1), create=True):
            self.assertIsNone(self.ctx.read_if_unchanged(path))
        fake = os.stat_result((stat.S_IFIFO | 0o644, 0, 0, 1, 0, 0, 5, 0, 0, 0))
        with patch.object(os, "fstat", return_value=fake):
            self.assertIsNone(self.ctx.read_if_unchanged(path))
        with patch.object(pathlib.Path, "resolve", side_effect=RuntimeError("loop")):
            self.assertIsNone(self.ctx.read_if_unchanged(path))
        with patch.object(context_module, "open", side_effect=MemoryError, create=True):
            with self.assertRaises(MemoryError):
                self.ctx.read_if_unchanged(path)
        self.assertEqual(self.ctx.read_if_unchanged(path), b"AAAA\n")


class TestProtectedRegularPolicy(unittest.TestCase):
    """compare_write's Linux fs.protected_regular rule as a pure decision, on every
    platform: each level, directory shape and owner combination against an independent
    transcription of fs/namei.c may_create_in_sticky, and every case it cannot classify
    failing closed. GitHub CI verifies the rule against a real kernel at one level only
    (TestProtectedRegularKernelEvidence); these cases cover the others."""

    CALLER, DIR_OWNER, FILE_OWNER, OVERFLOW = 1001, 64002, 64001, 65534

    def decide(
        self,
        mode: int,
        dir_uid: int,
        file_uid: int,
        *,
        level: Any = "2",
        overflow: Any = "65534",
        fsuid: Any = 1001,
        reads: list[str] | None = None,
    ) -> str | None:
        log = [] if reads is None else reads
        values = {
            compare_write_module._PROTECTED_REGULAR: level,
            compare_write_module._OVERFLOW_UID: overflow,
        }

        def read_int(path: str) -> int:
            log.append(path)
            value = values[path]
            if isinstance(value, BaseException):
                raise value
            return int(value)

        def read_fsuid() -> int:
            log.append("fsuid")
            if isinstance(fsuid, BaseException):
                raise fsuid
            return int(fsuid)

        with patch.object(compare_write_module, "_read_proc_int", read_int), patch.object(
            compare_write_module, "_read_fsuid", read_fsuid
        ):
            return compare_write_module._protected_regular_refusal(stat.S_IFDIR | mode, dir_uid, file_uid)

    def test_every_level_shape_and_owner_matches_the_kernel_rule(self) -> None:
        modes = (0o1777, 0o1770, 0o1707, 0o1720, 0o1702, 0o1755, 0o1700, 0o0777, 0o0770, 0o0755)
        owners = (
            (self.CALLER, self.FILE_OWNER),  # sticky dir of the caller, another user's file
            (self.DIR_OWNER, self.FILE_OWNER),  # the /tmp shape: three different owners
            (self.FILE_OWNER, self.FILE_OWNER),  # the directory owner owns the file
            (self.DIR_OWNER, self.CALLER),  # the caller owns the file
        )
        for level in (0, 1, 2):
            for mode in modes:
                for dir_uid, file_uid in owners:
                    with self.subTest(level=level, mode=oct(mode), dir_uid=dir_uid, file_uid=file_uid):
                        refused = _pr_kernel_refuses_o_creat(level, mode, dir_uid, file_uid, self.CALLER)
                        self.assertEqual(
                            self.decide(mode, dir_uid, file_uid, level=str(level)),
                            compare_write_module._DENIED if refused else None,
                        )

    def test_proc_is_read_only_where_the_rule_can_refuse(self) -> None:
        broken = OSError(errno.ENOENT, "no /proc")
        for mode in (0o0777, 0o0770, 0o1755, 0o1700):
            with self.subTest(mode=oct(mode)):
                reads: list[str] = []
                decision = self.decide(
                    mode, self.CALLER, self.FILE_OWNER, level=broken, overflow=broken, fsuid=broken, reads=reads
                )
                self.assertIsNone(decision)
                self.assertEqual(reads, [])
        # The kernel's ownership exemptions hold at every level: fs.protected_regular is
        # never read for a file owned by the directory's owner or by the caller.
        for label, (dir_uid, file_uid) in {
            "directory_owner_owns_the_file": (self.DIR_OWNER, self.DIR_OWNER),
            "caller_owns_the_file": (self.DIR_OWNER, self.CALLER),
        }.items():
            with self.subTest(exempt=label):
                reads = []
                self.assertIsNone(self.decide(0o1777, dir_uid, file_uid, level=broken, reads=reads))
                self.assertEqual(reads, [compare_write_module._OVERFLOW_UID, "fsuid"])
        for level, mode in (("0", 0o1777), ("1", 0o1770)):
            with self.subTest(level=level, mode=oct(mode)):
                reads = []
                decision = self.decide(mode, self.CALLER, self.FILE_OWNER, level=level, reads=reads)
                self.assertIsNone(decision)
                self.assertEqual(
                    reads, [compare_write_module._OVERFLOW_UID, "fsuid", compare_write_module._PROTECTED_REGULAR]
                )

    def test_what_cannot_be_classified_fails_closed(self) -> None:
        refusable = (0o1777, self.CALLER, self.FILE_OWNER)
        cases = {
            "sysctl_unreadable": {"level": PermissionError(errno.EACCES, "denied")},
            "sysctl_absent": {"level": FileNotFoundError(errno.ENOENT, "absent")},
            "sysctl_garbage": {"level": "two"},
            "sysctl_unknown_level": {"level": "3"},
            "overflow_uid_unreadable": {"overflow": OSError(errno.EIO, "I/O")},
            "fsuid_unreadable": {"fsuid": ValueError("no Uid: line")},
        }
        for label, override in cases.items():
            with self.subTest(case=label):
                decision = self.decide(*refusable, **override)
                self.assertIsNotNone(decision)
                self.assertNotEqual(decision, compare_write_module._DENIED)
        # A uid shown as the overflow uid may be unmapped: it never proves a match. Only a
        # match that is possible but unprovable is unclassifiable; otherwise the kernel's
        # answer is known (another user's file in a world-writable sticky directory: denied).
        for label, (dir_uid, file_uid, fsuid, want) in {
            "file_owner_unmapped": (self.CALLER, self.OVERFLOW, self.CALLER, compare_write_module._DENIED),
            "dir_owner_unmapped": (self.OVERFLOW, self.FILE_OWNER, self.CALLER, compare_write_module._DENIED),
            "both_unmapped": (self.OVERFLOW, self.OVERFLOW, self.CALLER, "an owner is unmapped (overflow uid)"),
            "caller_is_the_overflow_uid": (self.CALLER, self.OVERFLOW, self.OVERFLOW, "an owner is unmapped (overflow uid)"),
            "callers_file_in_an_unmapped_directory": (self.OVERFLOW, self.CALLER, self.CALLER, None),
        }.items():
            with self.subTest(case=label):
                self.assertEqual(self.decide(0o1777, dir_uid, file_uid, fsuid=fsuid), want)
        # The exemptions need no fs.protected_regular (Sol): the caller's own file, or the
        # directory owner's, is allowed although the level cannot be read ...
        absent = FileNotFoundError(errno.ENOENT, "absent")
        self.assertIsNone(self.decide(0o1777, self.DIR_OWNER, self.CALLER, level=absent))
        self.assertIsNone(self.decide(0o1777, self.DIR_OWNER, self.DIR_OWNER, level=absent))
        # ... but only when the overflow uid is known, so an unmapped owner cannot pose as one.
        unreadable = OSError(errno.EIO, "I/O")
        self.assertIsNotNone(self.decide(0o1777, self.DIR_OWNER, self.CALLER, overflow=unreadable))
        self.assertIsNotNone(self.decide(0o1777, self.DIR_OWNER, self.CALLER, overflow=unreadable, level=absent))
        # The kernel compares the filesystem uid, not the effective uid.
        self.assertIsNone(self.decide(0o1777, self.CALLER, 4242, fsuid=4242))
        self.assertEqual(self.decide(0o1777, self.DIR_OWNER, 1001, fsuid=4242), compare_write_module._DENIED)

    def test_refusals_carry_todays_error_and_fail_closed_ones_say_why(self) -> None:
        path = Path(tempfile.gettempdir()).resolve() / "sticky" / "target.txt"
        directory = os.stat_result((stat.S_IFDIR | 0o1777, 0, 0, 2, self.CALLER, 0, 0, 0, 0, 0))
        with patch.object(compare_write_module, "_protected_regular_refusal", return_value="denied"):
            with self.assertRaises(PermissionError) as denied:
                compare_write_module._enforce_protected_regular(path, directory, self.FILE_OWNER)
        today = PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
        self.assertEqual(_pr_error_fields(denied.exception), _pr_error_fields(today))
        with patch.object(compare_write_module, "_protected_regular_refusal", return_value="why not"):
            with self.assertRaises(PermissionError) as unsure:
                compare_write_module._enforce_protected_regular(path, directory, self.FILE_OWNER)
        self.assertEqual((unsure.exception.errno, unsure.exception.filename), (errno.EACCES, str(path)))
        self.assertIn("fs.protected_regular not verified: why not", unsure.exception.strerror)
        with patch.object(compare_write_module, "_protected_regular_refusal", return_value=None):
            compare_write_module._enforce_protected_regular(path, directory, self.FILE_OWNER)

    def test_anchored_open_errors_map_to_refusal_or_todays_error(self) -> None:
        # Vanished or replaced (ENOENT, ENOTDIR, ELOOP from the no-follow open; EISDIR or
        # ENXIO for a directory or socket swapped in just before the open) is "not the
        # verified file" (the refusal); anything else is the same error naming the full
        # target path, as open(path, ...) names it, without exception chaining.
        path = Path(tempfile.gettempdir()).resolve() / "folder" / "target.txt"
        for code in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP, errno.EISDIR, errno.ENXIO):
            with self.subTest(errno=code):
                mapped = compare_write_module._as_opened_by_path(OSError(code, os.strerror(code), "target.txt"), path)
                self.assertIsInstance(mapped, compare_write_module._NotTheVerifiedFile)
        for error in (
            PermissionError(errno.EACCES, os.strerror(errno.EACCES), "target.txt"),
            OSError(errno.EROFS, os.strerror(errno.EROFS), "target.txt"),
            BlockingIOError(errno.EAGAIN, os.strerror(errno.EAGAIN), "target.txt"),
        ):
            with self.subTest(error=type(error).__name__):
                mapped = compare_write_module._as_opened_by_path(error, path)
                self.assertEqual(
                    _pr_error_fields(mapped),
                    _pr_error_fields(type(error)(error.errno, error.strerror, str(path))),
                )

    def test_proc_readers_parse_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            status = Path(folder) / "status"
            status.write_bytes(b"Name:\tpython\nUid:\t1000\t1001\t1002\t1003\nGid:\t5\t5\t5\t5\n")
            with patch.object(compare_write_module, "_THREAD_STATUS", str(status)):
                self.assertEqual(compare_write_module._read_fsuid(), 1003)
                status.write_bytes(b"Name:\tpython\nGid:\t5\t5\t5\t5\n")
                with self.assertRaises(ValueError):
                    compare_write_module._read_fsuid()
            level = Path(folder) / "level"
            level.write_bytes(b"2\n")
            self.assertEqual(compare_write_module._read_proc_int(str(level)), 2)
            level.write_bytes(b"x\n")
            with self.assertRaises(ValueError):
                compare_write_module._read_proc_int(str(level))


def _tree(rel: str) -> ast.Module:
    return ast.parse((SRC / rel).read_text(encoding="utf-8-sig"))


def _functions(tree: ast.Module) -> Iterator[tuple[str, ast.FunctionDef]]:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    yield f"{node.name}.{item.name}", item


def _function(rel: str, qualname: str) -> ast.FunctionDef:
    for name, node in _functions(_tree(rel)):
        if name == qualname:
            return node
    raise AssertionError(f"{rel}: function not found: {qualname}")


def _call_name(node: ast.Call) -> str:
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return "?"


def _calls(node: ast.AST) -> list[ast.Call]:
    return [child for child in ast.walk(node) if isinstance(child, ast.Call)]


def _bindings(function: ast.FunctionDef, name: str) -> list[ast.expr]:
    """One entry per binding of ``name`` inside ``function``: the assigned value for a
    plain or tuple-unpacking assignment, a placeholder for any other kind of binding."""
    values: list[ast.expr] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                names = target.elts if isinstance(target, ast.Tuple) else [target]
                if any(isinstance(item, ast.Name) and item.id == name for item in names):
                    values.append(node.value)
    stores = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Store)
    ]
    stores += [a for a in ast.walk(function.args) if isinstance(a, ast.arg) and a.arg == name]
    return values + [ast.Constant(value=None)] * (len(stores) - len(values))


# Every production recorder: the buffer passed as data= must be bound exactly once in
# the recording function, from this call, and (where given) also be what the function
# decodes or writes.
_MARK_SITES = {
    ("tool_system/tools/read.py", "FileReadTool.run"): ("read_bytes", "_decode_like_read_text"),
    ("tool_system/tools/read.py", "FileReadTool._read_image"): ("read_bytes", "b64encode"),
    ("tool_system/tools/read.py", "FileReadTool._read_pdf"): ("read_bytes", "b64encode"),
    ("tool_system/tools/read.py", "FileReadTool._read_notebook"): ("read_bytes", "_decode_like_read_text"),
    ("tool_system/tools/write.py", "FileWriteTool.run"): ("encode", "write_if_unchanged"),
    ("tool_system/tools/edit.py", "FileEditTool.run"): ("encode", "write_if_unchanged"),
    ("tool_system/tools/notebook_edit.py", "NotebookEditTool.run"): ("_write_notebook_atomic", None),
    ("tool_system/tools/data.py", "DataInspectTool.run"): ("_load_rows", None),
    ("tool_system/tools/data.py", "DataTransformTool.run"): ("_write_new_text", None),
    ("tool_system/tools/qwen_media.py", "_prepare_media"): ("_local_media_data_url", None),
}
_METADATA_ATTRS = {
    "st_mtime", "st_mtime_ns", "st_ctime", "st_ctime_ns", "st_birthtime", "st_birthtime_ns",
    "st_ino", "st_dev",
}


class TestFingerprintStaticGuards(unittest.TestCase):
    """Defence in depth; the behaviour tests above are the primary proof."""

    def test_exact_production_mark_file_read_inventory(self) -> None:
        found: dict[tuple[str, str], int] = {}
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for qualname, function in _functions(tree):
                for call in _calls(function):
                    if isinstance(call.func, ast.Attribute) and call.func.attr == "mark_file_read":
                        found[(rel, qualname)] = found.get((rel, qualname), 0) + 1
            for node in ast.walk(tree):  # recorders outside any function would be missed above
                if isinstance(node, ast.Call) and _call_name(node) == "mark_file_read":
                    self.assertTrue(
                        any(node in _calls(f) for _, f in _functions(tree)), f"{rel}:{node.lineno}"
                    )
        self.assertEqual(found, {site: 1 for site in _MARK_SITES})

    def test_every_recorder_passes_the_buffer_it_read_or_wrote(self) -> None:
        for (rel, qualname), (source_call, consumer) in _MARK_SITES.items():
            with self.subTest(site=f"{rel}:{qualname}"):
                function = _function(rel, qualname)
                (mark,) = [c for c in _calls(function) if _call_name(c) == "mark_file_read"]
                self.assertEqual(len(mark.args), 1)
                self.assertEqual([k.arg for k in mark.keywords], ["data"])
                data = mark.keywords[0].value
                self.assertIsInstance(data, ast.Name, "data= must be an already-held variable")
                assert isinstance(data, ast.Name)
                bound = _bindings(function, data.id)
                self.assertEqual(len(bound), 1, f"{data.id} must be bound exactly once")
                self.assertIsInstance(bound[0], ast.Call)
                assert isinstance(bound[0], ast.Call)
                self.assertEqual(_call_name(bound[0]), source_call)
                if consumer is not None:
                    self.assertTrue(
                        any(
                            _call_name(c) == consumer
                            and any(isinstance(a, ast.Name) and a.id == data.id for a in c.args)
                            for c in _calls(function)
                        ),
                        f"{consumer}() must use the same {data.id} buffer",
                    )

    def test_write_helpers_return_exactly_the_bytes_they_wrote(self) -> None:
        for rel, qualname, expected_modes in (
            ("tool_system/tools/notebook_edit.py", "_write_notebook_atomic", ["wb"]),
            ("tool_system/tools/data.py", "_write_new_text", ["xb"]),  # exclusive create
        ):
            with self.subTest(helper=qualname):
                function = _function(rel, qualname)
                written = {
                    a.id
                    for c in _calls(function)
                    if _call_name(c) == "write"
                    for a in c.args
                    if isinstance(a, ast.Name)
                }
                returned = {
                    node.value.id
                    for node in ast.walk(function)
                    if isinstance(node, ast.Return) and isinstance(node.value, ast.Name)
                }
                self.assertEqual(len(written), 1, written)
                self.assertEqual(written, returned)
                (name,) = written
                self.assertEqual(len(_bindings(function, name)), 1, f"{name} must be bound once")
                names = {_call_name(c) for c in _calls(function)}
                for banned in ("read", "read_bytes", "read_text", "readinto"):
                    self.assertNotIn(banned, names, "a writer must not read the file back")
                opens = [c for c in _calls(function) if _call_name(c) in {"open", "NamedTemporaryFile"}]
                modes = [
                    k.value.value
                    for c in opens
                    for k in c.keywords
                    if k.arg == "mode" and isinstance(k.value, ast.Constant)
                ] + [
                    a.value for c in opens for a in c.args if isinstance(a, ast.Constant) and isinstance(a.value, str)
                ]
                self.assertEqual(modes, expected_modes)

    def test_write_and_edit_compare_and_write_one_prepared_buffer(self) -> None:
        for rel, qualname in (
            ("tool_system/tools/write.py", "FileWriteTool.run"),
            ("tool_system/tools/edit.py", "FileEditTool.run"),
        ):
            with self.subTest(site=rel):
                function = _function(rel, qualname)
                names = {_call_name(c) for c in _calls(function)}
                for banned in ("write_text", "write_bytes", "open", "read_text", "read_bytes", "was_file_read_and_unchanged"):
                    self.assertNotIn(banned, names)
                (write,) = [c for c in _calls(function) if _call_name(c) == "write_if_unchanged"]
                (mark,) = [c for c in _calls(function) if _call_name(c) == "mark_file_read"]
                (verify,) = [c for c in _calls(function) if _call_name(c) == "read_if_unchanged"]
                self.assertEqual(len(write.args), 2)
                self.assertIsInstance(write.args[1], ast.Name)
                self.assertEqual(ast.dump(write.args[1]), ast.dump(mark.keywords[0].value))
                self.assertEqual([k.arg for k in write.keywords], ["expected"])
                expected = write.keywords[0].value
                self.assertIsInstance(expected, ast.Name)
                assert isinstance(expected, ast.Name)
                sources = [
                    node.value
                    for node in ast.walk(function)
                    if isinstance(node, (ast.Assign, ast.AnnAssign))
                    and any(
                        isinstance(t, ast.Name) and t.id == expected.id
                        for t in (node.targets if isinstance(node, ast.Assign) else [node.target])
                    )
                ]
                self.assertEqual(len(sources), len(_bindings(function, expected.id)), "plain assignments only")
                for value in sources:
                    self.assertTrue(
                        (isinstance(value, ast.Constant) and value.value is None) or value is verify,
                        f"{expected.id} must be the verified bytes, or None for a create",
                    )
                self.assertLess(verify.lineno, write.lineno)

    def test_mutating_consumers_never_read_the_file_again_after_the_gate(self) -> None:
        for rel, qualname in (
            ("tool_system/tools/write.py", "FileWriteTool.run"),
            ("tool_system/tools/edit.py", "FileEditTool.run"),
            ("tool_system/tools/notebook_edit.py", "NotebookEditTool.run"),
            ("tool_system/tools/data.py", "DataTransformTool.run"),
        ):
            with self.subTest(site=qualname):
                names = [_call_name(c) for c in _calls(_function(rel, qualname))]
                self.assertEqual(names.count("read_if_unchanged"), 1)
                for banned in ("was_file_read_and_unchanged", "read_text", "read_bytes", "open", "read", "_load_rows"):
                    self.assertNotIn(banned, names)

    def test_compare_and_write_never_creates_truncates_or_replaces_before_comparing(self) -> None:
        function = _function("tool_system/compare_write.py", "write_if_unchanged")
        # Exactly these calls in write_if_unchanged's own body: no repair, retry, read-back
        # or re-open there (helpers are pinned by the behaviour tests and guards below).
        self.assertEqual(
            {_call_name(c) for c in _calls(function)},
            {
                "open", "_write_all", "_fingerprint_of_bytes", "_anchored_opener", "_read_matching",
                "getattr", "fstat", "fileno", "_permission_denied", "seek", "truncate", "len",
            },
        )
        opens = [c for c in _calls(function) if _call_name(c) == "open"]
        modes = [a.value for c in opens for a in c.args[1:2] if isinstance(a, ast.Constant)]
        self.assertEqual(sorted(modes), ["r+b", "xb"])
        (in_place,) = [c for c in opens if getattr(c.args[1], "value", None) == "r+b"]
        keywords = {k.arg: k.value for k in in_place.keywords}
        self.assertEqual(getattr(keywords.get("opener"), "id", None), "opener")
        (bound,) = _bindings(function, "opener")
        self.assertIsInstance(bound, ast.IfExp)
        assert isinstance(bound, ast.IfExp)
        self.assertEqual(getattr(bound.test, "id", None), "_ON_LINUX")
        self.assertEqual(_call_name(bound.body) if isinstance(bound.body, ast.Call) else None, "_anchored_opener")
        self.assertEqual(getattr(bound.orelse, "id", None), "_open_without_blocking")
        # Order: open -> compare -> hidden/system refusal -> seek -> write -> truncate.
        line = {name: [c.lineno for c in _calls(function) if _call_name(c) == name] for name in
                ("_read_matching", "_permission_denied", "seek", "_write_all", "truncate")}
        (compare,) = line["_read_matching"]
        (refuse,) = line["_permission_denied"]
        (seek,) = line["seek"]
        (truncate,) = line["truncate"]
        in_place_write = max(line["_write_all"])
        self.assertTrue(in_place.lineno < compare < refuse < seek < in_place_write < truncate)
        # No repair path: no finally, and only "vanished"/"exists" are caught.
        self.assertFalse([n for n in ast.walk(function) if isinstance(n, ast.Try) and n.finalbody])
        caught = sorted(
            ast.unparse(h.type) for h in ast.walk(function) if isinstance(h, ast.ExceptHandler) and h.type
        )
        self.assertEqual(caught, ["(FileNotFoundError, _NotTheVerifiedFile)", "FileExistsError"])
        self.assertEqual(len([h for h in ast.walk(function) if isinstance(h, ast.ExceptHandler)]), 2)
        still = _function("tool_system/compare_write.py", "still_holds")
        (read_open,) = [c for c in _calls(still) if _call_name(c) == "open"]
        self.assertEqual(getattr(read_open.args[1], "value", None), "rb")
        still_keywords = {k.arg: k.value for k in read_open.keywords}
        self.assertEqual(getattr(still_keywords.get("opener"), "id", None), "_open_without_blocking")

    def test_linux_in_place_open_is_anchored_no_follow_and_checked_on_the_opened_file(self) -> None:
        tree = _tree("tool_system/compare_write.py")
        # Nothing in compare_write ever asks for O_CREAT (the create path is open "xb").
        self.assertFalse(
            [n for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.Attribute))
             and (getattr(n, "id", None) or getattr(n, "attr", None)) in {"O_CREAT", "O_TRUNC"}]
        )
        outer = _function("tool_system/compare_write.py", "_anchored_opener")
        (opener,) = [n for n in ast.walk(outer) if isinstance(n, ast.FunctionDef) and n.name == "opener"]
        (parent_open,) = [c for c in _calls(opener) if _call_name(c) == "open"]
        self.assertEqual(
            {n.id for n in ast.walk(parent_open.args[1]) if isinstance(n, ast.Name)},
            {"_O_PATH", "_O_DIRECTORY", "_O_NOFOLLOW", "_O_CLOEXEC"},
        )
        opened = _function("tool_system/compare_write.py", "_open_in")
        calls = _calls(opened)
        (target_open,) = [c for c in calls if _call_name(c) == "open"]
        self.assertEqual({n.id for n in ast.walk(target_open.args[1]) if isinstance(n, ast.Name)},
                         {"flags", "_NON_BLOCKING_OPEN_FLAGS", "_O_NOFOLLOW"})
        for flags in (parent_open.args[1], target_open.args[1]):  # a pure chain of "|"
            self.assertTrue(all(isinstance(n.op, ast.BitOr) for n in ast.walk(flags) if isinstance(n, ast.BinOp)))
        self.assertEqual([k.arg for k in target_open.keywords], ["dir_fd"])
        (lookup,) = [c for c in calls if _call_name(c) == "stat"]
        self.assertEqual({k.arg: getattr(k.value, "value", None) for k in lookup.keywords},
                         {"dir_fd": None, "follow_symlinks": False})
        enforced = [c for c in calls if _call_name(c) == "_enforce_protected_regular"]
        self.assertEqual([ast.unparse(c.args[2]) for c in enforced], ["found.st_uid", "opened.st_uid"])
        self.assertEqual([ast.unparse(c.args[1]) for c in enforced], ["os.fstat(parent)"] * 2)
        checks = [c.lineno for c in enforced]
        self.assertEqual(len(checks), 2)
        self.assertTrue(checks[0] < target_open.lineno < checks[1])
        # Not a regular file -> "not the verified file", checked on the lookup (before the
        # rule and the open) and again on the file actually opened.
        regular = [c for c in calls if _call_name(c) == "S_ISREG"]
        self.assertEqual([ast.unparse(c.args[0]) for c in regular], ["found.st_mode", "opened.st_mode"])
        self.assertTrue(regular[0].lineno < checks[0] and target_open.lineno < regular[1].lineno < checks[1])
        refusals = [n for n in ast.walk(opened) if isinstance(n, ast.Raise) and n.exc is not None
                    and "_NotTheVerifiedFile" in ast.unparse(n.exc)]
        self.assertEqual(len(refusals), 2)

    def test_verified_bytes_are_the_ones_the_comparison_hashed(self) -> None:
        # _read_matching keeps the chunks it hashes and returns exactly them: no seek back,
        # no second read, no other return of file content.
        function = _function("tool_system/context.py", "_read_matching")
        names = [_call_name(c) for c in _calls(function)]
        self.assertNotIn("seek", names)
        self.assertEqual(names.count("read"), 1)
        returns = [ast.unparse(n.value) for n in ast.walk(function) if isinstance(n, ast.Return) and n.value]
        self.assertEqual(sorted(set(returns)), ["None", "b''.join(kept)"])

    def test_write_and_edit_mark_only_after_the_compare_and_write_succeeded(self) -> None:
        for rel, qualname in (
            ("tool_system/tools/write.py", "FileWriteTool.run"),
            ("tool_system/tools/edit.py", "FileEditTool.run"),
        ):
            with self.subTest(site=rel):
                function = _function(rel, qualname)
                self.assertFalse([n for n in ast.walk(function) if isinstance(n, ast.Try)])
                (write,) = [c for c in _calls(function) if _call_name(c) == "write_if_unchanged"]
                (mark,) = [c for c in _calls(function) if _call_name(c) == "mark_file_read"]
                self.assertLess(write.lineno, mark.lineno)

    def test_notebook_replace_is_preceded_by_a_final_compare_with_the_verified_bytes(self) -> None:
        helper = _function("tool_system/tools/notebook_edit.py", "_write_notebook_atomic")
        (temp,) = [c for c in _calls(helper) if _call_name(c) == "NamedTemporaryFile"]
        (check,) = [c for c in _calls(helper) if _call_name(c) == "still_holds"]
        (replace,) = [c for c in _calls(helper) if _call_name(c) == "replace"]
        self.assertTrue(temp.lineno < check.lineno < replace.lineno)
        self.assertEqual([getattr(a, "id", None) for a in check.args], ["path", helper.args.args[2].arg])
        run = _function("tool_system/tools/notebook_edit.py", "NotebookEditTool.run")
        (call,) = [c for c in _calls(run) if _call_name(c) == "_write_notebook_atomic"]
        expected = call.args[2]
        self.assertIsInstance(expected, ast.Name)
        assert isinstance(expected, ast.Name)
        (bound,) = _bindings(run, expected.id)
        self.assertIsInstance(bound, ast.Call)
        assert isinstance(bound, ast.Call)
        self.assertEqual(_call_name(bound), "read_if_unchanged")


    def test_only_the_context_primitive_touches_the_fingerprint_map(self) -> None:
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            allowed: set[ast.AST] = set()
            if rel == "tool_system/context.py":
                for qualname, function in _functions(tree):
                    if qualname in {
                        "ToolContext.mark_file_read",
                        "ToolContext.was_file_read_and_unchanged",
                        "ToolContext.read_if_unchanged",
                    }:
                        allowed.update(ast.walk(function))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "read_file_fingerprints":
                    if node not in allowed:
                        offenders.append(f"{rel}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_no_timestamp_or_identity_metadata_anywhere_in_src_but_the_glob_sort(self) -> None:
        # Stricter than the fingerprint path itself: any new use must be reviewed here.
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
                if isinstance(node, ast.Attribute) and node.attr in _METADATA_ATTRS:
                    if (rel, node.attr) != ("tool_system/tools/glob.py", "st_mtime"):
                        offenders.append(f"{rel}:{node.lineno}:{node.attr}")
        self.assertEqual(offenders, [])  # glob.py's st_mtime is only a sort key

    def test_comparison_opens_in_binary_mode(self) -> None:
        tree = _tree("tool_system/context.py")
        opens = [c for c in _calls(tree) if isinstance(c.func, ast.Name) and c.func.id == "open"]
        self.assertEqual(len(opens), 1)
        (call,) = opens
        self.assertEqual([a.value for a in call.args[1:2] if isinstance(a, ast.Constant)], ["rb"])
        keywords = {k.arg: k.value for k in call.keywords}
        self.assertIsInstance(keywords.get("buffering"), ast.Constant)
        self.assertEqual(getattr(keywords.get("buffering"), "value", None), 0)
        self.assertIn("opener", keywords)
        names = {_call_name(c) for c in _calls(tree)}
        for banned in ("read_bytes", "read_text", "fdopen", "file_digest", "stat", "lstat"):
            self.assertNotIn(banned, names)
        for call in _calls(tree):
            if (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "os"
                and call.func.attr in {"open", "read"}
            ):
                self.assertEqual(call.func.attr, "open", "no raw os.read")
                flags = call.args[1]
                self.assertIsInstance(flags, ast.BinOp, "os.open must extend open()'s own flags")
                assert isinstance(flags, ast.BinOp)
                self.assertIsInstance(flags.left, ast.Name)

    def test_posix_open_flags_are_non_blocking_and_no_ctty(self) -> None:
        # Behaviourally checked only on POSIX (FIFO test); both flags are 0 on Windows.
        tree = _tree("tool_system/context.py")
        (assign,) = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_NON_BLOCKING_OPEN_FLAGS" for t in node.targets)
        ]
        names = {
            c.args[1].value
            for c in _calls(assign.value)
            if _call_name(c) == "getattr" and isinstance(c.args[1], ast.Constant)
        }
        self.assertEqual(names, {"O_NONBLOCK", "O_NOCTTY"})
        self.assertEqual(
            context_module._NON_BLOCKING_OPEN_FLAGS,
            getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOCTTY", 0),
        )

    def test_gate_fails_closed_and_mark_does_no_io(self) -> None:
        for name, refusal in (("was_file_read_and_unchanged", False), ("read_if_unchanged", None)):
            gate = _function("tool_system/context.py", f"ToolContext.{name}")
            for node in ast.walk(gate):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
                    self.assertIs(node.value.value, refusal, f"{name} line {node.lineno}")
            handlers = [n for n in ast.walk(gate) if isinstance(n, ast.ExceptHandler)]
            self.assertEqual(len(handlers), 1)
            caught = handlers[0].type
            self.assertIsInstance(caught, ast.Tuple)
            assert isinstance(caught, ast.Tuple)
            self.assertEqual(
                sorted(e.id for e in caught.elts if isinstance(e, ast.Name)),
                ["OSError", "RuntimeError", "ValueError"],
            )
        mark = _function("tool_system/context.py", "ToolContext.mark_file_read")
        self.assertEqual({_call_name(c) for c in _calls(mark)}, {"_fingerprint_of_bytes", "resolve"})

    def test_read_evaluates_the_unchanged_gate_last(self) -> None:
        run = _function("tool_system/tools/read.py", "FileReadTool.run")
        gated = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.If)
            and any(_call_name(c) == "was_file_read_and_unchanged" for c in _calls(node.test))
        ]
        self.assertEqual(len(gated), 1)
        test = gated[0].test
        self.assertIsInstance(test, ast.BoolOp)
        assert isinstance(test, ast.BoolOp)
        self.assertIsInstance(test.op, ast.And)
        last = test.values[-1]
        self.assertTrue(isinstance(last, ast.Call) and _call_name(last) == "was_file_read_and_unchanged")
        self.assertEqual(
            sum(1 for value in test.values if any(_call_name(c) == "was_file_read_and_unchanged" for c in _calls(value))),
            1,
        )
        self.assertEqual(len(test.values), 4)
        # ... and only after image/PDF/notebook reads have been dispatched (never stubbed).
        dispatch = [
            node
            for node in ast.walk(run)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and any(isinstance(c, ast.Constant) and c.value == ".ipynb" for c in node.test.comparators)
        ]
        self.assertEqual(len(dispatch), 1)
        self.assertGreater(gated[0].lineno, dispatch[0].lineno)


# --------------------------------------------------------------------------------------
# Linux fs.protected_regular (carried over from the evidence probe: GitHub Actions run
# 36399673306, ubuntu24 image, kernel 6.17, fs.protected_regular=2)
#
# Today's Write/Edit overwrite opened the file with open(path, "wb") (O_WRONLY|O_CREAT|
# O_TRUNC). compare_write opens it in place without O_CREAT, which fs.protected_regular
# (fs/namei.c may_create_in_sticky) does not check: it refuses only O_CREAT opens of an
# existing regular file owned by neither the caller (fsuid) nor the directory owner, in a
# sticky directory that is world-writable (level >= 1) or group-writable (level 2). The
# probe VERIFIED that widening on that runner, so compare_write applies the rule itself
# (compare_write._protected_regular_refusal).
#
# Test 1 records the raw kernel behaviour (a path-based in-place open is let through where
# open("wb") is refused: RAW_WIDENING, expected). Test 2 is the proof that matters: Write
# and Edit refuse exactly like open("wb") on every shape, before opening the file, with at
# least one refusal exercised. A replacement between compare_write's check and its open is
# re-checked on the file actually opened. Nothing here changes a sysctl, so the kernel
# evidence covers only the level the runner reports (TestProtectedRegularPolicy covers the
# rule's other cases without a kernel). Files and directories owned by other uids (64001,
# 64002; never the overflow uid) are made with "sudo -n chown" inside this test's own
# private temporary directory, only on GitHub-hosted runners or with
# TEST_PROTECTED_REGULAR_SUDO=1 (a run as root uses os.lchown there without an opt-in).
# On Linux, where evidence is required (GitHub-hosted runners, or
# TEST_PROTECTED_REGULAR_REQUIRED=1), a missing prerequisite FAILS instead of skipping.
# Tests 1 and 2 emit one annotation each only when GitHub Actions runs them with
# TEST_PROTECTED_REGULAR_REQUIRED=1 (the probe's evidence step did; no step on this
# branch sets it).
# --------------------------------------------------------------------------------------
_PR_SYSCTL = "/proc/sys/fs/protected_regular"
_PR_OVERFLOWUID = "/proc/sys/kernel/overflowuid"
_PR_SUDO_OPT_IN = "TEST_PROTECTED_REGULAR_SUDO"
_PR_REQUIRED = "TEST_PROTECTED_REGULAR_REQUIRED"
_PR_FOREIGN_UID = 64001  # owns the target files
_PR_THIRD_UID = 64002  # owns some directories (the /tmp shape: neither caller nor file owner)
_PR_ORIGINAL = b"head\nkey=old\ntail\n"
_PR_CHANGED = b"HEAD" + _PR_ORIGINAL[4:]
_PR_REFUSE_EDIT = "refusing to edit: file must be read first and unchanged since last read"
_PR_REFUSE_WRITE = "refusing to overwrite: file must be read first and unchanged since last read"
_PR_ON_LINUX = sys.platform.startswith("linux")
_PR_GATE_KIND = {"EXPECTED": "notice", "CLEAR": "notice", "BLOCKED": "error"}


def _pr_github_hosted() -> bool:
    return (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
    )


def _pr_required() -> bool:
    return _pr_github_hosted() or os.environ.get(_PR_REQUIRED) == "1"


def _pr_sudo_allowed() -> bool:
    return _pr_github_hosted() or os.environ.get(_PR_SUDO_OPT_IN) == "1"


def _pr_read_proc(path: str) -> str:
    try:
        with open(path, "rb") as handle:
            return handle.read(32).decode("ascii", "replace").strip()
    except OSError as exc:
        return f"<unreadable {type(exc).__name__} errno={exc.errno}>"


def _pr_env_report() -> str:
    return (
        f"protected_regular={_pr_read_proc(_PR_SYSCTL)} "
        f"overflowuid={_pr_read_proc(_PR_OVERFLOWUID)} "
        f"euid={os.geteuid() if hasattr(os, 'geteuid') else None} "
        f"kernel={os.uname().release if hasattr(os, 'uname') else None} "
        f"image={os.environ.get('ImageOS')}/{os.environ.get('ImageVersion')} "
        f"GITHUB_ACTIONS={os.environ.get('GITHUB_ACTIONS')} "
        f"RUNNER_ENVIRONMENT={os.environ.get('RUNNER_ENVIRONMENT')} "
        f"{_PR_REQUIRED}={os.environ.get(_PR_REQUIRED)} "
        f"capture={'off' if sys.stdout is sys.__stdout__ else 'on'} "
        f"required={_pr_required()} sudo_allowed={_pr_sudo_allowed()} "
        f"python={sys.version.split()[0]} platform={sys.platform}"
    )


def _pr_escape(text: str, *, property_value: bool = False) -> str:
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if property_value else text


def _pr_annotate(kind: str, title: str, message: str) -> None:
    """Only when GitHub Actions runs with TEST_PROTECTED_REGULAR_REQUIRED=1 (the probe's
    evidence step did; no step on this branch sets it): one workflow-command annotation on
    its own line, message bounded to 3500 characters."""
    if os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get(_PR_REQUIRED) != "1":
        return
    title = f"{title} py{sys.version_info.major}.{sys.version_info.minor}"
    print(
        f"\n::{kind} title={_pr_escape(title, property_value=True)}::{_pr_escape(message[:3500])}",
        flush=True,
    )


def _pr_require(testcase: unittest.TestCase, ok: bool, what: str) -> None:
    """A missing prerequisite FAILS where evidence is required and skips elsewhere."""
    if ok:
        return
    detail = f"{_pr_env_report()} | {what}"
    if _pr_required():
        _pr_annotate("error", f"protected_regular prerequisite {type(testcase).__name__}", detail)
        testcase.fail(f"prerequisite missing where evidence is required (never skipped): {detail}")
    raise unittest.SkipTest(detail)


def _pr_level(testcase: unittest.TestCase) -> int:
    text = _pr_read_proc(_PR_SYSCTL)
    _pr_require(testcase, text in ("0", "1", "2"), f"unexpected {_PR_SYSCTL} value {text!r}")
    return int(text)


def _pr_kernel_refuses_o_creat(
    level: int, dir_mode: int, dir_uid: int, file_uid: int, fsuid: int
) -> bool:
    """Oracle transcribed from fs/namei.c may_create_in_sticky() for an EXISTING regular
    file opened with O_CREAT and without O_EXCL. No capability exempts the caller.
    Test 1 checks it against the runner's kernel, at the runner's level only."""
    if level == 0 or not dir_mode & stat.S_ISVTX:
        return False
    if file_uid == dir_uid or file_uid == fsuid:
        return False
    if dir_mode & stat.S_IWOTH:
        return True
    return level >= 2 and bool(dir_mode & stat.S_IWGRP)


def _pr_run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60, check=False
    )


class _PrOwners:
    """Gives paths inside this test's own private temporary directory to other uids and,
    when sudo was used, the whole tree back to the caller before it is removed (root
    needs no give-back)."""

    def __init__(self, testcase: unittest.TestCase, root: Path) -> None:
        self.root = root
        self.euid, self.egid = os.geteuid(), os.getegid()
        overflow = _pr_read_proc(_PR_OVERFLOWUID)
        _pr_require(testcase, overflow.isdigit(), f"cannot read {_PR_OVERFLOWUID}: {overflow!r}")
        foreign = {_PR_FOREIGN_UID, _PR_THIRD_UID}
        _pr_require(
            testcase,
            len(foreign) == 2 and not foreign & {int(overflow), self.euid},
            f"foreign uids {sorted(foreign)} must differ from each other, from the overflow "
            f"uid {overflow} and from the caller {self.euid}",
        )
        info = os.lstat(root)
        _pr_require(
            testcase,
            root.parent == Path(tempfile.gettempdir()).resolve()
            and stat.S_ISDIR(info.st_mode)
            and info.st_uid == self.euid
            and stat.S_IMODE(info.st_mode) == 0o700,
            f"ownership changes only inside a fresh private temporary directory, not {root}",
        )
        self.sudo_chown: list[str] | None = None
        if self.euid == 0:
            return  # root: os.lchown, no sudo
        _pr_require(
            testcase,
            _pr_sudo_allowed(),
            f"files owned by other uids need 'sudo -n chown': not a GitHub-hosted runner "
            f"and {_PR_SUDO_OPT_IN}=1 is not set",
        )
        sudo, chown = shutil.which("sudo"), shutil.which("chown")
        _pr_require(testcase, sudo is not None and chown is not None, "sudo and chown on PATH")
        assert sudo is not None and chown is not None
        probe = _pr_run([sudo, "-n", "true"])
        _pr_require(
            testcase,
            probe.returncode == 0,
            f"non-interactive sudo unavailable: rc={probe.returncode} {probe.stderr.strip()!r}",
        )
        self.sudo_chown = [sudo, "-n", chown]

    def give(self, testcase: unittest.TestCase, uid: int, paths: list[Path]) -> None:
        for path in paths:
            _pr_require(
                testcase,
                path != self.root
                and path.is_relative_to(self.root)
                and not path.is_symlink()
                and path.resolve() == path,
                f"refusing an ownership change outside the test directory: {path}",
            )
        try:
            if self.sudo_chown is None:
                for path in paths:
                    os.lchown(path, uid, uid)
            else:
                done = _pr_run([*self.sudo_chown, "-h", f"{uid}:{uid}", "--", *map(str, paths)])
                _pr_require(
                    testcase,
                    done.returncode == 0,
                    f"sudo chown to {uid} failed: rc={done.returncode} {done.stderr.strip()!r}",
                )
        except OSError as exc:
            _pr_require(testcase, False, f"chown to uid {uid} failed: {exc!r}")
        for path in paths:
            owner = os.lstat(path).st_uid
            _pr_require(testcase, owner == uid, f"{path} is owned by uid {owner}, not {uid}")

    def take_back(self) -> str | None:
        if self.sudo_chown is None:
            return None  # root may delete in any sticky directory
        done = _pr_run([*self.sudo_chown, "-R", f"{self.euid}:{self.egid}", "--", str(self.root)])
        return None if done.returncode == 0 else f"rc={done.returncode} {done.stderr.strip()!r}"


@dataclass(frozen=True)
class _PrShape:
    name: str
    dir_mode: int
    dir_owner: str  # "caller", "foreign" (owner of the files) or "third"
    file_owner: str


_PR_SHAPES = (
    # Refused at level >= 1: sticky, world-writable, file owned by neither caller nor dir owner.
    _PrShape("sticky1777_callerdir_foreignfile", 0o1777, "caller", "foreign"),
    # The /tmp shape: the sticky world-writable directory belongs to a third uid.
    _PrShape("sticky1777_thirddir_foreignfile", 0o1777, "third", "foreign"),
    # Refused at level 2 only: sticky and group-writable, not world-writable.
    _PrShape("sticky1770_callerdir_foreignfile", 0o1770, "caller", "foreign"),
    # Never refused: group-writable but not sticky.
    _PrShape("nonsticky0770_callerdir_foreignfile", 0o0770, "caller", "foreign"),
    # Never refused: sticky but neither group- nor world-writable.
    _PrShape("sticky1755_callerdir_foreignfile", 0o1755, "caller", "foreign"),
    # Never refused: world-writable but not sticky.
    _PrShape("nonsticky0777_callerdir_foreignfile", 0o0777, "caller", "foreign"),
    # Never refused: the directory owner owns the file.
    _PrShape("sticky1777_fileownerdir_foreignfile", 0o1777, "foreign", "foreign"),
    # Never refused: the caller owns the file.
    _PrShape("sticky1777_thirddir_callerfile", 0o1777, "third", "caller"),
)


@unittest.skipUnless(_PR_ON_LINUX, "Linux fs.protected_regular (fs/namei.c may_create_in_sticky)")
class _ProtectedRegularBase(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.owners = _PrOwners(self, self.root)
        self.addCleanup(self._take_back)  # LIFO: runs before temporary.cleanup

    def _take_back(self) -> None:
        problem = self.owners.take_back()
        if problem is not None:
            self.fail(f"could not return {self.root} to uid {self.owners.euid}: {problem}")

    def _uid(self, who: str) -> int:
        return {"caller": self.owners.euid, "foreign": _PR_FOREIGN_UID, "third": _PR_THIRD_UID}[who]

    def build(
        self,
        shape: _PrShape,
        names: Iterable[str],
        prepare: Callable[[dict[str, Path]], None] | None = None,
    ) -> tuple[dict[str, Path], str, bool]:
        """Files (0666, bytes _PR_ORIGINAL) in a directory of the shape's mode and owners.
        ``prepare`` runs while the caller still owns everything. Returns the files, a fact
        line, and whether the oracle says O_CREAT opens are refused at the runner's level."""
        folder = self.root / shape.name
        folder.mkdir()
        files: dict[str, Path] = {}
        for name in names:
            path = folder / f"{name}.txt"
            path.write_bytes(_PR_ORIGINAL)
            os.chmod(path, 0o666)
            files[name] = path
        if prepare is not None:
            prepare(files)
        os.chmod(folder, shape.dir_mode)
        file_uid, dir_uid = self._uid(shape.file_owner), self._uid(shape.dir_owner)
        if shape.file_owner != "caller":
            self.owners.give(self, file_uid, list(files.values()))
        if shape.dir_owner != "caller":
            self.owners.give(self, dir_uid, [folder])
        info = os.stat(folder)
        file_uids = {os.stat(path).st_uid for path in files.values()}
        file_modes = {stat.S_IMODE(os.stat(path).st_mode) for path in files.values()}
        _pr_require(
            self,
            stat.S_IMODE(info.st_mode) == shape.dir_mode
            and info.st_uid == dir_uid
            and file_uids == {file_uid}
            and file_modes == {0o666},
            f"fixture {shape.name}: dir mode {oct(stat.S_IMODE(info.st_mode))} uid {info.st_uid}, "
            f"file uids {sorted(file_uids)} modes {sorted(map(oct, file_modes))}",
        )
        refuses = _pr_kernel_refuses_o_creat(
            _pr_level(self), info.st_mode, info.st_uid, file_uid, self.owners.euid
        )
        facts = f"dir {oct(stat.S_IMODE(info.st_mode))} uid {dir_uid} file uid {file_uid}"
        return files, facts, refuses


# The raw probes, on sacrificial files, by path: today's open("wb"), O_CREAT alone, in-place
# opens (open("r+b"), and os.open with O_RDWR|O_CLOEXEC|O_NONBLOCK|O_NOCTTY), and O_TRUNC
# without O_CREAT. They show the kernel's rule only; compare_write's own open (relative to a
# directory handle, with O_NOFOLLOW) is exercised through the tools by tests 2 and 3.
_PR_PROBES: dict[str, Callable[[Path], Any]] = {
    "wb": lambda p: open(p, "wb"),
    "creat": lambda p: os.open(p, os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC, 0o666),
    "rplusb": lambda p: open(p, "r+b"),
    "rdwr_nb": lambda p: os.open(p, os.O_RDWR | os.O_CLOEXEC | os.O_NONBLOCK | os.O_NOCTTY),
    "trunc": lambda p: os.open(p, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC),
}
_PR_O_CREAT = frozenset({"wb", "creat"})
_PR_IN_PLACE = ("rplusb", "rdwr_nb")
_PR_TRUNCATING = frozenset({"wb", "trunc"})


def _pr_attempt(open_it: Callable[[Path], Any], path: Path) -> str:
    try:
        opened = open_it(path)
    except OSError as exc:
        return f"refused:{type(exc).__name__}:{exc.errno}"
    if isinstance(opened, int):
        os.close(opened)
    else:
        opened.close()
    return "opened"


def _pr_short(outcome: str) -> str:
    """Compact form for the annotation: R<errno>/<size> refused, O/<size> opened."""
    return outcome.replace("refused:PermissionError:", "R").replace("opened", "O")


def _pr_kernel_verdict(
    level: int, observed: dict[str, dict[str, str]], matches: bool
) -> tuple[str, str, list[str]]:
    """The raw kernel behaviour on validated fixtures: open("wb") refused where an in-place
    open is allowed -> RAW_WIDENING (EXPECTED: why compare_write applies the rule itself);
    no refusal at all (level 0) -> nothing to compare (BLOCKED); refusals but every in-place
    open also refused and the oracle matched -> NO_WIDENING (CLEAR); anything else ->
    BLOCKED. Returns (verdict, gate, widening shapes)."""
    refused_wb = [n for n, row in observed.items() if row["wb"].startswith("refused")]
    widening = [
        n for n in refused_wb if any(observed[n][p].startswith("opened") for p in _PR_IN_PLACE)
    ]
    if widening:
        return "RAW_WIDENING", "EXPECTED", widening
    if level == 0 or not refused_wb:
        return "INCONCLUSIVE_NO_REFUSAL", "BLOCKED", widening
    if not matches:
        return "NO_WIDENING_ORACLE_MISMATCH", "BLOCKED", widening
    return "NO_WIDENING", "CLEAR", widening


class TestProtectedRegularKernelEvidence(_ProtectedRegularBase):
    """Test 1: raw opens only. The kernel lets an in-place open through where today's
    open('wb') is refused (RAW_WIDENING, expected). The test fails on invalid evidence
    (fixture, level 0 where required, the kernel disagreeing with the transcribed rule) or
    a failed ownership give-back."""

    def test_o_creat_open_vs_in_place_open_of_another_users_file(self) -> None:
        title = "protected_regular kernel evidence"
        observed: dict[str, dict[str, str]] = {}
        expected: dict[str, dict[str, str]] = {}
        facts: dict[str, str] = {}
        try:
            level = _pr_level(self)
            for shape in _PR_SHAPES:
                files, facts[shape.name], refuses = self.build(shape, _PR_PROBES)
                observed[shape.name], expected[shape.name] = {}, {}
                for probe, open_it in _PR_PROBES.items():
                    outcome = _pr_attempt(open_it, files[probe])
                    observed[shape.name][probe] = f"{outcome}/{len(files[probe].read_bytes())}"
                    if probe in _PR_O_CREAT and refuses:
                        want = f"refused:PermissionError:{errno.EACCES}/{len(_PR_ORIGINAL)}"
                    else:
                        want = f"opened/{0 if probe in _PR_TRUNCATING else len(_PR_ORIGINAL)}"
                    expected[shape.name][probe] = want
        except unittest.SkipTest:
            raise
        except BaseException as exc:
            _pr_annotate("error", f"{title} ABORTED", f"{_pr_env_report()} | {exc!r} | partial={observed}")
            raise
        matches = observed == expected
        verdict, gate, widening = _pr_kernel_verdict(level, observed, matches)
        mismatches = [
            f"{s}.{p}: got {_pr_short(observed[s][p])} want {_pr_short(expected[s][p])}"
            for s in observed
            for p in observed[s]
            if observed[s][p] != expected[s][p]
        ]
        report = (
            f"verdict={verdict} gate={gate} oracle={'MATCH' if matches else 'MISMATCH'} "
            f"level={level} widening_shapes={widening} | {_pr_env_report()} | "
            f"mismatches={mismatches} | "
            + " ; ".join(
                f"{name} [{facts[name]}]: " + " ".join(f"{p}={_pr_short(v)}" for p, v in row.items())
                for name, row in observed.items()
            )
        )
        _pr_annotate(_PR_GATE_KIND[gate], f"{title} {verdict} gate {gate}", report)
        self.assertEqual(observed, expected, report)
        if _pr_required():
            self.assertIn(level, (1, 2), f"level 0 cannot prove or disprove the widening | {report}")


def _pr_error_fields(exc: BaseException | None) -> tuple[Any, ...] | None:
    if exc is None:
        return None
    return (
        type(exc),
        exc.args,
        getattr(exc, "errno", None),
        getattr(exc, "strerror", None),
        getattr(exc, "filename", None),
        getattr(exc, "filename2", None),
        getattr(exc, "winerror", None),
        str(exc),
        exc.__cause__ is None,
        exc.__context__ is None,
        exc.__suppress_context__,
        getattr(exc, "__notes__", None),
    )


def _pr_open_wb_error(path: Path) -> OSError | None:
    """What today's write path (open(path, 'wb')) raises on ``path``, or None."""
    try:
        open(path, "wb").close()
    except OSError as exc:
        return exc
    return None


def _pr_rebuilt(reference: OSError, path: Path) -> OSError:
    return type(reference)(reference.errno, reference.strerror, str(path))


def _pr_run_tool(ctx: ToolContext, run: Callable[[], Any], target: Path) -> dict[str, Any]:
    before = dict(ctx.read_file_fingerprints)
    raised: BaseException | None = None
    try:
        run()
    except Exception as exc:  # compared field by field against open('wb')
        raised = exc
    return {
        "raised": raised,
        "bytes": target.read_bytes(),
        "marked": ctx.read_file_fingerprints != before,
        "gate": ctx.was_file_read_and_unchanged(target),
    }


def _pr_parity_problems(
    tool: str,
    result: dict[str, Any],
    reference: OSError | None,
    reference_path: Path,
    target: Path,
    written: bytes,
) -> list[str]:
    """Empty when ``tool`` behaved exactly like open('wb') did on the reference file."""
    raised = result["raised"]
    if reference is None:
        problems = [] if raised is None else [f"NARROWING: {tool} raised {_pr_error_fields(raised)}"]
        if result["bytes"] != written:
            problems.append(f"{tool} wrote {result['bytes']!r}, expected {written!r}")
        if not result["gate"]:
            problems.append(f"{tool}: gate False after a successful write")
        return problems
    problems = []
    if _pr_error_fields(_pr_rebuilt(reference, reference_path)) != _pr_error_fields(reference):
        problems.append(f"harness: rebuilt reference differs from {_pr_error_fields(reference)}")
    if raised is None:
        problems.append(f"WIDENING: {tool} succeeded where open('wb') raised {_pr_error_fields(reference)}")
    elif _pr_error_fields(raised) != _pr_error_fields(_pr_rebuilt(reference, target)):
        problems.append(f"{tool} raised {_pr_error_fields(raised)}")
    if result["bytes"] != _PR_ORIGINAL:
        problems.append(f"{tool}: file touched: {result['bytes']!r}")
    if result["marked"]:
        problems.append(f"{tool}: fingerprints changed after a refused write")
    if not result["gate"]:
        problems.append(f"{tool}: gate False although the file is untouched")
    return problems


def _pr_read_ok(testcase: unittest.TestCase, ctx: ToolContext, path: Path, facts: str) -> None:
    result = FileReadTool().run({"file_path": str(path)}, ctx)
    testcase.assertFalse(result.is_error, f"fixture Read failed: {result.output} | {facts}")
    testcase.assertTrue(ctx.was_file_read_and_unchanged(path), f"fixture not recorded | {facts}")


def _pr_write(ctx: ToolContext, path: Path) -> Callable[[], Any]:
    return lambda: FileWriteTool().run({"file_path": str(path), "content": "written\n"}, ctx)


def _pr_edit(ctx: ToolContext, path: Path) -> Callable[[], Any]:
    return lambda: FileEditTool().run(
        {"file_path": str(path), "old_string": "key=old", "new_string": "key=new"}, ctx
    )


_PR_WRITTEN = "written\n".replace("\n", os.linesep).encode("utf-8")
_PR_EDITED = _PR_ORIGINAL.replace(b"key=old", b"key=new").replace(b"\n", os.linesep.encode())


class TestProtectedRegularToolParity(_ProtectedRegularBase):
    """Test 2: Write and Edit (compare-and-write) behave exactly like today's open('wb')
    on every shape, the refusals are the ones the oracle predicts (EACCES), and a changed
    file gets the refusal text first. Where evidence is required, at least one refusal
    must have been exercised."""

    def test_write_and_edit_match_open_wb_on_every_shape(self) -> None:
        title = "protected_regular tool parity"
        names = ("reference", "write", "edit", "changed_write", "changed_edit")
        rows: list[str] = []
        outcomes: list[tuple[str, str, list[str]]] = []
        refusals = 0
        try:
            for shape in _PR_SHAPES:
                ctx = ToolContext(workspace_root=self.root)

                def prepare(files: dict[str, Path], ctx: ToolContext = ctx) -> None:
                    # Read, then change two files, while the caller still owns them.
                    for name in names[1:]:
                        _pr_read_ok(self, ctx, files[name], shape.name)
                    for name in ("changed_write", "changed_edit"):
                        files[name].write_bytes(_PR_CHANGED)

                files, facts, refuses = self.build(shape, names, prepare)
                reference = _pr_open_wb_error(files["reference"])
                refusals += reference is not None
                problems: list[str] = []
                if (reference is not None) != refuses:
                    problems.append(f"open('wb') {'refused' if reference else 'opened'}; oracle refuses={refuses}")
                if reference is not None and reference.errno != errno.EACCES:
                    problems.append(f"open('wb') refused with errno {reference.errno}, not EACCES")
                for tool, run, target, written in (
                    ("Write", _pr_write(ctx, files["write"]), files["write"], _PR_WRITTEN),
                    ("Edit", _pr_edit(ctx, files["edit"]), files["edit"], _PR_EDITED),
                ):
                    anchored: list[str] = []
                    real_os_open = os.open

                    def recording_open(name: Any, flags: int, *args: Any, **kwargs: Any) -> int:
                        if kwargs.get("dir_fd") is not None:
                            anchored.append(str(name))
                        return real_os_open(name, flags, *args, **kwargs)

                    with patch.object(os, "open", recording_open):
                        result = _pr_run_tool(ctx, run, target)
                    if reference is not None and anchored:
                        problems.append(f"{tool}: opened the file before refusing it ({anchored})")
                    problems += _pr_parity_problems(
                        tool, result, reference, files["reference"], target, written
                    )
                for tool, make, name, text in (
                    ("Write", _pr_write, "changed_write", _PR_REFUSE_WRITE),
                    ("Edit", _pr_edit, "changed_edit", _PR_REFUSE_EDIT),
                ):
                    result = _pr_run_tool(ctx, make(ctx, files[name]), files[name])
                    raised = result["raised"]
                    if not (isinstance(raised, ToolInputError) and str(raised) == text):
                        problems.append(f"{tool} changed file: expected the refusal, got {raised!r}")
                    if result["bytes"] != _PR_CHANGED or result["marked"] or result["gate"]:
                        problems.append(
                            f"{tool} changed file: bytes {result['bytes']!r} "
                            f"marked={result['marked']} gate={result['gate']}"
                        )
                ref = "none" if reference is None else f"{type(reference).__name__}:{reference.errno}"
                detail = "; ".join(problems)[:300] or "none"
                rows.append(f"{shape.name} [{facts}]: open_wb={ref} problems={detail}")
                outcomes.append((shape.name, facts, problems))
        except unittest.SkipTest:
            raise
        except BaseException as exc:
            _pr_annotate("error", f"{title} ABORTED", f"{_pr_env_report()} | {exc!r} | partial={rows}")
            raise
        clean = all(not problems for _, _, problems in outcomes)
        if not clean:
            verdict, gate = "PARITY_MISMATCH", "BLOCKED"
        elif refusals == 0 and _pr_required():
            verdict, gate = "INCONCLUSIVE_NO_REFUSAL", "BLOCKED"
        else:
            verdict, gate = "PARITY", "CLEAR"
        report = (
            f"verdict={verdict} gate={gate} refusals_exercised={refusals} | {_pr_env_report()} | "
            + " ; ".join(rows)
        )
        _pr_annotate(_PR_GATE_KIND[gate], f"{title} {verdict} refusals {refusals}", report)
        for name, facts, problems in outcomes:
            with self.subTest(shape=name):
                self.assertEqual(problems, [], f"{facts} | {report}")
        self.assertTrue(clean, report)  # the parent fails too, not only SUBFAILED lines
        if _pr_required():
            self.assertGreaterEqual(refusals, 1, f"no refusal was exercised | {report}")


class TestProtectedRegularReplacementIsRechecked(_ProtectedRegularBase):
    """A file replaced between compare_write's check and its open is checked again on the
    file actually opened: a caller-owned file (allowed) swapped for another user's file
    with the identical bytes (refused today) is refused and never written."""

    def test_replacement_after_the_pre_open_check_is_refused(self) -> None:
        level = _pr_level(self)
        _pr_require(self, level in (1, 2), f"level {level} refuses nothing to swap in")
        shape = _PrShape("sticky1777_callerdir_swap", 0o1777, "caller", "caller")
        files, facts, _refuses = self.build(shape, ("target", "stand_in", "reference"))
        self.owners.give(self, _PR_FOREIGN_UID, [files["stand_in"], files["reference"]])
        ctx = ToolContext(workspace_root=self.root)
        _pr_read_ok(self, ctx, files["target"], facts)
        want = _pr_open_wb_error(files["reference"])
        self.assertIsNotNone(want, f"open('wb') of another user's file must be refused | {facts}")
        assert want is not None
        real_open = os.open
        swapped: list[str] = []

        def swap_then_open(name: Any, flags: int, *args: Any, **kwargs: Any) -> int:
            if kwargs.get("dir_fd") is not None and name == files["target"].name and not swapped:
                os.replace(files["stand_in"], files["target"])
                swapped.append(name)
            return real_open(name, flags, *args, **kwargs)

        before = dict(ctx.read_file_fingerprints)
        with patch.object(os, "open", swap_then_open):
            with self.assertRaises(PermissionError) as caught:
                _pr_edit(ctx, files["target"])()
        self.assertEqual(swapped, [files["target"].name], "the swap did not happen after the check")
        self.assertEqual(
            _pr_error_fields(caught.exception), _pr_error_fields(_pr_rebuilt(want, files["target"]))
        )
        self.assertEqual(files["target"].read_bytes(), _PR_ORIGINAL)
        self.assertEqual(os.stat(files["target"]).st_uid, _PR_FOREIGN_UID)
        self.assertEqual(ctx.read_file_fingerprints, before)


class TestReadOnlyParityControl(unittest.TestCase):
    """Harness control on every platform, no privileges: on a read-only file, Write and
    Edit raise exactly what open('wb') raises, untouched and unmarked."""

    def test_read_only_file_matches_open_wb(self) -> None:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            self.skipTest("root ignores file permission bits")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        ctx = ToolContext(workspace_root=root)
        paths = {name: root / f"{name}.txt" for name in ("reference", "write", "edit")}
        for path in paths.values():
            path.write_bytes(_PR_ORIGINAL)
        for name in ("write", "edit"):
            _pr_read_ok(self, ctx, paths[name], "read-only control")
        for path in paths.values():
            os.chmod(path, stat.S_IREAD)
            self.addCleanup(os.chmod, path, stat.S_IREAD | stat.S_IWRITE)
        reference = _pr_open_wb_error(paths["reference"])
        self.assertIsNotNone(reference, "precondition: open('wb') of a read-only file is refused")
        for tool, run, target, written in (
            ("Write", _pr_write(ctx, paths["write"]), paths["write"], _PR_WRITTEN),
            ("Edit", _pr_edit(ctx, paths["edit"]), paths["edit"], _PR_EDITED),
        ):
            result = _pr_run_tool(ctx, run, target)
            self.assertEqual(
                _pr_parity_problems(tool, result, reference, paths["reference"], target, written),
                [],
            )


if __name__ == "__main__":
    unittest.main()
