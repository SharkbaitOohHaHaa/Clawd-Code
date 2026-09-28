"""Read-state fingerprints bind the exact bytes each tool read or wrote.

A path counts as "read and unchanged" only when it is a regular file whose
current bytes are exactly the recorded bytes (byte length plus full SHA-256).
Metadata may prove a change (not a regular file, different size) but never
equality. Content-change cases restore the previous mtime with ``os.utime``, and
the same-size ones force the exact old ``(st_mtime_ns, st_size)`` collision, so
these safety proofs never depend on timestamp granularity or scheduling.

Not covered here, and not solved by the fingerprint: a change that lands after
the check but before a tool re-reads or writes the file (compare -> mutation
window), partial views (offset/limit, previews), a read and a mutation in the
same assistant batch, and fingerprint lifetime across context loss.
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

    with patch.object(pathlib.Path, "open", fake_open), patch.object(os, "replace", fake_replace):
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
    ("tool_system/tools/write.py", "FileWriteTool.run"): ("encode", "write_bytes"),
    ("tool_system/tools/edit.py", "FileEditTool.run"): ("encode", "write_bytes"),
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

    def test_write_and_edit_write_and_mark_one_prepared_buffer(self) -> None:
        for rel, qualname in (
            ("tool_system/tools/write.py", "FileWriteTool.run"),
            ("tool_system/tools/edit.py", "FileEditTool.run"),
        ):
            with self.subTest(site=rel):
                function = _function(rel, qualname)
                names = {_call_name(c) for c in _calls(function)}
                self.assertNotIn("write_text", names)
                self.assertNotIn("open", names)
                (write,) = [c for c in _calls(function) if _call_name(c) == "write_bytes"]
                (mark,) = [c for c in _calls(function) if _call_name(c) == "mark_file_read"]
                self.assertIsInstance(write.args[0], ast.Name)
                self.assertEqual(ast.dump(write.args[0]), ast.dump(mark.keywords[0].value))

    def test_only_the_context_primitive_touches_the_fingerprint_map(self) -> None:
        offenders = []
        for path in sorted(SRC.rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            allowed: set[ast.AST] = set()
            if rel == "tool_system/context.py":
                for qualname, function in _functions(tree):
                    if qualname in {"ToolContext.mark_file_read", "ToolContext.was_file_read_and_unchanged"}:
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
        gate = _function("tool_system/context.py", "ToolContext.was_file_read_and_unchanged")
        for node in ast.walk(gate):
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant):
                self.assertIs(node.value.value, False, f"line {node.lineno}")
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
# Linux fs.protected_regular evidence (vector-16 evidence gate; probe branch, tests only)
#
# Today Write/Edit overwrite through Path.write_bytes -> open(path, "wb")
# (O_WRONLY|O_CREAT|O_TRUNC). The planned in-place compare-and-write (not on this branch)
# would open the existing file with open(path, "r+b") (O_RDWR, no O_CREAT). Linux
# fs.protected_regular (fs/namei.c may_create_in_sticky) refuses only O_CREAT opens of an
# existing regular file owned by neither the caller (fsuid) nor the directory owner, in a
# sticky directory that is world-writable (level >= 1) or group-writable (level 2).
# Test 1 records, on the CI kernel, whether the in-place open is allowed where today's
# open("wb") is refused; test 2 pins that today's Write/Edit behave exactly like open("wb").
#
# Nothing here changes a sysctl, so the evidence covers only the level the runner reports.
# Files and directories owned by other uids (64001, 64002; never the overflow uid) are made
# with "sudo -n chown" inside this test's own private temporary directory, only on
# GitHub-hosted runners or with TEST_PROTECTED_REGULAR_SUDO=1. On Linux, where evidence is
# required (GitHub-hosted runners, or TEST_PROTECTED_REGULAR_REQUIRED=1), a missing
# prerequisite FAILS instead of skipping. In the CI evidence step (GitHub Actions with
# TEST_PROTECTED_REGULAR_REQUIRED=1), tests 1 and 2 each emit one verdict annotation, plus
# an error annotation for a prerequisite failure or an abort; annotations are readable
# through the public check-runs API. Verdict -> gate: WIDENING -> STOP (warning),
# NO_WIDENING -> CLEAR for the reported level (notice), anything else -> BLOCKED (error).
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
_PR_GATE_KIND = {"STOP": "warning", "CLEAR": "notice", "BLOCKED": "error"}


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
    """Only in the CI evidence step (GitHub Actions with TEST_PROTECTED_REGULAR_REQUIRED=1):
    one workflow-command annotation on its own line, message bounded to 3500 characters."""
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


# The probes, on sacrificial files: today's open("wb"), O_CREAT alone, the planned in-place
# opens (open("r+b"), and os.open with O_RDWR|O_CLOEXEC|O_NONBLOCK|O_NOCTTY, the flags the
# planned opener adds; no implementation on this branch), and O_TRUNC without O_CREAT.
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
    """Sol's outcome rule, from the raw observations on validated fixtures:
    open("wb") refused where an in-place open is allowed -> WIDENING (STOP); no refusal at
    all (level 0) -> cannot prove anything (BLOCKED); refusals but every in-place open also
    refused and the oracle matched -> NO_WIDENING (CLEAR); anything else -> BLOCKED.
    Returns (verdict, gate, widening shapes)."""
    refused_wb = [n for n, row in observed.items() if row["wb"].startswith("refused")]
    widening = [
        n for n in refused_wb if any(observed[n][p].startswith("opened") for p in _PR_IN_PLACE)
    ]
    if widening:
        return "WIDENING", "STOP", widening
    if level == 0 or not refused_wb:
        return "INCONCLUSIVE_NO_REFUSAL", "BLOCKED", widening
    if not matches:
        return "NO_WIDENING_ORACLE_MISMATCH", "BLOCKED", widening
    return "NO_WIDENING", "CLEAR", widening


class TestProtectedRegularKernelEvidence(_ProtectedRegularBase):
    """Test 1: raw opens only. Is an in-place open allowed where today's open('wb') is
    refused? The annotation carries the verdict and gate (WIDENING -> STOP). The test
    fails on invalid evidence (fixture, level 0 where required, the kernel disagreeing
    with the transcribed rule) or a failed ownership give-back, not on WIDENING."""

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
    """Test 2: today's Write and Edit behave exactly like open('wb') on every shape, the
    refusals are the ones the oracle predicts (EACCES), and a changed file gets the refusal
    text first. Where evidence is required, at least one refusal must have been exercised."""

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
                    result = _pr_run_tool(ctx, run, target)
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
