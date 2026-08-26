"""Compute and verify the exact RecoveryBox application-tree digest.

The versioned digest binds every included entry's type, UTF-8 POSIX relative
path, and (for files) byte length and contents. Directory entries are included
so empty source directories are not invisible. Traversal is anchored to open
directory descriptors and every component is opened without following links.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Protocol

_FORMAT_HEADER: Final = b"recoverybox-application-tree-sha256\x00v1\x00"
_SOURCE_DIRECTORY: Final = "src"
_DEPLOY_DIRECTORY: Final = "deploy"
_DEPLOY_FILES: Final = frozenset(
    {
        "recoverybox_status.py",
        "recoverybox_tree_digest.py",
    }
)
_BYTECODE_SUFFIX: Final = ".pyc"
_CACHE_DIRECTORY: Final = "__pycache__"
_READ_SIZE: Final = 1024 * 1024
_MAX_PATH_BYTES: Final = 1024 * 1024
_CLOSE_ON_EXEC: Final = getattr(os, "O_CLOEXEC", 0)
_NO_FOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY: Final = getattr(os, "O_DIRECTORY", 0)
_NONBLOCKING: Final = getattr(os, "O_NONBLOCK", 0)


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


class TreeDigestError(RuntimeError):
    """The application tree cannot be represented by the closed digest format."""


@dataclass(frozen=True, slots=True)
class _Record:
    relative_path: str
    is_directory: bool
    size: int
    identity: tuple[int, int, int, int, int, int, int]


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _safe_path_bytes(relative_path: str) -> bytes:
    pure_path = PurePosixPath(relative_path)
    if (
        not relative_path
        or pure_path.is_absolute()
        or str(pure_path) != relative_path
        or any(part in {"", ".", ".."} for part in pure_path.parts)
        or any(ord(character) < 32 or ord(character) == 127 for character in relative_path)
    ):
        raise TreeDigestError(f"unsafe relative path: {relative_path!r}")
    try:
        encoded = relative_path.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise TreeDigestError(f"relative path is not valid UTF-8: {relative_path!r}") from exc
    if len(encoded) > _MAX_PATH_BYTES:
        raise TreeDigestError("relative path is too long")
    return encoded


def _lstat_at(directory_fd: int, name: str, display: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError as exc:
        raise TreeDigestError(f"cannot inspect application-tree entry: {display}") from exc


def _list_names(directory_fd: int, display: str) -> list[str]:
    try:
        names = os.listdir(directory_fd)
    except OSError as exc:
        raise TreeDigestError(f"cannot enumerate application-tree directory: {display}") from exc
    if not all(isinstance(name, str) for name in names):
        raise TreeDigestError(f"application-tree directory has an invalid entry: {display}")
    return sorted(names)


def _open_directory_at(directory_fd: int, name: str, display: str) -> int:
    before = _lstat_at(directory_fd, name, display)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise TreeDigestError(f"expected a real directory: {display}")
    flags = os.O_RDONLY | _CLOSE_ON_EXEC | _NO_FOLLOW | _DIRECTORY
    try:
        opened_fd = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        raise TreeDigestError(f"cannot open application-tree directory: {display}") from exc
    try:
        opened = os.fstat(opened_fd)
        current = _lstat_at(directory_fd, name, display)
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity(before) != _identity(opened)
            or _identity(before) != _identity(current)
        ):
            raise TreeDigestError(f"application-tree directory changed: {display}")
    except Exception:
        os.close(opened_fd)
        raise
    return opened_fd


def _open_root(root: Path) -> tuple[int, tuple[int, int, int, int, int, int, int]]:
    if not _NO_FOLLOW or not _DIRECTORY or not _NONBLOCKING:
        raise TreeDigestError("platform lacks secure descriptor traversal flags")
    try:
        before = root.lstat()
    except OSError as exc:
        raise TreeDigestError(f"cannot inspect application root: {root}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise TreeDigestError(f"expected a real directory: {root}")
    flags = os.O_RDONLY | _CLOSE_ON_EXEC | _NO_FOLLOW | _DIRECTORY
    try:
        root_fd = os.open(root, flags)
    except OSError as exc:
        raise TreeDigestError(f"cannot open application root: {root}") from exc
    try:
        opened = os.fstat(root_fd)
        current = root.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or _identity(before) != _identity(opened)
            or _identity(before) != _identity(current)
        ):
            raise TreeDigestError(f"application root changed: {root}")
    except Exception:
        os.close(root_fd)
        raise
    return root_fd, _identity(opened)


def _validate_ignored_tree(directory_fd: int, display: str) -> None:
    for name in _list_names(directory_fd, display):
        child_display = f"{display}/{name}"
        metadata = _lstat_at(directory_fd, name, child_display)
        if stat.S_ISLNK(metadata.st_mode):
            raise TreeDigestError(f"application tree contains a symlink: {child_display}")
        if stat.S_ISDIR(metadata.st_mode):
            child_fd = _open_directory_at(directory_fd, name, child_display)
            try:
                _validate_ignored_tree(child_fd, child_display)
            finally:
                os.close(child_fd)
        elif not stat.S_ISREG(metadata.st_mode):
            raise TreeDigestError(f"application tree contains a special entry: {child_display}")


def _source_records(root_fd: int, root_display: str, *, strict: bool) -> list[_Record]:
    source_display = f"{root_display}/{_SOURCE_DIRECTORY}"
    source_fd = _open_directory_at(root_fd, _SOURCE_DIRECTORY, source_display)
    try:
        records = [_Record(_SOURCE_DIRECTORY, True, 0, _identity(os.fstat(source_fd)))]

        def visit(directory_fd: int, relative_directory: PurePosixPath, display: str) -> None:
            for name in _list_names(directory_fd, display):
                relative = relative_directory / name
                relative_text = relative.as_posix()
                _safe_path_bytes(relative_text)
                child_display = f"{display}/{name}"
                metadata = _lstat_at(directory_fd, name, child_display)

                if stat.S_ISLNK(metadata.st_mode):
                    raise TreeDigestError(f"application tree contains a symlink: {child_display}")
                if stat.S_ISDIR(metadata.st_mode):
                    child_fd = _open_directory_at(directory_fd, name, child_display)
                    try:
                        if name == _CACHE_DIRECTORY:
                            if strict:
                                raise TreeDigestError(
                                    f"strict tree contains {_CACHE_DIRECTORY}: {child_display}"
                                )
                            _validate_ignored_tree(child_fd, child_display)
                            continue
                        records.append(
                            _Record(relative_text, True, 0, _identity(os.fstat(child_fd)))
                        )
                        visit(child_fd, relative, child_display)
                    finally:
                        os.close(child_fd)
                    continue
                if stat.S_ISREG(metadata.st_mode):
                    if name.endswith(_BYTECODE_SUFFIX):
                        if strict:
                            raise TreeDigestError(f"strict tree contains bytecode: {child_display}")
                        continue
                    if metadata.st_nlink != 1:
                        raise TreeDigestError(
                            f"application-tree file has multiple hard links: {child_display}"
                        )
                    records.append(
                        _Record(relative_text, False, metadata.st_size, _identity(metadata))
                    )
                    continue
                raise TreeDigestError(f"application tree contains a special entry: {child_display}")

        visit(source_fd, PurePosixPath(_SOURCE_DIRECTORY), source_display)
        return records
    finally:
        os.close(source_fd)


def _deploy_records(root_fd: int, root_display: str, *, strict: bool) -> list[_Record]:
    deploy_display = f"{root_display}/{_DEPLOY_DIRECTORY}"
    deploy_fd = _open_directory_at(root_fd, _DEPLOY_DIRECTORY, deploy_display)
    try:
        names = _list_names(deploy_fd, deploy_display)
        if strict and set(names) != _DEPLOY_FILES:
            raise TreeDigestError("strict deploy directory does not match the closed file set")
        records = [_Record(_DEPLOY_DIRECTORY, True, 0, _identity(os.fstat(deploy_fd)))]
        for name in sorted(_DEPLOY_FILES):
            display = f"{deploy_display}/{name}"
            metadata = _lstat_at(deploy_fd, name, display)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise TreeDigestError(f"expected a real deployment helper: {display}")
            if metadata.st_nlink != 1:
                raise TreeDigestError(f"deployment helper has multiple hard links: {display}")
            relative = f"{_DEPLOY_DIRECTORY}/{name}"
            _safe_path_bytes(relative)
            records.append(_Record(relative, False, metadata.st_size, _identity(metadata)))
        return records
    finally:
        os.close(deploy_fd)


def _collect_records(root_fd: int, root_display: str, *, strict: bool) -> tuple[_Record, ...]:
    if strict and set(_list_names(root_fd, root_display)) != {
        _SOURCE_DIRECTORY,
        _DEPLOY_DIRECTORY,
    }:
        raise TreeDigestError("strict application root does not match the closed directory set")
    return tuple(
        sorted(
            (
                *_source_records(root_fd, root_display, strict=strict),
                *_deploy_records(root_fd, root_display, strict=strict),
            ),
            key=lambda record: record.relative_path,
        )
    )


def _open_parent(root_fd: int, relative_path: str, root_display: str) -> tuple[int, str]:
    parts = PurePosixPath(relative_path).parts
    current_fd = os.dup(root_fd)
    display = root_display
    try:
        for component in parts[:-1]:
            display = f"{display}/{component}"
            next_fd = _open_directory_at(current_fd, component, display)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, parts[-1]
    except Exception:
        os.close(current_fd)
        raise


def _hash_file(digest: _Digest, root_fd: int, root_display: str, record: _Record) -> None:
    parent_fd, name = _open_parent(root_fd, record.relative_path, root_display)
    display = f"{root_display}/{record.relative_path}"
    file_fd = -1
    try:
        before = _lstat_at(parent_fd, name, display)
        if _identity(before) != record.identity:
            raise TreeDigestError(f"application-tree file changed: {display}")
        flags = os.O_RDONLY | _CLOSE_ON_EXEC | _NO_FOLLOW | _NONBLOCKING
        try:
            file_fd = os.open(name, flags, dir_fd=parent_fd)
        except OSError as exc:
            raise TreeDigestError(f"cannot open application-tree file: {display}") from exc
        opened = os.fstat(file_fd)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _identity(opened) != record.identity
        ):
            raise TreeDigestError(f"application-tree file changed: {display}")

        bytes_read = 0
        while True:
            chunk = os.read(file_fd, _READ_SIZE)
            if not chunk:
                break
            bytes_read += len(chunk)
            digest.update(chunk)
        after = os.fstat(file_fd)
        current = _lstat_at(parent_fd, name, display)
        if (
            bytes_read != record.size
            or _identity(after) != record.identity
            or _identity(current) != record.identity
        ):
            raise TreeDigestError(f"application-tree file changed: {display}")
    except OSError as exc:
        raise TreeDigestError(f"cannot read application-tree file: {display}") from exc
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(parent_fd)


def _compute_once(
    root_fd: int,
    root_display: str,
    *,
    strict: bool,
) -> tuple[str, tuple[_Record, ...]]:
    records = _collect_records(root_fd, root_display, strict=strict)
    digest = hashlib.sha256(_FORMAT_HEADER)
    for record in records:
        path_bytes = _safe_path_bytes(record.relative_path)
        digest.update(b"D" if record.is_directory else b"F")
        digest.update(struct.pack(">Q", len(path_bytes)))
        digest.update(path_bytes)
        digest.update(struct.pack(">Q", record.size))
        if not record.is_directory:
            _hash_file(digest, root_fd, root_display, record)

    final_records = _collect_records(root_fd, root_display, strict=strict)
    if records != final_records:
        raise TreeDigestError("application tree changed while it was being hashed")
    return digest.hexdigest(), final_records


def application_tree_digest(root: Path, *, strict: bool = False) -> str:
    """Return a stable, versioned SHA-256 digest for a RecoveryBox app tree."""

    root = root.absolute()
    root_fd, root_identity = _open_root(root)
    try:
        first_digest, first_records = _compute_once(root_fd, str(root), strict=strict)
        final_digest, final_records = _compute_once(root_fd, str(root), strict=strict)
        final_snapshot = _collect_records(root_fd, str(root), strict=strict)
        try:
            path_identity = _identity(root.lstat())
        except OSError as exc:
            raise TreeDigestError(f"cannot revalidate application root: {root}") from exc
        if (
            first_digest != final_digest
            or first_records != final_records
            or final_records != final_snapshot
            or _identity(os.fstat(root_fd)) != root_identity
            or path_identity != root_identity
        ):
            raise TreeDigestError("application tree changed while it was being hashed")
        return final_digest
    finally:
        os.close(root_fd)


def _expected_digest(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError("expected digest must be 64 hexadecimal characters")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="application tree root")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require the exact staged root and reject excluded bytecode/cache entries",
    )
    parser.add_argument(
        "--expect",
        type=_expected_digest,
        help="fail unless the computed digest equals this SHA-256 value",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        computed = application_tree_digest(arguments.root, strict=arguments.strict)
    except TreeDigestError as exc:
        print(f"recoverybox tree digest failed: {exc}", file=sys.stderr)
        return 1
    if arguments.expect is not None and computed != arguments.expect:
        print("recoverybox tree digest did not match the expected source tree", file=sys.stderr)
        return 1
    print(computed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
