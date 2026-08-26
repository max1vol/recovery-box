from __future__ import annotations

import hashlib
import importlib.util
import os
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HELPER = _REPO_ROOT / "deploy" / "recoverybox_tree_digest.py"
_FORMAT_HEADER = b"recoverybox-application-tree-sha256\x00v1\x00"


def _load_helper() -> types.ModuleType:
    name = "_recoverybox_test_tree_digest"
    spec = importlib.util.spec_from_file_location(name, _HELPER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tree_digest() -> types.ModuleType:
    module = _load_helper()
    yield module
    sys.modules.pop(module.__name__, None)


def _application_tree(root: Path, *, reverse_creation: bool = False) -> Path:
    directories = [
        root / "src" / "recoverybox" / "empty",
        root / "src" / "recoverybox" / "nested",
        root / "deploy",
    ]
    for directory in reversed(directories) if reverse_creation else directories:
        directory.mkdir(parents=True, exist_ok=True)

    files = {
        root / "src" / "recoverybox" / "__init__.py": b'VALUE = "one"\n',
        root / "src" / "recoverybox" / "nested" / "worker.py": b"answer = 42\n",
        root / "deploy" / "recoverybox_status.py": b"def main():\n    return 0\n",
        root / "deploy" / "recoverybox_tree_digest.py": _HELPER.read_bytes(),
    }
    items = list(files.items())
    for path, payload in reversed(items) if reverse_creation else items:
        path.write_bytes(payload)
    return root


def _reference_digest(root: Path) -> str:
    entries: list[tuple[str, bool, bytes]] = []
    entries.append(("deploy", True, b""))
    for name in ("recoverybox_status.py", "recoverybox_tree_digest.py"):
        entries.append((f"deploy/{name}", False, (root / "deploy" / name).read_bytes()))
    entries.append(("src", True, b""))
    for path in (root / "src").rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            entries.append((relative, True, b""))
        else:
            entries.append((relative, False, path.read_bytes()))

    digest = hashlib.sha256(_FORMAT_HEADER)
    for relative, is_directory, payload in sorted(entries):
        encoded = relative.encode()
        digest.update(b"D" if is_directory else b"F")
        digest.update(struct.pack(">Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def test_digest_is_sorted_and_matches_independent_length_framing(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    first = _application_tree(tmp_path / "first")
    second = _application_tree(tmp_path / "second", reverse_creation=True)

    first_digest = tree_digest.application_tree_digest(first, strict=True)
    second_digest = tree_digest.application_tree_digest(second, strict=True)

    assert first_digest == second_digest
    assert first_digest == _reference_digest(first)


def test_digest_binds_file_path_length_content_helper_and_empty_directories(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    root = _application_tree(tmp_path / "app")
    baseline = tree_digest.application_tree_digest(root, strict=True)

    source = root / "src" / "recoverybox" / "nested" / "worker.py"
    source.write_bytes(b"answer = 43\n")
    assert tree_digest.application_tree_digest(root, strict=True) != baseline
    source.write_bytes(b"answer = 42\nextra = 1\n")
    assert tree_digest.application_tree_digest(root, strict=True) != baseline
    source.write_bytes(b"answer = 42\n")

    source.rename(source.with_name("renamed.py"))
    assert tree_digest.application_tree_digest(root, strict=True) != baseline
    source.with_name("renamed.py").rename(source)

    (root / "src" / "recoverybox" / "another-empty").mkdir()
    assert tree_digest.application_tree_digest(root, strict=True) != baseline
    (root / "src" / "recoverybox" / "another-empty").rmdir()

    helper = root / "deploy" / "recoverybox_tree_digest.py"
    helper.write_bytes(helper.read_bytes() + b"\n")
    assert tree_digest.application_tree_digest(root, strict=True) != baseline


def test_nonstrict_local_digest_ignores_only_bytecode_and_cache_directories(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    root = _application_tree(tmp_path / "app")
    baseline = tree_digest.application_tree_digest(root)
    (root / "src" / "recoverybox" / "ignored.pyc").write_bytes(b"bytecode")
    cache = root / "src" / "recoverybox" / "__pycache__"
    cache.mkdir()
    (cache / "module.cpython-311.pyc").write_bytes(b"cache")

    assert tree_digest.application_tree_digest(root) == baseline
    with pytest.raises(tree_digest.TreeDigestError, match=r"bytecode|__pycache__"):
        tree_digest.application_tree_digest(root, strict=True)


@pytest.mark.parametrize("location", ["root", "deploy"])
def test_strict_digest_rejects_entries_outside_closed_stage_set(
    tmp_path: Path,
    tree_digest: types.ModuleType,
    location: str,
) -> None:
    root = _application_tree(tmp_path / "app")
    parent = root if location == "root" else root / "deploy"
    (parent / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(tree_digest.TreeDigestError, match="closed"):
        tree_digest.application_tree_digest(root, strict=True)


def test_digest_uses_lstat_and_rejects_source_and_helper_symlinks(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    root = _application_tree(tmp_path / "app")
    source = root / "src" / "recoverybox" / "linked.py"
    source.symlink_to(root / "src" / "recoverybox" / "__init__.py")
    with pytest.raises(tree_digest.TreeDigestError, match="symlink"):
        tree_digest.application_tree_digest(root)

    source.unlink()
    helper = root / "deploy" / "recoverybox_tree_digest.py"
    helper.unlink()
    helper.symlink_to(_HELPER)
    with pytest.raises(tree_digest.TreeDigestError, match="real deployment helper"):
        tree_digest.application_tree_digest(root)


def test_digest_rejects_symlinks_inside_ignored_cache(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    root = _application_tree(tmp_path / "app")
    cache = root / "src" / "recoverybox" / "__pycache__"
    cache.mkdir()
    (cache / "escape.pyc").symlink_to(root / "deploy" / "recoverybox_status.py")

    with pytest.raises(tree_digest.TreeDigestError, match="symlink"):
        tree_digest.application_tree_digest(root)


def test_digest_rejects_hard_linked_files(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    root = _application_tree(tmp_path / "app")
    source = root / "src" / "recoverybox" / "__init__.py"
    os.link(source, source.with_name("hard-linked.py"))

    with pytest.raises(tree_digest.TreeDigestError, match="multiple hard links"):
        tree_digest.application_tree_digest(root)


def test_digest_rejects_special_and_unsafe_source_entries(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    root = _application_tree(tmp_path / "app")
    unsafe = root / "src" / "recoverybox" / "bad\nname.py"
    unsafe.write_text("bad", encoding="utf-8")
    with pytest.raises(tree_digest.TreeDigestError, match="unsafe relative path"):
        tree_digest.application_tree_digest(root)
    unsafe.unlink()

    fifo = root / "src" / "recoverybox" / "pipe"
    os.mkfifo(fifo)
    with pytest.raises(tree_digest.TreeDigestError, match="special entry"):
        tree_digest.application_tree_digest(root)


def test_cli_verifies_exact_expected_digest_and_fails_closed_on_mismatch(
    tmp_path: Path,
    tree_digest: types.ModuleType,
) -> None:
    root = _application_tree(tmp_path / "app")
    expected = tree_digest.application_tree_digest(root, strict=True)
    command = [
        sys.executable,
        str(_HELPER),
        "--root",
        str(root),
        "--strict",
        "--expect",
        expected,
    ]

    verified = subprocess.run(command, check=False, capture_output=True, text=True)
    assert verified.returncode == 0
    assert verified.stdout == f"{expected}\n"
    assert verified.stderr == ""

    mismatch = subprocess.run(
        [*command[:-1], "0" * 64],
        check=False,
        capture_output=True,
        text=True,
    )
    assert mismatch.returncode == 1
    assert mismatch.stdout == ""
    assert "did not match" in mismatch.stderr


def test_directory_swap_to_symlink_is_rejected_by_descriptor_anchored_open(
    tmp_path: Path,
    tree_digest: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _application_tree(tmp_path / "app")
    nested = root / "src" / "recoverybox" / "nested"
    moved = nested.with_name("nested-original")
    outside = tmp_path / "outside"
    outside.mkdir()
    original_open = tree_digest.os.open
    observed_flags: list[int] = []

    def racing_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == "nested" and not observed_flags:
            observed_flags.append(flags)
            nested.rename(moved)
            nested.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tree_digest.os, "open", racing_open)

    with pytest.raises(tree_digest.TreeDigestError, match="directory"):
        tree_digest.application_tree_digest(root)
    assert observed_flags
    assert observed_flags[0] & tree_digest._NO_FOLLOW
    assert observed_flags[0] & tree_digest._DIRECTORY


def test_regular_to_fifo_swap_is_nonblocking_and_rejected_after_open(
    tmp_path: Path,
    tree_digest: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _application_tree(tmp_path / "app")
    target = root / "src" / "recoverybox" / "nested" / "worker.py"
    original_open = tree_digest.os.open
    observed_flags: list[int] = []

    def racing_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
        if path == "worker.py" and not observed_flags:
            observed_flags.append(flags)
            target.unlink()
            os.mkfifo(target)
            if not flags & tree_digest._NONBLOCKING:
                raise AssertionError("a swapped FIFO would block without O_NONBLOCK")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tree_digest.os, "open", racing_open)

    with pytest.raises(tree_digest.TreeDigestError, match="changed"):
        tree_digest.application_tree_digest(root)
    assert observed_flags[0] & tree_digest._NONBLOCKING
    assert observed_flags[0] & tree_digest._NO_FOLLOW


@pytest.mark.parametrize("mutation", ["content", "entry"])
def test_final_closed_tree_revalidation_rejects_post_read_mutation(
    tmp_path: Path,
    tree_digest: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    root = _application_tree(tmp_path / "app")
    source = root / "src" / "recoverybox" / "nested" / "worker.py"
    original_collect = tree_digest._collect_records
    collect_count = 0

    def mutating_collect(*args: object, **kwargs: object) -> object:
        nonlocal collect_count
        snapshot = original_collect(*args, **kwargs)
        collect_count += 1
        if collect_count == 2:
            if mutation == "content":
                source.write_bytes(b"answer = 43\n")
            else:
                source.with_name("added.py").write_bytes(b"added = True\n")
        return snapshot

    monkeypatch.setattr(tree_digest, "_collect_records", mutating_collect)

    with pytest.raises(tree_digest.TreeDigestError, match="changed"):
        tree_digest.application_tree_digest(root)
    assert collect_count >= 4
