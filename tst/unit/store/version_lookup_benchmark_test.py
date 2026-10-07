import hashlib
import subprocess

import pytest

from tst.benchmarks import version_lookup


def test_missing_pinned_baseline_has_fetch_instructions(monkeypatch):
    def missing_revision(command, **kwargs):
        assert command == [
            "git",
            "show",
            f"{version_lookup.BASELINE_REVISION}:src/agent_env/store/document_store/document_store.py",
        ]
        raise subprocess.CalledProcessError(128, command, stderr="fatal: bad object")

    monkeypatch.setattr(version_lookup.subprocess, "check_output", missing_revision)
    with pytest.raises(RuntimeError, match="is unavailable in this checkout") as error:
        version_lookup._baseline_versioned_store()
    assert f"git fetch origin {version_lookup.BASELINE_REVISION}" in str(error.value)


def test_baseline_loader_uses_the_pinned_source(monkeypatch):
    source = """class VersionedEntityStore:
    def get(self, id, version=None):
        return (id, version)
    def next_version(self, id):
        return 3
    def put(self, entity, max_retries=5):
        return 4
"""
    commands = []

    def baseline_source(command, **kwargs):
        commands.append(command)
        return source

    monkeypatch.setattr(version_lookup.subprocess, "check_output", baseline_source)
    baseline, source_hash = version_lookup._baseline_versioned_store()

    assert commands == [[
        "git",
        "show",
        f"{version_lookup.BASELINE_REVISION}:src/agent_env/store/document_store/document_store.py",
    ]]
    assert source_hash == hashlib.sha256(source.encode()).hexdigest()
    instance = baseline.__new__(baseline)
    assert instance.get("sample") == ("sample", None)
    assert instance.next_version("sample") == 3
    assert instance.put({"id": "sample"}) == 4
