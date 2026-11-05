"""resolve_within_dir keeps storage paths inside their base directory.

Defense-in-depth for CodeQL py/path-injection (alerts #5-#8): collection ZIP
paths are server-generated today, but the join with RESULTS_DIR must never
escape it even if a stored path is ever tampered with.
"""
import os

import pytest

from backend.core.storage_paths import resolve_within_dir


def test_relative_path_resolves_inside_base(tmp_path):
    result = resolve_within_dir(str(tmp_path), "collections/ab/abc.zip")
    assert result == os.path.join(os.path.realpath(tmp_path), "collections", "ab", "abc.zip")


@pytest.mark.parametrize(
    "rel",
    ["../escape.zip", "collections/../../escape.zip", "/etc/passwd", ".", ""],
)
def test_paths_escaping_base_are_rejected(tmp_path, rel):
    with pytest.raises(ValueError):
        resolve_within_dir(str(tmp_path), rel)


def test_filesystem_root_as_base_accepts_descendants():
    # base "/" must not turn the prefix into "//" and reject everything.
    assert resolve_within_dir("/", "tmp/x.zip") == os.path.realpath("/tmp/x.zip")


def test_symlink_pointing_outside_base_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    base = tmp_path / "results"
    base.mkdir()
    (base / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        resolve_within_dir(str(base), "link/file.zip")


def test_delete_with_escaping_path_touches_neither_disk_nor_azure(tmp_path, monkeypatch):
    """A rejected storage_path must not fall through to the Azure delete."""
    import sys
    import types
    from unittest.mock import MagicMock

    from backend.services import collection_service

    monkeypatch.setattr(collection_service.settings, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(collection_service, "is_azure_configured", lambda: True)
    # The function lazily does `from backend.core.azure_sync import _get_blob_client`,
    # so install the fake module in sys.modules (order-independent across tests).
    blob_factory = MagicMock()
    fake_azure = types.ModuleType("backend.core.azure_sync")
    fake_azure._get_blob_client = blob_factory
    monkeypatch.setitem(sys.modules, "backend.core.azure_sync", fake_azure)

    collection_service._delete_collection_zip("abc", "../other/victim.zip")

    blob_factory.assert_not_called()
