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
    ["../escape.zip", "collections/../../escape.zip", "/etc/passwd"],
)
def test_paths_escaping_base_are_rejected(tmp_path, rel):
    with pytest.raises(ValueError):
        resolve_within_dir(str(tmp_path), rel)


def test_symlink_pointing_outside_base_is_rejected(tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    base = tmp_path / "results"
    base.mkdir()
    (base / "link").symlink_to(outside)
    with pytest.raises(ValueError):
        resolve_within_dir(str(base), "link/file.zip")
