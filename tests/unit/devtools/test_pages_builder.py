from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from devtools import pages_builder


@pytest.mark.parametrize(
    "failure",
    [FileNotFoundError("pagefind"), subprocess.CalledProcessError(2, ["pagefind"])],
)
def test_pagefind_failure_is_typed_and_does_not_return_site(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: BaseException
) -> None:
    site = tmp_path / "site"
    monkeypatch.setattr(pages_builder, "build_site", lambda **_kwargs: site)
    monkeypatch.setattr(
        pages_builder.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(RuntimeError, match="diagnosis: render_pagefind_failed"):
        pages_builder.build_site_with_pagefind(output_dir=site)
