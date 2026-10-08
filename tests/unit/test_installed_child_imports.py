"""Installed callers carry their in-memory dependency paths into real children."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("owner", ["browser", "resident", "tour"])
def test_installed_like_callers_preserve_child_imports_and_arguments(tmp_path: Path, owner: str) -> None:
    """Removing any caller bootstrap fails under a bare, non-checkout parent.

    Help follows each actual launch path without starting a service or opening
    an archive. The browser's argparse usage also proves module argv[0], while
    the resident's existing tests hold its shutdown and configuration custody.
    """
    code = """
import asyncio, json, os, subprocess, sys
from pathlib import Path
sys.path[:] = json.loads(sys.argv[1])
owner, scratch = sys.argv[2], Path(sys.argv[3])
if owner == 'browser':
    from polylogue.daemon.cli import _run_browser_host
    launch = asyncio.create_subprocess_exec
    observed = []
    async def help_child(*args, **kwargs):
        child = await launch(*args, '--help', stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, **kwargs)
        output, errors = await child.communicate()
        assert child.returncode == 0, errors.decode()
        assert b' -m polylogue.daemon.browser_host ' in output, output.decode()
        observed.append(child)
        return child
    asyncio.create_subprocess_exec = help_child
    try:
        asyncio.run(_run_browser_host(host='127.0.0.1', port=0, daemon_origin='http://127.0.0.1:1'))
    except RuntimeError:
        pass
    assert len(observed) == 1
elif owner == 'resident':
    from polylogue.operations.demo_resident import demo_resident
    launch = subprocess.Popen
    observed = []
    def help_child(args, **kwargs):
        child = launch([*args, '--help'], **kwargs)
        observed.append(child)
        return child
    subprocess.Popen = help_child
    try:
        with demo_resident(scratch / 'archive'):
            raise AssertionError('help cannot bind a daemon listener')
    except RuntimeError:
        pass
    assert len(observed) == 1 and observed[0].returncode == 0
else:
    from polylogue.demo.tour import _run_cli_step
    step, rendered = _run_cli_step(name='help', args=('--help',), explanation='', env=dict(os.environ),
        archive_root=scratch / 'archive', output_path=scratch / 'help.txt')
    assert step.exit_code == 0, rendered
    assert ' -m polylogue ' in rendered
    assert step.command == ('polylogue', '--help')
    assert 'sys.path' not in rendered
print(json.dumps({'owner': owner, 'completed': True}))
"""
    paths = [str(Path(path or os.getcwd()).resolve()) for path in sys.path]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONUSERBASE"}
    }
    result = subprocess.run(
        [os.path.realpath(sys.executable), "-c", code, json.dumps(paths), owner, str(tmp_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"owner": owner, "completed": True}
