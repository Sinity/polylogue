"""Contract for the derived-tier sweep and read-path substitution census.

Anti-vacuity for this module as a whole: every test below builds a throwaway
package and asks the census what it observed, so deleting any branch of
``devtools/derived_sweep_census.py``'s analysis turns one of them red. The
checked-in declaration is exercised too (``test_head_declaration_matches``), so
the pair covers both directions the campaign asked for -- a new sweep must go
red naming it, and the clean tree must stay green.
"""

from __future__ import annotations

from pathlib import Path

from devtools import repo_root
from devtools.derived_sweep_census import (
    DECLARATION_PATH,
    census_package,
    collect_violations,
    render_declaration,
)


def _package(root: Path, name: str, body: str) -> Path:
    package = root / "polylogue" / "storage"
    package.mkdir(parents=True, exist_ok=True)
    module = package / name
    module.write_text(body, encoding="utf-8")
    return module


def _keys(root: Path) -> set[str]:
    return {site.key for site in census_package(root / "polylogue", repo_root=root).sites}


def _kinds(root: Path, function: str) -> set[str]:
    return {site.kind for site in census_package(root / "polylogue", repo_root=root).sites if site.function == function}


def test_state_predicate_rewrite_is_censused(tmp_path: Path) -> None:
    """A derived UPDATE that picks its rows out of archive state is seen.

    Anti-vacuity: deleting the ``state_predicate`` branch of ``_statement_scope``
    -- or narrowing the census to ``no_where`` -- makes this empty.
    """
    _package(
        tmp_path,
        "sweeper.py",
        "def sweep(conn):\n"
        '    conn.execute("UPDATE session_profiles SET is_continuation = 1 WHERE parent_id IS NOT NULL")\n',
    )
    assert _kinds(tmp_path, "sweep") == {"unbound_rewrite"}


def test_caller_bound_rewrite_is_not_censused(tmp_path: Path) -> None:
    """The same statement scoped to a caller-supplied id is out of subject.

    Anti-vacuity: if the scope check stopped recognising a bound parameter,
    every scoped writer in the tree would enter the census and this goes red.
    """
    _package(
        tmp_path,
        "writer.py",
        "def write(conn, session_id):\n"
        '    conn.execute("UPDATE session_profiles SET is_continuation = 1 WHERE session_id = ?", (session_id,))\n',
    )
    assert _keys(tmp_path) == set()


def test_binding_inside_a_subquery_still_binds(tmp_path: Path) -> None:
    """``id IN (SELECT ... WHERE k = ?)`` is a caller-supplied subject.

    Anti-vacuity: stripping subqueries before looking for a parameter -- which
    an earlier draft of this census did -- turns every prefix-delete in
    ``archive_tiers/write.py`` into a false sweep, and this goes red.
    """
    _package(
        tmp_path,
        "cascade.py",
        "def clear(conn, session_id):\n"
        "    conn.execute(\n"
        '        "DELETE FROM blocks WHERE message_id IN (SELECT message_id FROM messages WHERE session_id = ?)",\n'
        "        (session_id,),\n"
        "    )\n",
    )
    assert _keys(tmp_path) == set()


def test_limit_parameter_does_not_bind_a_subject(tmp_path: Path) -> None:
    """A bounded sweep is still a sweep: ``LIMIT ?`` binds the size, not the rows.

    Anti-vacuity: this is the exact shape the deleted lineage sweep had -- an
    unscoped selection with a bound ``LIMIT``. Letting the selection region run
    past ``LIMIT`` would certify it as scoped and make this empty.
    """
    _package(
        tmp_path,
        "bounded.py",
        "def sweep(conn, limit):\n"
        '    conn.execute("DELETE FROM session_profiles WHERE is_continuation IS NULL LIMIT ?", (limit,))\n',
    )
    assert _kinds(tmp_path, "sweep") == {"unbound_rewrite"}


def test_optional_scope_clause_is_censused(tmp_path: Path) -> None:
    """A predicate one code path omits is the signature of an optional scope.

    Anti-vacuity: this is how ``_repair_stale_prefix_branch_points_db`` looked
    before PR #5372. Resolving a name to its last assignment instead of the
    union of its assignments hides the unscoped branch and empties this.
    """
    _package(
        tmp_path,
        "optional.py",
        "def sweep(conn, session_ids=None):\n"
        '    clause = ""\n'
        "    params = []\n"
        "    if session_ids is not None:\n"
        '        holes = ",".join("?" for _ in session_ids)\n'
        '        clause = f"AND session_id IN ({holes})"\n'
        "        params = list(session_ids)\n"
        '    conn.execute(f"SELECT session_id FROM session_profiles WHERE 1=1 {clause}", tuple(params))\n'
        '    conn.execute("UPDATE session_profiles SET is_continuation = 1 WHERE session_id = ?", ("x",))\n',
    )
    assert "omissible_scope" in _kinds(tmp_path, "sweep")


def test_state_selected_subject_is_censused(tmp_path: Path) -> None:
    """A row-bound rewrite whose subjects come from an unbound scan is seen.

    Anti-vacuity: the rewrite here carries ``WHERE session_id = ?``, so a census
    that only inspected rewrite statements reports nothing. Deleting the
    derived-read half makes this empty.
    """
    _package(
        tmp_path,
        "selector.py",
        "def sweep(conn):\n"
        '    rows = conn.execute("SELECT session_id FROM session_profiles WHERE is_continuation IS NULL").fetchall()\n'
        "    for row in rows:\n"
        '        conn.execute("UPDATE session_profiles SET is_continuation = 0 WHERE session_id = ?", (row[0],))\n',
    )
    assert "state_selected_subject" in _kinds(tmp_path, "sweep")


def test_read_path_substitution_is_censused(tmp_path: Path) -> None:
    """A read that fills a stored field it found absent is seen.

    Anti-vacuity: this is the deleted ``_repair_profile_parent_ids`` shape. The
    guard reads the same attribute the copy replaces; dropping that correlation
    from ``_substitution_sites`` empties this.
    """
    _package(
        tmp_path,
        "reader.py",
        "def load(conn, parents):\n"
        '    rows = conn.execute("SELECT session_id FROM session_profiles WHERE session_id = ?", ("x",)).fetchall()\n'
        "    out = []\n"
        "    for row in rows:\n"
        "        record = parents[row[0]]\n"
        "        if not record.parent_id:\n"
        '            out.append(record.model_copy(update={"parent_id": "recovered"}))\n'
        "        else:\n"
        "            out.append(record)\n"
        "    return out\n",
    )
    sites = census_package(tmp_path / "polylogue", repo_root=tmp_path).sites
    assert [(site.kind, site.subject, site.scope) for site in sites] == [
        ("read_path_substitution", "parent_id", "read_only_module")
    ]


def test_undeclared_site_is_a_gate_violation(tmp_path: Path) -> None:
    """An observed site missing from the declaration fails the gate by name.

    Anti-vacuity: without the set difference the gate would accept any
    declaration, including an empty one, and this goes red.
    """
    _package(
        tmp_path,
        "sweeper.py",
        'def sweep(conn):\n    conn.execute("DELETE FROM session_profiles WHERE is_continuation IS NULL")\n',
    )
    declaration = tmp_path / "census.yaml"
    declaration.write_text("package: polylogue\nsites: []\n", encoding="utf-8")
    violations = collect_violations(repo_root=tmp_path, declaration_path=declaration)
    assert [violation["rule"] for violation in violations] == ["derived_sweep_undeclared"]
    assert "sweeper.py::sweep::session_profiles::unbound_rewrite" in str(violations[0]["key"])


def test_stale_declared_entry_is_a_violation(tmp_path: Path) -> None:
    """A declared site the census no longer observes must be dropped.

    Anti-vacuity: without this the declaration would accumulate entries for
    deleted code and stop describing the tree.
    """
    (tmp_path / "polylogue").mkdir()
    declaration = tmp_path / "census.yaml"
    declaration.write_text(
        'package: polylogue\nsites:\n  - file: "polylogue/gone.py"\n    function: "gone"\n'
        '    subject: "session_profiles"\n    kind: "unbound_rewrite"\n    scope: "no_where"\n'
        '    classification: unclassified\n    reason: "x"\n',
        encoding="utf-8",
    )
    violations = collect_violations(repo_root=tmp_path, declaration_path=declaration)
    assert [violation["rule"] for violation in violations] == ["derived_sweep_census_stale"]


def test_masking_classification_fails_the_gate(tmp_path: Path) -> None:
    """The defect cannot be parked in the declaration as a classification.

    Anti-vacuity: drop ``FORBIDDEN_CLASSIFICATIONS`` from the check and a real
    archive-wide corrective sweep passes the gate by admitting what it is.
    """
    _package(
        tmp_path,
        "sweeper.py",
        'def sweep(conn):\n    conn.execute("DELETE FROM session_profiles WHERE is_continuation IS NULL")\n',
    )
    declaration = tmp_path / "census.yaml"
    declaration.write_text(
        'package: polylogue\nsites:\n  - file: "polylogue/storage/sweeper.py"\n    function: "sweep"\n'
        '    subject: "session_profiles"\n    kind: "unbound_rewrite"\n    scope: "state_predicate"\n'
        '    classification: archive_wide_corrective_sweep\n    reason: "it is what it is"\n',
        encoding="utf-8",
    )
    rules = {violation["rule"] for violation in collect_violations(repo_root=tmp_path, declaration_path=declaration)}
    assert rules == {"derived_sweep_masks_a_producer"}


def test_declared_reason_is_required(tmp_path: Path) -> None:
    """A classification without a reason is not a review.

    Anti-vacuity: without this a declaration can be regenerated with empty
    reasons and still pass, which is the rubber-stamp failure mode.
    """
    _package(
        tmp_path,
        "sweeper.py",
        'def sweep(conn):\n    conn.execute("DELETE FROM session_profiles WHERE is_continuation IS NULL")\n',
    )
    declaration = tmp_path / "census.yaml"
    declaration.write_text(
        'package: polylogue\nsites:\n  - file: "polylogue/storage/sweeper.py"\n    function: "sweep"\n'
        '    subject: "session_profiles"\n    kind: "unbound_rewrite"\n    scope: "state_predicate"\n'
        '    classification: unclassified\n    reason: ""\n',
        encoding="utf-8",
    )
    rules = {violation["rule"] for violation in collect_violations(repo_root=tmp_path, declaration_path=declaration)}
    assert rules == {"derived_sweep_reason_missing"}


def test_head_declaration_matches_the_census() -> None:
    """The checked-in declaration is exactly what the tree exhibits.

    This is the green half of the ratchet: the clean tree must stay green, so
    the census is a census rather than a prohibition. Anti-vacuity: any edit
    that adds, removes or rescopes a derived sweep without regenerating
    ``docs/plans/derived-sweep-census.yaml`` turns this red.
    """
    assert collect_violations(repo_root=repo_root()) == []


def test_render_declaration_round_trips() -> None:
    """Regenerating the declaration preserves adjudications already recorded.

    Anti-vacuity: if ``render_declaration`` dropped existing classifications,
    a regeneration would silently reset every adjudicated entry to the floor
    and this comparison goes red.
    """
    from devtools.derived_sweep_census import load_declaration

    root = repo_root()
    declared = load_declaration(root / DECLARATION_PATH)
    observation = census_package(root / "polylogue", repo_root=root)
    rendered = render_declaration(observation, existing=declared.entries)
    assert rendered == (root / DECLARATION_PATH).read_text(encoding="utf-8")
