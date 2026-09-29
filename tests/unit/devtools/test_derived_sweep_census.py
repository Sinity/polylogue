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
    derived_table_tiers,
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


def test_duplicate_table_names_retain_every_tier_owner() -> None:
    """The derived owner survives when a durable DDL reuses the table name.

    Anti-vacuity: last-wins mapping records query_unit_frame_state as only
    durable and removes it from the derived rewrite population.
    """
    assert {"index", "user"} <= derived_table_tiers()["query_unit_frame_state"]


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


def test_archive_cutoff_parameter_does_not_bind_selected_identity(tmp_path: Path) -> None:
    """A state predicate remains a sweep even when its cutoff is parameterized.

    Anti-vacuity: treating every WHERE parameter as a caller-selected key
    hides this archive-wide update and leaves the census empty.
    """
    _package(
        tmp_path,
        "cutoff.py",
        "def sweep(conn, cutoff):\n"
        '    conn.execute("UPDATE session_profiles SET is_continuation = 1 WHERE created_at < ?", (cutoff,))\n',
    )
    assert _kinds(tmp_path, "sweep") == {"unbound_rewrite"}


def test_nested_where_does_not_bind_outer_update(tmp_path: Path) -> None:
    """A subquery predicate cannot scope an UPDATE with no outer WHERE.

    Anti-vacuity: searching for the first WHERE anywhere in the statement
    treats the nested settings lookup as caller-selected update subjects.
    """
    _package(
        tmp_path,
        "nested.py",
        "def sweep(conn):\n"
        '    conn.execute("UPDATE session_profiles SET value = (SELECT value FROM settings WHERE key = ?)" , ("x",))\n',
    )
    assert _kinds(tmp_path, "sweep") == {"unbound_rewrite"}


def test_schema_qualified_rewrite_is_censused(tmp_path: Path) -> None:
    """An attached database qualifier does not hide the actual derived table.

    Anti-vacuity: capturing only the qualifier (index_tier) drops the
    session_profiles rewrite and this expected census entry disappears.
    """
    _package(
        tmp_path,
        "qualified.py",
        'def sweep(conn):\n    conn.execute("UPDATE index_tier.session_profiles SET parent_id = NULL")\n',
    )
    assert _kinds(tmp_path, "sweep") == {"unbound_rewrite"}


def test_multiple_rewrite_sites_merge_to_worst_scope(tmp_path: Path) -> None:
    """A newly unscoped call site is not hidden by an earlier declared site.

    Anti-vacuity: retaining only the first table/kind observation leaves the
    call-site scope bound and suppresses the newly added archive-wide update.
    """
    _package(
        tmp_path,
        "two_sites.py",
        "def sweep(conn, session_id):\n"
        '    conn.execute("UPDATE session_profiles SET parent_id = NULL WHERE session_id = ?", (session_id,))\n'
        '    conn.execute("UPDATE session_profiles SET parent_id = NULL")\n',
    )
    sites = census_package(tmp_path / "polylogue", repo_root=tmp_path).sites
    assert len([site for site in sites if site.function == "sweep" and site.kind == "unbound_rewrite"]) == 1


def test_insert_or_replace_from_archive_select_is_censused(tmp_path: Path) -> None:
    """An archive-selected replacement is visible even though it starts INSERT.

    Anti-vacuity: limiting rewrite candidates to UPDATE/DELETE ignores this
    full derived-table replacement and produces no observation.
    """
    _package(
        tmp_path,
        "replace_sweep.py",
        "def sweep(conn):\n"
        '    conn.execute("INSERT OR REPLACE INTO session_profiles (session_id) SELECT session_id FROM sessions")\n',
    )
    assert "unbound_rewrite" in _kinds(tmp_path, "sweep")


def test_or_fallback_in_replace_call_is_censused(tmp_path: Path) -> None:
    """Every supported copy constructor exposes stored-value fallbacks.

    Anti-vacuity: restricting the fallback pass to model_copy misses this
    replace(profile, parent_id=profile.parent_id or infer()) mask.
    """
    _package(
        tmp_path,
        "fallback.py",
        "from dataclasses import replace\n"
        "def load(conn, profile):\n"
        '    conn.execute("SELECT session_id FROM session_profiles WHERE session_id = ?", ("x",))\n'
        "    return replace(profile, parent_id=profile.parent_id or infer_parent())\n",
    )
    assert [(site.kind, site.subject) for site in census_package(tmp_path / "polylogue", repo_root=tmp_path).sites] == [
        ("read_path_substitution", "parent_id")
    ]


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


def test_values_are_lexically_scoped_and_empty_sentinel_survives_the_cap(tmp_path: Path) -> None:
    """Anti-vacuity: foreign locals cannot taint this function, and branch 9 remains visible."""
    _package(
        tmp_path,
        "scoped.py",
        "def unrelated():\n    clause = ''\n"
        "def mandatory(conn):\n    clause = 'WHERE session_id = ?'\n"
        "    conn.execute(f'DELETE FROM session_profiles {clause}', ('x',))\n"
        "def optional(conn, branch):\n"
        "    clause = 'WHERE id = 0'\n    clause = 'WHERE id = 1'\n"
        "    clause = 'WHERE id = 2'\n    clause = 'WHERE id = 3'\n"
        "    clause = 'WHERE id = 4'\n    clause = 'WHERE id = 5'\n"
        "    clause = 'WHERE id = 6'\n    clause = 'WHERE id = 7'\n"
        + "    if branch:\n        clause = ''\n"
        + "    conn.execute(f'DELETE FROM session_profiles {clause}')\n"
        "def outer():\n    clause = 'WHERE session_id = ?'\n"
        "    def nested(conn):\n"
        "        conn.execute(f'DELETE FROM session_profiles {clause}', ('x',))\n",
    )
    assert _kinds(tmp_path, "mandatory") == set()
    assert _kinds(tmp_path, "outer.nested") == set()
    assert "omissible_scope" in _kinds(tmp_path, "optional")


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


def test_existence_probe_is_not_mistaken_for_rewrite_subjects(tmp_path: Path) -> None:
    """An unrelated SELECT 1 probe does not supply a later caller-key rewrite.

    Anti-vacuity: function-level co-occurrence alone labels the probe as a
    derived subject scan and adds a spurious declaration requirement.
    """
    _package(
        tmp_path,
        "writer.py",
        "def write(conn, session_id):\n"
        '    conn.execute("SELECT 1 FROM sessions LIMIT 1")\n'
        '    conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = ?", (session_id,))\n',
    )
    assert _kinds(tmp_path, "write") == set()


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


def test_substitution_in_else_branch_is_the_absent_field_branch(tmp_path: Path) -> None:
    """Branch polarity selects the absent-value arm of an if/else.

    Anti-vacuity: scanning only the positive body misses this
    ``is not None`` guard's else arm where the field is actually substituted.
    """
    _package(
        tmp_path,
        "else_reader.py",
        "from dataclasses import replace\n"
        "def load(conn, profile):\n"
        '    conn.execute("SELECT session_id FROM session_profiles WHERE session_id = ?", ("x",))\n'
        "    if profile.parent_id is not None:\n"
        "        return profile\n"
        "    else:\n"
        "        return replace(profile, parent_id=infer_parent())\n",
    )
    assert [(site.kind, site.subject) for site in census_package(tmp_path / "polylogue", repo_root=tmp_path).sites] == [
        ("read_path_substitution", "parent_id")
    ]


def test_caller_loaded_profile_substitution_is_censused_without_sql(tmp_path: Path) -> None:
    _package(
        tmp_path,
        "reader.py",
        "def load(profile):\n    if not profile.parent_id:\n"
        '        return profile.model_copy(update={"parent_id": infer_parent()})\n    return profile\n',
    )
    assert _kinds(tmp_path, "load") == {"read_path_substitution"}


def test_bad_package_declaration_fails_even_with_no_sites(tmp_path: Path) -> None:
    (tmp_path / "polylogue").mkdir()
    declaration = tmp_path / "census.yaml"
    declaration.write_text("package: polylgue\nsites: []\n", encoding="utf-8")
    violations = collect_violations(repo_root=tmp_path, declaration_path=declaration)
    assert violations[0]["rule"] == "derived_sweep_census_package_invalid"


def test_rendered_yaml_quotes_arbitrary_reason_and_marks_new_entries_pending() -> None:
    import yaml

    from devtools.derived_sweep_census import CensusEntry, CensusObservation, SweepSite

    site = SweepSite("polylogue/a.py", "f", "session_id", "read_path_substitution", "read_only_module", 1)
    reason = 'uses "session_id" as the key'
    rendered = render_declaration(
        CensusObservation((site,)),
        existing={
            site.key: CensusEntry(
                site.key,
                site.file,
                site.function,
                site.subject,
                site.kind,
                site.scope,
                "presentation_projection",
                reason,
            )
        },
    )
    declaration = yaml.safe_load(rendered)
    assert declaration["sites"][0]["reason"] == reason


def test_new_site_render_does_not_claim_adjudication() -> None:
    from devtools.derived_sweep_census import CensusObservation, SweepSite

    site = SweepSite("polylogue/a.py", "f", "session_id", "read_path_substitution", "read_only_module", 1)
    rendered = render_declaration(CensusObservation((site,)))
    assert "Every entry below has been adjudicated" not in rendered
    assert "classification: unclassified" in rendered


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


def test_repeated_rewrite_key_keeps_the_broadest_scope_and_its_location(tmp_path: Path) -> None:
    """Two rewrites of one table in one function share a census key.

    Anti-vacuity: first-wins retention keeps the earlier ``state_predicate``
    site, so the later archive-wide ``no_where`` rewrite and its line vanish.
    """
    state_selected = '    conn.execute("UPDATE session_profiles SET parent_id = NULL WHERE parent_id IS NOT NULL")\n'
    unscoped = '    conn.execute("UPDATE session_profiles SET parent_id = NULL")\n'
    for broad_first in (False, True):
        root = tmp_path / str(broad_first)
        statements = (unscoped, state_selected) if broad_first else (state_selected, unscoped)
        _package(root, "sweep.py", "def sweep(conn):\n" + "".join(statements))
        sites = [
            site
            for site in census_package(root / "polylogue", repo_root=root).sites
            if site.function == "sweep" and site.kind == "unbound_rewrite"
        ]
        assert len(sites) == 1
        assert sites[0].scope == "no_where"
        assert sites[0].line == (2 if broad_first else 3)
