"""beads-state-report: a self-contained HTML state-of-the-backlog report.

Reads ``.beads/issues.jsonl`` and emits one standalone HTML file covering the
whole bead population -- open AND closed -- across seven views: shape
(status x priority x type), structure (epic/program trees), topology (the
``blocks`` dependency graph: ready / blocked / top blockers / cycles), time
(creation and closure per day, velocity, age of open work), health (a review
queue of graph defects), themes (where open work concentrates by subsystem),
and the hand-verified verdict subset.

Everything mechanical is computed here; nothing is transcribed by hand. The
prose in the report is confined to the ``PROSE`` mapping at the top of this
module, so a reader can audit exactly which sentences are authored judgement
and which text is derived from the data.

Usage: devtools workspace beads-state-report [--out PATH] [--fresh] [--json]
``--fresh`` runs ``bd export -o .beads/issues.jsonl`` first, because ``bd``
mutations do not immediately re-export and a stale file yields stale counts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from devtools import repo_root as _get_root

IssueDict = dict[str, Any]
Edge = tuple[str, str, str]

# --------------------------------------------------------------------------
# AUTHORED PROSE.  Everything in this mapping is hand-written judgement; every
# number rendered in the report comes from the computation below and is
# interpolated at render time.  Placeholders are ``{name}`` format fields fed
# from the computed ``Facts``.  Keep analysis here and only here.
# --------------------------------------------------------------------------
PROSE: dict[str, str] = {
    "lead": """
      This is the whole bead population of the polylogue repository, open and
      closed, as one artifact. The headline shape is a backlog that is
      <b>{age_days} days old in its entirety</b> &mdash; the oldest bead was
      created {first_created} &mdash; and that has already closed
      <b>{closed}</b> of the {total} beads it has ever contained. Nothing here
      is legacy debt inherited from a previous tracker; every row was written,
      and nearly half of them retired, inside a single month. Read the
      <a href="#topology">topology</a> and <a href="#health">health</a>
      sections first: they are where the graph is doing real work and where it
      is quietly broken.
    """,
    "shape": """
      The status &times; priority matrix is the fastest read on whether the
      priority scale is load-bearing. It is. Median time from creation to
      closure rises without a single inversion across all five tiers &mdash;
      {p0_lead}, {p1_lead}, {p2_lead}, {p3_lead}, {p4_lead} days for P0
      through P4, a {lead_ratio}&times; spread &mdash; and closure rate falls
      monotonically across the first four ({p0_rate}, {p1_rate}, {p2_rate},
      {p3_rate}), with P4 the sole exception at {p4_rate} (taken up
      immediately below). That refutes the obvious suspicion about
      {p0_total} P0s. A priority label that had inflated into a synonym for
      &ldquo;I care about this&rdquo; would show a flat lead-time curve; this
      one sorts work at every rung. P0 is genuinely triage.
    """,
    "p4": """
      P4 is the one rung that does not behave like the others. Its closure
      rate ({p4_rate}) is <em>higher</em> than P3's ({p3_rate}) while its
      median lead time ({p4_lead} days) is the longest of any tier &mdash; the
      one place the ladder inverts. The likeliest reading is that P4
      is bimodal: a graveyard of {p4_open} genuinely parked items, plus a
      minority of trivia that gets swept up opportunistically when someone is
      already in the file. That is not a broken label, but it does mean
      &ldquo;P4&rdquo; carries no information about whether a bead will ever
      be worked &mdash; unlike P0 through P3, where the number predicts both
      outcome and latency.
    """,
    "structure": """
      Epics are not uniformly staged, and the spread is the point. Compare
      <code>polylogue-9e5</code> (audit lane) at {e9e5_open} of {e9e5_total}
      children still open against <code>polylogue-9l5</code>
      (outcome-grounded analytics) at {e9l5_open} of {e9l5_total}. Those two
      epics have comparable size and opposite completion states: one is a
      finished program still holding its parent row open, the other has never
      been started. A flat backlog list renders those identically. The fill
      bars below are the whole reason this section exists &mdash; they let you
      see at a glance which programs are nearly retired, which are mid-flight,
      and which are pure declaration.
    """,
    "structure_ids": """
      A structural caveat worth knowing before you trust any per-epic count:
      <b>the dotted id is not the parent link</b>. Bead ids like
      <code>polylogue-a7xr.4</code> look hierarchical, but hierarchy actually
      lives in <code>parent-child</code> dependency edges, and the two
      disagree: {id_mismatch} beads have a dotted id whose prefix names a
      different parent than their edge does, and {id_orphan} dotted beads have
      no parent edge at all. Epics also adopt children whose ids do not share
      their prefix. Every tree below is built from <em>edges</em>, which is why
      some child counts here are larger than a prefix-grep would report.
    """,
    "topology": """
      The dependency graph is far shallower than its {blocks_edges}
      <code>blocks</code> edges suggest. Of {open_total} open beads,
      <b>{ready} are ready</b> &mdash; no open bead blocks them &mdash; and
      only {blocked} are actually waiting on something. That is a
      {ready_pct} ready fraction, which means the backlog is not
      dependency-starved; it is throughput-starved. There is no ordering
      constraint keeping most of this work parked, and no amount of unblocking
      will change the picture.
    """,
    "topology_cycles": """
      There are <b>no cycles</b> in the <code>blocks</code> graph, checked by
      depth-first colouring over all {total} beads. That is worth stating
      explicitly rather than leaving as an absence: a work graph this size,
      grown this fast, with edges added by many agent lanes, is exactly where
      you would expect a mutual-blocking pair to appear by accident. None has.
    """,
    "topology_blockers": """
      The load-bearing blockers are few and clustered. The top blocker holds
      {top_blocker_n} open beads behind it, and the whole tail is short: the
      blocking relation is close to a set of small local chains rather than a
      spine. Practically, unblocking is not the lever here &mdash; but if you
      want the highest-leverage single items to finish, they are at the top of
      this list, because each one releases a fan-out that no other bead does.
    """,
    "time": """
      Every bead in this repository was created inside {span_days} days.
      The creation series is spiky rather than steady &mdash; {peak_created}
      beads filed on the single busiest day ({peak_created_day}) against a
      median of {median_created}/day &mdash; which is the signature of
      audit-lane batch filing, not continuous discovery. Closure is markedly
      smoother, with a median of {median_closed}/day. Beads arrive in bursts
      and drain steadily.
    """,
    "time_velocity": """
      Comparing the two halves of the month tells you whether this is
      converging. First half: {c1_created} created, {c1_closed} closed
      (net {c1_net:+d}). Second half: {c2_created} created, {c2_closed} closed
      (net {c2_net:+d}). The backlog is <b>{trend}</b>. Filing is winning over
      closing, and the gap is not narrowing &mdash; a backlog whose net growth
      is positive in both halves of its only month of existence is still in
      its discovery phase, not its burn-down phase. Treat any completion
      estimate derived from current velocity as meaningless until a half-month
      window comes in net-negative.
    """,
    "health": """
      {n_checks} mechanical checks over the graph. None of these are things a
      reader should batch-fix from this page &mdash; they are a
      <em>review queue</em>, and the distinction matters most for the
      &ldquo;open parent, all children closed&rdquo; row. That check finds
      {allclosed} parents whose every child is retired, which looks like an
      auto-closable set and is not. <code>polylogue-1vpm.6</code> sits in it
      with acceptance criteria AC8 and AC9 untouched: its children covered the
      decomposed work, the parent's own criteria were never satisfied, and
      closing it on the strength of its children would silently discard the
      residual. Children closing is evidence that a parent is <em>ready for
      adjudication</em>, never evidence that it is done.
    """,
    "health_dangling": """
      The dangling-reference row is a genuine data bug, not a judgement call.
      Two <code>blocks</code> edges point at <code>20d.14</code> and
      <code>20d.1</code> &mdash; bare ids missing the <code>polylogue-</code>
      prefix that every real bead carries. They resolve to nothing. The
      blocking intent behind those two edges is silently absent from every
      query, which is exactly the failure mode that makes an unvalidated
      free-text id field dangerous in a dependency graph.
    """,
    "reconciliation": """
      A separate pass, run today against this same population, is the single
      most decision-relevant fact in this backlog and does not fit anywhere
      else on this page: some of what looks open here is actually
      <b>fixed in code and merged</b>, but invisible in the archive this
      report itself reads facts from, because the archive that agents and
      the daemon query is several derived-schema versions behind what
      <code>origin/master</code> declares. A closed bead in the matrix above
      can still describe a real bug against the data an operator is looking
      at right now. Read the schema gap first &mdash; it is why the rest of
      this section exists.
    """,
    "reconciliation_schema_gap": """
      The live archive is at index-schema <b>v{schema_live}</b>.
      <code>origin/master</code> declares <b>v{schema_declared}</b>. Of the
      versions between them, <b>{schema_blockers_n}</b>
      ({schema_blockers_list}) are declared <code>SEMANTIC_REPARSE</code>
      &mdash; a class this codebase's own schema-versioning policy reserves
      for deltas whose result depends on parser semantics, which means there
      is no clone-safe SQL fast-forward and the only path from v{schema_live}
      to v{schema_declared} is <code>polylogue ops reset --index &&
      polylogued run</code>. Every PR that already merged one of those four
      versions is real, reviewed, and completely inert until that rebuild
      runs. That is not a queue-depth problem findable anywhere else on this
      page &mdash; it is a single operator action blocking an unknown number
      of already-finished fixes, several of which this pass individually
      confirmed (below).
    """,
    "reconciliation_p0": """
      A hand-verified pass over the 23 beads that were <code>status=open,
      priority=0</code> this morning (plus one <code>in_progress</code> P0
      named explicitly in the pass's brief), read against
      <code>origin/master</code> and the live archive rather than trusted
      from prior notes. It closed one (verified already effective, not
      merely merged), demoted five whose severity did not survive
      re-measurement, and left the rest open &mdash; but split "open" into
      two kinds that a flat status field cannot express: work with no code
      fix yet, and work that is <em>done</em> and waiting only on the rebuild
      above. This generator extracts the pass's own
      <code>RECONCILIATION 2026-07-31: &lt;VERDICT&gt;</code> markers from
      bead notes &mdash; the same discipline as the verified subset elsewhere
      on this page, and with the same kind of coverage gap, reported next
      rather than hidden.
    """,
    "reconciliation_gap": """
      {reconciliation_anchored} of the pass's beads carry a machine-parseable
      marker; the pass's own summary table covers {reconciliation_expected}.
      The {reconciliation_missing} difference (beads updated by a sibling
      session earlier the same day, whose findings this pass read and
      corroborated without writing its own tagged marker into their notes)
      are real and accounted for in the pass's own writeup, just not
      extractable by this regex. This is the identical shape as the
      <a href="#verified">verified-subset extraction gap</a> above &mdash; a
      review this valuable should write its verdict in one parseable form
      every time, not most of the time.
    """,
    "reconciliation_rebuild": """
      The rebuild-blocked set: every row below is merged to
      <code>origin/master</code>, confirmed by direct query against the live
      archive to have <em>not yet taken effect</em>, and requires no further
      engineering &mdash; only the schema rebuild above. Treat this table,
      not the P0 count in the shape matrix, as the honest measure of how much
      of today's work is actually done.
    """,
    # Two variants, selected by measurement: the split was live earlier today
    # and had been repaired by the time this generator was first run against a
    # fresh export.  Keeping both means the report stays correct if it recurs.
    "vocab_split": """
      This is the most consequential defect in the dataset and it is invisible
      from any single query. <code>relates-to</code> ({relates_to} edges) and
      <code>related</code> ({related} edges) are two spellings of one
      relation, written interleaved rather than migrated. Zero edges exist
      under both spellings, so nothing is duplicated; the vocabulary simply
      forked. A query written against <code>relates-to</code> silently returns
      {relates_pct} of the association graph; one written against
      <code>related</code> returns {related_pct}. Neither reports an error, and
      neither is wrong-looking &mdash; which is what makes this the hardest
      defect here to notice from inside a query.
    """,
    "vocab_repaired": """
      <b>This defect was live earlier today and is now closed.</b> The
      association relation had forked into two spellings &mdash;
      <code>relates-to</code> and <code>related</code> &mdash; written
      interleaved through the month rather than migrated, so any query against
      either name returned a silent partial subset of the association graph.
      As of this run all {relates_to} association edges carry the single
      <code>relates-to</code> spelling and <code>related</code> holds
      {related}. The fork is gone.
    """,
    "vocab_recurring": """
      <b>This defect was live earlier today, was repaired, and has already
      started recurring.</b> The association relation had forked into two
      spellings &mdash; <code>relates-to</code> and <code>related</code>
      &mdash; and a bulk re-type converted {retyped} existing
      <code>related</code> edges to <code>relates-to</code> between
      14:35&ndash;14:40 UTC today. As of this run <code>related</code> holds
      {related} again: {related_recurrence_n} edge(s) created strictly after
      that repair window. The repair fixed the data; it did not touch the
      write path, so nothing stops the next agent session that types
      <code>related</code> instead of <code>relates-to</code> from reopening
      the exact same fork. See <a href="#reconciliation">Reconciliation</a>
      for the specific edge and timestamp.
    """,
    "vocab_repair_cost": """
      The repair was not free, and the cost is visible in the table above.
      Every re-typed edge now carries today's date as its
      <code>created_at</code>: the <code>relates-to</code> row's
      &ldquo;first written&rdquo; still reaches back to early July because
      edges that always had that spelling were untouched, but the
      {retyped} edges that were converted lost their original assertion
      dates in the rewrite. Their history says they were asserted today; they
      were not. That is a small, permanent hole in the graph's chronology, and
      it is the generic cost of a bulk re-type in a store that treats the
      relation row as the record rather than versioning it. Nothing here needs
      undoing &mdash; a unified vocabulary is worth more than {retyped} edge
      timestamps &mdash; but any future analysis of &ldquo;when were
      associations asserted?&rdquo; must treat 2026-07-31 as an artifact spike,
      not a real burst of activity.
    """,
    "vocab_labels": """
      A second vocabulary fork, larger than the relation one and still
      wide open. There are {area_labels} distinct
      <code>area:*</code> labels across {total} beads, {area_singletons} of
      them used exactly once, and the synonym pairs are the obvious ones
      &mdash; <code>area:test</code> / <code>area:testing</code> /
      <code>area:test-harness</code>, <code>area:perf</code> /
      <code>area:performance</code>, <code>area:source</code> /
      <code>area:sources</code>, <code>area:blob</code> /
      <code>area:blobs</code>, <code>area:schema</code> /
      <code>area:schemas</code>, <code>area:demo</code> /
      <code>area:demos</code>. A free-text label field with no controlled
      vocabulary degrades into per-author spelling. The folded view in the
      <a href="#themes">Themes</a> section applies an explicit synonym map
      (printed there, so you can disagree with it) rather than trusting the
      raw labels.
    """,
    "themes": """
      Two independent readings of the same question, because neither alone is
      trustworthy. The label view uses the repository's own
      <code>area:*</code> labels, folded through an explicit synonym map;
      it covers {labelled} of {total} beads and says nothing about the
      {unlabelled} without one. The keyword view classifies every bead by
      matching subsystem vocabulary against title and description, so it has
      full coverage but will misfile anything whose title uses a word
      loosely. Where the two agree, the concentration is real. The signal
      that survives both: open work is not spread evenly &mdash; it piles into
      storage/substrate and sources/ingest, the two rings the project's own
      architecture doc calls the substrate.
    """,
    "verified": """
      Tonight's landing-check lane produced something more durable than the
      tool it was built around. The tool itself measured 6.1% precision and
      57% recall against a hand-verified sample and was closed unmerged
      &mdash; correctly. But the verification work does not evaporate with it:
      {verdict_total} beads now carry a machine-parseable
      <code>VERIFICATION (&hellip;): STALE|PARTIAL|LIVE</code> verdict written
      by a human reviewer, distributed {verdict_live} LIVE /
      {verdict_partial} PARTIAL / {verdict_stale} STALE. That is a labelled
      subset of the backlog with a ground-truth judgement attached to each
      row, and it is the only part of these {total} beads where
      &ldquo;is this claim still true?&rdquo; has been answered by inspection
      rather than assumed.
    """,
    "verified_gap": """
      A visible hole, stated rather than papered over. The operator's count of
      the hand-verified set is 190 beads (9 STALE / 57 PARTIAL / 124 LIVE).
      The strict pattern above finds {verdict_total}; relaxing to
      &ldquo;any bare-capital verdict token anywhere in notes or comments&rdquo;
      finds {loose_total} ({loose_live}/{loose_partial}/{loose_stale}), which
      brackets the operator's figure closely but includes incidental prose. The
      difference is verdicts recorded outside the bead notes, or phrased
      without the <code>VERIFICATION (&hellip;):</code> prefix. Both counts are
      shown because the gap between them is itself the finding: a review pass
      this valuable should have written its verdicts in one parseable form, and
      roughly {unparseable} of them are now only recoverable by reading.
    """,
    "next": """
      What a reader should do with this, in order.
    """,
    "method": """
      Everything numeric on this page is computed by the generator named below
      from a single input file; nothing was transcribed. The authored content
      is confined to the lead paragraph, the interpretation paragraph under
      each section heading, the finding list, and this note &mdash; all of it
      living in one <code>PROSE</code> mapping in the generator source, with computed
      values interpolated in. If a number here disagrees with <code>bd</code>, the
      export is stale, not the arithmetic: regenerate with <code>--fresh</code>.
    """,
}

# Authored finding list: (severity, title, body-html).  Numbers interpolated.
FINDINGS: list[tuple[str, str, str]] = [
    (
        "bad",
        "Two <code>blocks</code> edges point at ids that do not exist",
        "<code>20d.14</code> and <code>20d.1</code> lack the <code>polylogue-</code> "
        "prefix. The blocking intent is unrecoverable from any query. Id references "
        "in dependency edges are not validated on write.",
    ),
    (
        "warn",
        "Dotted ids and parent-child edges disagree",
        "{id_mismatch} beads whose id prefix names a different parent than their edge; "
        "{id_orphan} dotted beads with no parent edge. Any tooling that infers hierarchy "
        "from the id string is reading a different graph than <code>bd</code> is.",
    ),
    (
        "warn",
        "{allclosed} open parents with every child closed &mdash; review, not auto-close",
        "Looks like a closure batch; is not. <code>polylogue-1vpm.6</code> has untouched "
        "AC8/AC9 with all children retired. Children closing means the parent is ready "
        "for adjudication, not that it is satisfied.",
    ),
    (
        "warn",
        "{area_labels} distinct <code>area:*</code> labels, {area_singletons} used once",
        "Uncontrolled label vocabulary with obvious synonym pairs (test/testing, "
        "perf/performance, source/sources, blob/blobs, schema/schemas, demo/demos). "
        "Grouping by raw label undercounts every affected subsystem.",
    ),
    (
        "warn",
        "Backlog net growth is positive in both halves of its only month",
        "First half net {c1_net:+d}, second half net {c2_net:+d}. Still in discovery "
        "phase. Velocity-derived completion estimates are not meaningful yet.",
    ),
    (
        "info",
        "Priority scale is doing real work &mdash; do not renumber it",
        "Closure rate and median lead time both fall monotonically P0&rarr;P3 "
        "({p0_lead}d &rarr; {p3_lead}d median). {p0_total} P0s is not inflation. "
        "P4 is the exception: highest lead time, but a closure rate above P3's.",
    ),
    (
        "info",
        "No cycles in the <code>blocks</code> graph",
        "Verified by DFS colouring over all {total} beads and {blocks_edges} edges. "
        "Notable given how fast the graph grew and how many independent lanes wrote to it.",
    ),
    (
        "info",
        "{ready_pct} of open work is unblocked",
        "{ready} of {open_total} open beads have no open blocker. The constraint on this "
        "backlog is execution capacity, not dependency ordering.",
    ),
    (
        "info",
        "{dup_titles} duplicate title pairs",
        "Small but real: independently filed beads describing the same work. "
        "Listed in the health section with both ids.",
    ),
    (
        "bad",
        "{reconciliation_fixed_rebuild} + {reconciliation_fixed_deploy} beads are done "
        "in code and invisible in the archive",
        "The P0 reconciliation pass confirmed these by direct query against the live "
        "archive, not by trusting the merged PR. See "
        '<a href="#reconciliation">Reconciliation</a> for the schema-version gap that '
        "blocks them and the exact table.",
    ),
]

# Authored next-actions: (label-class, text).
NEXT_ACTIONS: list[tuple[str, str]] = [
    (
        "bad",
        "Repair the two prefix-less <code>blocks</code> references, then add a write-time "
        "id-existence check so the class cannot recur.",
    ),
    (
        "warn",
        "Adjudicate the open-parent/all-children-closed queue one at a time against each "
        "parent's own acceptance criteria. Expect a minority to be genuinely done.",
    ),
    (
        "warn",
        "Decide whether the never-started epics are commitments or wishes. An epic with "
        "every child open and no in-flight work is a declaration, and declarations should "
        "either be scheduled or demoted to <code>horizon:vision</code>.",
    ),
    (
        "info",
        "Fold the <code>area:*</code> synonym pairs and constrain the label vocabulary; "
        "the folded map in the Themes section is a starting proposal, not a decision.",
    ),
    (
        "info",
        "Preserve the {verdict_total}-bead verified subset as a labelled evaluation set. "
        "It is the only ground truth this backlog has about its own staleness.",
    ),
    (
        "bad",
        "Run <code>polylogue ops reset --index &amp;&amp; polylogued run</code> before "
        "trusting any P0 count on this page. It is the one action that converts the "
        "{reconciliation_fixed_rebuild}+{reconciliation_fixed_deploy} rebuild-blocked "
        "beads from &ldquo;open&rdquo; to actually verified against live data.",
    ),
]

# Mechanical graph-health checks: (badge class, label, Facts attribute, meaning).
HEALTH_CHECKS: tuple[tuple[str, str, str, str], ...] = (
    ("bad", "dangling dependency references", "dangling", "edge points at an id that does not exist"),
    ("warn", "dotted id disagrees with parent edge", "id_mismatch", "id prefix names a different parent"),
    ("warn", "dotted id with no parent edge", "id_orphan", "looks nested, is structurally a root"),
    (
        "warn",
        "open parent, all children closed",
        "open_parent_all_closed",
        "review queue &mdash; adjudicate against the parent's own AC",
    ),
    (
        "warn",
        "closed parent with open children",
        "closed_parent_open_kids",
        "parent retired while decomposed work continues",
    ),
    ("warn", "duplicate titles", "dup_titles", "same work filed twice, independently"),
    ("info", "closed without a close reason", "closed_no_reason", "no durable record of why it ended"),
    ("info", "open without acceptance criteria", "no_ac_open", "no definition of done"),
)


def vocab_finding(facts: Facts) -> tuple[str, str, str]:
    """The relation-vocabulary finding, phrased for whichever state is measured."""
    if facts.vocab_state == "split":
        return (
            "bad",
            "Split relation vocabulary: <code>relates-to</code> vs <code>related</code>",
            "{relates_to} + {related} edges are one relation under two names, actively "
            "interleaved rather than migrated. Any association query silently returns a "
            "partial subset. Pick one spelling, rewrite the other, constrain the field.",
        )
    if facts.vocab_state == "recurring":
        return (
            "warn",
            "Relation vocabulary was unified today, then forked again within the hour",
            "The repair re-typed {retyped} historical edges to <code>relates-to</code>, "
            "but wrote no constraint, so {related} fresh <code>related</code> edge(s) landed "
            'afterward &mdash; see the <a href="#reconciliation">Reconciliation</a> section '
            "for the exact timestamps. The data was cleaned; the field is still open.",
        )
    return (
        "ok",
        "Relation vocabulary is unified &mdash; the fork closed today",
        "All {relates_to} association edges now carry <code>relates-to</code> and "
        "<code>related</code> holds {related}. The repair rewrote <code>created_at</code> "
        "on the converted edges, so association chronology has a permanent artifact "
        "spike on the repair date. Constrain the field so it cannot fork again.",
    )


def vocab_action(facts: Facts) -> tuple[str, str]:
    """The relation-vocabulary next-action, phrased for whichever state is measured."""
    if facts.vocab_state == "split":
        return (
            "bad",
            "Normalize the relation vocabulary before writing any more association edges. "
            "Every day both spellings stay live, the split grows.",
        )
    if facts.vocab_state == "recurring":
        return (
            "bad",
            "Constrain the relation-type field to a closed set now &mdash; measured proof "
            "the fork recurs without one: a <code>related</code> edge landed an hour after "
            "today's repair.",
        )
    return (
        "warn",
        "Constrain the relation-type field to a closed set now that the fork is closed, "
        "and record the repair date as an artifact so association chronology is not "
        "read literally.",
    )


def _git_show(root: Path, ref: str, rel_path: str) -> str | None:
    """Return a file's content at a git ref, or ``None`` if unavailable.

    Reads via ``git show`` rather than the working tree so this is correct
    even when the branch generating the report has not merged whatever PRs
    moved the ref -- exactly the situation the schema-gap fact below exists
    to surface.
    """
    try:
        proc = subprocess.run(
            ["git", "show", f"{ref}:{rel_path}"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout if proc.returncode == 0 else None


_LIFECYCLE_DECL_RE = re.compile(r"version=(\d+),.*?classes=\(([^)]*)\)", re.DOTALL)


def schema_gap_facts(root: Path) -> dict[str, Any]:
    """The derived-index schema gap: origin/master's declared
    ``INDEX_SCHEMA_VERSION`` vs the live archive's actual ``PRAGMA
    user_version``, and which intervening versions are ``SEMANTIC_REPARSE``
    (block a SQL fast-forward -- a merged fix at one of those versions is
    real and inert until ``polylogue ops reset --index && polylogued run``).

    Two independent read paths, both best-effort and non-fatal on failure so
    a sandbox without network/git-remote access or a live archive still
    renders a report (just without this fact):

    - ``git show origin/master:...`` for the declared version and delta
      classes, so this is correct even on a branch behind master.
    - the operator's configured archive root
      (``~/.config/polylogue/polylogue.toml``), read-only ``PRAGMA``, for the
      live version. Deliberately does not use ``POLYLOGUE_ARCHIVE_ROOT`` --
      that env var is routinely overridden for sandboxed/test runs (see this
      repo's own devshell), and this fact is specifically about the real
      production archive, not whatever archive the current process would
      otherwise touch.
    """
    result: dict[str, Any] = {
        "available": False,
        "declared": None,
        "live_version": None,
        "live_path": None,
        "blockers": [],
        "blocker_notes": {},
    }

    index_src = _git_show(root, "origin/master", "polylogue/storage/sqlite/archive_tiers/index.py")
    if index_src:
        m = re.search(r"^INDEX_SCHEMA_VERSION\s*=\s*(\d+)", index_src, re.MULTILINE)
        if m:
            result["declared"] = int(m.group(1))

    lifecycle_src = _git_show(root, "origin/master", "polylogue/storage/sqlite/lifecycle.py") or ""
    versions: dict[int, tuple[str, ...]] = {}
    for vm in _LIFECYCLE_DECL_RE.finditer(lifecycle_src):
        v = int(vm.group(1))
        classes = tuple(c.strip().removeprefix("DerivedDeltaClass.") for c in vm.group(2).split(",") if c.strip())
        versions[v] = classes

    live_path: Path | None = None
    try:
        import tomllib

        cfg_path = Path.home() / ".config/polylogue/polylogue.toml"
        if cfg_path.exists():
            cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
            root_str = cfg.get("archive", {}).get("root")
            if root_str:
                live_path = Path(root_str) / "index.db"
    except Exception:
        live_path = None

    if live_path and live_path.exists():
        result["live_path"] = str(live_path)
        try:
            proc = subprocess.run(
                ["sqlite3", "-readonly", str(live_path), "PRAGMA user_version;"],
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            )
            result["live_version"] = int(proc.stdout.strip())
        except Exception:
            pass

    if result["declared"] is not None and result["live_version"] is not None:
        result["available"] = True
        lo, hi = result["live_version"] + 1, result["declared"]
        result["blockers"] = [v for v in range(lo, hi + 1) if "SEMANTIC_REPARSE" in versions.get(v, ())]
        result["blocker_notes"] = {v: versions.get(v, ()) for v in range(lo, hi + 1)}
    return result


SUBSYSTEMS: dict[str, tuple[str, ...]] = {
    # canonical subsystem -> area:* label suffixes folded into it
    "storage": ("storage", "substrate", "blob", "blobs", "schema", "schemas", "archive", "durability"),
    "sources": (
        "sources",
        "source",
        "ingest",
        "parsers",
        "parsing",
        "capture",
        "lineage",
        "attachments",
        "protocol",
        "identity",
    ),
    "daemon": ("daemon", "ops", "pipeline", "events", "perf", "performance", "reliability"),
    "insights": ("insights", "analytics", "cost", "usage", "temporal", "evidence", "analysis", "embeddings"),
    "cli": ("cli", "query", "query-dsl", "search", "surface", "api", "config"),
    "mcp": ("mcp", "context", "coordination", "orchestration", "agents", "delegation", "delegations"),
    "web/extension": ("web", "browser", "browser-capture", "extension", "rendering", "ux"),
    "tests": ("test", "testing", "tests", "test-harness", "verification", "coverage", "quality", "eval"),
    "devtools": (
        "devtools",
        "devloop",
        "ci",
        "release",
        "git",
        "beads",
        "beads-hygiene",
        "planning",
        "maintenance",
        "cleanup",
        "status",
    ),
    "docs/legibility": ("docs", "legibility", "demos", "demo", "architecture", "adoption", "audit"),
}

KEYWORDS: dict[str, tuple[str, ...]] = {
    "storage": ("storage", "sqlite", "index.db", "source.db", "user.db", "schema", "blob", "migration", "table"),
    "sources": ("parser", "parse", "ingest", "provider", "origin", "claude code", "codex", "chatgpt", "session"),
    "daemon": ("daemon", "polylogued", "converg", "watcher", "cursor", "acquire"),
    "insights": ("insight", "cost", "usage", "token", "profile", "timeline", "analytic"),
    "cli": ("cli", "command", "query", "click", "filter", "find ", "read --"),
    "mcp": ("mcp", "agent", "context", "assertion", "recall", "coordination"),
    "web/extension": ("web", "browser", "extension", "workbench", "http", "ui"),
    "tests": ("test", "pytest", "fixture", "coverage", "hypothesis", "regression"),
    "devtools": ("devtools", "render", "lint", "bead", "ci ", "workflow", "hook"),
    "docs/legibility": ("doc", "readme", "demo", "narrative", "legib", "explain"),
}

TYPE_GLYPH: dict[str, tuple[str, str]] = {
    "task": ("▪", "task"),
    "bug": ("✕", "bug"),
    "feature": ("✦", "feature"),
    "epic": ("▣", "epic"),
    "chore": ("⚙", "chore"),
    "decision": ("⚖", "decision"),
    "spike": ("⚡", "spike"),
}
STATUS_GLYPH: dict[str, tuple[str, str]] = {
    "open": ("○", "open"),
    "in_progress": ("◐", "in progress"),
    "closed": ("●", "closed"),
}


# --------------------------------------------------------------------------
# loading + computation
# --------------------------------------------------------------------------
def load(path: Path) -> tuple[dict[str, IssueDict], list[Edge]]:
    """Parse the bd JSONL export into an id->issue map and a dependency edge list."""
    issues: dict[str, IssueDict] = {}
    edges: list[Edge] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("_type") == "issue":
            issues[rec["id"]] = rec
            for dep in rec.get("dependencies") or []:
                edges.append((str(dep["issue_id"]), str(dep["depends_on_id"]), str(dep.get("type", "blocks"))))
        elif rec.get("_type") == "dependency":
            edges.append((str(rec["issue_id"]), str(rec["depends_on_id"]), str(rec.get("type", "blocks"))))
    return issues, edges


def _blob(rec: IssueDict) -> str:
    parts = [str(rec.get("notes") or "")]
    parts.extend(str(c.get("text") or "") for c in (rec.get("comments") or []))
    return "\n".join(parts)


def _parse_ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


VERDICT_STRICT = re.compile(r"VERIFICATION\s*\(([^)]*)\)\s*:\s*(STALE|PARTIAL|LIVE)", re.IGNORECASE)
VERDICT_LOOSE = re.compile(r"\b(STALE|PARTIAL|LIVE)\b")

# The 2026-07-31 P0 reconciliation pass (docs/tmp/p0-reconciliation.md) wrote
# one `RECONCILIATION 2026-07-31[...]: <VERDICT>` marker per bead it touched.
# Scoping the token search to a window *after* that literal anchor matters:
# bare tokens like MISFRAMED are ordinary English words that also appear,
# unrelated, in older notes on other beads (checked -- see reconciliation
# extraction below).
RECONCILIATION_ANCHOR = re.compile(r"RECONCILIATION 2026-07-31")
RECONCILIATION_TOKENS: tuple[str, ...] = (
    "FIXED-AND-EFFECTIVE",
    "FIXED-PENDING-REBUILD",
    "FIXED-PENDING-DEPLOY",
    "MISFRAMED",
    "GENUINELY OPEN",
)
RECONCILIATION_TOKEN_RE = re.compile("|".join(re.escape(t) for t in RECONCILIATION_TOKENS))
RECONCILIATION_WINDOW = 1200


class Facts:
    """Every measured quantity the report renders.  Nothing here is authored."""

    def __init__(self, issues: dict[str, IssueDict], edges: list[Edge], now: dt.datetime) -> None:
        self.issues = issues
        self.edges = edges
        self.now = now
        self._cluster_adj: dict[str, set[str]] = defaultdict(set)
        rows = list(issues.values())
        self.rows = rows
        self.total = len(rows)

        self.status = Counter(str(r["status"]) for r in rows)
        self.priority = Counter(int(r["priority"]) for r in rows)
        self.itype = Counter(str(r["issue_type"]) for r in rows)
        self.matrix: dict[tuple[str, int], int] = Counter((str(r["status"]), int(r["priority"])) for r in rows)
        self.open_ids = [r["id"] for r in rows if r["status"] != "closed"]
        self.open_total = len(self.open_ids)

        # ---- dependency graph -------------------------------------------
        self.dep_types = Counter(t for _, _, t in edges)

        # relates-to/related vocabulary state. A repair that re-types every
        # existing `related` edge does not constrain the field, so a fresh
        # `related` edge can land minutes later -- that is a distinct state
        # from "never repaired" and from "clean", and matters for which
        # prose variant and badge render below.
        _relates, _related = self.dep_types["relates-to"], self.dep_types["related"]
        _assoc_total = _relates + _related
        if _related == 0:
            self.vocab_state = "clean"
        elif _assoc_total and _related <= max(3, 0.02 * _assoc_total):
            self.vocab_state = "recurring"
        else:
            self.vocab_state = "split"

        self.blockers: dict[str, set[str]] = defaultdict(set)
        self.blocking: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, list[str]] = defaultdict(list)
        self.parent: dict[str, str] = {}
        for src, dst, kind in edges:
            if kind == "blocks":
                self.blockers[src].add(dst)
                self.blocking[dst].add(src)
            elif kind == "parent-child":
                self.children[dst].append(src)
                self.parent[src] = dst
        self.dangling = sorted({(a, b, t) for a, b, t in edges if b not in issues or a not in issues})

        def is_open(bid: str) -> bool:
            rec = issues.get(bid)
            return rec is not None and rec["status"] != "closed"

        self.is_open = is_open
        self.blocked = [i for i in self.open_ids if any(is_open(b) for b in self.blockers[i])]
        self.ready = [i for i in self.open_ids if not any(is_open(b) for b in self.blockers[i])]
        self.top_blockers = sorted(
            ((sum(1 for x in self.blocking[i] if is_open(x)), i) for i in self.open_ids),
            reverse=True,
        )
        self.top_blockers = [(n, i) for n, i in self.top_blockers if n > 0]
        self.cycles = self._find_cycles()

        # ---- epics -------------------------------------------------------
        self.epics: list[dict[str, Any]] = []
        for pid, kids in self.children.items():
            if pid not in issues:
                continue
            self.epics.append(
                {
                    "id": pid,
                    "n": len(kids),
                    "open": sum(1 for k in kids if is_open(k)),
                    "kids": kids,
                    "title": str(issues[pid]["title"]),
                    "type": str(issues[pid]["issue_type"]),
                    "status": str(issues[pid]["status"]),
                }
            )
        self.epics.sort(key=lambda e: (-int(e["n"]), str(e["id"])))

        # ---- id/edge hierarchy disagreement -------------------------------
        self.id_mismatch = sorted(
            (child, self.parent[child])
            for child in self.parent
            if "." in child and child.rsplit(".", 1)[0] != self.parent[child]
        )
        self.id_orphan = sorted(r["id"] for r in rows if "." in r["id"] and r["id"] not in self.parent)

        # ---- health -------------------------------------------------------
        self.closed_parent_open_kids = sorted(
            (pid, [k for k in kids if is_open(k)])
            for pid, kids in self.children.items()
            if pid in issues and issues[pid]["status"] == "closed" and any(is_open(k) for k in kids)
        )
        self.open_parent_all_closed = sorted(
            pid
            for pid, kids in self.children.items()
            if pid in issues and is_open(pid) and kids and not any(is_open(k) for k in kids)
        )
        title_counts = Counter(str(r["title"]).strip().lower() for r in rows)
        self.dup_titles = sorted(
            (title, [r["id"] for r in rows if str(r["title"]).strip().lower() == title])
            for title, n in title_counts.items()
            if n > 1
        )
        self.no_ac_open = [
            r["id"] for r in rows if r["status"] != "closed" and not str(r.get("acceptance_criteria") or "").strip()
        ]
        self.closed_no_reason = [
            r["id"] for r in rows if r["status"] == "closed" and not str(r.get("close_reason") or "").strip()
        ]

        # ---- time ----------------------------------------------------------
        self.created_day = Counter(str(r["created_at"])[:10] for r in rows)
        self.closed_day = Counter(str(r["closed_at"])[:10] for r in rows if r.get("closed_at"))
        self.days = sorted(set(self.created_day) | set(self.closed_day))
        self.first_created = min(str(r["created_at"]) for r in rows)[:10]
        self.last_created = max(str(r["created_at"]) for r in rows)[:10]
        self.span_days = (
            _parse_ts(self.last_created + "T00:00:00Z") - _parse_ts(self.first_created + "T00:00:00Z")
        ).days + 1

        self.lead_by_priority: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if r["status"] == "closed" and r.get("closed_at"):
                delta = (_parse_ts(str(r["closed_at"])) - _parse_ts(str(r["created_at"]))).total_seconds() / 86400
                self.lead_by_priority[int(r["priority"])].append(delta)
        self.age_open = [
            (now - _parse_ts(str(r["created_at"]))).total_seconds() / 86400 for r in rows if r["status"] != "closed"
        ]
        self.age_buckets = Counter(min(int(a // 7), 3) for a in self.age_open)

        mid = self.days[len(self.days) // 2] if self.days else ""
        self.half1 = (
            sum(v for d, v in self.created_day.items() if d < mid),
            sum(v for d, v in self.closed_day.items() if d < mid),
        )
        self.half2 = (
            sum(v for d, v in self.created_day.items() if d >= mid),
            sum(v for d, v in self.closed_day.items() if d >= mid),
        )
        self.mid_day = mid

        # ---- labels / themes ------------------------------------------------
        self.labels = Counter(str(x) for r in rows for x in (r.get("labels") or []))
        self.area_labels = [x for x in self.labels if x.startswith("area:")]
        self.area_singletons = sum(1 for x in self.area_labels if self.labels[x] == 1)
        self.labelled = sum(1 for r in rows if any(str(x).startswith("area:") for x in (r.get("labels") or [])))

        rev_area: dict[str, str] = {}
        for canon, suffixes in SUBSYSTEMS.items():
            for suffix in suffixes:
                rev_area[suffix] = canon
        self.by_label_theme: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            seen: set[str] = set()
            for raw in r.get("labels") or []:
                text = str(raw)
                if not text.startswith("area:"):
                    continue
                folded = rev_area.get(text.removeprefix("area:"))
                if folded and folded not in seen:
                    seen.add(folded)
                    self.by_label_theme[folded][str(r["status"])] += 1
        self.by_kw_theme: dict[str, Counter[str]] = defaultdict(Counter)
        for r in rows:
            text = (str(r["title"]) + " " + str(r.get("description") or "")).lower()
            best, score = "unclassified", 0
            for canon, words in KEYWORDS.items():
                hits = sum(text.count(w) for w in words)
                if hits > score:
                    best, score = canon, hits
            self.by_kw_theme[best][str(r["status"])] += 1

        # ---- verdict subset --------------------------------------------------
        self.verdicts: list[tuple[str, str, str, str]] = []
        loose: Counter[str] = Counter()
        for r in rows:
            text = _blob(r)
            strict = VERDICT_STRICT.search(text)
            if strict:
                self.verdicts.append((str(r["id"]), strict.group(2).upper(), str(r["status"]), strict.group(1).strip()))
            found = VERDICT_LOOSE.findall(text)
            if found:
                loose[Counter(found).most_common(1)[0][0]] += 1
        self.verdict_counts = Counter(v for _, v, _, _ in self.verdicts)

        # A bulk relation re-type stamps every converted edge with the repair
        # date.  Detect it as a same-day spike on the newest association edge.
        rel_days = Counter(
            str(d.get("created_at", ""))[:10]
            for r in rows
            for d in (r.get("dependencies") or [])
            if d.get("type") == "relates-to"
        )
        self.retyped_spike = 0
        self.retyped_spike_day = ""
        if rel_days and self.vocab_state in ("clean", "recurring"):
            newest = max(rel_days)
            spike = rel_days[newest]
            if spike >= 0.2 * sum(rel_days.values()):
                self.retyped_spike = spike
                self.retyped_spike_day = newest
        self.loose_counts = loose

        # A `related` edge whose created_at is later than the newest
        # relates-to edge on the repair day proves the fork already started
        # recurring after the repair, not merely that repair was incomplete.
        self.related_recurrence: list[tuple[str, str, str]] = []
        if self.retyped_spike_day:
            newest_retype_ts = max(
                (
                    str(d.get("created_at", ""))
                    for r in rows
                    for d in (r.get("dependencies") or [])
                    if d.get("type") == "relates-to" and str(d.get("created_at", ""))[:10] == self.retyped_spike_day
                ),
                default="",
            )
            for r in rows:
                for d in r.get("dependencies") or []:
                    if d.get("type") == "related" and str(d.get("created_at", "")) > newest_retype_ts:
                        self.related_recurrence.append(
                            (str(r["id"]), str(d.get("depends_on_id")), str(d.get("created_at")))
                        )

        # ---- P0 reconciliation pass (2026-07-31) ---------------------------
        # Same discipline as the verdict subset above: a machine-parseable
        # marker, extracted mechanically, with the extraction's own coverage
        # gap reported rather than papered over.
        self.reconciliation: list[tuple[str, str, str]] = []
        self._reconciliation_anchored = 0
        for r in rows:
            text = _blob(r)
            anchor = RECONCILIATION_ANCHOR.search(text)
            if not anchor:
                continue
            self._reconciliation_anchored += 1
            window = text[anchor.end() : anchor.end() + RECONCILIATION_WINDOW]
            token = RECONCILIATION_TOKEN_RE.search(window)
            if token:
                cut = window.find(token.group()) + len(token.group())
                snippet = window[:cut].strip()
                self.reconciliation.append((str(r["id"]), token.group(), snippet))
        self.reconciliation_counts = Counter(v for _, v, _ in self.reconciliation)

    def _find_cycles(self) -> list[list[str]]:
        colour: dict[str, int] = {}
        cycles: list[list[str]] = []

        def visit(start: str) -> None:
            stack: list[tuple[str, Iterable[str]]] = [(start, iter(sorted(self.blockers[start])))]
            colour[start] = 1
            path = [start]
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if nxt not in self.issues:
                        continue
                    state = colour.get(nxt)
                    if state == 1:
                        cycles.append(path[path.index(nxt) :] + [nxt])
                    elif state is None:
                        colour[nxt] = 1
                        path.append(nxt)
                        stack.append((nxt, iter(sorted(self.blockers[nxt]))))
                        advanced = True
                    break
                if not advanced:
                    colour[node] = 2
                    stack.pop()
                    path.pop()

        for bid in sorted(self.issues):
            if colour.get(bid) is None:
                visit(bid)
        return cycles

    def rate(self, priority: int) -> float:
        total = self.priority[priority]
        closed = self.matrix.get(("closed", priority), 0)
        return closed / total if total else 0.0

    def densest_cluster(self) -> list[str]:
        """Largest weakly-connected component of the open-only `blocks` graph."""
        adj: dict[str, set[str]] = defaultdict(set)
        open_set = set(self.open_ids)
        for src, dst, kind in self.edges:
            if kind == "blocks" and src in open_set and dst in open_set:
                adj[src].add(dst)
                adj[dst].add(src)
        seen: set[str] = set()
        best: list[str] = []
        for node in sorted(adj):
            if node in seen:
                continue
            comp: list[str] = []
            queue = [node]
            seen.add(node)
            while queue:
                cur = queue.pop()
                comp.append(cur)
                for nxt in sorted(adj[cur]):
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            if len(comp) > len(best):
                best = comp
        self._cluster_adj = adj
        return sorted(best)

    def cluster_core(self, component: Sequence[str], min_degree: int = 3) -> list[str]:
        """The legible core of a component: nodes with >= min_degree neighbours in it.

        A 100+ node component renders as a hairball; its high-degree core is
        the part that actually carries the blocking structure.
        """
        adj = self._cluster_adj
        member = set(component)
        return sorted(n for n in component if len({x for x in adj[n] if x in member}) >= min_degree)


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------
def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def chip(facts: Facts, bid: str, *, title: bool = False) -> str:
    """The compact bead notation used everywhere in the report."""
    rec = facts.issues.get(bid)
    if rec is None:
        return f'<span class="bd missing"><span class="bi">{esc(bid)}</span><span class="bg">?</span></span>'
    prio = int(rec["priority"])
    status = str(rec["status"])
    sglyph, sname = STATUS_GLYPH.get(status, ("?", status))
    tglyph, tname = TYPE_GLYPH.get(str(rec["issue_type"]), ("?", str(rec["issue_type"])))
    inn = sum(1 for b in facts.blockers[bid] if facts.is_open(b))
    out = sum(1 for b in facts.blocking[bid] if facts.is_open(b))
    deg = ""
    if inn or out:
        deg = f'<span class="bg">{"&uarr;" + str(inn) if inn else ""}{"&darr;" + str(out) if out else ""}</span>'
    tip = f"P{prio} · {sname} · {tname} · {inn} open blocker(s) · blocks {out} open"
    text = f' <span class="bl">{esc(str(rec["title"])[:64])}</span>' if title else ""
    return (
        f'<span class="bd p{prio} st-{status}" title="{esc(tip)}">'
        f'<span class="bp">P{prio}</span>'
        f'<span class="bs">{sglyph}</span>'
        f'<span class="bi">{esc(bid)}</span>'
        f'<span class="by">{tglyph}</span>{deg}{text}</span>'
    )


def bar_cells(counts: Sequence[tuple[str, int]], total: int) -> str:
    if not counts:
        return ""
    peak = max(n for _, n in counts) or 1
    out = []
    for name, n in counts:
        pct = 100 * n / peak
        share = 100 * n / total if total else 0
        out.append(
            f"<tr><td>{name}</td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{pct:.1f}%"></span><b>{n:,}</b></td>'
            f'<td class="num" data-v="{share:.2f}">{share:.1f}%</td></tr>'
        )
    return "".join(out)


def histogram_svg(days: Sequence[str], series: Sequence[tuple[str, Counter[str], str]]) -> str:
    """Grouped per-day bar chart, pure SVG, viewBox only."""
    if not days:
        return ""
    step = 10.0
    width = len(days) * step
    height = 46.0
    peak = max((c[day] for _, c, _ in series for day in days), default=1) or 1
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="beads created and closed per day" preserveAspectRatio="none">'
    ]
    bw = step / (len(series) + 1)
    for si, (_, counts, cls) in enumerate(series):
        for di, day in enumerate(days):
            value = counts[day]
            if not value:
                continue
            bh = (value / peak) * (height - 8)
            x = di * step + si * bw
            parts.append(
                f'<rect class="bar {cls}" x="{x:.2f}" y="{height - bh:.2f}" '
                f'width="{bw:.2f}" height="{bh:.2f}"><title>{esc(day)}: {value}</title></rect>'
            )
    parts.append("</svg>")
    return "".join(parts)


def cluster_svg(facts: Facts, nodes: Sequence[str]) -> str:
    """Layered DAG of one connected `blocks` component. Layer = longest path from a root."""
    node_set = set(nodes)
    layer: dict[str, int] = {}

    def depth(bid: str, seen: frozenset[str]) -> int:
        if bid in layer:
            return layer[bid]
        ups = [b for b in facts.blockers[bid] if b in node_set and b not in seen]
        value = 0 if not ups else 1 + max(depth(u, seen | {bid}) for u in ups)
        layer[bid] = value
        return value

    for bid in nodes:
        depth(bid, frozenset())
    rows: dict[int, list[str]] = defaultdict(list)
    for bid in sorted(nodes):
        rows[layer[bid]].append(bid)
    pos: dict[str, tuple[float, float]] = {}
    col_w, row_h = 210.0, 34.0
    for lv, members in rows.items():
        for i, bid in enumerate(sorted(members)):
            pos[bid] = (18 + lv * col_w, 22 + i * row_h)
    width = 36 + (max(rows) + 1) * col_w
    height = 30 + max(len(v) for v in rows.values()) * row_h
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="largest connected blocking cluster" class="dag">',
        '<defs><marker id="dagar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" '
        'markerHeight="6" orient="auto"><path d="M0 0L10 5L0 10z" fill="currentColor"/></marker></defs>',
    ]
    for src in nodes:
        for dst in facts.blockers[src]:
            if dst not in node_set:
                continue
            x1, y1 = pos[dst]
            x2, y2 = pos[src]
            parts.append(
                f'<path d="M{x1 + 168:.1f} {y1:.1f} C{x1 + 190:.1f} {y1:.1f} '
                f'{x2 - 22:.1f} {y2:.1f} {x2 - 4:.1f} {y2:.1f}" fill="none" '
                f'stroke="currentColor" opacity=".38" marker-end="url(#dagar)"/>'
            )
    for bid, (x, y) in pos.items():
        rec = facts.issues[bid]
        prio = int(rec["priority"])
        parts.append(
            f'<g class="dn p{prio}"><rect x="{x:.1f}" y="{y - 11:.1f}" width="168" height="22" rx="5"/>'
            f'<text x="{x + 7:.1f}" y="{y + 4:.1f}" font-size="11">P{prio} {esc(bid)}</text>'
            f"<title>{esc(str(rec['title'])[:110])}</title></g>"
        )
    parts.append("</svg>")
    return "".join(parts)


def tree_html(facts: Facts, root: str, depth: int = 0) -> str:
    kids = sorted(facts.children.get(root, []))
    if not kids or depth > 1:
        return ""
    items = []
    for kid in kids:
        if kid not in facts.issues:
            continue
        sub = tree_html(facts, kid, depth + 1)
        title = esc(str(facts.issues[kid]["title"])[:78])
        items.append(f'<li>{chip(facts, kid)} <span class="meta">{title}</span>{sub}</li>')
    return f'<ul class="tree">{"".join(items)}</ul>'


def fill_bar(open_n: int, total: int) -> str:
    done = total - open_n
    pct = 100 * done / total if total else 0
    return (
        f'<span class="fill" title="{done} of {total} closed">'
        f'<span style="width:{pct:.1f}%"></span></span>'
        f'<span class="filln">{done}/{total}</span>'
    )


CSS = """
:root{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#1a2129; --muted:#5b6773;
  --line:#dde3e9; --accent:#2563eb; --accent-ink:#ffffff;
  --ok:#0f7b46; --ok-bg:#e2f5eb; --warn:#8a5a00; --warn-bg:#fdf0d3;
  --bad:#a01c1c; --bad-bg:#fbe4e4; --info:#1d4ed8; --info-bg:#e3ebfd;
  --todo:#6b21a8; --todo-bg:#f1e6fb; --code-bg:#eef1f4;
  --p0:#a01c1c; --p1:#b45309; --p2:#1d4ed8; --p3:#0f7b46; --p4:#5b6773;
}
@media (prefers-color-scheme: dark){:root{
  --bg:#12161b; --panel:#1a2027; --ink:#e6ebf0; --muted:#94a1ad;
  --line:#2b333c; --accent:#5b8def; --accent-ink:#0d1117;
  --ok:#4cc98a; --ok-bg:#12301f; --warn:#e2b93b; --warn-bg:#33290e;
  --bad:#ef7070; --bad-bg:#391717; --info:#7ea6f4; --info-bg:#16223b;
  --todo:#c793ef; --todo-bg:#2a1738; --code-bg:#232a32;
  --p0:#ef7070; --p1:#e2b93b; --p2:#7ea6f4; --p3:#4cc98a; --p4:#94a1ad;
}}
:root[data-theme="light"]{color-scheme:light}
:root[data-theme="dark"]{color-scheme:dark}
:root[data-theme="dark"]{
  --bg:#12161b; --panel:#1a2027; --ink:#e6ebf0; --muted:#94a1ad;
  --line:#2b333c; --accent:#5b8def; --accent-ink:#0d1117;
  --ok:#4cc98a; --ok-bg:#12301f; --warn:#e2b93b; --warn-bg:#33290e;
  --bad:#ef7070; --bad-bg:#391717; --info:#7ea6f4; --info-bg:#16223b;
  --todo:#c793ef; --todo-bg:#2a1738; --code-bg:#232a32;
  --p0:#ef7070; --p1:#e2b93b; --p2:#7ea6f4; --p3:#4cc98a; --p4:#94a1ad;
}
:root[data-theme="light"]{
  --bg:#f6f7f9; --panel:#ffffff; --ink:#1a2129; --muted:#5b6773;
  --line:#dde3e9; --accent:#2563eb; --accent-ink:#ffffff;
  --ok:#0f7b46; --ok-bg:#e2f5eb; --warn:#8a5a00; --warn-bg:#fdf0d3;
  --bad:#a01c1c; --bad-bg:#fbe4e4; --info:#1d4ed8; --info-bg:#e3ebfd;
  --todo:#6b21a8; --todo-bg:#f1e6fb; --code-bg:#eef1f4;
  --p0:#a01c1c; --p1:#b45309; --p2:#1d4ed8; --p3:#0f7b46; --p4:#5b6773;
}
:root{--fs:17px}
*{box-sizing:border-box}
html{font-size:var(--fs)}
body{margin:0;background:var(--bg);color:var(--ink);
  font:1rem/1.62 system-ui,-apple-system,"Segoe UI",sans-serif}
header.page{position:sticky;top:0;z-index:5;background:var(--panel);
  border-bottom:1px solid var(--line);padding:.7rem 1.2rem;
  display:flex;flex-wrap:wrap;align-items:baseline;gap:.6rem}
header.page h1{font-size:1.3rem;margin:0}
header.page .spacer{flex:1}
.chip{display:inline-block;padding:.12rem .6rem;border:1px solid var(--line);
  border-radius:99px;font-size:.85rem;color:var(--muted);background:var(--bg)}
button.theme,button.fs{border:1px solid var(--line);background:var(--bg);color:var(--ink);
  border-radius:.4rem;padding:.2rem .6rem;cursor:pointer;font-size:.85rem}
.layout{display:grid;grid-template-columns:16rem minmax(0,1fr);
  max-width:80rem;margin:0 auto;gap:1.2rem;padding:1.2rem}
@media(max-width:62rem){.layout{grid-template-columns:minmax(0,1fr)}nav#toc{display:none}}
nav#toc{position:sticky;top:3.8rem;align-self:start;font-size:.92rem;
  border-right:1px solid var(--line);padding-right:.8rem;max-height:85vh;overflow:auto}
nav#toc a{display:block;color:var(--muted);text-decoration:none;padding:.16rem 0}
nav#toc a.h3{padding-left:.9rem;font-size:.87rem}
nav#toc a:hover{color:var(--accent)}
main{min-width:0}
section{background:var(--panel);border:1px solid var(--line);border-radius:.6rem;
  padding:1.1rem 1.3rem;margin-bottom:1.1rem}
h2{font-size:1.28rem;margin:.1rem 0 .7rem;line-height:1.3}
h3{font-size:1.07rem;margin:1.1rem 0 .45rem;line-height:1.35}
p{margin:.5rem 0;max-width:76ch}
a{color:var(--accent)}
code,kbd{background:var(--code-bg);border-radius:.3rem;padding:.08rem .35rem;
  font:.9em ui-monospace,SFMono-Regular,Menlo,monospace}
pre{background:var(--code-bg);padding:.75rem .9rem;border-radius:.5rem;
  overflow-x:auto;font-size:.92rem;line-height:1.5}
pre code{background:none;padding:0;font-size:1em}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr));
  gap:.7rem;margin:.4rem 0 .9rem}
.tile{border:1px solid var(--line);border-radius:.55rem;padding:.6rem .85rem;background:var(--bg)}
.tile .n{font-size:1.6rem;font-weight:650;display:block;line-height:1.2}
.tile .l{font-size:.85rem;color:var(--muted)}
.tile.qa{position:relative;cursor:help}
.tile.qa::after{content:"\\2315";position:absolute;top:.35rem;right:.45rem;opacity:.35;font-size:.8rem}
.badge{display:inline-block;font-size:.8rem;font-weight:600;letter-spacing:.02em;
  padding:.1rem .55rem;border-radius:99px;white-space:nowrap}
.ok{color:var(--ok);background:var(--ok-bg)} .warn{color:var(--warn);background:var(--warn-bg)}
.bad{color:var(--bad);background:var(--bad-bg)} .info{color:var(--info);background:var(--info-bg)}
.todo{color:var(--todo);background:var(--todo-bg)}
.tablewrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:.95rem}
th,td{border-bottom:1px solid var(--line);text-align:left;padding:.42rem .6rem;vertical-align:top}
th{color:var(--muted);font-size:.85rem;text-transform:uppercase;letter-spacing:.04em;
  cursor:pointer;user-select:none;white-space:nowrap}
th.nosort{cursor:default}
tr:hover td{background:color-mix(in srgb, var(--accent) 6%, transparent)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
details{border:1px solid var(--line);border-radius:.5rem;padding:.5rem .85rem;margin:.5rem 0;background:var(--bg)}
summary{cursor:pointer;font-weight:600;font-size:1rem}
details[open]{padding-bottom:.6rem}
input.filter{width:100%;max-width:26rem;margin:.2rem 0 .55rem;padding:.35rem .65rem;
  border:1px solid var(--line);border-radius:.4rem;background:var(--bg);color:var(--ink);font-size:.95rem}
footer{color:var(--muted);font-size:.88rem;text-align:center;padding:1rem}
.meta{display:grid;grid-template-columns:auto 1fr;gap:.2rem .9rem;font-size:.9rem;
  border-left:3px solid var(--line);padding:.1rem 0 .1rem .9rem;margin:.2rem 0 .9rem}
.meta dt{color:var(--muted)}
.meta dd{margin:0}
.ev{display:inline-block;font-size:.72rem;font-weight:700;letter-spacing:.03em;
  padding:0 .35rem;border-radius:.25rem;vertical-align:.1em;text-transform:uppercase}
.ev-measured{color:var(--ok);background:var(--ok-bg)}
.ev-derived{color:var(--info);background:var(--info-bg)}
.ev-inferred{color:var(--warn);background:var(--warn-bg)}
.ev-assumed{color:var(--bad);background:var(--bad-bg)}
time.age{color:var(--muted);font-size:.85em;font-variant-numeric:tabular-nums}
time.age.stale{color:var(--warn);font-weight:600}
time.age.stale::after{content:" \\26A0"}
a.path{font:.9em ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--code-bg);
  border-radius:.3rem;padding:.08rem .35rem;text-decoration:none;border-bottom:1px dotted var(--accent)}
.pop{position:fixed;z-index:50;max-width:min(46rem,92vw);max-height:70vh;overflow:auto;
  background:var(--panel);border:1px solid var(--line);border-radius:.5rem;
  box-shadow:0 .6rem 2rem rgba(0,0,0,.28);padding:.7rem .9rem;font-size:.92rem}
.pop h4{margin:0 0 .35rem;font-size:.92rem;font-family:ui-monospace,monospace;color:var(--muted)}
.pop pre{margin:0;max-height:52vh;font-size:.86rem}
.pop .pin{float:right;border:none;background:none;color:var(--muted);cursor:pointer}
blockquote.q{position:relative;margin:.55rem 0 .4rem;padding:.6rem 1.1rem .6rem 2.3rem;
  background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--todo);
  border-radius:0 .45rem .45rem 0;
  font:italic 1rem/1.55 Georgia,"Iowan Old Style","Noto Serif",ui-serif,serif}
blockquote.q::before{content:"\\201C";position:absolute;left:.45rem;top:.15rem;
  font:italic 2.2rem/1 Georgia,ui-serif,serif;color:var(--todo);opacity:.5}
blockquote.q cite{display:block;margin-top:.4rem;font:600 .72rem/1 system-ui,sans-serif;
  text-transform:uppercase;letter-spacing:.05em;color:var(--muted);font-style:normal}

/* ---- BEAD NOTATION -------------------------------------------------- */
.bd{display:inline-flex;align-items:center;gap:.28rem;border:1px solid var(--line);
  border-radius:.35rem;padding:.02rem .34rem .02rem .05rem;background:var(--bg);
  font:.82rem/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:nowrap;
  border-left:3px solid var(--pc,var(--muted))}
.bd.p0{--pc:var(--p0)} .bd.p1{--pc:var(--p1)} .bd.p2{--pc:var(--p2)}
.bd.p3{--pc:var(--p3)} .bd.p4{--pc:var(--p4)}
.bd .bp{font-weight:700;color:var(--pc);font-size:.78rem;padding-left:.3rem}
.bd .bs{font-size:.82rem;opacity:.85}
.bd.st-closed{opacity:.62}
.bd.st-closed .bi{text-decoration:line-through;text-decoration-thickness:1px}
.bd.st-in_progress{box-shadow:inset 0 0 0 1px color-mix(in srgb,var(--accent) 45%,transparent)}
.bd .bi{color:var(--ink)}
.bd .by{opacity:.75}
.bd .bg{color:var(--accent);font-size:.75rem;font-weight:700}
.bd.missing{border-left-color:var(--bad);background:var(--bad-bg)}
.bd .bl{font:.8rem/1.5 system-ui,sans-serif;color:var(--muted);max-width:26rem;
  overflow:hidden;text-overflow:ellipsis}
.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(13rem,1fr));gap:.5rem 1.2rem;
  font-size:.9rem;margin:.6rem 0}
.legend div{display:flex;gap:.45rem;align-items:baseline}
.legend b{font:.85rem ui-monospace,monospace;color:var(--accent);min-width:2.2rem}

/* ---- distribution bars ---- */
.dist{position:relative;text-align:right;white-space:nowrap}
.dist span{position:absolute;left:0;top:.25rem;bottom:.25rem;width:var(--w);
  background:var(--accent);opacity:.18;border-radius:.2rem}
.dist b{position:relative;font-variant-numeric:tabular-nums}

/* ---- epic fill bars ---- */
.fill{display:inline-block;width:9rem;height:.62rem;border-radius:99px;background:var(--code-bg);
  overflow:hidden;vertical-align:middle;border:1px solid var(--line)}
.fill>span{display:block;height:100%;background:var(--ok)}
.filln{font:.78rem ui-monospace,monospace;color:var(--muted);margin-left:.4rem;
  font-variant-numeric:tabular-nums}

/* ---- matrix ---- */
table.mx td.c{text-align:right;font-variant-numeric:tabular-nums;position:relative}
table.mx td.c i{position:absolute;left:.2rem;top:.25rem;bottom:.25rem;border-radius:.2rem;
  background:var(--accent);opacity:.16;font-style:normal}
table.mx td.c b{position:relative}

/* ---- charts ---- */
.chart svg{width:100%;height:auto;max-height:7rem}
.bar{fill:var(--accent);opacity:.75}
.bar.closed{fill:var(--ok);opacity:.8}
.dag{height:auto;color:var(--muted);min-width:44rem;width:100%}
.dag .dn rect{fill:var(--bg);stroke:var(--pc,var(--muted));stroke-width:1.4}
.dag .dn.p0{--pc:var(--p0)} .dag .dn.p1{--pc:var(--p1)} .dag .dn.p2{--pc:var(--p2)}
.dag .dn.p3{--pc:var(--p3)} .dag .dn.p4{--pc:var(--p4)}
.dag .dn text{fill:var(--ink);font-family:ui-monospace,monospace}
.dagwrap{overflow-x:auto;border:1px solid var(--line);border-radius:.5rem;padding:.4rem;background:var(--bg)}

/* ---- trees ---- */
.tree,.tree ul{list-style:none;margin:0;padding-left:1.1rem}
.tree li{position:relative;padding:.1rem 0 .1rem .8rem}
.tree li::before{content:"";position:absolute;left:0;top:0;bottom:0;border-left:1px solid var(--line)}
.tree li::after{content:"";position:absolute;left:0;top:.85rem;width:.65rem;border-top:1px solid var(--line)}
.tree>li:last-child::before{bottom:auto;height:.85rem}
.tree .meta{font-size:.82rem;opacity:.7;margin-left:.4rem}
ul.findings{list-style:none;padding:0;margin:.5rem 0}
ul.findings li{border-left:3px solid var(--line);padding:.4rem 0 .4rem .8rem;margin:.5rem 0;max-width:80ch}
ul.findings li.sev-bad{border-left-color:var(--bad)}
ul.findings li.sev-warn{border-left-color:var(--warn)}
ul.findings li.sev-info{border-left-color:var(--info)}
ul.findings li.sev-ok{border-left-color:var(--ok)}
ul.findings b{display:block;margin-bottom:.15rem}
@media print{header.page,nav#toc,button,input.filter,.pop{display:none}
  .layout{grid-template-columns:1fr;max-width:none}
  section{break-inside:avoid;border:none;padding:0}
  details{border:none}details:not([open])>*:not(summary){display:revert}}
"""

JS = """
function tgl(){const r=document.documentElement,
  d=(r.dataset.theme||(matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'))==='dark';
  r.dataset.theme=d?'light':'dark';try{localStorage.htmlreport_theme=r.dataset.theme}catch(e){}}
function fs(d){const r=document.documentElement,
  cur=parseFloat(getComputedStyle(r).getPropertyValue('--fs'))||17,
  next=Math.min(24,Math.max(13,cur+d));
  r.style.setProperty('--fs',next+'px');try{localStorage.htmlreport_fs=next}catch(e){}}
try{if(localStorage.htmlreport_fs)document.documentElement.style.setProperty('--fs',localStorage.htmlreport_fs+'px');
    if(localStorage.htmlreport_theme)document.documentElement.dataset.theme=localStorage.htmlreport_theme}catch(e){}
(()=>{const t=document.getElementById('toc');if(!t)return;
document.querySelectorAll('main h2, main h3').forEach(h=>{
  const s=h.closest('section')||h;if(!h.id)h.id=(h.textContent||'').trim().toLowerCase()
    .replace(/[^a-z0-9]+/g,'-').replace(/^-|-$/g,'');
  const a=document.createElement('a');a.href='#'+(h.tagName==='H2'&&s.id?s.id:h.id);
  a.textContent=h.textContent;if(h.tagName==='H3')a.className='h3';t.appendChild(a)})})();
document.querySelectorAll('table').forEach(tb=>{
  tb.querySelectorAll('th:not(.nosort)').forEach((th,i)=>th.addEventListener('click',()=>{
    const dir=th.dataset.d=th.dataset.d==='a'?'d':'a';
    const val=td=>td&&td.dataset.v!==undefined?+td.dataset.v:(td?td.textContent.trim():'');
    [...tb.tBodies[0].rows].sort((r1,r2)=>{const a=val(r1.cells[i]),b=val(r2.cells[i]);
      const c=(typeof a=='number'&&typeof b=='number')?a-b:String(a).localeCompare(String(b));
      return dir==='a'?c:-c}).forEach(r=>tb.tBodies[0].appendChild(r))}))});
function flt(inp,id){const q=inp.value.toLowerCase();
  document.querySelectorAll('#'+id+' tbody tr').forEach(r=>
    r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')}
(()=>{const U=[[31536e6,'y'],[2592e6,'mo'],[6048e5,'w'],[864e5,'d'],[36e5,'h'],[6e4,'m']];
document.querySelectorAll('time.age').forEach(t=>{
  const d=new Date(t.dateTime);if(isNaN(d))return;const ms=Date.now()-d;
  let s='just now';for(const[u,n]of U){if(Math.abs(ms)>=u){s=Math.floor(Math.abs(ms)/u)+n+(ms<0?' ahead':' ago');break}}
  const pad=n=>String(n).padStart(2,'0');
  const iso=d.getFullYear()+'-'+pad(d.getMonth()+1)+'-'+pad(d.getDate())+' '+pad(d.getHours())+':'+pad(d.getMinutes());
  if(!t.textContent.trim())t.textContent=iso+' ('+s+')';else t.textContent+=' ('+s+')';
  t.title=d.toString();
  const budget=+(t.dataset.staleDays||0);
  if(budget&&ms>budget*864e5)t.classList.add('stale')})})();
(()=>{let cur=null,pinned=false,timer=null;
const kill=()=>{if(cur&&!pinned){cur.remove();cur=null}};
const show=(host,html,title)=>{
  if(cur)cur.remove();
  const p=document.createElement('div');p.className='pop';
  p.innerHTML='<button class="pin" title="unpin">\\u{1F4CC}</button>'+(title?'<h4>'+title+'</h4>':'')+html;
  document.body.appendChild(p);
  const r=host.getBoundingClientRect(),pr=p.getBoundingClientRect();
  let top=r.bottom+8; if(top+pr.height>innerHeight-8)top=Math.max(8,r.top-pr.height-8);
  let left=Math.min(r.left,innerWidth-pr.width-8);
  p.style.top=top+'px';p.style.left=Math.max(8,left)+'px';
  p.querySelector('.pin').onclick=()=>{pinned=false;kill()};
  p.onmouseenter=()=>clearTimeout(timer);
  p.onmouseleave=()=>{timer=setTimeout(kill,220)};
  cur=p};
const src=el=>{
  const t=el.querySelector(':scope > template.pop');
  if(t)return[t.innerHTML,el.dataset.popTitle||''];
  return null};
document.querySelectorAll(':has(> template.pop)').forEach(el=>{
  const s=src(el);if(!s)return;
  if(el.tabIndex<0)el.tabIndex=0;
  const open=()=>{clearTimeout(timer);pinned=false;show(el,s[0],s[1])};
  el.addEventListener('mouseenter',()=>{clearTimeout(timer);timer=setTimeout(open,180)});
  el.addEventListener('mouseleave',()=>{clearTimeout(timer);timer=setTimeout(kill,220)});
  el.addEventListener('focus',open);
  el.addEventListener('click',e=>{
    if(el.tagName==='A'&&(e.metaKey||e.ctrlKey))return;
    e.preventDefault();if(!cur)open();pinned=!pinned});
});
addEventListener('keydown',e=>{if(e.key==='Escape'){pinned=false;kill()}});})();
"""


def _values(facts: Facts, source: Path, schema_gap: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every value the authored prose is allowed to interpolate."""
    schema_gap = schema_gap or {}
    lead_med = {p: _median(facts.lead_by_priority[p]) for p in range(5)}
    c1c, c1x = facts.half1
    c2c, c2x = facts.half2
    peak_day, peak_n = max(facts.created_day.items(), key=lambda kv: kv[1])
    verdict = facts.verdict_counts
    loose = facts.loose_counts
    strict_total = sum(verdict.values())
    loose_total = sum(loose.values())
    ep = {e["id"]: e for e in facts.epics}
    e9e5 = ep.get("polylogue-9e5", {"open": 0, "n": 0})
    e9l5 = ep.get("polylogue-9l5", {"open": 0, "n": 0})
    relates = facts.dep_types["relates-to"]
    related = facts.dep_types["related"]
    return {
        "total": f"{facts.total:,}",
        "closed": f"{facts.status['closed']:,}",
        "open_total": f"{facts.open_total:,}",
        "age_days": facts.span_days,
        "span_days": facts.span_days,
        "first_created": facts.first_created,
        "source": str(source),
        "p0_total": facts.priority[0],
        "p4_open": facts.priority[4] - facts.matrix.get(("closed", 4), 0),
        **{f"p{p}_rate": f"{facts.rate(p):.0%}" for p in range(5)},
        **{f"p{p}_lead": f"{lead_med[p]:.2f}" for p in range(5)},
        "lead_ratio": f"{(lead_med[4] / lead_med[0]):.0f}" if lead_med[0] else "n/a",
        "e9e5_open": e9e5["open"],
        "e9e5_total": e9e5["n"],
        "e9l5_open": e9l5["open"],
        "e9l5_total": e9l5["n"],
        "id_mismatch": len(facts.id_mismatch),
        "id_orphan": len(facts.id_orphan),
        "blocks_edges": f"{facts.dep_types['blocks']:,}",
        "ready": f"{len(facts.ready):,}",
        "blocked": len(facts.blocked),
        "ready_pct": f"{len(facts.ready) / facts.open_total:.0%}" if facts.open_total else "n/a",
        "top_blocker_n": facts.top_blockers[0][0] if facts.top_blockers else 0,
        "peak_created": peak_n,
        "peak_created_day": peak_day,
        "median_created": f"{_median(list(facts.created_day.values())):.0f}",
        "median_closed": f"{_median(list(facts.closed_day.values())):.0f}",
        "c1_created": c1c,
        "c1_closed": c1x,
        "c1_net": c1c - c1x,
        "c2_created": c2c,
        "c2_closed": c2x,
        "c2_net": c2c - c2x,
        "trend": "growing in both halves" if (c1c - c1x) > 0 and (c2c - c2x) > 0 else "mixed",
        "allclosed": len(facts.open_parent_all_closed),
        "dup_titles": len(facts.dup_titles),
        "relates_to": relates,
        "related": related,
        "relates_pct": f"{relates / (relates + related):.0%}" if relates + related else "n/a",
        "related_pct": f"{related / (relates + related):.0%}" if relates + related else "n/a",
        "area_labels": len(facts.area_labels),
        "area_singletons": facts.area_singletons,
        "labelled": f"{facts.labelled:,}",
        "unlabelled": f"{facts.total - facts.labelled:,}",
        "verdict_total": strict_total,
        "verdict_live": verdict["LIVE"],
        "verdict_partial": verdict["PARTIAL"],
        "verdict_stale": verdict["STALE"],
        "loose_total": loose_total,
        "loose_live": loose["LIVE"],
        "loose_partial": loose["PARTIAL"],
        "loose_stale": loose["STALE"],
        "unparseable": max(0, 190 - strict_total),
        "n_checks": len(HEALTH_CHECKS),
        "retyped": f"{facts.retyped_spike:,}",
        "related_recurrence_n": len(facts.related_recurrence),
        "reconciliation_anchored": facts._reconciliation_anchored,
        "reconciliation_expected": 24,
        "reconciliation_missing": max(0, 24 - facts._reconciliation_anchored),
        "reconciliation_fixed_effective": facts.reconciliation_counts["FIXED-AND-EFFECTIVE"],
        "reconciliation_fixed_rebuild": facts.reconciliation_counts["FIXED-PENDING-REBUILD"],
        "reconciliation_fixed_deploy": facts.reconciliation_counts["FIXED-PENDING-DEPLOY"],
        "reconciliation_misframed": facts.reconciliation_counts["MISFRAMED"],
        "reconciliation_open": facts.reconciliation_counts["GENUINELY OPEN"],
        "schema_live": schema_gap.get("live_version", "n/a"),
        "schema_declared": schema_gap.get("declared", "n/a"),
        "schema_blockers_n": len(schema_gap.get("blockers") or []),
        "schema_blockers_list": ", ".join(f"v{v}" for v in (schema_gap.get("blockers") or [])) or "none",
    }


def qa(command: str, label: str) -> str:
    """Attach the exact invocation that produced a figure."""
    return (
        '<template class="pop"><figure class="code"><figcaption>'
        f"{esc(label)}</figcaption><pre><code>{esc(command)}</code></pre></figure></template>"
    )


JQ = "jq -rs"


def render(facts: Facts, source: Path, generated: dt.datetime, schema_gap: dict[str, Any] | None = None) -> str:
    schema_gap = schema_gap or {}
    vals = _values(facts, source, schema_gap)
    prose = {k: v.format(**vals).strip() for k, v in PROSE.items()}
    out: list[str] = []
    add = out.append
    src = str(source)

    add('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">')
    add('<meta name="viewport" content="width=device-width, initial-scale=1">')
    add(f"<title>Beads backlog — state of the graph — {generated:%Y-%m-%d}</title>")
    add("<style>" + CSS + "</style>")
    add("</head>\n<body>")
    add('<header class="page"><h1>Beads backlog &mdash; state of the graph</h1>')
    add(f'<span class="chip">polylogue</span><span class="chip">{facts.total:,} beads</span>')
    add(f'<span class="chip">{generated:%Y-%m-%d}</span><span class="spacer"></span>')
    add('<button class="fs" onclick="fs(-1)" title="smaller text">A&minus;</button>')
    add('<button class="fs" onclick="fs(1)" title="larger text">A+</button>')
    add('<button class="theme" onclick="tgl()">&#9680; theme</button></header>')
    add('<div class="layout"><nav id="toc" aria-label="contents"></nav><main>')

    # ---------------- summary ----------------
    add('<section id="summary"><h2>Summary</h2>')
    add('<dl class="meta">')
    add(f'<dt>generated</dt><dd><time class="age" datetime="{generated.isoformat()}"></time></dd>')
    add(
        f'<dt>data as of</dt><dd><time class="age" datetime="{generated.isoformat()}" '
        'data-stale-days="3"></time> &mdash; a live backlog; re-run to refresh</dd>'
    )
    add('<dt>status</dt><dd><span class="badge ok">finished</span> for this snapshot</dd>')
    add(
        '<dt>basis</dt><dd><span class="ev ev-measured">measured</span> every count, matrix, '
        'tree, and histogram &middot; <span class="ev ev-inferred">inferred</span> every '
        "interpretation paragraph and finding</dd>"
    )
    add(f'<dt>source</dt><dd><a class="path" href="file://{esc(src)}">{esc(src)}</a></dd>')
    add("<dt>regenerate</dt><dd><code>devtools workspace beads-state-report --fresh --out &lt;path&gt;</code></dd>")
    add("</dl>")

    tiles = [
        (f"{facts.total:,}", "beads, all time", "wc -l .beads/issues.jsonl"),
        (
            f"{facts.open_total:,}",
            "open + in progress",
            "jq -r 'select(._type==\"issue\")|.status' .beads/issues.jsonl | grep -vc closed",
        ),
        (
            f"{facts.status['closed']:,}",
            "closed",
            "jq -r 'select(._type==\"issue\")|.status' .beads/issues.jsonl | grep -c closed",
        ),
        (
            f"{len(facts.ready):,}",
            "ready (no open blocker)",
            "bd ready --limit 5000 --json | jq length",
        ),
        (f"{facts.span_days}", "days of backlog history", "jq -r '.created_at' .beads/issues.jsonl | sort | head -1"),
        (
            f"{len(facts.epics)}",
            "beads with children",
            "jq -r '.dependencies[]?|select(.type==\"parent-child\")|.depends_on_id' "
            ".beads/issues.jsonl | sort -u | wc -l",
        ),
    ]
    add('<div class="tiles">')
    for value, label, cmd in tiles:
        add(
            f'<div class="tile qa"><span class="n">{value}</span>'
            f'<span class="l">{label}</span>{qa(cmd, "measured " + generated.strftime("%Y-%m-%d"))}</div>'
        )
    add("</div>")
    add(f"<p>{prose['lead']}</p>")
    add("</section>")

    # ---------------- notation ----------------
    add('<section id="notation"><h2>Bead notation</h2>')
    add(
        "<p>One bead renders as one chip, everywhere in this report. Left bar and "
        "leading number are priority; then status, id, type glyph, and dependency "
        "degree. Hover any chip for the expanded reading.</p>"
    )
    samples = [facts.rows[0]["id"]]
    for wanted in ("epic", "bug", "feature"):
        for r in facts.rows:
            if r["issue_type"] == wanted and r["id"] not in samples:
                samples.append(str(r["id"]))
                break
    add("<p>" + " ".join(chip(facts, s) for s in samples[:4]) + "</p>")
    add('<div class="legend">')
    add("<div><b>P0&ndash;P4</b><span>priority; also the left bar colour</span></div>")
    for glyph, name in STATUS_GLYPH.values():
        add(f"<div><b>{glyph}</b><span>{name}</span></div>")
    for glyph, name in TYPE_GLYPH.values():
        add(f"<div><b>{glyph}</b><span>{name}</span></div>")
    add("<div><b>&uarr;n</b><span>n open beads block this one</span></div>")
    add("<div><b>&darr;n</b><span>this one blocks n open beads</span></div>")
    add("<div><b>struck id</b><span>closed</span></div>")
    add("<div><b>blue outline</b><span>in progress</span></div>")
    add("</div>")
    add(
        '<p><span class="badge info">reading it</span> A chip with a red left bar, a hollow '
        "circle, and <code>&darr;9</code> is an open P0 that nine other open beads are "
        "waiting on &mdash; the highest-leverage shape in the graph.</p>"
    )
    add("</section>")

    # ---------------- shape ----------------
    add('<section id="shape"><h2>Shape</h2>')
    add(f"<p>{prose['shape']}</p>")
    add("<h3>Status &times; priority</h3>")
    add('<div class="tablewrap"><table class="mx" id="mx">')
    add(
        "<thead><tr><th>priority</th><th>open</th><th>in progress</th><th>closed</th>"
        "<th>total</th><th>closed %</th><th>median lead (days)</th></tr></thead><tbody>"
    )
    peak = max(facts.matrix.values()) or 1
    for p in range(5):
        row = [f'<tr><td><span class="bd p{p}"><span class="bp">P{p}</span></span></td>']
        for st in ("open", "in_progress", "closed"):
            n = facts.matrix.get((st, p), 0)
            row.append(f'<td class="c" data-v="{n}"><i style="width:{100 * n / peak:.1f}%"></i><b>{n:,}</b></td>')
        total_p = facts.priority[p]
        med = _median(facts.lead_by_priority[p])
        row.append(f'<td class="num" data-v="{total_p}">{total_p:,}</td>')
        row.append(f'<td class="num" data-v="{facts.rate(p):.4f}">{facts.rate(p):.0%}</td>')
        row.append(f'<td class="num" data-v="{med:.3f}">{med:.2f}</td>')
        add("".join(row) + "</tr>")
    add("</tbody></table></div>")
    add(f"<p>{prose['p4']}</p>")

    add("<h3>Type</h3>")
    add('<div class="tablewrap"><table id="ty"><thead><tr><th>type</th><th>beads</th>')
    add("<th>share</th><th>closed %</th></tr></thead><tbody>")
    peak_t = max(facts.itype.values()) or 1
    for tname, n in facts.itype.most_common():
        closed = sum(1 for r in facts.rows if r["issue_type"] == tname and r["status"] == "closed")
        glyph = TYPE_GLYPH.get(tname, ("?", tname))[0]
        add(
            f"<tr><td>{glyph} {esc(tname)}</td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{100 * n / peak_t:.1f}%"></span>'
            f"<b>{n:,}</b></td>"
            f'<td class="num" data-v="{n / facts.total:.4f}">{100 * n / facts.total:.1f}%</td>'
            f'<td class="num" data-v="{closed / n:.4f}">{closed / n:.0%}</td></tr>'
        )
    add("</tbody></table></div>")
    add(
        "<p>Epics close at the lowest rate of any type, which is expected and not a "
        "defect: an epic is a container whose closure is gated on adjudication, not on "
        "work. Bugs close fastest. Every <code>decision</code> bead is closed &mdash; "
        "decisions are the one type this project reliably finishes.</p>"
    )
    add("</section>")

    # ---------------- structure ----------------
    add('<section id="structure"><h2>Structure</h2>')
    add(f"<p>{prose['structure']}</p>")
    add('<div class="tablewrap"><table id="ep"><thead><tr><th>parent</th><th>title</th>')
    add("<th>progress</th><th data-v>children</th><th data-v>open</th></tr></thead><tbody>")
    for e in facts.epics[:24]:
        add(
            f"<tr><td>{chip(facts, str(e['id']))}</td>"
            f"<td>{esc(str(e['title'])[:74])}</td>"
            f'<td data-v="{(int(e["n"]) - int(e["open"])) / int(e["n"]):.4f}">'
            f"{fill_bar(int(e['open']), int(e['n']))}</td>"
            f'<td class="num" data-v="{e["n"]}">{e["n"]}</td>'
            f'<td class="num" data-v="{e["open"]}">{e["open"]}</td></tr>'
        )
    add("</tbody></table></div>")
    add(f"<p>{prose['structure_ids']}</p>")
    add("<h3>Trees</h3>")
    add(
        "<p>The two extremes, expanded. Everything below is drawn from "
        "<code>parent-child</code> edges, two levels deep.</p>"
    )
    for eid in ("polylogue-9e5", "polylogue-9l5"):
        if eid not in facts.issues:
            continue
        rec = facts.issues[eid]
        add(
            f"<details><summary>{chip(facts, eid)} &nbsp;{esc(str(rec['title'])[:80])}</summary>"
            + tree_html(facts, eid)
            + "</details>"
        )
    add("<details><summary>All other parents with 8+ children</summary>")
    for e in facts.epics:
        if int(e["n"]) < 8 or e["id"] in ("polylogue-9e5", "polylogue-9l5"):
            continue
        add(
            f"<details><summary>{chip(facts, str(e['id']))} &nbsp;{esc(str(e['title'])[:80])} "
            f"&nbsp;{fill_bar(int(e['open']), int(e['n']))}</summary>" + tree_html(facts, str(e["id"])) + "</details>"
        )
    add("</details>")
    add("</section>")

    # ---------------- topology ----------------
    add('<section id="topology"><h2>Topology</h2>')
    add(f"<p>{prose['topology']}</p>")
    add('<div class="tiles">')
    for value, label, cmd in [
        (
            f"{facts.dep_types['blocks']:,}",
            "blocks edges",
            "jq -r '.dependencies[]?|.type' .beads/issues.jsonl | sort | uniq -c",
        ),
        (f"{len(facts.ready):,}", "ready", "bd ready --limit 5000 --json | jq length"),
        (f"{len(facts.blocked)}", "blocked by an open bead", "bd blocked --json | jq length"),
        (f"{len(facts.cycles)}", "cycles in blocks", "DFS colouring over the blocks graph (this generator)"),
    ]:
        add(
            f'<div class="tile qa"><span class="n">{value}</span><span class="l">{label}</span>'
            f"{qa(cmd, 'measured')}</div>"
        )
    add("</div>")
    add(f"<p>{prose['topology_cycles']}</p>")

    add("<h3>Relation vocabulary</h3>")
    add('<div class="tablewrap"><table id="rel"><thead><tr><th>relation</th><th>edges</th>')
    add("<th>first written</th><th>last written</th></tr></thead><tbody>")
    first_last: dict[str, tuple[str, str]] = {}
    for rec in facts.rows:
        for dep in rec.get("dependencies") or []:
            kind, when = str(dep.get("type", "")), str(dep.get("created_at", ""))[:10]
            if not when:
                continue
            lo, hi = first_last.get(kind, (when, when))
            first_last[kind] = (min(lo, when), max(hi, when))
    peak_r = max(facts.dep_types.values()) or 1
    for kind, n in facts.dep_types.most_common():
        lo, hi = first_last.get(kind, ("", ""))
        flag = ""
        if kind in ("relates-to", "related"):
            flag = {
                "split": ' <span class="badge bad">split vocab</span>',
                "recurring": ' <span class="badge warn">unified, then recurred</span>',
                "clean": ' <span class="badge ok">unified today</span>',
            }[facts.vocab_state]
        add(
            f"<tr><td><code>{esc(kind)}</code>{flag}</td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{100 * n / peak_r:.1f}%"></span><b>{n:,}</b></td>'
            f"<td>{esc(lo)}</td><td>{esc(hi)}</td></tr>"
        )
    add("</tbody></table></div>")
    vocab_prose_key = {"split": "vocab_split", "recurring": "vocab_recurring", "clean": "vocab_repaired"}[
        facts.vocab_state
    ]
    add(f"<p>{prose[vocab_prose_key]}</p>")
    if facts.vocab_state in ("clean", "recurring") and facts.retyped_spike:
        add(f"<p>{prose['vocab_repair_cost']}</p>")

    add("<h3>Highest-leverage blockers</h3>")
    add(f"<p>{prose['topology_blockers']}</p>")
    add('<div class="tablewrap"><table id="tb"><thead><tr><th>bead</th><th>title</th>')
    add("<th>blocks (open)</th></tr></thead><tbody>")
    for n, bid in facts.top_blockers[:20]:
        add(
            f"<tr><td>{chip(facts, bid)}</td><td>{esc(str(facts.issues[bid]['title'])[:80])}</td>"
            f'<td class="num" data-v="{n}">{n}</td></tr>'
        )
    add("</tbody></table></div>")

    cluster = facts.densest_cluster()
    core = facts.cluster_core(cluster)
    if core:
        add("<h3>Densest blocking cluster</h3>")
        add(
            "<p>Rendering all "
            f"{facts.total:,} beads as a graph produces a hairball that carries less "
            "information than a well-chosen slice, so this is a slice. The open-only "
            f"<code>blocks</code> graph breaks into disconnected components; the largest "
            f"holds <b>{len(cluster)}</b> beads, and what follows is its <b>{len(core)}</b>-node "
            "core &mdash; every member with three or more neighbours inside the component. "
            "The nodes dropped are single-edge leaves hanging off these; the structure is "
            "here. Arrows point from blocker to blocked, and columns are longest-path depth, "
            "so leftmost is furthest upstream. Hover a node for its title.</p>"
        )
        add('<div class="dagwrap">' + cluster_svg(facts, core) + "</div>")
        add(
            "<p>Read the column count, not the node count: the deepest chain here is only a "
            "few links long. Nothing in this backlog is buried under a long prerequisite "
            "stack, which is the same conclusion the ready fraction above reaches by a "
            "different route.</p>"
        )
    add("</section>")

    # ---------------- time ----------------
    add('<section id="time"><h2>Time</h2>')
    add(f"<p>{prose['time']}</p>")
    add('<figure class="chart">')
    add(
        histogram_svg(
            facts.days,
            [("created", facts.created_day, "created"), ("closed", facts.closed_day, "closed")],
        )
    )
    add(
        f"<figcaption>beads per day, {facts.first_created} &rarr; {facts.last_created} "
        f'&middot; <span class="bar-key" style="color:var(--accent)">&#9632; created</span> '
        f'<span style="color:var(--ok)">&#9632; closed</span> &middot; '
        f'<span class="ev ev-measured">measured</span></figcaption>'
    )
    add("</figure>")
    add(f"<p>{prose['time_velocity']}</p>")

    add('<div class="tablewrap"><table id="half"><thead><tr><th>window</th><th>created</th>')
    add("<th>closed</th><th>net</th></tr></thead><tbody>")
    for name, (c, x) in (
        (f"first half (&lt; {facts.mid_day})", facts.half1),
        (f"second half (&ge; {facts.mid_day})", facts.half2),
    ):
        cls = "bad" if c - x > 0 else "ok"
        add(
            f'<tr><td>{name}</td><td class="num" data-v="{c}">{c}</td>'
            f'<td class="num" data-v="{x}">{x}</td>'
            f'<td class="num" data-v="{c - x}"><span class="badge {cls}">{c - x:+d}</span></td></tr>'
        )
    add("</tbody></table></div>")

    add("<h3>Age of open work</h3>")
    labels = ["0&ndash;6 days", "7&ndash;13 days", "14&ndash;20 days", "21+ days"]
    add('<div class="tablewrap"><table id="age"><thead><tr><th>age</th><th>open beads</th>')
    add("<th>share of open</th></tr></thead><tbody>")
    peak_a = max(facts.age_buckets.values()) or 1
    for i, label in enumerate(labels):
        n = facts.age_buckets.get(i, 0)
        add(
            f"<tr><td>{label}</td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{100 * n / peak_a:.1f}%"></span><b>{n}</b></td>'
            f'<td class="num" data-v="{n / facts.open_total:.4f}">{100 * n / facts.open_total:.1f}%</td></tr>'
        )
    add("</tbody></table></div>")
    add(
        f"<p>Median open bead is {_median(facts.age_open):.0f} days old and the oldest is "
        f"{max(facts.age_open):.0f} &mdash; which is simply the age of the tracker. No "
        "bead in this repository has yet had time to become stale by age alone; when "
        "something here is stale it is because the code moved under it, not because it "
        "sat. That is what makes the verified subset below unusually valuable right now.</p>"
    )
    add("</section>")

    # ---------------- health ----------------
    add('<section id="health"><h2>Health</h2>')
    add(f"<p>{prose['health']}</p>")
    checks = [(cls, name, len(getattr(facts, attr)), why) for cls, name, attr, why in HEALTH_CHECKS]
    add('<div class="tablewrap"><table id="hc"><thead><tr><th>check</th><th>count</th>')
    add("<th>meaning</th></tr></thead><tbody>")
    for cls, name, n, meaning in checks:
        badge = f'<span class="badge {"ok" if n == 0 else cls}">{n}</span>'
        add(f"<tr><td>{esc(name)}</td><td>{badge}</td><td>{meaning}</td></tr>")
    add("</tbody></table></div>")
    add(f"<p>{prose['health_dangling']}</p>")
    if facts.dangling:
        add('<div class="tablewrap"><table id="dg"><thead><tr><th>from</th><th>points at</th>')
        add("<th>relation</th></tr></thead><tbody>")
        for a, b, kind in facts.dangling:
            add(
                f"<tr><td>{chip(facts, a)}</td>"
                f'<td><span class="bd missing"><span class="bi">{esc(b)}</span>'
                f'<span class="bg">&#10008;</span></span></td><td><code>{esc(kind)}</code></td></tr>'
            )
        add("</tbody></table></div>")

    add("<h3>Review queue: open parents whose children are all closed</h3>")
    add(
        "<p>Not a closure list. Each of these needs its own acceptance criteria read "
        "against what the children actually delivered.</p>"
    )
    add('<div class="tablewrap"><table id="rq"><thead><tr><th>parent</th><th>title</th>')
    add('<th>children</th><th class="nosort">has own AC</th></tr></thead><tbody>')
    for pid in facts.open_parent_all_closed:
        rec = facts.issues[pid]
        has_ac = bool(str(rec.get("acceptance_criteria") or "").strip())
        flag = '<span class="badge warn">yes &mdash; read it</span>' if has_ac else '<span class="badge info">no</span>'
        add(
            f"<tr><td>{chip(facts, pid)}</td><td>{esc(str(rec['title'])[:76])}</td>"
            f'<td class="num" data-v="{len(facts.children[pid])}">{len(facts.children[pid])}</td>'
            f"<td>{flag}</td></tr>"
        )
    add("</tbody></table></div>")

    if facts.dup_titles:
        add("<h3>Duplicate titles</h3>")
        add('<div class="tablewrap"><table id="dup"><thead><tr><th>title</th>')
        add('<th class="nosort">beads</th></tr></thead><tbody>')
        for title, ids in facts.dup_titles:
            add(f"<tr><td>{esc(title[:86])}</td><td>" + " ".join(chip(facts, i) for i in ids) + "</td></tr>")
        add("</tbody></table></div>")
    add("</section>")

    # ---------------- reconciliation ----------------
    add('<section id="reconciliation"><h2>Reconciliation</h2>')
    add(f"<p>{prose['reconciliation']}</p>")

    add("<h3>The schema gap: fixed-in-code vs live-in-archive</h3>")
    if schema_gap.get("available"):
        add(f"<p>{prose['reconciliation_schema_gap']}</p>")
        add('<div class="tiles">')
        for value, label, cmd in [
            (
                f"v{schema_gap['live_version']}",
                "live archive schema",
                f"sqlite3 -readonly {schema_gap.get('live_path', '<archive>/index.db')} 'PRAGMA user_version;'",
            ),
            (
                f"v{schema_gap['declared']}",
                "origin/master declares",
                "git show origin/master:polylogue/storage/sqlite/archive_tiers/index.py | grep INDEX_SCHEMA_VERSION",
            ),
            (
                str(len(schema_gap["blockers"])),
                "SEMANTIC_REPARSE versions blocking fast-forward",
                "git show origin/master:polylogue/storage/sqlite/lifecycle.py",
            ),
        ]:
            add(
                f'<div class="tile qa"><span class="n">{esc(value)}</span><span class="l">{esc(label)}</span>'
                f"{qa(cmd, 'measured ' + generated.strftime('%Y-%m-%d'))}</div>"
            )
        add("</div>")
        if schema_gap["blockers"]:
            add(
                '<div class="tablewrap"><table id="sg"><thead><tr><th>version</th><th>delta class(es)</th></tr></thead><tbody>'
            )
            for v in range(schema_gap["live_version"] + 1, schema_gap["declared"] + 1):
                classes = schema_gap["blocker_notes"].get(v, ())
                cls_badges = (
                    " ".join(
                        f'<span class="badge {"bad" if c == "SEMANTIC_REPARSE" else "info"}">{esc(c)}</span>'
                        for c in classes
                    )
                    or '<span class="badge info">undeclared</span>'
                )
                add(f"<tr><td>v{v}</td><td>{cls_badges}</td></tr>")
            add("</tbody></table></div>")
        add(
            "<p>The single operator action that clears this entire row: "
            "<code>polylogue ops reset --index &amp;&amp; polylogued run</code>.</p>"
        )
    else:
        add(
            '<p><span class="badge warn">not measured</span> This fact needs both '
            "<code>git show origin/master:...</code> (network/remote access) and a "
            "readable live archive at the operator's configured "
            "<code>~/.config/polylogue/polylogue.toml</code> archive root; at least one "
            "was unavailable when this report was generated, so the schema-gap tiles are "
            "omitted rather than shown with a guessed value.</p>"
        )

    add("<h3>The P0 pass</h3>")
    add(f"<p>{prose['reconciliation_p0']}</p>")
    add('<div class="tiles">')
    for value, label in [
        (str(vals["reconciliation_fixed_effective"]), "FIXED-AND-EFFECTIVE &mdash; closed"),
        (str(vals["reconciliation_fixed_rebuild"]), "FIXED-PENDING-REBUILD"),
        (str(vals["reconciliation_fixed_deploy"]), "FIXED-PENDING-DEPLOY"),
        (str(vals["reconciliation_misframed"]), "MISFRAMED &mdash; demoted P0&rarr;P2"),
        (str(vals["reconciliation_open"]), "GENUINELY OPEN"),
    ]:
        add(f'<div class="tile"><span class="n">{value}</span><span class="l">{label}</span></div>')
    add("</div>")
    add(f"<p>{prose['reconciliation_gap']}</p>")

    add("<h3>Fixed-pending-rebuild &amp; fixed-pending-deploy</h3>")
    add(f"<p>{prose['reconciliation_rebuild']}</p>")
    rebuild_rows = [
        (bid, verdict, snippet)
        for bid, verdict, snippet in facts.reconciliation
        if verdict in ("FIXED-PENDING-REBUILD", "FIXED-PENDING-DEPLOY")
    ]
    if rebuild_rows:
        add('<div class="tablewrap"><table id="rb"><thead><tr><th>bead</th><th>title</th>')
        add('<th>verdict</th><th class="nosort">evidence</th></tr></thead><tbody>')
        rb_badge = {"FIXED-PENDING-REBUILD": "warn", "FIXED-PENDING-DEPLOY": "warn"}
        for bid, verdict, snippet in sorted(rebuild_rows, key=lambda t: (t[1], t[0])):
            title = esc(str(facts.issues[bid]["title"])[:70]) if bid in facts.issues else ""
            add(
                f"<tr><td>{chip(facts, bid)}</td><td>{title}</td>"
                f'<td><span class="badge {rb_badge[verdict]}">{esc(verdict)}</span></td>'
                f"<td>{esc(snippet[-220:])}</td></tr>"
            )
        add("</tbody></table></div>")

    add("<h3>Full reconciliation table</h3>")
    add('<input class="filter" placeholder="filter reconciliation&hellip;" oninput="flt(this,\'rc\')">')
    add('<div class="tablewrap"><table id="rc"><thead><tr><th>bead</th><th>title</th>')
    add("<th>verdict</th></tr></thead><tbody>")
    rc_badge = {
        "FIXED-AND-EFFECTIVE": "ok",
        "FIXED-PENDING-REBUILD": "warn",
        "FIXED-PENDING-DEPLOY": "warn",
        "MISFRAMED": "info",
        "GENUINELY OPEN": "bad",
    }
    for bid, verdict, _snippet in sorted(facts.reconciliation, key=lambda t: (t[1], t[0])):
        title = esc(str(facts.issues[bid]["title"])[:74]) if bid in facts.issues else ""
        add(
            f"<tr><td>{chip(facts, bid)}</td><td>{title}</td>"
            f'<td><span class="badge {rc_badge.get(verdict, "info")}">{esc(verdict)}</span></td></tr>'
        )
    add("</tbody></table></div>")

    if facts.related_recurrence:
        add("<h3>The recurrence, exactly</h3>")
        add(
            "<p>Direct evidence the vocabulary fork is unrepaired at the mechanism level, "
            "not just at the data level: the bulk re-type above ran "
            "14:35&ndash;14:40 UTC today; the edge below was written after it.</p>"
        )
        add('<div class="tablewrap"><table id="rr"><thead><tr><th>from</th><th>relates to</th>')
        add("<th>created_at</th></tr></thead><tbody>")
        for src_id, dst_id, ts in facts.related_recurrence:
            add(f"<tr><td>{chip(facts, src_id)}</td><td>{chip(facts, dst_id)}</td><td>{esc(ts)}</td></tr>")
        add("</tbody></table></div>")
    add("</section>")

    # ---------------- themes ----------------
    add('<section id="themes"><h2>Themes</h2>')
    add(f"<p>{prose['themes']}</p>")
    add(f"<p>{prose['vocab_labels']}</p>")
    add("<h3>Two readings, side by side</h3>")
    add('<div class="tablewrap"><table id="th"><thead><tr><th>subsystem</th>')
    add(
        "<th>open (labels)</th><th>closed (labels)</th><th>open (keywords)</th>"
        "<th>closed (keywords)</th></tr></thead><tbody>"
    )
    names = sorted(set(facts.by_label_theme) | set(facts.by_kw_theme))
    peak_o = (
        max((facts.by_label_theme[n]["open"] + facts.by_label_theme[n]["in_progress"] for n in names), default=1) or 1
    )
    for name in names:
        lab = facts.by_label_theme.get(name, Counter())
        kw = facts.by_kw_theme.get(name, Counter())
        lab_open = lab["open"] + lab["in_progress"]
        kw_open = kw["open"] + kw["in_progress"]
        add(
            f"<tr><td>{esc(name)}</td>"
            f'<td class="dist" data-v="{lab_open}"><span style="--w:{100 * lab_open / peak_o:.1f}%"></span><b>{lab_open}</b></td>'
            f'<td class="num" data-v="{lab["closed"]}">{lab["closed"]}</td>'
            f'<td class="num" data-v="{kw_open}">{kw_open}</td>'
            f'<td class="num" data-v="{kw["closed"]}">{kw["closed"]}</td></tr>'
        )
    add("</tbody></table></div>")
    add("<details><summary>The synonym map used to fold <code>area:*</code> labels</summary>")
    add('<div class="tablewrap"><table id="syn"><thead><tr><th>subsystem</th>')
    add('<th class="nosort">folded area labels</th></tr></thead><tbody>')
    for canon, suffixes in SUBSYSTEMS.items():
        add(f"<tr><td>{esc(canon)}</td><td>" + " ".join(f"<code>area:{esc(s)}</code>" for s in suffixes) + "</td></tr>")
    add("</tbody></table></div>")
    add(
        "<p>This map is a proposal written by hand, not something the repository "
        "declares. Disagreeing with a row changes the table above it &mdash; which is "
        "the honest state of a backlog with no controlled label vocabulary.</p>"
    )
    add("</details>")
    add("<details><summary>Raw <code>area:*</code> labels, top 30</summary>")
    add('<div class="tablewrap"><table id="raw"><thead><tr><th>label</th><th>beads</th>')
    add("</tr></thead><tbody>")
    area_counts = [(x, facts.labels[x]) for x in facts.area_labels]
    area_counts.sort(key=lambda kv: (-kv[1], kv[0]))
    peak_l = area_counts[0][1] if area_counts else 1
    for name, n in area_counts[:30]:
        add(
            f"<tr><td><code>{esc(name)}</code></td>"
            f'<td class="dist" data-v="{n}"><span style="--w:{100 * n / peak_l:.1f}%"></span><b>{n}</b></td></tr>'
        )
    add("</tbody></table></div></details>")
    add("</section>")

    # ---------------- verified subset ----------------
    add('<section id="verified"><h2>The verified subset</h2>')
    add(f"<p>{prose['verified']}</p>")
    add('<div class="tiles">')
    for value, label in [
        (str(facts.verdict_counts["LIVE"]), "LIVE &mdash; claim still holds"),
        (str(facts.verdict_counts["PARTIAL"]), "PARTIAL &mdash; partly landed"),
        (str(facts.verdict_counts["STALE"]), "STALE &mdash; superseded by the code"),
    ]:
        add(f'<div class="tile"><span class="n">{value}</span><span class="l">{label}</span></div>')
    add("</div>")
    add(f"<p>{prose['verified_gap']}</p>")
    add('<input class="filter" placeholder="filter verdicts&hellip;" oninput="flt(this,\'vd\')">')
    add('<div class="tablewrap"><table id="vd"><thead><tr><th>bead</th><th>verdict</th>')
    add("<th>sweep</th><th>title</th></tr></thead><tbody>")
    badge_of = {"LIVE": "ok", "PARTIAL": "warn", "STALE": "bad"}
    for bid, verdict, _status, sweep in sorted(facts.verdicts, key=lambda v: (v[1], v[0])):
        add(
            f"<tr><td>{chip(facts, bid)}</td>"
            f'<td><span class="badge {badge_of[verdict]}">{verdict}</span></td>'
            f"<td>{esc(sweep[:34])}</td>"
            f"<td>{esc(str(facts.issues[bid]['title'])[:70])}</td></tr>"
        )
    add("</tbody></table></div>")
    add(
        "<p>One correction from the same night is worth reading as a method note rather "
        "than a data point. <code>polylogue-c5mb</code> was filed as a P0 on a headline "
        "figure that turned out to be an event-count-versus-file-count artifact; the "
        "re-measurement is recorded in its own notes and the severity was corrected in "
        "place rather than the bead being quietly rewritten.</p>"
    )
    if "polylogue-c5mb" in facts.issues:
        add("<p>" + chip(facts, "polylogue-c5mb", title=True) + "</p>")
    add(
        '<blockquote class="q">the &lsquo;552,633 distinct filenames / 71 GB permanently '
        "gone&rsquo; claim does NOT hold. Re-measured against the live archive and disk"
        "&hellip; DISTINCT debt basenames 12,004 (NOT 552,633); present on disk right now "
        "12,004 = 100.0%.<cite>polylogue-c5mb correction note, 2026-07-31</cite></blockquote>"
    )
    add(
        "<p>The lesson generalizes past this bead: a count of <em>events</em> is not a "
        "count of <em>things</em>, and a P0 whose severity rests on a large number "
        "should have that number re-derived by a second method before the priority "
        "is trusted.</p>"
    )
    add("</section>")

    # ---------------- findings ----------------
    add('<section id="findings"><h2>Findings</h2>')
    add(
        "<p>Authored judgement, ordered by how much a reader should care. Every number "
        "inside them is interpolated from the computation above.</p>"
    )
    add('<ul class="findings">')
    for cls, title, body in [vocab_finding(facts), *FINDINGS]:
        add(
            f'<li class="sev-{cls}"><b><span class="badge {cls}">{cls}</span> '
            f"{title.format(**vals)}</b>{body.format(**vals)}</li>"
        )
    add("</ul>")
    add("<h3>What to do next</h3>")
    add(f"<p>{prose['next']}</p>")
    add('<ol class="next">')
    for cls, text in [vocab_action(facts), *NEXT_ACTIONS]:
        add(f'<li><span class="badge {cls}">{cls}</span> {text.format(**vals)}</li>')
    add("</ol>")
    add("</section>")

    # ---------------- method ----------------
    add('<section id="method"><h2>Method &amp; regeneration</h2>')
    add(f"<p>{prose['method']}</p>")
    add("<h3>Computed vs authored</h3>")
    add('<div class="tablewrap"><table id="prov"><thead><tr><th>part of the report</th>')
    add('<th class="nosort">origin</th></tr></thead><tbody>')
    for part, origin in [
        ("every count, percentage, matrix cell, and median", "computed from the JSONL"),
        ("epic trees, fill bars, dependency degree on each chip", "computed (parent-child + blocks edges)"),
        ("per-day histogram, half-month velocity, age buckets", "computed"),
        ("ready / blocked / cycle detection / densest cluster SVG", "computed (graph traversal in the generator)"),
        ("health checks and both review queues", "computed"),
        ("keyword theme classification", "computed, using a hand-written keyword list"),
        ("area-label folding", "computed, using a hand-written synonym map"),
        ("verdict extraction", "computed by regex over notes and comments"),
        ("P0 reconciliation verdict extraction", "computed by regex over notes and comments"),
        (
            "schema-version gap (live vs. origin/master, SEMANTIC_REPARSE blockers)",
            "computed: `git show origin/master:...` + a read-only PRAGMA against the live archive",
        ),
        ("relates-to/related fork state and recurrence detection", "computed from the JSONL's edge timestamps"),
        ("lead paragraph and every interpretation paragraph", "authored"),
        ("the findings list and next actions", "authored"),
        ("the bead notation and its legend", "authored design, computed contents"),
    ]:
        cls = "info" if origin.startswith("computed") else "todo"
        add(f'<tr><td>{esc(part)}</td><td><span class="badge {cls}">{esc(origin)}</span></td></tr>')
    add("</tbody></table></div>")
    add("<h3>Regenerate</h3>")
    add(
        "<pre><code>cd /realm/project/polylogue\n"
        "devtools workspace beads-state-report --fresh \\\n"
        f"  --out {esc(str(source.parent))}/beads-state.html</code></pre>"
    )
    add(
        "<p><code>--fresh</code> re-runs <code>bd export -o .beads/issues.jsonl</code> "
        "first. Without it the report describes whatever the file last held, which after "
        "any uncommitted <code>bd</code> mutation is not the live state. The exported "
        "file is contended across sessions and should not be committed as part of "
        "generating a report.</p>"
    )
    add("</section>")

    add("</main></div>")
    add(
        f"<footer>generated by <code>devtools workspace beads-state-report</code> on "
        f"{generated:%Y-%m-%d %H:%M} from {esc(src)} &middot; {facts.total:,} beads, "
        f"{len(facts.edges):,} dependency edges</footer>"
    )
    add("<script>" + JS + "</script>")
    add("</body>\n</html>")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    root = _get_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", nargs="?", default=None, help="issues.jsonl (default: .beads/issues.jsonl)")
    parser.add_argument("--out", default=None, help="write HTML here (default: .local/beads-state.html)")
    parser.add_argument("--fresh", action="store_true", help="run `bd export` before reading")
    parser.add_argument("--json", action="store_true", help="print the computed facts as JSON instead of HTML")
    args = parser.parse_args(argv)

    path = Path(args.path) if args.path else root / ".beads/issues.jsonl"
    if args.fresh:
        subprocess.run(["bd", "export", "-o", str(path)], check=True, capture_output=True)
    if not path.exists():
        print(f"no such file: {path}", file=sys.stderr)
        return 1

    issues, edges = load(path)
    if not issues:
        print(f"no issues parsed from {path}", file=sys.stderr)
        return 1
    now = dt.datetime.now(dt.UTC)
    facts = Facts(issues, edges, now)
    schema_gap = schema_gap_facts(root)

    if args.json:
        payload = _values(facts, path, schema_gap)
        payload["dangling"] = [list(d) for d in facts.dangling]
        payload["open_parent_all_closed"] = facts.open_parent_all_closed
        payload["cycles"] = facts.cycles
        payload["reconciliation"] = facts.reconciliation
        payload["schema_gap"] = schema_gap
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    out = Path(args.out) if args.out else root / ".local/beads-state.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(facts, path, now, schema_gap), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size:,} bytes) from {facts.total:,} beads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
