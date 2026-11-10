//! How far each stored revision chain is parsed, kept per chain in
//! `chain_progress` and counted by origin, reading version and state in
//! `chain_progress_count`, so whether the index has caught up is a read of
//! those counts rather than a walk of every chain.
//!
//! A chain's row follows from its stored length and its input's origin
//! (`archive.db`) and its parse state (`index.db`). On the writer's index
//! connection, temporary triggers recompute a chain's row whenever its parse
//! state changes, and in primary mode, where the source rows live in
//! `index.db`, whenever its revisions change or an imported observation of
//! its input arrives, all in the transaction of the change. A standalone
//! archive is written on another connection; temporary triggers there
//! record the chains and inputs a transaction changed ([`recorded`]), and
//! the writer recomputes them in its next index transaction ([`refresh`])
//! and then forgets them ([`forget`]). A writer that stops without doing so
//! leaves `meta.chain_progress` other than `kept`, and the next writer
//! start recomputes every chain ([`open`]).
use crate::Result;
use rusqlite::{Connection, OptionalExtension, params};

/// The `meta` key whose value is `kept` while the rows reflect every
/// committed source row.
const KEPT: &str = "chain_progress";

/// The rows of the chains in `chains`, a subquery of `(input, chain,
/// length)` over `source_revision`, in the order of `chain_progress`'s
/// columns. The state is the part of the reading contract that does not
/// depend on the declared version: `no_checkpoint` (no parse state),
/// `current` (done at the stored length, or waiting at exactly the stored
/// length), `no_parser` (no parser yet), `partial` (a source prefix is
/// missing) and `behind` (anything else, including a refusal or failure
/// recorded at the stored length). The origin is the format an import
/// read the input as, else the input's own.
fn chain_rows(chains: &str) -> String {
    format!(
        "SELECT c.input, c.chain,
            COALESCE((SELECT json_extract(o.details_json, '$.import_format')
                FROM source_observation o
                WHERE o.input = c.input AND o.class = 'imported' AND json_valid(o.details_json)
                  AND json_type(o.details_json, '$.import_format') = 'text'
                ORDER BY (json_extract(o.details_json, '$.import_format') = 'export-file'),
                         o.rowid DESC
                LIMIT 1), i.format),
            COALESCE(p.reading_version, ''),
            CASE WHEN p.input_pk IS NULL THEN 'no_checkpoint'
                 WHEN p.status = 'no_parser_yet' THEN 'no_parser'
                 WHEN p.status = 'partial' THEN 'partial'
                 WHEN p.status = 'waiting' AND p.at_length = c.length THEN 'current'
                 WHEN p.status = 'done' AND p.parsed_upto >= c.length THEN 'current'
                 ELSE 'behind' END
        FROM ({chains}) c
        JOIN source_input i ON i.id = c.input
        LEFT JOIN input_ref ir ON ir.id = c.input
        LEFT JOIN parse_state p ON p.input_pk = ir.pk AND p.chain = c.chain"
    )
}

/// Every chain of every input, at its stored length.
const EVERY_CHAIN: &str =
    "SELECT input, chain, max(length) AS length FROM source_revision GROUP BY input, chain";

/// Every chain of input `input` (an SQL expression).
fn chains_of(input: &str) -> String {
    format!(
        "SELECT input, chain, max(length) AS length FROM source_revision
         WHERE input = {input} GROUP BY input, chain"
    )
}

/// Chain `chain` of input `input` (SQL expressions) at the length of its
/// latest revision, which is its longest: a chain only grows. Reading it
/// from the newest revision down keeps an append to a long history cheap.
fn one_chain(input: &str, chain: &str) -> String {
    format!(
        "SELECT input, chain, length FROM source_revision
         WHERE input = {input} AND chain = {chain} ORDER BY number DESC LIMIT 1"
    )
}

/// Statements that bring the rows of `chains` (from [`chains_of`] or
/// [`one_chain`]) of input `input` up to date, writing only rows whose
/// values differ, and remove rows in `scope` (an SQL condition on
/// `chain_progress`) whose chain the source no longer holds (a primary
/// scope reset). Table names are unqualified because the statements also
/// run inside triggers.
fn refresh_sql(chains: &str, scope: &str) -> [String; 2] {
    [
        format!(
            "INSERT INTO chain_progress(input, chain, origin, reading_version, state)
             SELECT * FROM ({rows}) WHERE true
             ON CONFLICT(input, chain) DO UPDATE SET origin = excluded.origin,
                 reading_version = excluded.reading_version, state = excluded.state
             WHERE chain_progress.origin <> excluded.origin
                OR chain_progress.reading_version <> excluded.reading_version
                OR chain_progress.state <> excluded.state",
            rows = chain_rows(chains)
        ),
        format!(
            "DELETE FROM chain_progress WHERE {scope}
                 AND NOT EXISTS (SELECT 1 FROM ({chains}) c WHERE c.chain = chain_progress.chain)"
        ),
    ]
}

/// The statements for one chain.
fn chain_sql(input: &str, chain: &str) -> [String; 2] {
    refresh_sql(
        &one_chain(input, chain),
        &format!("input = {input} AND chain = {chain}"),
    )
}

/// The statements for every chain of one input.
fn input_sql(input: &str) -> [String; 2] {
    refresh_sql(&chains_of(input), &format!("input = {input}"))
}

/// Installs the triggers on the writer's index connection (see the module
/// documentation) and, when the rows may be behind the source, recomputes
/// every chain. `standalone`: the source is written on another connection,
/// so the rows are marked not kept until [`close`].
pub(crate) fn open(index: &Connection, standalone: bool) -> Result<()> {
    install_triggers(index, !standalone)?;
    let kept: bool = index
        .query_row(
            "SELECT value = 'kept' FROM meta WHERE key = ?1",
            [KEPT],
            |row| row.get(0),
        )
        .optional()?
        .unwrap_or(false);
    if kept && !standalone {
        return Ok(());
    }
    // The mark must reach the disk before the archive changes: an index
    // commit lost with the machine must never leave an earlier `kept`.
    let synchronous: i64 = index.pragma_query_value(None, "synchronous", |r| r.get(0))?;
    index.pragma_update(None, "synchronous", synchronous.max(2))?;
    let tx = rusqlite::Transaction::new_unchecked(index, rusqlite::TransactionBehavior::Immediate)?;
    if !kept {
        tx.execute_batch(&format!(
            "DELETE FROM chain_progress;
             INSERT INTO chain_progress(input, chain, origin, reading_version, state) {};",
            chain_rows(EVERY_CHAIN)
        ))?;
    }
    tx.execute(
        "INSERT INTO meta(key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![KEPT, if standalone { "open" } else { "kept" }],
    )?;
    tx.commit()?;
    index.pragma_update(None, "synchronous", synchronous)?;
    Ok(())
}

/// Marks the rows as reflecting every committed source row: called by a
/// stopping standalone writer after it recomputed the recorded inputs.
pub(crate) fn close(index: &Connection) -> Result<()> {
    index.execute("UPDATE meta SET value = 'kept' WHERE key = ?1", [KEPT])?;
    Ok(())
}

/// Temporary triggers on `index` that recompute a chain's row when its
/// parse state changes, and with `source` also when its source rows in
/// `index.db` change. Test and benchmark setup that edits parse states on
/// a connection of its own installs them too.
pub(crate) fn install_triggers(index: &Connection, source: bool) -> Result<()> {
    let parse_input = |row: &str| format!("(SELECT id FROM input_ref WHERE pk = {row}.input_pk)");
    let mut triggers = vec![
        (
            "parse_added",
            "INSERT ON main.parse_state",
            chain_sql(&parse_input("NEW"), "NEW.chain"),
        ),
        (
            "parse_changed",
            "UPDATE OF reading_version, status, parsed_upto, at_length ON main.parse_state",
            chain_sql(&parse_input("NEW"), "NEW.chain"),
        ),
        (
            "parse_removed",
            "DELETE ON main.parse_state",
            chain_sql(&parse_input("OLD"), "OLD.chain"),
        ),
    ];
    if source {
        triggers.extend([
            (
                "revision_added",
                "INSERT ON main.source_revision",
                chain_sql("NEW.input", "NEW.chain"),
            ),
            (
                "revision_grown",
                "UPDATE OF length ON main.source_revision",
                chain_sql("NEW.input", "NEW.chain"),
            ),
            (
                "revision_removed",
                "DELETE ON main.source_revision",
                chain_sql("OLD.input", "OLD.chain"),
            ),
            (
                "import_added",
                "INSERT ON main.source_observation WHEN NEW.class = 'imported'",
                input_sql("NEW.input"),
            ),
            (
                "import_removed",
                "DELETE ON main.source_observation WHEN OLD.class = 'imported'",
                input_sql("OLD.input"),
            ),
        ]);
    }
    let mut sql = String::new();
    for (name, event, [upsert, stale]) in triggers {
        sql.push_str(&format!(
            "CREATE TEMP TRIGGER IF NOT EXISTS chain_progress_{name} AFTER {event}
             BEGIN {upsert}; {stale}; END;\n"
        ));
    }
    index.execute_batch(&sql)?;
    Ok(())
}

/// Temporary triggers on the standalone writer's `archive.db` connection
/// that record the chains whose revisions a transaction changed, and the
/// inputs an imported observation names; a rolled-back transaction records
/// nothing. The inserts use `ON CONFLICT DO NOTHING`: an `OR IGNORE` in a
/// trigger gives way to the conflict policy of the statement that fired it.
pub(crate) fn record_source_changes(archive: &Connection) -> Result<()> {
    archive.execute_batch(
        "CREATE TEMP TABLE IF NOT EXISTS chain_progress_chains (input TEXT NOT NULL, chain INTEGER NOT NULL,
             PRIMARY KEY (input, chain)) WITHOUT ROWID;
         CREATE TEMP TABLE IF NOT EXISTS chain_progress_inputs (input TEXT PRIMARY KEY) WITHOUT ROWID;
         CREATE TEMP TRIGGER IF NOT EXISTS chain_progress_revision_added AFTER INSERT ON main.revision
         BEGIN INSERT INTO chain_progress_chains VALUES (NEW.input, NEW.chain) ON CONFLICT DO NOTHING; END;
         CREATE TEMP TRIGGER IF NOT EXISTS chain_progress_revision_grown AFTER UPDATE OF length ON main.revision
         BEGIN INSERT INTO chain_progress_chains VALUES (NEW.input, NEW.chain) ON CONFLICT DO NOTHING; END;
         CREATE TEMP TRIGGER IF NOT EXISTS chain_progress_import_added AFTER INSERT ON main.observation
         WHEN NEW.class = 'imported'
         BEGIN INSERT INTO chain_progress_inputs VALUES (NEW.input) ON CONFLICT DO NOTHING; END;",
    )?;
    Ok(())
}

/// What [`record_source_changes`] recorded on `archive`.
#[derive(Default)]
pub(crate) struct SourceChanges {
    chains: Vec<(String, i64)>,
    inputs: Vec<String>,
}

/// The chains and inputs recorded on `archive`.
pub(crate) fn recorded(archive: &Connection) -> Result<SourceChanges> {
    let chains = archive
        .prepare_cached("SELECT input, chain FROM temp.chain_progress_chains")?
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<rusqlite::Result<_>>()?;
    let inputs = archive
        .prepare_cached("SELECT input FROM temp.chain_progress_inputs")?
        .query_map([], |row| row.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(SourceChanges { chains, inputs })
}

impl SourceChanges {
    pub(crate) fn is_empty(&self) -> bool {
        self.chains.is_empty() && self.inputs.is_empty()
    }
}

/// Recomputes the rows of `changes` in the index transaction `index`.
pub(crate) fn refresh(index: &Connection, changes: &SourceChanges) -> Result<()> {
    for (input, chain) in &changes.chains {
        for sql in chain_sql("?1", "?2") {
            index.prepare_cached(&sql)?.execute(params![input, chain])?;
        }
    }
    for input in &changes.inputs {
        for sql in input_sql("?1") {
            index.prepare_cached(&sql)?.execute([input])?;
        }
    }
    Ok(())
}

/// Forgets `changes` on `archive` once the index transaction that
/// recomputed them committed.
pub(crate) fn forget(archive: &Connection, changes: &SourceChanges) -> Result<()> {
    for (input, chain) in &changes.chains {
        archive
            .prepare_cached(
                "DELETE FROM temp.chain_progress_chains WHERE input = ?1 AND chain = ?2",
            )?
            .execute(params![input, chain])?;
    }
    for input in &changes.inputs {
        archive
            .prepare_cached("DELETE FROM temp.chain_progress_inputs WHERE input = ?1")?
            .execute([input])?;
    }
    Ok(())
}

/// The chains counted by origin, reading version (empty without a parse
/// state) and state, for [`crate::read::completeness`].
pub(crate) fn counts(conn: &Connection) -> Result<Vec<(String, String, State, u64)>> {
    let mut statement = conn.prepare_cached(
        "SELECT origin, reading_version, state, n FROM main.chain_progress_count",
    )?;
    let mut rows = statement.query([])?;
    let mut counts = Vec::new();
    while let Some(row) = rows.next()? {
        let state: String = row.get(2)?;
        let n: i64 = row.get(3)?;
        counts.push((
            row.get(0)?,
            row.get(1)?,
            State::from_stored(&state)?,
            u64::try_from(n).map_err(|_| {
                crate::StoreError::Invalid("a chain progress count is negative".into())
            })?,
        ));
    }
    Ok(counts)
}

/// A chain's parse state as `chain_progress` stores it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum State {
    NoCheckpoint,
    Current,
    NoParser,
    Partial,
    Behind,
}

impl State {
    fn from_stored(state: &str) -> Result<Self> {
        Ok(match state {
            "no_checkpoint" => Self::NoCheckpoint,
            "current" => Self::Current,
            "no_parser" => Self::NoParser,
            "partial" => Self::Partial,
            "behind" => Self::Behind,
            other => {
                return Err(crate::StoreError::Invalid(format!(
                    "chain progress state {other:?} is not known"
                )));
            }
        })
    }
}
