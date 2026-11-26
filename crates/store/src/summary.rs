//! Session summaries: every value of one session that is not a column of
//! one of its rows, each a declared, versioned summary computed from the
//! session's assembled rows (docs/measures.md, convergence).
//!
//! The code that assembles a session's rows marks it due in the same
//! transaction (`summary_due`, [`mark`]), with the earliest row key its
//! changes reached; a summary whose version moved marks every session. The
//! summary worker takes the due sessions, a bounded number per round, and
//! finds them through the marks alone: its cost follows the sessions that
//! changed, never the archive.
//!
//! A round runs in two waves. The first computes the summaries that mark
//! rows of the session (duplicates, the active path) and those with causes
//! outside its rows (lineage), each over what it needs; the second folds
//! the session's rows from the nearest stored state before the earliest
//! change, which the first wave's changed rows can move earlier
//! ([`fold::Run`]), computes the session's own columns from its records
//! and the folds, and writes its transcript lines from the last line the
//! earliest change can reach ([`lines`]). A reader that needs a due
//! session's lines asks the worker for it ([`ask`]); the worker computes
//! asked sessions before the next session of its round. Both waves read on
//! the worker's own connection; the writer
//! only applies the resulting [`Write`]s ([`apply`]), and only while the
//! session's [`Token`] is the one its inputs were read under, so a value
//! computed from rows that changed meanwhile is never stored and the
//! session stays due. Nothing here is computed in a write transaction, and
//! no value is kept by adding and subtracting what a change contributed.
//!
//! A row summary (a message's cost, a shell call's normalized command) is
//! written with its row by the same function, so it runs here only when its
//! version moves. A rebuild marks every session due from the start once it
//! is installed.

use crate::ConnExt as _;
use crate::{Result, StoreError};
use polylogue_model::Catalog;
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::BTreeMap;

mod active_path;
pub(crate) mod ask;
mod cost;
pub(crate) mod counts;
mod duplicates;
pub(crate) mod fold;
mod lineage;
pub(crate) mod lines;
pub(crate) mod links;
mod live;
mod repository;
mod session;
pub(crate) mod span;
mod title;
mod tool_calls;
pub(crate) mod values;

pub use fold::Key;
#[cfg(any(test, feature = "fixture"))]
pub use fold::rows_folded;

/// Sessions taken per round, so one of the writer's summary transactions
/// stays short.
const SESSIONS_PER_ROUND: usize = 200;

/// One declared summary.
#[derive(Clone, Copy)]
struct Summary {
    /// Its key in `summary_state`; stable across versions.
    name: &'static str,
    products: &'static [polylogue_model::derived::Construct],
    /// Bumped when the rules change; every session is computed again.
    version: Version,
    kind: Kind,
}

#[derive(Clone, Copy)]
enum Kind {
    /// A value of one row, written by the writer with the row: computed
    /// here only when its version moves.
    Row(fn(&Connection, i64, &Catalog, &mut Output) -> Result<()>),
    /// Computed from what it reads of the session, in the first wave, when
    /// its rows were reached or a cause it declares was marked.
    Marks {
        compute: fn(&Connection, i64, &mut Output) -> Result<()>,
        when: When,
    },
    /// A fold ([`fold`]): every fold runs in the second wave.
    Fold,
    /// Computed in the second wave from the folds and the session's
    /// records.
    AfterFolds,
    /// The transcript lines ([`lines`]), computed in the second wave from
    /// the last line the earliest change can reach.
    Lines,
}

/// What makes a first-wave summary run, besides its version moving.
#[derive(Clone, Copy)]
enum When {
    /// A change reached the session's rows.
    RowsReached,
    /// Its explicit ancestry changed (`summary_due.lineage`).
    Lineage,
}

/// Rule versions stay readable; catalog-derived rows also name the complete
/// catalog content.
#[derive(Clone, Copy)]
enum Version {
    Rules(i64),
    Catalog(i64),
}

impl Version {
    fn resolve(&self, catalog: &Catalog) -> Result<String> {
        Ok(match self {
            Self::Rules(n) => n.to_string(),
            Self::Catalog(n) => {
                let hash = blake3::hash(&serde_json::to_vec(catalog)?);
                serde_json::to_string(&("catalog", n, &catalog.version, hash.to_hex().as_str()))?
            }
        })
    }
}

const fn folded(
    name: &'static str,
    version: i64,
    products: &'static [polylogue_model::derived::Construct],
) -> Summary {
    Summary {
        name,
        products,
        version: Version::Rules(version),
        kind: Kind::Fold,
    }
}

/// Every summary, in the order its outputs are applied.
const ALL: &[Summary] = &[
    lineage::SUMMARY,
    duplicates::SUMMARY,
    active_path::SUMMARY,
    tool_calls::COMMAND_NORM,
    cost::SUMMARY,
    folded(
        <counts::Counts as fold::Fold>::NAME,
        <counts::Counts as fold::Fold>::VERSION,
        &[polylogue_model::derived::SESSION_COUNTS],
    ),
    folded(
        <session::Inputs as fold::Fold>::NAME,
        <session::Inputs as fold::Fold>::VERSION,
        &[],
    ),
    folded(
        <span::SessionSpan as fold::Fold>::NAME,
        <span::SessionSpan as fold::Fold>::VERSION,
        &[
            polylogue_model::derived::SESSION_SPAN,
            polylogue_model::derived::SESSION_TIME,
        ],
    ),
    folded(
        <values::Values as fold::Fold>::NAME,
        <values::Values as fold::Fold>::VERSION,
        &[],
    ),
    folded(
        <title::Title as fold::Fold>::NAME,
        <title::Title as fold::Fold>::VERSION,
        &[],
    ),
    folded(
        <tool_calls::Reasons as fold::Fold>::NAME,
        <tool_calls::Reasons as fold::Fold>::VERSION,
        &[],
    ),
    folded(
        <links::Links as fold::Fold>::NAME,
        <links::Links as fold::Fold>::VERSION,
        &[],
    ),
    folded(
        <live::Live as fold::Fold>::NAME,
        <live::Live as fold::Fold>::VERSION,
        &[],
    ),
    session::SUMMARY,
    repository::SUMMARY,
    Summary {
        name: lines::NAME,
        products: &[],
        version: Version::Rules(lines::VERSION),
        kind: Kind::Lines,
    },
];

/// What one session's summaries computed in one wave: the rows the writer
/// writes for them, and only those.
#[derive(Default)]
pub(crate) struct Output {
    pub lineage: Option<polylogue_model::read::SessionLineage>,
    /// Messages whose repeat mark changes, with the message each repeats.
    pub duplicates: Vec<(i64, Option<String>)>,
    /// Messages whose place on the active path changes.
    pub active_path: Vec<(i64, Option<i64>)>,
    pub command_norm: Vec<(i64, Option<String>)>,
    pub cost: Vec<(i64, Option<i64>)>,
    pub counts: Option<counts::Value>,
    pub inputs: Option<session::Spans>,
    pub span: Option<
        Box<(
            crate::measure::time::Span,
            Vec<crate::measure::time::PartyTime>,
        )>,
    >,
    pub values: Option<values::Value>,
    pub title: Option<Option<String>>,
    /// Calls whose outcome reason the folded rows decide.
    pub reasons: Vec<(i64, &'static str)>,
    pub links: Option<links::Found>,
    pub live: live::Emitted,
    pub session: Option<Box<session::Facts>>,
    pub repository: Option<(Option<String>, Option<String>)>,
    pub lines: Option<lines::Lines>,
}

/// Declared stored products, in their summary order.
pub fn products() -> Vec<polylogue_model::derived::Construct> {
    ALL.iter()
        .flat_map(|d| d.products.iter().copied())
        .collect()
}

pub(crate) fn current_versions() -> Result<BTreeMap<String, String>> {
    Ok(ALL
        .iter()
        .zip(builtin_versions()?)
        .map(|(d, v)| (d.name.to_owned(), v.clone()))
        .collect())
}

pub(crate) fn is_current(conn: &Connection, name: &str, pk: i64) -> Result<bool> {
    conn.scalar(
        "SELECT NOT EXISTS(SELECT 1 FROM summary_due WHERE session_pk = ?2)
            AND EXISTS(SELECT 1 FROM summary_state d WHERE d.name = ?1 AND d.session_pk = ?2
                       AND pl_summary_current(d.name, d.version))",
        params![name, pk],
    )
}

/// Marks `session_pk` due from `from`, the earliest row key a change
/// reached ([`Key::past_every_row`] when none did): a mark already there
/// keeps the earlier key.
pub(crate) fn mark(conn: &Connection, session_pk: i64, from: &Key) -> Result<()> {
    conn.prepare_cached(
        "INSERT INTO summary_due(session_pk, from_order) VALUES (?1, ?2)
         ON CONFLICT(session_pk) DO UPDATE SET from_order = min(from_order, excluded.from_order),
           marked = marked + 1",
    )?
    .execute(params![session_pk, from.0])?;
    Ok(())
}

/// Marks every session due from the start: a rebuilt index, whose rows
/// were assembled without their summaries.
pub(crate) fn mark_every_session(conn: &Connection) -> Result<()> {
    conn.execute(
        "INSERT INTO summary_due(session_pk, from_order) SELECT pk, X'' FROM session WHERE true
         ON CONFLICT(session_pk) DO UPDATE SET from_order = X'', marked = marked + 1",
        [],
    )?;
    Ok(())
}

/// Computes every due session's summaries on `index` alone, reading and
/// writing through the one connection; returns the number of sessions
/// done. The writer's summary worker runs the same pass with its own read
/// connection and the writer's [`apply`].
pub fn run(index: &mut Connection) -> Result<u64> {
    run_with_catalog(index, Catalog::builtin())
}

pub(crate) fn run_with_catalog(index: &mut Connection, catalog: &Catalog) -> Result<u64> {
    let pass = Pass::start(index, catalog)?;
    let index: &Connection = index;
    pass.run(
        index,
        &mut |write| apply(index, &write),
        &|| false,
        &Vec::new,
    )
}

/// One summary pass: the resolved declarations, and the sessions due taken
/// a round at a time.
pub(crate) struct Pass<'a> {
    catalog: &'a Catalog,
    versions: Vec<String>,
}

/// The shipped summaries' versions, resolved once: the catalog's version
/// hashes its whole content.
fn builtin_versions() -> Result<&'static [String]> {
    static VERSIONS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    if let Some(versions) = VERSIONS.get() {
        return Ok(versions);
    }
    let versions = versions(Catalog::builtin())?;
    Ok(VERSIONS.get_or_init(|| versions))
}

fn versions(catalog: &Catalog) -> Result<Vec<String>> {
    ALL.iter().map(|d| d.version.resolve(catalog)).collect()
}

/// Marks every session due for a shipped summary whose version moved since
/// the last pass: the writer's part of starting a pass.
pub(crate) fn mark_moved_versions(index: &mut Connection) -> Result<()> {
    Pass::start(index, Catalog::builtin()).map(drop)
}

impl Pass<'static> {
    /// The pass over the shipped summaries and catalog, whose moved
    /// versions [`mark_moved_versions`] marked.
    pub(crate) fn builtin() -> Result<Self> {
        Ok(Pass {
            catalog: Catalog::builtin(),
            versions: builtin_versions()?.to_vec(),
        })
    }
}

impl<'a> Pass<'a> {
    /// Resolves every version before writing any rows, and marks every
    /// session due for a summary whose version moved since the last pass.
    fn start(index: &mut Connection, catalog: &'a Catalog) -> Result<Self> {
        let versions = if std::ptr::eq(catalog, Catalog::builtin()) {
            builtin_versions()?.to_vec()
        } else {
            self::versions(catalog)?
        };
        let current: BTreeMap<&str, &str> = ALL
            .iter()
            .zip(&versions)
            .map(|(d, v)| (d.name, v.as_str()))
            .collect();
        let stored: Option<String> = crate::meta_get(index, "summary_versions")?;
        let stored: BTreeMap<String, String> = match stored {
            Some(json) => serde_json::from_str(&json)?,
            None => BTreeMap::new(),
        };
        let moved = current
            .iter()
            .any(|(name, version)| stored.get(*name).map(String::as_str) != Some(*version));
        if moved {
            let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
            tx.execute(
                "INSERT INTO summary_due(session_pk, from_order) SELECT pk, X'FF' FROM session WHERE true
                 ON CONFLICT(session_pk) DO NOTHING",
                [],
            )?;
            crate::meta_set(&tx, "summary_versions", serde_json::to_string(&current)?)?;
            tx.commit()?;
        }
        Ok(Self { catalog, versions })
    }

    /// Computes rounds of due sessions until none is left below the last
    /// round, reading through `read` and writing each wave through `write`;
    /// returns the sessions done. Sessions are taken in descending order
    /// once each, so one whose rows change while it is computed stays due
    /// for the next pass instead of being retried here.
    /// Sessions readers wait for (`asked`) are computed before the next
    /// session of a round.
    pub(crate) fn run(
        &self,
        read: &Connection,
        write: &mut dyn FnMut(Write) -> Result<Vec<bool>>,
        stopping: &dyn Fn() -> bool,
        asked: &dyn Fn() -> Vec<ask::Asked>,
    ) -> Result<u64> {
        let mut below = i64::MAX;
        let mut done = 0;
        let mut whole = Whole::default();
        while let Some(n) = self.round(read, write, &mut below, stopping, asked, &mut whole)? {
            done += n;
        }
        if whole.sessions > 0 {
            crate::log!(
                info,
                "summaries: {} sessions folded from their first row, {} rows read",
                whole.sessions,
                whole.rows
            );
        }
        Ok(done)
    }

    /// Computes the sessions readers wait for, answering each: `true` when
    /// it was due and its summaries were computed.
    pub(crate) fn answer(
        &self,
        read: &Connection,
        write: &mut dyn FnMut(Write) -> Result<Vec<bool>>,
        stopping: &dyn Fn() -> bool,
        asked: Vec<ask::Asked>,
    ) -> Result<()> {
        let mut due = Vec::with_capacity(asked.len());
        let mut sessions = Vec::new();
        for asked in &asked {
            let marked: bool = read
                .prepare_cached("SELECT EXISTS(SELECT 1 FROM summary_due WHERE session_pk = ?1)")?
                .query_row([asked.session_pk], |r| r.get(0))?;
            due.push(marked);
            if marked && !sessions.contains(&asked.session_pk) {
                sessions.push(asked.session_pk);
            }
        }
        let result = self.compute(
            read,
            write,
            sessions,
            stopping,
            &Vec::new,
            &mut Whole::default(),
        );
        for (asked, due) in asked.into_iter().zip(due) {
            asked.answer(match &result {
                Ok(_) => Ok(due),
                Err(StoreError::WriterGone) => Err(StoreError::WriterGone),
                Err(e) => Err(StoreError::Failed(polylogue_model::Failure::new(
                    "summary", e,
                ))),
            });
        }
        result.map(drop)
    }

    /// Computes the next due sessions below `below` through both waves;
    /// `None` when none are left.
    fn round(
        &self,
        read: &Connection,
        write: &mut dyn FnMut(Write) -> Result<Vec<bool>>,
        below: &mut i64,
        stopping: &dyn Fn() -> bool,
        asked: &dyn Fn() -> Vec<ask::Asked>,
        whole: &mut Whole,
    ) -> Result<Option<u64>> {
        let due: Vec<i64> = read
            .prepare_cached(
                "SELECT session_pk FROM summary_due WHERE session_pk < ?1
                 ORDER BY session_pk DESC LIMIT ?2",
            )?
            .query_map(params![*below, SESSIONS_PER_ROUND as i64], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let Some(&last) = due.last() else {
            return Ok(None);
        };
        *below = last;
        self.compute(read, write, due, stopping, asked, whole)
            .map(Some)
    }

    /// Computes `due` through both waves; returns the sessions applied.
    fn compute(
        &self,
        read: &Connection,
        write: &mut dyn FnMut(Write) -> Result<Vec<bool>>,
        due: Vec<i64>,
        stopping: &dyn Fn() -> bool,
        asked: &dyn Fn() -> Vec<ask::Asked>,
        whole: &mut Whole,
    ) -> Result<u64> {
        // The first wave: summaries that mark rows or have their own causes.
        let mut taken = Vec::new();
        let mut first = Vec::new();
        // Sessions gone since they were marked: their marks are cleared.
        let mut second = Vec::new();
        for pk in due {
            if stopping() {
                return Err(crate::StoreError::WriterGone);
            }
            self.answer_asked(read, write, stopping, asked)?;
            let snapshot = read.unchecked_transaction()?;
            let Some((token, due)) = Token::read(&snapshot, pk)? else {
                continue;
            };
            if token.changed.is_none() {
                second.push(SessionWrite {
                    pk,
                    token,
                    computed: Vec::new(),
                    output: Output::default(),
                    folded: None,
                });
                continue;
            }
            let doing = crate::activity::doing_on("summarizing", session_id(&snapshot, pk)?);
            let (stale, folds_moved, lines_moved) = self.stale(&snapshot, pk, &due)?;
            let mut output = Output::default();
            let mut computed = Vec::new();
            for &i in &stale {
                match ALL[i].kind {
                    Kind::Row(compute) => compute(&snapshot, pk, self.catalog, &mut output)?,
                    Kind::Marks { compute, .. } => compute(&snapshot, pk, &mut output)?,
                    Kind::Fold | Kind::AfterFolds | Kind::Lines => continue,
                }
                computed.push(i);
            }
            // Rows the first wave marks move where the folds start; a
            // message that becomes or stops being a repeat also changes
            // whether its parent is a branch point.
            let mut from = if folds_moved {
                Key::start()
            } else {
                due.from.clone()
            };
            let mut key = snapshot.prepare_cached(
                "SELECT m.session_order, m.id, p.session_order, p.id FROM message m
                 LEFT JOIN message p ON p.id = m.parent_id AND p.session_pk = m.session_pk
                 WHERE m.pk = ?1",
            )?;
            let marked = output
                .duplicates
                .iter()
                .map(|(pk, _)| (*pk, true))
                .chain(output.active_path.iter().map(|(pk, _)| (*pk, false)));
            for (pk, repeat) in marked {
                type Keys = (String, String, Option<String>, Option<String>);
                if let Some((order, id, parent_order, parent)) =
                    key.query_row([pk], |r| <Keys>::try_from(r)).optional()?
                {
                    from = from.min(Key::message(&order, &id));
                    if repeat && let (Some(order), Some(id)) = (parent_order, parent) {
                        from = from.min(Key::message(&order, &id));
                    }
                }
            }
            drop(key);
            drop(snapshot);
            drop(doing);
            taken.push((pk, token, stale, (due.from, from), lines_moved, true));
            first.push(SessionWrite {
                pk,
                token,
                computed,
                output,
                folded: None,
            });
        }
        if first.iter().any(|s| !s.computed.is_empty()) {
            let applied = write(Write {
                sessions: first,
                finish: false,
            })?;
            for (t, ok) in taken.iter_mut().zip(applied) {
                t.5 &= ok;
            }
        }
        // The second wave: the folds, then what is computed from them, and
        // the transcript lines.
        let start = Key::start();
        for (pk, token, stale, (marked, from), lines_moved, applied) in &taken {
            if !applied {
                continue;
            }
            if stopping() {
                return Err(crate::StoreError::WriterGone);
            }
            self.answer_asked(read, write, stopping, asked)?;
            let snapshot = read.unchecked_transaction()?;
            if Token::read(&snapshot, *pk)?.map(|(t, _)| t) != Some(*token) {
                // Its rows changed since its first wave: it stays due.
                continue;
            }
            let id = session_id(&snapshot, *pk)?;
            let doing = crate::activity::doing_on("folding session summaries", id.clone());
            let run = fold::Run { conn: &snapshot }.session(*pk, from)?;
            drop(doing);
            if run.after.is_start() && run.read > 0 {
                whole.sessions += 1;
                whole.rows += run.read;
            }
            let mut output = run.output;
            session::compute(&snapshot, *pk, &mut output)?;
            repository::compute(&snapshot, *pk, &mut output)?;
            let doing = crate::activity::doing_on("writing transcript lines", id.clone());
            let lines = lines::compute(&snapshot, *pk, if *lines_moved { &start } else { from })?;
            drop(doing);
            if run.read >= fold::EVERY || lines.read >= fold::EVERY {
                crate::log!(
                    info,
                    "summaries of {id}: due from {}, folds start at {}; folds {}, {} of {} rows \
                     read; lines from line {} of {}, {} units read",
                    marked.row(),
                    from.row(),
                    run.began,
                    run.read,
                    run.rows,
                    lines.start,
                    lines.before,
                    lines.read
                );
            }
            output.lines = Some(lines);
            let computed = stale
                .iter()
                .copied()
                .filter(|&i| matches!(ALL[i].kind, Kind::Fold | Kind::AfterFolds | Kind::Lines))
                .collect();
            drop(snapshot);
            second.push(SessionWrite {
                pk: *pk,
                token: *token,
                computed,
                output,
                folded: Some((run.after, run.kept)),
            });
        }
        if second.is_empty() {
            return Ok(0);
        }
        let applied = write(Write {
            sessions: second,
            finish: true,
        })?;
        Ok(applied.into_iter().filter(|ok| *ok).count() as u64)
    }

    /// Computes the sessions readers wait for now, before going on.
    fn answer_asked(
        &self,
        read: &Connection,
        write: &mut dyn FnMut(Write) -> Result<Vec<bool>>,
        stopping: &dyn Fn() -> bool,
        asked: &dyn Fn() -> Vec<ask::Asked>,
    ) -> Result<()> {
        let asked = asked();
        if asked.is_empty() {
            return Ok(());
        }
        self.answer(read, write, stopping, asked)
    }

    /// The summaries the session needs: every fold and what follows from
    /// them; those that mark rows when its rows were reached; each one a
    /// marked cause asks for; and each one whose stored version is not its
    /// current one. Also whether a fold's version moved, which starts every
    /// fold from the first row, and whether the lines' version moved, which
    /// writes the lines from the first row.
    fn stale(&self, read: &Connection, pk: i64, due: &Due) -> Result<(Vec<usize>, bool, bool)> {
        let mut stored = std::collections::HashMap::new();
        let mut statement =
            read.prepare_cached("SELECT name, version FROM summary_state WHERE session_pk = ?1")?;
        for row in statement.query_map([pk], |r| <(String, String)>::try_from(r))? {
            let (name, version) = row?;
            stored.insert(name, version);
        }
        let reached = due.from < Key::past_every_row();
        let moved = |i: usize| stored.get(ALL[i].name) != Some(&self.versions[i]);
        let folds_moved = (0..ALL.len()).any(|i| matches!(ALL[i].kind, Kind::Fold) && moved(i));
        let lines_moved = (0..ALL.len()).any(|i| matches!(ALL[i].kind, Kind::Lines) && moved(i));
        let stale = (0..ALL.len())
            .filter(|&i| {
                moved(i)
                    || match ALL[i].kind {
                        Kind::Row(_) => false,
                        Kind::Marks {
                            when: When::RowsReached,
                            ..
                        } => reached,
                        Kind::Marks {
                            when: When::Lineage,
                            ..
                        } => due.lineage,
                        Kind::Fold | Kind::AfterFolds | Kind::Lines => true,
                    }
            })
            .collect();
        Ok((stale, folds_moved, lines_moved))
    }
}

/// A session's ID, for what the worker says it is doing.
fn session_id(conn: &Connection, pk: i64) -> Result<String> {
    Ok(conn
        .prepare_cached("SELECT id FROM session WHERE pk = ?1")?
        .query_row([pk], |r| r.get(0))
        .optional()?
        .unwrap_or_else(|| format!("session {pk}")))
}

/// The sessions a pass folded from their first row, and the rows that
/// read: the costly part of a pass, which its log line reports.
#[derive(Default)]
struct Whole {
    sessions: u64,
    rows: u64,
}

/// Why a session is due (`summary_due`).
struct Due {
    from: Key,
    lineage: bool,
}

/// What a session's summaries were computed from: the change counter of
/// its own rows (`session.changed`; `None` once the session is gone) and the
/// count of markings of it (`summary_due.marked`), which moves with every
/// cause, including those outside its rows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Token {
    changed: Option<i64>,
    marked: i64,
}

impl Token {
    /// The session's token now, with why it is due; `None` when it is not
    /// due.
    fn read(conn: &Connection, pk: i64) -> Result<Option<(Self, Due)>> {
        let due = conn
            .prepare_cached(
                "SELECT marked, from_order, lineage FROM summary_due WHERE session_pk = ?1",
            )?
            .query_row([pk], |r| {
                Ok((
                    r.get(0)?,
                    Due {
                        from: Key(r.get(1)?),
                        lineage: r.get(2)?,
                    },
                ))
            })
            .optional()?;
        let Some((marked, due)) = due else {
            return Ok(None);
        };
        let changed = conn
            .prepare_cached("SELECT changed FROM session WHERE pk = ?1")?
            .query_row([pk], |r| r.get(0))
            .optional()?;
        Ok(Some((Self { changed, marked }, due)))
    }
}

/// One wave of one round: what the summary worker computed, for the writer
/// to apply ([`apply`]).
pub(crate) struct Write {
    sessions: Vec<SessionWrite>,
    /// The round's last wave: an applied session's mark is cleared.
    finish: bool,
}

struct SessionWrite {
    pk: i64,
    token: Token,
    /// The summaries computed, as places in [`ALL`].
    computed: Vec<usize>,
    output: Output,
    /// Where the folds started and the states they kept, when they ran.
    folded: Option<(Key, Vec<fold::Kept>)>,
}

impl Write {
    /// The sessions it holds.
    pub(crate) fn len(&self) -> usize {
        self.sessions.len()
    }
}

/// Applies one wave in one transaction of `index`: each session's outputs
/// and summary versions, only while its token is the one its inputs were
/// read under. Returns, per session, whether it was applied; one that was
/// not keeps its mark and its earlier values.
pub(crate) fn apply(index: &Connection, write: &Write) -> Result<Vec<bool>> {
    // Immediate, like the writer's and the search thread's: a deferred
    // transaction fails to upgrade while the search thread writes.
    let tx = Transaction::new_unchecked(index, TransactionBehavior::Immediate)?;
    let counter = crate::archive::bump_counter(&tx)?;
    let mut applied = Vec::with_capacity(write.sessions.len());
    let versions = builtin_versions()?;
    let mut state = tx.prepare_cached(
        "INSERT INTO summary_state(name, session_pk, version) VALUES (?1, ?2, ?3)
         ON CONFLICT(name, session_pk) DO UPDATE SET version = excluded.version",
    )?;
    for session in &write.sessions {
        if Token::read(&tx, session.pk)?.map(|(token, _)| token) != Some(session.token) {
            applied.push(false);
            continue;
        }
        apply_output(&tx, session.pk, &session.output, counter)?;
        if let Some((after, kept)) = &session.folded {
            live::apply(&tx, session.pk, after, &session.output.live, counter)?;
            if let Some(found) = &session.output.links {
                links::apply(&tx, session.pk, after, found)?;
            }
            fold::keep(&tx, session.pk, after, kept)?;
        }
        for &i in &session.computed {
            state.execute(params![ALL[i].name, session.pk, versions[i]])?;
        }
        if !session.computed.is_empty() {
            // Its summaries reach the change feed; its own rows did not
            // change.
            tx.prepare_cached("UPDATE session SET derived_changed = ?2 WHERE pk = ?1")?
                .execute(params![session.pk, counter])?;
        }
        if write.finish {
            tx.prepare_cached("DELETE FROM summary_due WHERE session_pk = ?1")?
                .execute([session.pk])?;
        }
        applied.push(true);
    }
    drop(state);
    tx.commit()?;
    Ok(applied)
}

/// Writes one session's outputs.
fn apply_output(tx: &Transaction<'_>, pk: i64, output: &Output, counter: i64) -> Result<()> {
    if let Some(value) = &output.lineage {
        lineage::apply(tx, pk, value)?;
    }
    duplicates::apply(tx, &output.duplicates)?;
    active_path::apply(tx, &output.active_path)?;
    tool_calls::apply_command_norm(tx, &output.command_norm)?;
    cost::apply(tx, &output.cost)?;
    if let Some(value) = &output.counts {
        counts::apply(tx, pk, value)?;
    }
    if let Some(span) = &output.span {
        crate::measure::time::write(tx, pk, &span.0, &span.1)?;
    }
    if let Some(value) = &output.values {
        values::apply(tx, pk, value, counter)?;
    }
    if let Some(title) = &output.title {
        title::apply(tx, pk, title.as_deref())?;
    }
    tool_calls::apply_reasons(tx, &output.reasons)?;
    if let Some(facts) = &output.session {
        session::apply(tx, pk, facts)?;
    }
    if let Some((repo, root)) = &output.repository {
        repository::apply(tx, pk, repo, root)?;
    }
    if let Some(lines) = &output.lines {
        lines::apply(tx, pk, lines)?;
    }
    Ok(())
}

/// Whether no summary is behind: no session is due and the stored versions
/// are the current ones.
pub(crate) fn caught_up(conn: &Connection) -> Result<bool> {
    let due: bool = conn.scalar(
        // CROSS JOIN keeps the due marks the outer loop: an empty
        // `summary_due` answers without visiting any session.
        "SELECT EXISTS(SELECT 1 FROM summary_due d CROSS JOIN session s ON s.pk = d.session_pk WHERE s.stub = 0)",
        [])?;
    if due {
        return Ok(false);
    }
    let stored: Option<String> = crate::meta_get(conn, "summary_versions")?;
    let current = serde_json::to_string(&current_versions()?)?;
    let sessions: bool = conn.exists("SELECT EXISTS(SELECT 1 FROM session)", [])?;
    Ok(!sessions || stored.as_deref() == Some(current.as_str()))
}

#[cfg(test)]
mod tests;
