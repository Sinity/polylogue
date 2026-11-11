//! Derived rows: everything computed from the parsed rows, one declared
//! derivation each. A derivation names the derivations it reads, a version,
//! its class and the rows it owns for a session.
//!
//! A transaction that changes a session's rows marks the session due
//! (`derive_due`, kept by triggers on `session.changed`); a derivation whose
//! version moved marks every session due for it alone. The derive step
//! takes the due sessions, a bounded number per round, through every
//! derivation they need in dependency order, and finds them through the
//! marks alone: its cost follows the sessions that changed, never the
//! archive. It never moves `session.changed`, which says when the session's
//! own rows changed.
//!
//! The derive step runs beside the writer, not on it. A derivation's
//! [`Derivation::compute`] reads what it needs of one session and computes
//! its values on the derive thread's own read connection; the writer only
//! applies the resulting [`Output`]s ([`apply`]), so a slow derivation
//! never holds parse batches or user writes. Derivations run in waves: a
//! derivation's wave is one past the latest wave of the derivations it
//! reads, so each wave reads what the earlier waves committed. Every output
//! carries the session's [`Token`] as its inputs were read; the writer
//! applies it only while the token is unchanged, so a value computed from
//! rows that changed meanwhile is never stored, and the session stays due.
//!
//! A row-class derivation computes a value of one row alone (a message's
//! cost, a shell call's normalized command). The writer writes it with the
//! row by the same function, so it runs here only when its version moves.
//! A fold-class derivation is kept in the transaction of each change to the
//! rows it folds: the session counts by schema triggers (including the
//! changes this step makes to duplicates and the active path), the links by
//! the writer from the transaction's change summary. It runs here only when
//! its version moves, for a session it never ran on, or for a cause it
//! declares (the links of a session whose stored messages were rewritten).
//! A first build and an incremental one derive the same rows from the same
//! records. Replacing a reading recomputes the derived rows it changes.

use crate::{Result, StoreError};
use polylogue_model::Catalog;
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use std::collections::BTreeMap;

mod active_path;
mod cost;
pub(crate) mod counts;
mod duplicates;
pub(crate) mod links;
mod repository;
mod session;
mod session_span;
mod title;
mod tool_calls;

/// Sessions taken per round, so one of the writer's derive transactions
/// stays short.
const SESSIONS_PER_ROUND: usize = 200;

/// One derivation over a session's stored rows.
#[derive(Clone, Copy)]
struct Derivation {
    /// Its key in `derive_state`; stable across versions.
    pub name: &'static str,
    pub class: Class,
    pub products: &'static [polylogue_model::derived::Construct],
    /// Bumped when the rules change; every session is derived again.
    pub version: Version,
    /// Derivations whose rows this one reads; they run in an earlier wave.
    pub after: &'static [&'static str],
    /// Changes outside the session's own rows that make it due for this
    /// derivation alone; a change of its own rows is implicit.
    pub dirtied_by: &'static [Cause],
    /// Reads the session's rows and computes what the derivation stores
    /// for it, replacing whatever an earlier run made. It runs on a read
    /// connection, never in the writer's transaction.
    pub compute: fn(&Connection, i64, &Inputs<'_>) -> Result<Output>,
}

/// What a derivation's values depend on, which says when it runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// One row's value, written by the writer with the row: derived again
    /// only when its version moves.
    Row,
    /// A session value kept from every change to the rows it folds, in the
    /// transaction of the change: derived again only when its version moves
    /// or a cause it declares is marked.
    Fold,
    /// A value of the whole session: derived again whenever the session's
    /// rows changed.
    Session,
}

/// A change that a derivation reads besides its session's rows being
/// written.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cause {
    /// A repository lookup at or above the session's working directory
    /// changed (`derive_due.repo_lookup`).
    RepoLookupChanged,
    /// A stored message of the session was rewritten, or the session changed
    /// in a way the writer does not describe message by message, so a fold
    /// cannot add the change's share (`derive_due.rewritten`).
    MessagesRewritten,
}

/// What a derivation reads besides the session's rows.
#[derive(Clone, Copy)]
pub(crate) struct Inputs<'a> {
    pub catalog: &'a Catalog,
    /// The derivation's version moved, or it never ran on the session:
    /// nothing it stored for the session may be resumed from.
    pub from_start: bool,
}

/// What one derivation computed for one session: the rows the writer
/// writes for it, and only those.
pub(crate) enum Output {
    Duplicates(Vec<(i64, Option<String>)>),
    Session(Box<session::Facts>),
    CommandNorm(Vec<(i64, Option<String>)>),
    ToolCalls(Vec<(i64, &'static str)>),
    ActivePath(Vec<(i64, Option<i64>)>),
    Repository {
        repo: Option<String>,
        root: Option<String>,
    },
    Title(Option<String>),
    Links(links::Links),
    Cost(Vec<(i64, Option<i64>)>),
    Counts(counts::Counts),
    Span(Box<crate::measure::time::Derived>),
    /// Session columns set to values, for synthetic derivations in tests.
    #[cfg(test)]
    Columns(Vec<(&'static str, Option<String>)>),
}

impl Output {
    /// Writes the output's rows for `session_pk`, recording the messages it
    /// changes in `changes`.
    fn apply(
        &self,
        tx: &Transaction<'_>,
        session_pk: i64,
        changes: &crate::change_summary::ChangeSummary,
    ) -> Result<()> {
        match self {
            Self::Duplicates(marks) => duplicates::apply(tx, marks, changes),
            Self::Session(facts) => session::apply(tx, session_pk, facts),
            Self::CommandNorm(calls) => tool_calls::apply_command_norm(tx, calls),
            Self::ToolCalls(calls) => tool_calls::apply_reasons(tx, calls),
            Self::ActivePath(moved) => active_path::apply(tx, moved, changes),
            Self::Repository { repo, root } => repository::apply(tx, session_pk, repo, root),
            Self::Title(title) => title::apply(tx, session_pk, title.as_deref()),
            Self::Links(links) => links::replace(tx, session_pk, links),
            Self::Cost(costs) => cost::apply(tx, costs),
            Self::Counts(counts) => counts::apply(tx, session_pk, counts),
            Self::Span(derived) => crate::measure::time::write_derived(tx, session_pk, derived),
            #[cfg(test)]
            Self::Columns(columns) => {
                for (column, value) in columns {
                    tx.execute(
                        &format!("UPDATE session SET {column} = ?2 WHERE pk = ?1"),
                        params![session_pk, value],
                    )?;
                }
                Ok(())
            }
        }
    }
}

/// Rule versions stay readable; catalog-derived rows also name the complete
/// catalog content. The full digest avoids truncating catalog identity to an integer.
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

/// Every derivation, in an order where each comes after what it reads.
const ALL: &[Derivation] = &[
    duplicates::DERIVATION,
    session::DERIVATION,
    tool_calls::COMMAND_NORM,
    tool_calls::DERIVATION,
    active_path::DERIVATION,
    repository::DERIVATION,
    title::DERIVATION,
    links::DERIVATION,
    cost::DERIVATION,
    counts::DERIVATION,
    session_span::DERIVATION,
];

/// Declared stored products, in their derivation order.
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
    Ok(conn.query_row(
        "SELECT NOT EXISTS(SELECT 1 FROM derive_due WHERE session_pk = ?2)
            AND EXISTS(SELECT 1 FROM derive_state d WHERE d.name = ?1 AND d.session_pk = ?2
                       AND pl_derived_current(d.name, d.version))",
        params![name, pk],
        |r| r.get(0),
    )?)
}

/// Derives every due session on `index` alone, reading and writing through
/// the one connection; returns the number of (derivation, session) pairs
/// done. The writer's derive thread runs the same pass with its own read
/// connection and the writer's [`apply`].
pub fn run(index: &mut Connection) -> Result<u64> {
    run_all(index, Catalog::builtin(), ALL)
}

#[cfg(test)]
pub(crate) fn run_with_catalog(index: &mut Connection, catalog: &Catalog) -> Result<u64> {
    run_all(index, catalog, ALL)
}

fn run_all(index: &mut Connection, catalog: &Catalog, derivations: &[Derivation]) -> Result<u64> {
    let pass = Pass::start(index, catalog, derivations)?;
    let index: &Connection = index;
    pass.run(index, &mut |write| apply(index, &write), &|| false)
}

/// One derive pass: the resolved declarations, and the sessions due taken
/// a round at a time.
pub(crate) struct Pass<'a> {
    catalog: &'a Catalog,
    derivations: &'a [Derivation],
    versions: Vec<String>,
    /// Each derivation's wave.
    waves: Vec<usize>,
}

/// The shipped derivations' versions, resolved once: the catalog's version
/// hashes its whole content.
fn builtin_versions() -> Result<&'static [String]> {
    static VERSIONS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    if let Some(versions) = VERSIONS.get() {
        return Ok(versions);
    }
    let versions = versions(ALL, Catalog::builtin())?;
    Ok(VERSIONS.get_or_init(|| versions))
}

/// Marks every session due for a shipped derivation whose version moved
/// since the last pass: the writer's part of starting a pass.
pub(crate) fn mark_moved_versions(index: &mut Connection) -> Result<()> {
    Pass::start(index, Catalog::builtin(), ALL).map(drop)
}

impl Pass<'static> {
    /// The pass over the shipped derivations and catalog, whose moved
    /// versions [`mark_moved_versions`] marked.
    pub(crate) fn builtin() -> Result<Self> {
        Pass::new(Catalog::builtin(), ALL, builtin_versions()?.to_vec())
    }
}

impl<'a> Pass<'a> {
    /// Resolves the whole declaration before writing any rows, and marks
    /// every session due for a derivation whose version moved since the last
    /// pass (a dependency's rule change moves its consumers' versions too).
    fn start(
        index: &mut Connection,
        catalog: &'a Catalog,
        derivations: &'a [Derivation],
    ) -> Result<Self> {
        let versions =
            if std::ptr::eq(derivations, ALL) && std::ptr::eq(catalog, Catalog::builtin()) {
                builtin_versions()?.to_vec()
            } else {
                versions(derivations, catalog)?
            };
        let current: BTreeMap<&str, &str> = derivations
            .iter()
            .zip(&versions)
            .map(|(d, v)| (d.name, v.as_str()))
            .collect();
        let stored: Option<String> = index
            .query_row(
                "SELECT value FROM meta WHERE key = 'derived_versions'",
                [],
                |r| r.get(0),
            )
            .optional()?;
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
                "INSERT INTO derive_due(session_pk, rows_changed) SELECT pk, 0 FROM session WHERE true
                 ON CONFLICT(session_pk) DO NOTHING",
                [],
            )?;
            tx.execute(
                "INSERT INTO meta(key, value) VALUES ('derived_versions', ?1)
                 ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [serde_json::to_string(&current)?],
            )?;
            tx.commit()?;
        }
        Self::new(catalog, derivations, versions)
    }

    fn new(
        catalog: &'a Catalog,
        derivations: &'a [Derivation],
        versions: Vec<String>,
    ) -> Result<Self> {
        let mut waves: Vec<usize> = Vec::with_capacity(derivations.len());
        for d in derivations {
            let wave = d
                .after
                .iter()
                .map(|after| {
                    derivations
                        .iter()
                        .position(|o| o.name == *after)
                        .and_then(|j| waves.get(j))
                        .map(|w| w + 1)
                        .ok_or_else(|| {
                            StoreError::Invalid(format!(
                                "derivation {} must follow {after}",
                                d.name
                            ))
                        })
                })
                .try_fold(0, |a, w| w.map(|w| a.max(w)))?;
            waves.push(wave);
        }
        Ok(Self {
            catalog,
            derivations,
            versions,
            waves,
        })
    }

    /// Derives rounds of due sessions until none is left below the last
    /// round, reading through `read` and writing each wave through `write`;
    /// returns the (derivation, session) pairs done. Sessions are taken in
    /// descending order once each, so one whose rows change while it is
    /// derived stays due for the next pass instead of being retried here.
    pub(crate) fn run(
        &self,
        read: &Connection,
        write: &mut dyn FnMut(Write) -> Result<Vec<bool>>,
        stopping: &dyn Fn() -> bool,
    ) -> Result<u64> {
        let mut below = i64::MAX;
        let mut done = 0;
        while let Some(n) = self.round(read, write, &mut below, stopping)? {
            done += n;
        }
        Ok(done)
    }

    /// Derives the next due sessions below `below` through every wave;
    /// `None` when none are left.
    fn round(
        &self,
        read: &Connection,
        write: &mut dyn FnMut(Write) -> Result<Vec<bool>>,
        below: &mut i64,
        stopping: &dyn Fn() -> bool,
    ) -> Result<Option<u64>> {
        let due: Vec<i64> = read
            .prepare_cached(
                "SELECT session_pk FROM derive_due WHERE session_pk < ?1
                 ORDER BY session_pk DESC LIMIT ?2",
            )?
            .query_map(params![*below, SESSIONS_PER_ROUND as i64], |r| r.get(0))?
            .collect::<rusqlite::Result<_>>()?;
        let Some(&last) = due.last() else {
            return Ok(None);
        };
        *below = last;
        let mut taken: Vec<Taken> = due
            .into_iter()
            .map(|pk| Taken {
                pk,
                token: None,
                stale: Vec::new(),
                admitted: true,
            })
            .collect();
        let last_wave = self.waves.iter().copied().max().unwrap_or(0);
        for wave in 0..=last_wave {
            let mut sessions = Vec::new();
            let mut at = Vec::new();
            for (i, t) in taken.iter_mut().enumerate() {
                if !t.admitted {
                    continue;
                }
                if stopping() {
                    return Err(StoreError::WriterGone);
                }
                let snapshot = read.unchecked_transaction()?;
                let marked = Token::read(&snapshot, t.pk)?;
                let token = marked.map(|(token, _)| token);
                if wave == 0 {
                    let Some((token, due)) = marked else {
                        // Its mark is gone already.
                        t.admitted = false;
                        continue;
                    };
                    t.token = Some(token);
                    t.stale = self.stale(&snapshot, t.pk, token, due)?;
                } else if token != t.token {
                    // Its rows changed since its first wave: it stays due.
                    t.admitted = false;
                    continue;
                }
                let token = t.token.expect("a taken session's token");
                let mut outputs = Vec::new();
                if token.changed.is_some() {
                    for &(d, from_start) in t.stale.iter().filter(|(d, _)| self.waves[*d] == wave) {
                        let derivation = &self.derivations[d];
                        let inputs = Inputs {
                            catalog: self.catalog,
                            from_start,
                        };
                        outputs.push(Derived {
                            name: derivation.name,
                            version: self.versions[d].clone(),
                            output: (derivation.compute)(&snapshot, t.pk, &inputs)?,
                        });
                    }
                }
                drop(snapshot);
                sessions.push(SessionWrite {
                    pk: t.pk,
                    token,
                    outputs,
                });
                at.push(i);
            }
            let finish = wave == last_wave;
            if !finish && sessions.iter().all(|s| s.outputs.is_empty()) {
                continue;
            }
            if sessions.is_empty() {
                break;
            }
            let admitted = write(Write { sessions, finish })?;
            for (i, ok) in at.into_iter().zip(admitted) {
                taken[i].admitted &= ok;
            }
        }
        Ok(Some(
            taken
                .iter()
                .filter(|t| t.admitted && t.token.is_some_and(|t| t.changed.is_some()))
                .map(|t| t.stale.len() as u64)
                .sum(),
        ))
    }

    /// The derivations the session needs, each with whether it starts from
    /// nothing: every session derivation when its rows changed, each
    /// derivation a marked cause dirties, and each derivation whose stored
    /// version is not its current one.
    fn stale(
        &self,
        read: &Connection,
        pk: i64,
        token: Token,
        due: Due,
    ) -> Result<Vec<(usize, bool)>> {
        if token.changed.is_none() {
            return Ok(Vec::new());
        }
        let mut stored = std::collections::HashMap::new();
        let mut statement =
            read.prepare_cached("SELECT name, version FROM derive_state WHERE session_pk = ?1")?;
        for row in statement.query_map([pk], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })? {
            let (name, version) = row?;
            stored.insert(name, version);
        }
        Ok((0..self.derivations.len())
            .filter_map(|i| {
                let d = &self.derivations[i];
                let moved = stored.get(d.name) != Some(&self.versions[i]);
                let stale = moved
                    || (due.rows_changed && d.class == Class::Session)
                    || (due.repo_lookup && d.dirtied_by.contains(&Cause::RepoLookupChanged))
                    || (due.rewritten && d.dirtied_by.contains(&Cause::MessagesRewritten));
                stale.then_some((i, moved))
            })
            .collect())
    }
}

/// A due session in a round.
struct Taken {
    pk: i64,
    /// As read in its first wave.
    token: Option<Token>,
    /// The derivations it needs, with whether each starts from nothing.
    stale: Vec<(usize, bool)>,
    /// Every wave so far was applied.
    admitted: bool,
}

/// Why a session is due (`derive_due`).
#[derive(Clone, Copy)]
struct Due {
    rows_changed: bool,
    repo_lookup: bool,
    rewritten: bool,
}

/// What a session's derived values were computed from: the change counter
/// of its own rows (`session.changed`; `None` once the session is gone) and
/// the count of markings of it (`derive_due.marked`), which moves with every
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
                "SELECT marked, rows_changed, repo_lookup, rewritten FROM derive_due
                 WHERE session_pk = ?1",
            )?
            .query_row([pk], |r| {
                Ok((
                    r.get(0)?,
                    Due {
                        rows_changed: r.get(1)?,
                        repo_lookup: r.get(2)?,
                        rewritten: r.get(3)?,
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

/// One wave of one round: what the derive thread computed, for the writer
/// to apply ([`apply`]).
pub(crate) struct Write {
    sessions: Vec<SessionWrite>,
    /// The round's last wave: an applied session's mark is cleared.
    finish: bool,
}

struct SessionWrite {
    pk: i64,
    token: Token,
    outputs: Vec<Derived>,
}

struct Derived {
    name: &'static str,
    version: String,
    output: Output,
}

/// Applies one wave in one transaction of `index`: each session's outputs
/// and derivation versions, only while its token is the one its inputs
/// were read under. Returns, per session, whether it was applied; one that
/// was not keeps its mark and its earlier values.
pub(crate) fn apply(index: &Connection, write: &Write) -> Result<Vec<bool>> {
    // Immediate, like the writer's and the search thread's: a deferred
    // transaction fails to upgrade while the search thread writes.
    let tx = Transaction::new_unchecked(index, TransactionBehavior::Immediate)?;
    let counter = crate::archive::bump_counter(&tx)?;
    let changes = crate::change_summary::ChangeSummary::start(&tx)?;
    let mut applied = Vec::with_capacity(write.sessions.len());
    let mut state = tx.prepare_cached(
        "INSERT INTO derive_state(name, session_pk, version) VALUES (?1, ?2, ?3)
         ON CONFLICT(name, session_pk) DO UPDATE SET version = excluded.version",
    )?;
    for session in &write.sessions {
        if Token::read(&tx, session.pk)?.map(|(token, _)| token) != Some(session.token) {
            applied.push(false);
            continue;
        }
        for derived in &session.outputs {
            derived.output.apply(&tx, session.pk, &changes)?;
            state.execute(params![derived.name, session.pk, derived.version])?;
        }
        if !session.outputs.is_empty() {
            // Its derived values reach the change feed; its own rows
            // did not change.
            tx.prepare_cached("UPDATE session SET derived_changed = ?2 WHERE pk = ?1")?
                .execute(params![session.pk, counter])?;
        }
        if write.finish {
            tx.prepare_cached("DELETE FROM derive_due WHERE session_pk = ?1")?
                .execute([session.pk])?;
        }
        applied.push(true);
    }
    drop(state);
    let sessions = changes.sessions(&tx)?;
    // A span resumed from a settled message must not pass a message whose
    // duplicate mark or path place changed.
    crate::measure::time::forget_settled(&tx, &changes, &sessions)?;
    crate::items::refresh(&tx, &changes)?;
    crate::session_values::refresh(&tx, &changes)?;
    tx.commit()?;
    Ok(applied)
}

/// Whether no derivation is behind: no session is due and the stored
/// versions are the current ones.
pub(crate) fn caught_up(conn: &Connection) -> Result<bool> {
    let due: bool = conn.query_row(
        // CROSS JOIN keeps the due marks the outer loop: an empty
        // `derive_due` answers without visiting any session.
        "SELECT EXISTS(SELECT 1 FROM derive_due d CROSS JOIN session s ON s.pk = d.session_pk WHERE s.stub = 0)",
        [],
        |r| r.get(0),
    )?;
    if due {
        return Ok(false);
    }
    let stored: Option<String> = conn
        .query_row(
            "SELECT value FROM meta WHERE key = 'derived_versions'",
            [],
            |r| r.get(0),
        )
        .optional()?;
    let current = serde_json::to_string(&current_versions()?)?;
    let sessions: bool =
        conn.query_row("SELECT EXISTS(SELECT 1 FROM session)", [], |r| r.get(0))?;
    Ok(!sessions || stored.as_deref() == Some(current.as_str()))
}

fn versions(derivations: &[Derivation], catalog: &Catalog) -> Result<Vec<String>> {
    let mut positions = BTreeMap::new();
    for (i, d) in derivations.iter().enumerate() {
        if positions.insert(d.name, i).is_some() {
            return Err(StoreError::Invalid(format!(
                "duplicate derivation {}",
                d.name
            )));
        }
    }
    let mut marks = vec![0; derivations.len()];
    for i in 0..derivations.len() {
        check_dependencies(i, derivations, &positions, &mut marks)?;
    }
    let own = derivations
        .iter()
        .map(|d| d.version.resolve(catalog))
        .collect::<Result<Vec<_>>>()?;
    let mut ancestors: Vec<BTreeMap<&str, &str>> = Vec::new();
    let mut versions = Vec::new();
    for (i, d) in derivations.iter().enumerate() {
        let mut dependencies = BTreeMap::new();
        for after in d.after {
            let j = positions[after];
            if j >= i {
                return Err(StoreError::Invalid(format!(
                    "derivation {} must follow {after}",
                    d.name
                )));
            }
            dependencies.insert(*after, own[j].as_str());
            dependencies.extend(ancestors[j].iter().map(|(k, v)| (*k, *v)));
        }
        // A flat transitive map avoids repeatedly embedding and escaping
        // whole dependency trees. Sorted keys make declaration order irrelevant.
        let version = if dependencies.is_empty() {
            own[i].clone()
        } else {
            serde_json::to_string(&(&own[i], &dependencies))?
        };
        ancestors.push(dependencies);
        versions.push(version);
    }
    Ok(versions)
}

fn check_dependencies(
    i: usize,
    derivations: &[Derivation],
    positions: &BTreeMap<&str, usize>,
    marks: &mut [u8],
) -> Result<()> {
    let d = &derivations[i];
    match marks[i] {
        1 => {
            return Err(StoreError::Invalid(format!(
                "derivation dependency cycle at {}",
                d.name
            )));
        }
        2 => return Ok(()),
        _ => {}
    }
    marks[i] = 1;
    for after in d.after {
        let j = *positions.get(after).ok_or_else(|| {
            StoreError::Invalid(format!("derivation {} depends on missing {after}", d.name))
        })?;
        check_dependencies(j, derivations, positions, marks)?;
    }
    marks[i] = 2;
    Ok(())
}

#[cfg(test)]
mod tests;
