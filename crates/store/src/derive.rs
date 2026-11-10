//! Derived rows: everything computed from the parsed rows, one declared
//! derivation each. A derivation names the derivations it reads, a version,
//! its class and the rows it owns for a session.
//!
//! A transaction that changes a session's rows marks the session due
//! (`derive_due`, kept by triggers on `session.changed`); a derivation whose
//! version moved marks every session due for it alone. The derive step
//! takes the due sessions, a bounded number per transaction, through every
//! derivation they need in dependency order, so each session's rows are
//! read once, and finds them through the marks alone: its cost follows the
//! sessions that changed, never the archive. It never moves
//! `session.changed`, which says when the session's own rows changed.
//!
//! A row-class derivation computes a value of one row alone (a message's
//! cost, a shell call's normalized command). The writer writes it with the
//! row by the same function, so it runs here only when its version moves.
//! A fold-class derivation (the session counts) is kept by schema triggers
//! in the transaction of each change to the rows it folds, including the
//! changes this step makes to duplicates and the active path, so it too
//! runs here only when its version moves or for a session it never ran on.
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
mod links;
mod repository;
mod session;
mod session_span;
mod title;
mod tool_calls;

/// Sessions derived per transaction, so a long derive never holds the
/// writer for minutes.
const SESSIONS_PER_TRANSACTION: usize = 200;

/// One derivation over a session's stored rows.
#[derive(Clone, Copy)]
struct Derivation {
    /// Its key in `derive_state`; stable across versions.
    pub name: &'static str,
    pub class: Class,
    pub products: &'static [polylogue_model::derived::Construct],
    /// Bumped when the rules change; every session is derived again.
    pub version: Version,
    /// Derivations whose rows this one reads; they run first.
    pub after: &'static [&'static str],
    /// Changes outside the session's own rows that make it due for this
    /// derivation alone; a change of its own rows is implicit.
    pub dirtied_by: &'static [Cause],
    /// Recomputes the session's rows: replaces whatever an earlier run
    /// made, from the session's current rows.
    pub run: fn(&Transaction<'_>, i64, &Inputs<'_>) -> Result<()>,
}

/// What a derivation's values depend on, which says when it runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// One row's value, written by the writer with the row: derived again
    /// only when its version moves.
    Row,
    /// A session value that schema triggers keep from every change to the
    /// rows it folds, in the transaction of the change: derived again only
    /// when its version moves.
    Fold,
    /// A value of the whole session: derived again whenever the session's
    /// rows changed.
    Session,
}

/// A change outside a session's own rows that a derivation reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cause {
    /// A repository lookup at or above the session's working directory
    /// changed (`derive_due.repo_lookup`).
    RepoLookupChanged,
}

/// What a derivation reads besides the session's rows, and where it records
/// the shown rows it changes.
pub(crate) struct Inputs<'a> {
    pub catalog: &'a Catalog,
    pub changes: &'a crate::change_summary::ChangeSummary,
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

/// Derives every due session; returns the number of (derivation, session)
/// pairs done.
pub fn run(
    index: &mut Connection,
    commits: Option<&crate::completion::CommitEvents>,
) -> Result<u64> {
    run_all(index, Catalog::builtin(), ALL, commits)
}

#[cfg(test)]
pub(crate) fn run_with_catalog(index: &mut Connection, catalog: &Catalog) -> Result<u64> {
    run_all(index, catalog, ALL, None)
}

fn run_all(
    index: &mut Connection,
    catalog: &Catalog,
    derivations: &[Derivation],
    commits: Option<&crate::completion::CommitEvents>,
) -> Result<u64> {
    let pass = Pass::start(index, catalog, derivations)?;
    let mut done = 0;
    while let Some(n) = pass.next(index, commits)? {
        done += n;
    }
    Ok(done)
}

/// One derive pass: the resolved declarations, and the sessions due taken
/// a transaction at a time, so a caller can serve other writes between.
pub(crate) struct Pass<'a> {
    catalog: &'a Catalog,
    derivations: &'a [Derivation],
    versions: Vec<String>,
}

impl Pass<'static> {
    /// The pass over the shipped derivations and catalog.
    pub(crate) fn builtin(index: &mut Connection) -> Result<Self> {
        Pass::start(index, Catalog::builtin(), ALL)
    }
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
        Ok(Self {
            catalog,
            derivations,
            versions,
        })
    }

    /// Derives the next due sessions in one transaction; `None` when none
    /// are due.
    pub(crate) fn next(
        &self,
        index: &mut Connection,
        commits: Option<&crate::completion::CommitEvents>,
    ) -> Result<Option<u64>> {
        let due: Vec<(i64, Due)> = index
            .prepare_cached(
                "SELECT session_pk, rows_changed, repo_lookup FROM derive_due
                 ORDER BY session_pk DESC LIMIT ?1",
            )?
            .query_map([SESSIONS_PER_TRANSACTION as i64], |r| {
                Ok((
                    r.get(0)?,
                    Due {
                        rows_changed: r.get(1)?,
                        repo_lookup: r.get(2)?,
                    },
                ))
            })?
            .collect::<rusqlite::Result<_>>()?;
        if due.is_empty() {
            return Ok(None);
        }
        // Immediate, like the writer's and the search thread's: a deferred
        // transaction fails to upgrade while the search thread writes.
        let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = crate::archive::bump_counter(&tx)?;
        let changes = crate::change_summary::ChangeSummary::start(&tx)?;
        let inputs = Inputs {
            catalog: self.catalog,
            changes: &changes,
        };
        let mut done = 0;
        for (pk, due) in due {
            let derived = self.derive_session(&tx, pk, due, &inputs)?;
            if derived > 0 {
                // Its derived values reach the change feed; its own rows
                // did not change.
                tx.prepare_cached("UPDATE session SET derived_changed = ?2 WHERE pk = ?1")?
                    .execute(params![pk, counter])?;
            }
            done += derived;
            tx.prepare_cached("DELETE FROM derive_due WHERE session_pk = ?1")?
                .execute([pk])?;
        }
        crate::items::refresh(&tx, &changes)?;
        tx.commit()?;
        if let Some(events) = commits {
            events.committed();
        }
        Ok(Some(done))
    }

    /// Runs, in order, the derivations the session needs: every session
    /// derivation when its rows changed, each derivation a marked outside
    /// cause dirties, and each derivation whose stored version is not its
    /// current one; returns how many ran.
    fn derive_session(
        &self,
        tx: &Transaction<'_>,
        pk: i64,
        due: Due,
        inputs: &Inputs<'_>,
    ) -> Result<u64> {
        let exists: bool = tx
            .prepare_cached("SELECT EXISTS(SELECT 1 FROM session WHERE pk = ?1)")?
            .query_row([pk], |r| r.get(0))?;
        if !exists {
            return Ok(0);
        }
        let mut stored = std::collections::HashMap::new();
        let mut statement =
            tx.prepare_cached("SELECT name, version FROM derive_state WHERE session_pk = ?1")?;
        for row in statement.query_map([pk], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
        })? {
            let (name, version) = row?;
            stored.insert(name, version);
        }
        let stale: Vec<usize> = (0..self.derivations.len())
            .filter(|&i| {
                let d = &self.derivations[i];
                (due.rows_changed && d.class == Class::Session)
                    || (due.repo_lookup && d.dirtied_by.contains(&Cause::RepoLookupChanged))
                    || stored.get(d.name) != Some(&self.versions[i])
            })
            .collect();
        let mut mark = tx.prepare_cached(
            "INSERT INTO derive_state(name, session_pk, version) VALUES (?1, ?2, ?3)
             ON CONFLICT(name, session_pk) DO UPDATE SET version = excluded.version",
        )?;
        for &i in &stale {
            (self.derivations[i].run)(tx, pk, inputs)?;
            mark.execute(params![self.derivations[i].name, pk, self.versions[i]])?;
        }
        Ok(stale.len() as u64)
    }
}

/// Why a session is due (`derive_due`).
#[derive(Clone, Copy)]
struct Due {
    rows_changed: bool,
    repo_lookup: bool,
}

/// Whether no derivation is behind: no session is due and the stored
/// versions are the current ones.
pub(crate) fn caught_up(conn: &Connection) -> Result<bool> {
    let due: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM derive_due d JOIN session s ON s.pk = d.session_pk WHERE s.stub = 0)",
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
