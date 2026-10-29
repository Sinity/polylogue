//! Derived rows: everything computed from the parsed rows, one declared
//! derivation each. A derivation names the
//! derivations it reads, a version, and the rows it owns for a session; the
//! derive step recomputes a session's rows whenever the session changed
//! since they were made or the derivation's version moved, in dependency
//! order. It takes one session at a time through every derivation it needs,
//! so each session's rows are read from disk once, not once per derivation.
//! The writer's ingest path stores records and no derived row, except a
//! value that depends on its own row alone (a message's cost, a shell
//! call's normalized command), which it writes with the row by the same
//! function so that a first build's derive does not rewrite those rows.
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
mod repository;
mod session;
mod session_span;
mod title;
mod tool_calls;
mod work;

pub use work::verification_title;

/// Sessions derived per transaction, so a long derive never holds the
/// writer for minutes.
const SESSIONS_PER_TRANSACTION: usize = 200;

/// One derivation over a session's stored rows.
#[derive(Clone, Copy)]
struct Derivation {
    /// Its key in `derive_state`; stable across versions.
    pub name: &'static str,
    pub products: &'static [polylogue_model::derived::Construct],
    /// Bumped when the rules change; every session is derived again.
    pub version: Version,
    /// Derivations whose rows this one reads; they run first.
    pub after: &'static [&'static str],
    /// Recomputes the session's rows: replaces whatever an earlier run
    /// made, from the session's current rows.
    pub run: fn(&Transaction<'_>, i64, &Catalog) -> Result<()>,
    /// Removes shared rows left without any session, even in an empty archive.
    pub finish: Option<fn(&Transaction<'_>) -> Result<()>>,
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
    tool_calls::DERIVATION,
    active_path::DERIVATION,
    repository::DERIVATION,
    title::DERIVATION,
    work::DERIVATION,
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
        .zip(versions(ALL, Catalog::builtin())?)
        .map(|(d, v)| (d.name.to_owned(), v))
        .collect())
}

pub(crate) fn is_current(conn: &Connection, name: &str, pk: i64) -> Result<bool> {
    Ok(conn.query_row("SELECT EXISTS(SELECT 1 FROM derive_state d JOIN session s ON s.pk=d.session_pk WHERE d.name=?1 AND s.pk=?2 AND d.from_changed=s.changed AND pl_derived_current(d.name,d.version))", params![name,pk], |r| r.get(0))?)
}

/// Runs every derivation over the sessions it has not derived at their
/// current state; returns the number of (derivation, session) pairs done.
pub fn run(
    index: &mut Connection,
    commits: Option<&crate::completion::CommitEvents>,
) -> Result<u64> {
    run_all(index, Catalog::builtin(), ALL, commits)
}

#[cfg(test)]
fn run_with_catalog(index: &mut Connection, catalog: &Catalog) -> Result<u64> {
    run_all(index, catalog, ALL, None)
}

fn run_all(
    index: &mut Connection,
    catalog: &Catalog,
    derivations: &[Derivation],
    commits: Option<&crate::completion::CommitEvents>,
) -> Result<u64> {
    // Resolve the whole declaration before writing any rows. Dependency rule
    // changes invalidate consumers even when their source session is unchanged.
    let versions = versions(derivations, catalog)?;
    let mut done = 0;
    let todo = sessions_to_derive(index, derivations, &versions)?;
    for sessions in todo.chunks(SESSIONS_PER_TRANSACTION) {
        // Immediate, like the writer's and the search thread's: a deferred
        // transaction fails to upgrade while the search thread writes.
        let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
        let counter = crate::archive::bump_counter(&tx)?;
        for &pk in sessions {
            done += derive_session(&tx, pk, derivations, &versions, catalog, counter)?;
        }
        tx.commit()?;
        if let Some(events) = commits {
            events.committed();
        }
    }
    let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let before = tx.total_changes();
    for finish in derivations.iter().filter_map(|d| d.finish) {
        finish(&tx)?;
    }
    let changed = tx.total_changes() != before;
    if changed {
        crate::archive::bump_counter(&tx)?;
    }
    tx.commit()?;
    if changed && let Some(events) = commits {
        events.committed();
    }
    Ok(done)
}

/// The sessions some derivation has not derived at their current state and
/// its current version.
fn sessions_to_derive(
    index: &Connection,
    derivations: &[Derivation],
    versions: &[String],
) -> Result<Vec<i64>> {
    if derivations.is_empty() {
        return Ok(Vec::new());
    }
    let wanted = vec!["(?, ?)"; derivations.len()].join(", ");
    let sql = format!(
        "WITH wanted(name, version) AS (VALUES {wanted})
         SELECT s.pk FROM session s
         WHERE (SELECT count(*) FROM wanted w JOIN derive_state d
                  ON d.name = w.name AND d.session_pk = s.pk
                 WHERE d.version = w.version AND d.from_changed = s.changed) < {}
         ORDER BY s.pk",
        derivations.len()
    );
    let mut values: Vec<&dyn rusqlite::ToSql> = Vec::new();
    for (d, version) in derivations.iter().zip(versions) {
        values.push(&d.name);
        values.push(version);
    }
    Ok(index
        .prepare(&sql)?
        .query_map(values.as_slice(), |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?)
}

/// Runs, in order, every derivation that has not derived the session at its
/// current state and version, while the session's rows are read once into
/// the cache; returns how many ran. Derived rows change the visible session,
/// not its inputs, so the markers of derivations that were current move to
/// the session's new state and do not make them run again.
fn derive_session(
    tx: &Transaction<'_>,
    pk: i64,
    derivations: &[Derivation],
    versions: &[String],
    catalog: &Catalog,
    counter: i64,
) -> Result<u64> {
    let Some(changed) = tx
        .prepare_cached("SELECT changed FROM session WHERE pk = ?1")?
        .query_row([pk], |r| r.get::<_, i64>(0))
        .optional()?
    else {
        return Ok(0);
    };
    let mut current = std::collections::HashSet::new();
    let mut statement = tx.prepare_cached(
        "SELECT name, version FROM derive_state WHERE session_pk = ?1 AND from_changed = ?2",
    )?;
    for row in statement.query_map(params![pk, changed], |r| {
        Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
    })? {
        current.insert(row?);
    }
    let stale: Vec<usize> = (0..derivations.len())
        .filter(|&i| !current.contains(&(derivations[i].name.to_owned(), versions[i].clone())))
        .collect();
    if stale.is_empty() {
        return Ok(0);
    }
    for &i in &stale {
        (derivations[i].run)(tx, pk, catalog)?;
    }
    tx.prepare_cached(
        "UPDATE derive_state SET from_changed = ?3 WHERE session_pk = ?1 AND from_changed = ?2",
    )?
    .execute(params![pk, changed, counter])?;
    tx.prepare_cached("UPDATE session SET changed = ?2 WHERE pk = ?1")?
        .execute(params![pk, counter])?;
    let mut mark = tx.prepare_cached(
        "INSERT INTO derive_state(name, session_pk, version, from_changed)
         VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(name, session_pk) DO UPDATE SET version = ?3, from_changed = ?4",
    )?;
    for &i in &stale {
        mark.execute(params![derivations[i].name, pk, versions[i], counter])?;
    }
    Ok(stale.len() as u64)
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
