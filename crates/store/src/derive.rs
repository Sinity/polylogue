//! Derived rows (architecture 5.6): every interpretation of the stored
//! records, one declared derivation each. A derivation names the
//! derivations it reads, a version, and the rows it owns for a session; the
//! derive step recomputes a session's rows whenever the session changed
//! since they were made or the derivation's version moved, in dependency
//! order. The writer's ingest path stores records only and never writes a
//! derived row, so a first build and an incremental one derive the same
//! rows from the same records. Excision needs nothing per derivation: a
//! removed session takes its rows with it, and an excised text removes the
//! sessions built from it, which come back from their inputs parsed again
//! and are derived anew.

use crate::{Result, StoreError};
use polylogue_model::Catalog;
use rusqlite::{Connection, Transaction, TransactionBehavior, params};
use std::collections::BTreeMap;

mod active_path;
mod cost;
mod repository;
mod session;
mod title;
mod tool_calls;
mod work;

/// Sessions derived per transaction, so a long derive never holds the
/// writer for minutes.
const SESSIONS_PER_TRANSACTION: usize = 200;

/// One derivation over a session's stored rows.
#[derive(Clone, Copy)]
struct Derivation {
    /// Its key in `derive_state`; stable across versions.
    pub name: &'static str,
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
    session::DERIVATION,
    tool_calls::DERIVATION,
    active_path::DERIVATION,
    repository::DERIVATION,
    title::DERIVATION,
    work::DERIVATION,
    cost::DERIVATION,
];

/// Runs every derivation over the sessions it has not derived at their
/// current state; returns the number of (derivation, session) pairs done.
pub fn run(index: &mut Connection) -> Result<u64> {
    run_with_catalog(index, Catalog::builtin())
}

fn run_with_catalog(index: &mut Connection, catalog: &Catalog) -> Result<u64> {
    run_all(index, catalog, ALL)
}

fn run_all(index: &mut Connection, catalog: &Catalog, derivations: &[Derivation]) -> Result<u64> {
    // Resolve the whole declaration before writing any rows. Dependency rule
    // changes invalidate consumers even when their source session is unchanged.
    let versions = versions(derivations, catalog)?;
    let mut done = 0;
    for (d, version) in derivations.iter().zip(&versions) {
        done += run_one(index, d, version, catalog)?;
    }
    Ok(done)
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

fn run_one(
    index: &mut Connection,
    d: &Derivation,
    version: &str,
    catalog: &Catalog,
) -> Result<u64> {
    let mut done = 0;
    loop {
        // Immediate, like the writer's and the search thread's: a deferred
        // transaction fails to upgrade while the search thread writes.
        let tx = crate::publication::transaction(index, TransactionBehavior::Immediate)?;
        let todo: Vec<(i64, i64)> = tx
            .prepare_cached(
                "SELECT s.pk, s.changed FROM session s
                 LEFT JOIN derive_state d ON d.session_pk = s.pk AND d.name = ?1
                 WHERE d.session_pk IS NULL OR d.version <> ?2 OR d.from_changed <> s.changed
                 LIMIT ?3",
            )?
            .query_map(
                params![d.name, version, SESSIONS_PER_TRANSACTION as i64],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?
            .collect::<rusqlite::Result<_>>()?;
        if todo.is_empty() {
            let before = tx.total_changes();
            if let Some(finish) = d.finish {
                finish(&tx)?;
            }
            if tx.total_changes() != before {
                crate::archive::bump_counter(&tx)?;
            }
            tx.commit()?;
            return Ok(done);
        }
        let counter = crate::archive::bump_counter(&tx)?;
        for (pk, changed) in &todo {
            (d.run)(&tx, *pk, catalog)?;
            // Derived publication changes the visible session, not its raw
            // inputs. Forward already-current cache markers so it does not dirty
            // earlier products; stale inputs or rule versions remain stale.
            tx.execute(
                "UPDATE derive_state SET from_changed=?3 WHERE session_pk=?1 AND from_changed=?2",
                params![pk, changed, counter],
            )?;
            tx.execute(
                "UPDATE session SET changed=?2 WHERE pk=?1",
                params![pk, counter],
            )?;
            tx.prepare_cached(
                "INSERT INTO derive_state(name, session_pk, version, from_changed)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(name, session_pk) DO UPDATE SET version = ?3, from_changed = ?4",
            )?
            .execute(params![d.name, pk, version, counter])?;
        }
        tx.commit()?;
        done += todo.len() as u64;
    }
}

#[cfg(test)]
mod tests;
