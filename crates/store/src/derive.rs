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

use crate::Result;
use polylogue_model::Catalog;
use rusqlite::{Connection, Transaction, TransactionBehavior, params};

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
                format!("catalog:{n}:{}:{hash}", catalog.version)
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
    let mut done = 0;
    for d in ALL {
        debug_assert!(
            d.after.iter().all(|a| ALL
                .iter()
                .take_while(|x| x.name != d.name)
                .any(|x| x.name == *a)),
            "derivation {} runs before what it reads",
            d.name
        );
        done += run_one(index, d, catalog)?;
    }
    Ok(done)
}

fn run_one(index: &mut Connection, d: &Derivation, catalog: &Catalog) -> Result<u64> {
    let version = d.version.resolve(catalog)?;
    let mut done = 0;
    loop {
        // Immediate, like the writer's and the search thread's: a deferred
        // transaction fails to upgrade while the search thread writes.
        let tx = index.transaction_with_behavior(TransactionBehavior::Immediate)?;
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
            if let Some(finish) = d.finish {
                finish(&tx)?;
            }
            tx.commit()?;
            return Ok(done);
        }
        for (pk, changed) in &todo {
            (d.run)(&tx, *pk, catalog)?;
            tx.prepare_cached(
                "INSERT INTO derive_state(name, session_pk, version, from_changed)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(name, session_pk) DO UPDATE SET version = ?3, from_changed = ?4",
            )?
            .execute(params![d.name, pk, version, changed])?;
        }
        tx.commit()?;
        done += todo.len() as u64;
    }
}

#[cfg(test)]
mod tests;
