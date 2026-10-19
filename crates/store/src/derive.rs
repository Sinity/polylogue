//! Derived rows (architecture 5.6): every interpretation of the stored
//! records, one declared derivation each. A derivation names the
//! derivations it reads, a version, and the rows it owns for a session; the
//! derive step recomputes a session's rows whenever the session changed
//! since they were made or the derivation's version moved, in dependency
//! order. The writer's ingest path stores records only and never writes a
//! derived row, so a first build and an incremental one derive the same
//! rows from the same records. Excision needs nothing per derivation: a
//! removed session takes its rows with it, and an excised text marks its
//! sessions changed, so their rows are derived again from what is left.

use crate::Result;
use rusqlite::{Connection, Transaction, TransactionBehavior, params};

mod active_path;
mod repository;
mod session;
mod title;
mod tool_calls;

/// Sessions derived per transaction, so a long derive never holds the
/// writer for minutes.
const SESSIONS_PER_TRANSACTION: usize = 200;

/// One derivation over a session's stored rows.
pub struct Derivation {
    /// Its key in `derive_state`; stable across versions.
    pub name: &'static str,
    /// Bumped when the rules change; every session is derived again.
    pub version: i64,
    /// Derivations whose rows this one reads; they run first.
    pub after: &'static [&'static str],
    /// Recomputes the session's rows: replaces whatever an earlier run
    /// made, from the session's current rows.
    pub run: fn(&Transaction<'_>, i64) -> Result<()>,
}

/// Every derivation, in an order where each comes after what it reads.
pub const ALL: &[Derivation] = &[
    session::DERIVATION,
    tool_calls::DERIVATION,
    active_path::DERIVATION,
    repository::DERIVATION,
    title::DERIVATION,
    crate::work::DERIVATION,
];

/// Runs every derivation over the sessions it has not derived at their
/// current state; returns the number of (derivation, session) pairs done.
pub fn run(index: &mut Connection) -> Result<u64> {
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
        done += run_one(index, d)?;
    }
    crate::work::sweep(index)?;
    Ok(done)
}

fn run_one(index: &mut Connection, d: &Derivation) -> Result<u64> {
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
                params![d.name, d.version, SESSIONS_PER_TRANSACTION as i64],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )?
            .collect::<rusqlite::Result<_>>()?;
        if todo.is_empty() {
            tx.commit()?;
            return Ok(done);
        }
        for (pk, changed) in &todo {
            (d.run)(&tx, *pk)?;
            tx.prepare_cached(
                "INSERT INTO derive_state(name, session_pk, version, from_changed)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(name, session_pk) DO UPDATE SET version = ?3, from_changed = ?4",
            )?
            .execute(params![d.name, pk, d.version, changed])?;
        }
        tx.commit()?;
        done += todo.len() as u64;
    }
}

#[cfg(test)]
mod tests;
