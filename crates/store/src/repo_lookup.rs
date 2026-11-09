//! Repository lookups are
//! stored as records; which sessions they attribute is the `repository`
//! derivation's job. A changed lookup marks the sessions under its
//! directory changed, so they are attributed again.

use crate::Result;
use polylogue_model::RepoLookup;
use rusqlite::{OptionalExtension, Transaction, params};

/// Stores a lookup; when it differs from the stored one, marks every
/// session under its directory changed.
pub(crate) fn upsert_lookup(
    tx: &Transaction<'_>,
    l: &RepoLookup,
    input_pk: i64,
    chain: i64,
    counter: i64,
) -> Result<()> {
    let roots = serde_json::to_string(&l.root_commits)?;
    let previous: Option<(Option<String>, String)> = tx
        .prepare_cached("SELECT remote, root_commits FROM repo_lookup WHERE path = ?1")?
        .query_row([&l.path], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    tx.prepare_cached(
        "INSERT INTO repo_lookup(path, remote, root_commits, input_pk, chain) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(path) DO UPDATE SET remote = excluded.remote,
             root_commits = excluded.root_commits, input_pk = excluded.input_pk,
             chain = excluded.chain",
    )?
    .execute(params![l.path, l.remote, roots, input_pk, chain])?;
    if previous != Some((l.remote.clone(), roots)) {
        tx.prepare_cached(
            "UPDATE session SET changed = ?2
             WHERE cwd = ?1 OR substr(cwd, 1, length(?1) + 1) = ?1 || '/'",
        )?
        .execute(params![l.path, counter])?;
    }
    Ok(())
}
