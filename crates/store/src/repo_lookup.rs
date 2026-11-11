//! Repository lookups are stored as records; which sessions they attribute
//! is the `repository` derivation's job. A changed lookup marks the sessions
//! at or under its directory due for that derivation alone; their own rows
//! did not change.

use crate::Result;
use polylogue_model::RepoLookup;
use rusqlite::{OptionalExtension, Transaction, params};

/// Stores a lookup; when it differs from the stored one, marks every
/// session at or under its directory due for its repository.
pub(crate) fn upsert_lookup(
    tx: &Transaction<'_>,
    l: &RepoLookup,
    input_pk: i64,
    chain: i64,
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
        // `cwd` under the directory sorts between `<path>/` and `<path>0`
        // ('0' follows '/'), so the session index serves the range.
        tx.prepare_cached(
            "INSERT INTO derive_due(session_pk, rows_changed, repo_lookup)
             SELECT pk, 0, 1 FROM session
             WHERE cwd = ?1 OR (cwd >= ?1 || '/' AND cwd < ?1 || '0')
             ON CONFLICT(session_pk) DO UPDATE SET repo_lookup = 1, marked = marked + 1",
        )?
        .execute([&l.path])?;
    }
    Ok(())
}
