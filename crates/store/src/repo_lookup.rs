//! Repository lookups (spec sources/README.md, the repo-lookup record) are
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
    counter: i64,
) -> Result<()> {
    let previous: Option<(Option<String>, Option<String>)> = tx
        .prepare_cached("SELECT remote, root_commit FROM repo_lookup WHERE path = ?1")?
        .query_row([&l.path], |r| Ok((r.get(0)?, r.get(1)?)))
        .optional()?;
    tx.prepare_cached(
        "INSERT INTO repo_lookup(path, remote, root_commit, input_pk) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(path) DO UPDATE SET remote = excluded.remote,
             root_commit = excluded.root_commit, input_pk = excluded.input_pk",
    )?
    .execute(params![l.path, l.remote, l.root_commit, input_pk])?;
    if previous != Some((l.remote.clone(), l.root_commit.clone())) {
        tx.prepare_cached(
            "UPDATE session SET changed = ?2
             WHERE cwd = ?1 OR substr(cwd, 1, length(?1) + 1) = ?1 || '/'",
        )?
        .execute(params![l.path, counter])?;
    }
    Ok(())
}
