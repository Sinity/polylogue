//! Repository lookups are stored as records; which sessions they attribute
//! is the `repository` derivation's job. A changed lookup marks the sessions
//! at or under its directory due for that derivation alone; their own rows
//! did not change.

use crate::Result;
use polylogue_model::RepoLookup;
use rusqlite::{OptionalExtension, Transaction, params};

impl crate::Reader<'_> {
    /// Recorded repository identities at these exact worktree roots.
    pub fn repository_roots_at_paths(&self, paths: &[String]) -> Result<Vec<String>> {
        let mut statement = self.conn()?.prepare_cached(
            "SELECT root_commits FROM repo_lookup WHERE path IN (SELECT value FROM json_each(?1)) ORDER BY path",
        )?;
        let mut rows = statement.query([serde_json::to_string(paths)?])?;
        let mut roots = std::collections::BTreeSet::new();
        while let Some(row) = rows.next()? {
            let encoded: String = row.get(0)?;
            let mut recorded: Vec<String> = serde_json::from_str(&encoded)?;
            recorded.sort();
            if let Some(first) = recorded.into_iter().next() {
                roots.insert(first);
            }
        }
        self.check()?;
        Ok(roots.into_iter().collect())
    }
}

/// Stores a lookup; when it differs from the stored one, marks every
/// session at or under its directory due for its repository.
pub(crate) fn upsert_lookup(
    tx: &Transaction<'_>,
    lookup: &RepoLookup,
    input_pk: i64,
    chain: i64,
) -> Result<()> {
    // Stored sorted, so the derivation's first root is the identity the reader names.
    let mut sorted = lookup.root_commits.clone();
    sorted.sort();
    let roots = serde_json::to_string(&sorted)?;
    let previous: Option<(Option<String>, String)> = tx
        .prepare_cached("SELECT remote, root_commits FROM repo_lookup WHERE path = ?1")?
        .query_row([&lookup.path], |r| <(_, _)>::try_from(r))
        .optional()?;
    tx.prepare_cached(
        "INSERT INTO repo_lookup(path, remote, root_commits, input_pk, chain) VALUES (?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(path) DO UPDATE SET remote = excluded.remote,
             root_commits = excluded.root_commits, input_pk = excluded.input_pk,
             chain = excluded.chain",
    )?
    .execute(params![lookup.path, lookup.remote, roots, input_pk, chain])?;
    if previous != Some((lookup.remote.clone(), roots)) {
        // `cwd` under the directory sorts between `<path>/` and `<path>0`
        // ('0' follows '/'), so the session index serves the range.
        tx.prepare_cached(
            "INSERT INTO summary_due(session_pk, from_order, repo_lookup)
             SELECT pk, X'FF', 1 FROM session
             WHERE cwd = ?1 OR (cwd >= ?1 || '/' AND cwd < ?1 || '0')
             ON CONFLICT(session_pk) DO UPDATE SET repo_lookup = 1, marked = marked + 1",
        )?
        .execute([&lookup.path])?;
    }
    Ok(())
}
