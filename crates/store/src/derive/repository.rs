//! The repository a session belongs to (spec sources/README.md, the
//! repo-lookup record): the lookup whose directory is the longest at or
//! above its working directory, compared by whole path components. Its
//! `repo_root` comes from that lookup; its `repo` too, unless a record of
//! its own named the repository (`record_repo`).

use super::{Derivation, Version};
use crate::Result;
use rusqlite::{OptionalExtension, Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "repository",
    version: Version::Rules(1),
    // Reads the session's working directory and recorded repository.
    after: &["session"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

fn run(tx: &Transaction<'_>, session_pk: i64) -> Result<()> {
    let (cwd, own): (Option<String>, Option<String>) = tx
        .prepare_cached("SELECT cwd, record_repo FROM session WHERE pk = ?1")?
        .query_row([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))?;
    let lookup: Option<(Option<String>, Option<String>)> = match &cwd {
        Some(cwd) => tx
            .prepare_cached(
                "SELECT remote, root_commit FROM repo_lookup
                 WHERE path = ?1 OR substr(?1, 1, length(path) + 1) = path || '/'
                 ORDER BY length(path) DESC LIMIT 1",
            )?
            .query_row([cwd], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?,
        None => None,
    };
    let (remote, root) = lookup.unwrap_or((None, None));
    let repo = own.or(remote);
    tx.prepare_cached(
        "UPDATE session SET repo = ?2, repo_root = ?3
         WHERE pk = ?1 AND (repo IS NOT ?2 OR repo_root IS NOT ?3)",
    )?
    .execute(params![session_pk, repo, root])?;
    Ok(())
}
