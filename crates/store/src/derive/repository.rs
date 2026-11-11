//! The repository a session belongs to, from the repo-lookup records: the
//! lookup whose directory is the longest at or above its working directory,
//! compared by whole path components. Its
//! `repo_root` is the first of that lookup's sorted root commits, so a
//! single-root repository is named by its root and a merged one by a stable
//! member of its set; its `repo` comes from the lookup too, unless a record
//! of its own named the repository (`record_repo`).

use super::{Cause, Class, Derivation, Output, Version};
use crate::Result;
use rusqlite::{Connection, OptionalExtension, Transaction, params};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "repository",
    class: Class::Session,
    products: &[],
    version: Version::Rules(2),
    // Reads the session's working directory and recorded repository.
    after: &["session"],
    dirtied_by: &[Cause::RepoLookupChanged],
    compute: |conn, pk, _| compute(conn, pk),
};

fn compute(conn: &Connection, session_pk: i64) -> Result<Output> {
    let (cwd, own): (Option<String>, Option<String>) = conn
        .prepare_cached("SELECT cwd, record_repo FROM session WHERE pk = ?1")?
        .query_row([session_pk], |r| Ok((r.get(0)?, r.get(1)?)))?;
    let lookup: Option<(Option<String>, Option<String>)> = match &cwd {
        Some(cwd) => conn
            .prepare_cached(
                "SELECT remote, json_extract(root_commits, '$[0]') FROM repo_lookup
                 WHERE path = ?1 OR substr(?1, 1, length(path) + 1) = path || '/'
                 ORDER BY length(path) DESC LIMIT 1",
            )?
            .query_row([cwd], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?,
        None => None,
    };
    let (remote, root) = lookup.unwrap_or((None, None));
    Ok(Output::Repository {
        repo: own.or(remote),
        root,
    })
}

pub(super) fn apply(
    tx: &Transaction<'_>,
    session_pk: i64,
    repo: &Option<String>,
    root: &Option<String>,
) -> Result<()> {
    tx.prepare_cached(
        "UPDATE session SET repo = ?2, repo_root = ?3
         WHERE pk = ?1 AND (repo IS NOT ?2 OR repo_root IS NOT ?3)",
    )?
    .execute(params![session_pk, repo, root])?;
    Ok(())
}
