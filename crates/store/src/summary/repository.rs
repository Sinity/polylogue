//! The repository a session belongs to, from the repo-lookup records: the
//! lookup whose directory is the longest at or above its working directory,
//! compared by whole path components. Its
//! `repo_root` is the first of that lookup's sorted root commits, so a
//! single-root repository is named by its root and a merged one by a stable
//! member of its set; its `repo` comes from the lookup too, unless a record
//! of its own named the repository (`record_repo`).

use super::{Kind, Output, Summary, Version};
use crate::Result;
use rusqlite::{Connection, OptionalExtension, Transaction, params};

pub(super) const SUMMARY: Summary = Summary {
    name: "repository",
    products: &[],
    version: Version::Rules(2),
    kind: Kind::AfterFolds,
};

/// The repository of the session's working directory and records, as the
/// session summary computed them in this run ([`super::session::compute`]).
pub(super) fn compute(conn: &Connection, session_pk: i64, out: &mut Output) -> Result<()> {
    let Some(facts) = &out.session else {
        return Ok(());
    };
    let (cwd, own) = super::session::place(conn, session_pk, facts)?;
    let lookup: Option<(Option<String>, Option<String>)> = match &cwd {
        Some(cwd) => conn
            .prepare_cached(&format!(
                "SELECT remote, json_extract(root_commits, '$[0]') FROM repo_lookup
                 WHERE {} ORDER BY length(path) DESC LIMIT 1",
                crate::session_sql::path_under_column("?1", "path")
            ))?
            .query_row([cwd], |r| <(_, _)>::try_from(r))
            .optional()?,
        None => None,
    };
    let (remote, root) = lookup.unwrap_or((None, None));
    out.repository = Some((own.or(remote), root));
    Ok(())
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
