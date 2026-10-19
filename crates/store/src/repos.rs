//! Joining sessions to repository lookups (spec sources/README.md, the
//! repo-lookup record). A session belongs to the lookup whose directory is
//! the longest at or above its working directory, compared by whole path
//! components. Its `repo_root` comes from that lookup; its `repo` too, unless
//! a record of its own named the repository.

use crate::Result;
use polylogue_model::RepoLookup;
use rusqlite::{OptionalExtension, Transaction, params};

/// Stores a lookup and gives every session under its directory the
/// repository it names.
pub(crate) fn upsert_lookup(
    tx: &Transaction<'_>,
    l: &RepoLookup,
    input_pk: i64,
    counter: i64,
) -> Result<()> {
    let previous: Option<Option<String>> = tx
        .prepare_cached("SELECT remote FROM repo_lookup WHERE path = ?1")?
        .query_row([&l.path], |r| r.get(0))
        .optional()?;
    tx.prepare_cached(
        "INSERT INTO repo_lookup(path, remote, root_commit, input_pk) VALUES (?1, ?2, ?3, ?4)
         ON CONFLICT(path) DO UPDATE SET remote = excluded.remote,
             root_commit = excluded.root_commit, input_pk = excluded.input_pk",
    )?
    .execute(params![l.path, l.remote, l.root_commit, input_pk])?;
    let sessions: Vec<i64> = tx
        .prepare_cached(
            "SELECT pk FROM session
             WHERE cwd = ?1 OR substr(cwd, 1, length(?1) + 1) = ?1 || '/'",
        )?
        .query_map([&l.path], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    for pk in sessions {
        assign(
            tx,
            pk,
            counter,
            previous.as_ref().and_then(Option::as_deref),
        )?;
    }
    Ok(())
}

/// Copies the repository of the session's lookup onto the session.
/// `replaced` is a remote this lookup used to name: a `repo` equal to it
/// came from the lookup and follows the change.
pub(crate) fn assign(
    tx: &Transaction<'_>,
    session_pk: i64,
    counter: i64,
    replaced: Option<&str>,
) -> Result<()> {
    let Some((cwd, repo, root)) = tx
        .prepare_cached("SELECT cwd, repo, repo_root FROM session WHERE pk = ?1")?
        .query_row([session_pk], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, Option<String>>(2)?,
            ))
        })
        .optional()?
    else {
        return Ok(());
    };
    let Some(cwd) = cwd else { return Ok(()) };
    let Some((remote, new_root)) = tx
        .prepare_cached(
            "SELECT remote, root_commit FROM repo_lookup
             WHERE path = ?1 OR substr(?1, 1, length(path) + 1) = path || '/'
             ORDER BY length(path) DESC LIMIT 1",
        )?
        .query_row([&cwd], |r| {
            Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<String>>(1)?,
            ))
        })
        .optional()?
    else {
        return Ok(());
    };
    // A `repo` no lookup names came from the session's own records.
    let own = match repo.as_deref() {
        None => false,
        Some(r) if Some(r) == replaced => false,
        Some(r) => tx
            .prepare_cached("SELECT 1 FROM repo_lookup WHERE remote = ?1 LIMIT 1")?
            .query_row([r], |_| Ok(()))
            .optional()?
            .is_none(),
    };
    let new_repo = if own { repo.clone() } else { remote };
    if new_repo != repo || new_root != root {
        tx.prepare_cached(
            "UPDATE session SET repo = ?2, repo_root = ?3, changed = ?4 WHERE pk = ?1",
        )?
        .execute(params![session_pk, new_repo, new_root, counter])?;
    }
    Ok(())
}
