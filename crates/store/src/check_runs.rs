//! Check runs the development tooling recorded: every `xtask` run (`check`,
//! and `test` for selected tests) writes a row to devkit's machine-wide
//! history, `$HOME/.local/state/devkit/history.db`, with its checkout,
//! outcome and AgentCTL job. That file's `runs` table is its published
//! interface; it is read here read-only, and never written.

use crate::Result;
use rusqlite::{Connection, OpenFlags, params};
use serde::Serialize;
use std::path::{Path, PathBuf};

/// The history's default place under a home directory.
pub fn default_path(home: &Path) -> PathBuf {
    home.join(".local/state/devkit/history.db")
}

/// A check run that failed with no later pass of it in the same checkout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FailedRun {
    /// `project command args`, as run: `polylogue check`,
    /// `polylogue test -p polylogue-store`.
    pub check: String,
    pub checkout: String,
    pub run: i64,
    pub started_ms: i64,
    pub failure: Option<String>,
    pub job: Option<String>,
    /// Failures of the same check in the checkout since the window began.
    pub times: u64,
}

/// Failed `check` and `test` runs since `since` (Unix ms) in checkouts at or
/// under `dirs`, newest first, each with no later passing run of the same
/// command and arguments, or of a whole `check`, in its checkout. A missing
/// history file holds no runs.
pub fn failed_since(path: &Path, dirs: &[String], since: i64) -> Result<Vec<FailedRun>> {
    if !path.exists() {
        return Ok(Vec::new());
    }
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )?;
    conn.busy_timeout(std::time::Duration::from_secs(2))?;
    let dirs = serde_json::to_string(dirs)?;
    let mut stmt = conn.prepare(
        "SELECT r.id, r.project, r.command, r.args, r.checkout, r.started_ms, r.failure, r.agentctl_job,
                (SELECT count(*) FROM runs f WHERE f.status = 'failed' AND f.checkout = r.checkout
                    AND f.project = r.project AND f.command = r.command AND f.args = r.args
                    AND f.started_ms >= ?2)
           FROM runs r
          WHERE r.status = 'failed' AND r.command IN ('check', 'test') AND r.started_ms >= ?2
            AND EXISTS (SELECT 1 FROM json_each(?1) d WHERE r.checkout = d.value
                        OR (r.checkout >= d.value || '/' AND r.checkout < d.value || '0'))
            AND NOT EXISTS (
                SELECT 1 FROM runs p WHERE p.status = 'passed' AND p.checkout = r.checkout
                   AND p.project = r.project AND p.id > r.id
                   AND (p.command = 'check' OR (p.command = r.command AND p.args = r.args)))
          ORDER BY r.id DESC",
    )?;
    let rows = stmt.query_map(params![dirs, since], |r| {
        let args: String = r.get(3)?;
        let args: Vec<String> = serde_json::from_str(&args).unwrap_or_default();
        let mut check = format!("{} {}", r.get::<_, String>(1)?, r.get::<_, String>(2)?);
        for a in args {
            check.push(' ');
            check.push_str(&a);
        }
        Ok(FailedRun {
            check,
            run: r.get(0)?,
            checkout: r.get(4)?,
            started_ms: r.get(5)?,
            failure: r.get(6)?,
            job: r.get(7)?,
            times: u64::try_from(r.get::<_, i64>(8)?).unwrap_or(0),
        })
    })?;
    let mut out: Vec<FailedRun> = Vec::new();
    for row in rows {
        let row = row?;
        // The latest failure of each check in each checkout stands for it.
        if !out
            .iter()
            .any(|o| o.check == row.check && o.checkout == row.checkout)
        {
            out.push(row);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
