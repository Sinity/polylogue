//! Measurement (architecture 1, "Measure things"; 7): tokens by kind, cost,
//! subscription quota and the analyses over sessions and tool
//! calls. Every figure says how many items it could not count; unknown is
//! never shown as zero. Hidden sessions stay in every figure: hiding
//! changes what is listed, not what was used.

pub mod analysis;
pub mod quota;
mod usage;

pub use usage::{ModelUsage, Money, SessionUsage, TokenCount, TokensByKind, Totals, UsageReport};

use crate::{Reader, Result};

/// Which messages a measurement covers. Nothing given is the whole archive.
#[derive(Debug, Clone, Default)]
pub struct Scope {
    /// Session primary keys, from a sessions query.
    pub sessions: Option<Vec<i64>>,
    /// Unix ms bounds on a message's time, `since` inclusive.
    pub since: Option<i64>,
    pub until: Option<i64>,
    /// Only sessions of these origins (`claude-code`, `codex`, …).
    pub origins: Option<Vec<String>>,
}

impl Scope {
    /// The same scope, narrowed to one tool's sessions.
    pub fn for_tool(mut self, origin: &str) -> Self {
        self.origins = Some(vec![origin.to_owned()]);
        self
    }

    /// The SQL condition on `m` (message) and `s` (session), with its
    /// parameters.
    fn condition(&self) -> (String, Vec<rusqlite::types::Value>) {
        let mut clauses = vec!["1".to_owned()];
        let mut params: Vec<rusqlite::types::Value> = vec![];
        if let Some(since) = self.since {
            params.push(since.into());
            clauses.push(format!("m.at >= ?{}", params.len()));
        }
        if let Some(until) = self.until {
            params.push(until.into());
            clauses.push(format!("m.at < ?{}", params.len()));
        }
        if let Some(origins) = &self.origins {
            let mut marks = vec![];
            for o in origins {
                params.push(o.clone().into());
                marks.push(format!("?{}", params.len()));
            }
            clauses.push(format!(
                "m.session_pk IN (SELECT pk FROM session WHERE origin IN ({}))",
                marks.join(",")
            ));
        }
        if let Some(pks) = &self.sessions {
            // Session keys are integers the caller read from the index, so
            // they are written into the statement rather than bound one by
            // one past SQLite's parameter limit.
            let list = pks.iter().map(i64::to_string).collect::<Vec<_>>().join(",");
            clauses.push(format!("m.session_pk IN ({list})"));
        }
        (clauses.join(" AND "), params)
    }
}

/// Tokens by kind and cost over the scope, at the physical grain: each model
/// response once, however many sessions show it.
pub fn usage(reader: &Reader<'_>, scope: &Scope) -> Result<UsageReport> {
    usage::report(reader, scope)
}

/// Each session's usage in the scope, physical and logical.
pub fn sessions(reader: &Reader<'_>, scope: &Scope) -> Result<Vec<SessionUsage>> {
    usage::per_session(reader, scope)
}

/// Sessions in `scope` that a current `hidden` note hides. They stay in
/// every figure; this says how many figures include.
pub(crate) fn hidden_sessions(reader: &Reader<'_>, scope: &Scope) -> Result<u64> {
    let (cond, params) = scope.condition();
    let sql = format!(
        "SELECT count(DISTINCT s.pk) FROM current_message m JOIN session s ON s.pk = m.session_pk
         WHERE {cond} AND EXISTS (
           SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id
           WHERE hn.kind = 'hidden' AND ht.ref = 'session:' || s.id
             AND NOT EXISTS (SELECT 1 FROM a.note later WHERE later.root = hn.root
                             AND later.commit_seq > hn.commit_seq)
             AND NOT EXISTS (SELECT 1 FROM a.note w JOIN a.note_target wt ON wt.note_id = w.id
                             WHERE w.kind = 'withdrawal' AND wt.ref = 'note:' || hn.root))"
    );
    let n: i64 = reader
        .conn()
        .query_row(&sql, rusqlite::params_from_iter(params), |r| r.get(0))?;
    Ok(n as u64)
}

#[cfg(test)]
mod tests;
