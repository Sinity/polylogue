//! Measurement reads: tokens by kind, cost, subscription quota, and the
//! measures over sessions and tool calls. Each runs at read time over the
//! sessions a query selects; facts that depend only on one session's rows
//! come from derivations (`session_survival`, message cost). Every figure says
//! how many items it could not count; unknown is never shown as zero.
//! Hidden sessions stay in every figure: hiding changes what is listed, not
//! what was used.

pub mod daily;
pub mod drift;
pub mod quota;
mod reports;
pub mod survival;
pub mod tool_calls;
mod usage;

pub use usage::{
    ModelUsage, Money, RecordedUsage, SessionUsage, TokenCount, TokensByKind, Totals, UsageReport,
};

use crate::{Reader, Result};

/// Which messages a measurement covers. Nothing given is the whole archive.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    /// Session primary keys, from a sessions query.
    pub sessions: Option<Vec<i64>>,
    /// Unix ms bounds on a message's time, `since` inclusive.
    pub since: Option<i64>,
    pub until: Option<i64>,
    /// Only sessions of these origins (`claude-code`, `codex`, …).
    pub origins: Option<Vec<String>>,
}

impl Selection {
    /// The same selection, narrowed to one tool's sessions.
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
        self.session_clauses("m.session_pk", &mut clauses, &mut params);
        (clauses.join(" AND "), params)
    }

    /// The SQL condition on a session primary key column, for figures read
    /// from per-session facts. Those facts have no message times, so a
    /// time-bounded selection is refused.
    fn session_condition(&self, column: &str) -> Result<(String, Vec<rusqlite::types::Value>)> {
        if self.since.is_some() || self.until.is_some() {
            return Err(crate::StoreError::Invalid(
                "per-session measures take no time bounds".into(),
            ));
        }
        let mut clauses = vec!["1".to_owned()];
        let mut params: Vec<rusqlite::types::Value> = vec![];
        self.session_clauses(column, &mut clauses, &mut params);
        Ok((clauses.join(" AND "), params))
    }

    fn session_clauses(
        &self,
        column: &str,
        clauses: &mut Vec<String>,
        params: &mut Vec<rusqlite::types::Value>,
    ) {
        if let Some(origins) = &self.origins {
            let mut marks = vec![];
            for o in origins {
                params.push(o.clone().into());
                marks.push(format!("?{}", params.len()));
            }
            clauses.push(format!(
                "{column} IN (SELECT pk FROM session WHERE origin IN ({}))",
                marks.join(",")
            ));
        }
        if let Some(pks) = &self.sessions {
            // One parameter holds every key, however many sessions the
            // query selected.
            params.push(serde_json::Value::from(pks.clone()).to_string().into());
            clauses.push(format!(
                "{column} IN (SELECT value FROM json_each(?{}))",
                params.len()
            ));
        }
    }
}

/// Tokens by kind and cost over the selection, at the own grain: each model
/// response once, however many sessions show it.
pub fn usage(reader: &Reader<'_>, selection: &Selection) -> Result<UsageReport> {
    usage::report(reader, selection)
}

/// Each session's usage in the selection, own and with parents.
pub fn sessions(reader: &Reader<'_>, selection: &Selection) -> Result<Vec<SessionUsage>> {
    usage::per_session(reader, selection)
}

/// Sessions in `selection` that a current `hidden` note hides. They stay in
/// every figure; this says how many figures include.
pub(crate) fn hidden_sessions(reader: &Reader<'_>, selection: &Selection) -> Result<u64> {
    let (cond, params) = selection.condition();
    let sql = format!(
        "SELECT count(DISTINCT s.pk) FROM message m JOIN session s ON s.pk = m.session_pk
         WHERE {cond} AND EXISTS (
           SELECT 1 FROM a.note hn JOIN a.note_target ht ON ht.note_id = hn.id
           WHERE hn.kind = 'hidden' AND ht.ref = 'session:' || s.id
             AND NOT EXISTS (SELECT 1 FROM a.note later WHERE later.first_note = hn.first_note
                             AND later.commit_seq > hn.commit_seq)
             AND NOT EXISTS (SELECT 1 FROM a.note_relation w
                             WHERE w.kind = 'withdraws' AND w.other = 'note:' || hn.first_note))"
    );
    let n: i64 = reader
        .conn()?
        .query_row(&sql, rusqlite::params_from_iter(params), |r| r.get(0))?;
    Ok(n as u64)
}

#[cfg(test)]
mod tests;
