//! Tool-call measures: outcomes kept apart, which call follows which, and
//! the commands whose latest run in a repository failed.

use super::Selection;
use crate::{Reader, Result};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};

/// Tool outcomes kept apart: they never collapse into one success figure.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Outcomes {
    pub ok: u64,
    pub error: u64,
    /// The call has no result record yet, and nothing later in its session
    /// rules one out; read as pending while the session is live.
    pub no_result: u64,
    /// The call has no result record and the session went on past it, so
    /// none will come.
    pub passed_without_result: u64,
    /// Any other unknown outcome.
    pub unknown: u64,
}

pub fn outcomes(reader: &Reader<'_>, selection: &Selection) -> Result<Outcomes> {
    let (cond, params) = selection.condition();
    Ok(reader.conn()?.query_row(
        &format!(
            "SELECT sum(t.outcome = 'ok'), sum(t.outcome = 'error'),
                    sum(t.outcome = 'unknown' AND t.outcome_reason = 'no_result'),
                    sum(t.outcome = 'unknown' AND t.outcome_reason = 'passed_without_result'),
                    sum(t.outcome = 'unknown'
                        AND coalesce(t.outcome_reason, '') NOT IN ('no_result', 'passed_without_result'))
             FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
             JOIN message m ON m.pk = b.message_pk
             WHERE {cond}"
        ),
        rusqlite::params_from_iter(params),
        |r| {
            let n = |i| r.get::<_, Option<i64>>(i).map(|v| v.unwrap_or(0) as u64);
            Ok(Outcomes {
                ok: n(0)?,
                error: n(1)?,
                no_result: n(2)?,
                passed_without_result: n(3)?,
                unknown: n(4)?,
            })
        },
    )?)
}

/// Normalized stages extracted from the submitted text of recorded shell calls.
#[derive(Debug, Default, Serialize)]
pub struct ShellStages {
    pub shell_calls: u64,
    pub calls_without_recognized_stages: u64,
    pub stages: BTreeMap<String, StageCount>,
}

#[derive(Debug, Default, Serialize)]
pub struct StageCount {
    pub occurrences: u64,
    pub calls: u64,
    pub sessions: u64,
    #[serde(skip)]
    last_seen: Option<(i64, i64)>,
}

pub fn shell_stages(reader: &Reader<'_>, selection: &Selection) -> Result<ShellStages> {
    let (condition, params) = selection.condition();
    // Visit the selected sessions in index order without sorting their command text.
    let mut statement = reader.conn()?.prepare(&format!(
        "SELECT t.session_pk, t.pk, t.command
         FROM tool_call t INDEXED BY tool_call_session
         JOIN block b ON b.pk = t.call_block_pk
         JOIN message m ON m.pk = b.message_pk AND m.session_pk = t.session_pk
         WHERE t.action = 'shell' AND {condition}
         ORDER BY t.session_pk"
    ))?;
    let mut rows = statement.query(rusqlite::params_from_iter(params))?;
    let mut report = ShellStages::default();
    while let Some(row) = rows.next()? {
        reader.check()?;
        let session: i64 = row.get(0)?;
        let call: i64 = row.get(1)?;
        let command: Option<String> = row.get(2)?;
        let stages = command
            .as_deref()
            .map(polylogue_model::command::effective_stages)
            .unwrap_or_default();
        report.shell_calls += 1;
        if stages.is_empty() {
            report.calls_without_recognized_stages += 1;
        }
        for stage in stages {
            let count = report.stages.entry(stage.normalized()).or_default();
            count.occurrences += 1;
            // A call contributes one row, and the rows of a session are contiguous.
            if count.last_seen != Some((session, call)) {
                count.calls += 1;
                if count
                    .last_seen
                    .is_none_or(|(previous, _)| previous != session)
                {
                    count.sessions += 1;
                }
                count.last_seen = Some((session, call));
            }
        }
    }
    Ok(report)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Transition {
    pub from: String,
    pub to: String,
    pub count: u64,
    /// Of `count`, how many followed a failed `from` call.
    pub after_failure: u64,
}

/// A failed call and the call that came next in the same session, as both
/// tools recorded them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FailureExample {
    pub session: String,
    pub failed: String,
    pub next: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Process {
    pub sessions: u64,
    pub calls: u64,
    /// Calls with no time; order still comes from their place in the
    /// transcript, never from the clock.
    pub calls_without_time: u64,
    pub outcomes: Outcomes,
    pub transitions: Vec<Transition>,
    /// The most frequent runs of three consecutive actions.
    pub common_sequences: Vec<(String, u64)>,
    pub failure_examples: Vec<FailureExample>,
}

/// Directly-follows relations between tool calls within each session, in
/// transcript order (message order key, then block position): two calls
/// with the same time keep their transcript order and nothing is inferred
/// from timestamps. Calls are named by action and command or tool.
pub fn process(reader: &Reader<'_>, selection: &Selection, top: usize) -> Result<Process> {
    let (cond, params) = selection.condition();
    let mut stmt = reader.conn()?.prepare(&format!(
        "SELECT s.id, t.action, coalesce(t.command_norm, t.tool), t.outcome, t.at
         FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
         JOIN message m ON m.pk = b.message_pk JOIN session s ON s.pk = m.session_pk
         WHERE {cond}
         ORDER BY m.session_pk, m.session_order, m.id, b.position"
    ))?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params.iter()), |r| {
        <(String, String, String, String, Option<i64>)>::try_from(r)
    })?;
    let mut pairs: HashMap<(String, String), (u64, u64)> = HashMap::new();
    let mut triples: HashMap<String, u64> = HashMap::new();
    let mut examples = vec![];
    let (mut calls, mut untimed, mut sessions) = (0u64, 0u64, 0u64);
    let mut prev: Option<(String, String, String)> = None;
    let mut prev2: Option<String> = None;
    for row in rows {
        let (session, action, what, outcome, at) = row?;
        calls += 1;
        untimed += u64::from(at.is_none());
        let name = format!("{action}:{what}");
        match &prev {
            Some((s, p, o)) if *s == session => {
                let e = pairs.entry((p.clone(), name.clone())).or_default();
                e.0 += 1;
                if o == "error" {
                    e.1 += 1;
                    if examples.len() < top {
                        examples.push(FailureExample {
                            session: session.clone(),
                            failed: p.clone(),
                            next: name.clone(),
                        });
                    }
                }
                if let Some(p2) = &prev2 {
                    *triples.entry(format!("{p2} → {p} → {name}")).or_default() += 1;
                }
                prev2 = Some(p.clone());
            }
            _ => {
                sessions += 1;
                prev2 = None;
            }
        }
        prev = Some((session, name, outcome));
    }
    let mut transitions: Vec<Transition> = pairs
        .into_iter()
        .map(|((from, to), (count, after_failure))| Transition {
            from,
            to,
            count,
            after_failure,
        })
        .collect();
    transitions.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.from.cmp(&b.from)));
    transitions.truncate(top);
    let mut common_sequences: Vec<(String, u64)> = triples.into_iter().collect();
    common_sequences.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    common_sequences.truncate(top);
    Ok(Process {
        sessions,
        calls,
        calls_without_time: untimed,
        outcomes: outcomes(reader, selection)?,
        transitions,
        common_sequences,
        failure_examples: examples,
    })
}

/// A command whose most recent run in a repository failed: what is left to
/// fix there. A run counts by its
/// decided outcome (ok or error); a run with an unknown outcome decides
/// nothing.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FailingCommand {
    /// The repository's root commit, or `cwd:<dir>` for sessions in no
    /// known repository.
    pub repository: String,
    /// Its remote or name, when known.
    pub repo: Option<String>,
    pub command: String,
    /// The failing run: when, in which session, with which exit code.
    pub at: Option<i64>,
    pub session: String,
    pub exit_code: Option<i64>,
    /// Failed runs since the last run that succeeded.
    pub failures_since_ok: u64,
    /// The last run that succeeded, if any did.
    pub last_ok_at: Option<i64>,
}

/// Every repository's unresolved commands in the selection, most recent first.
pub fn failing_commands(reader: &Reader<'_>, selection: &Selection) -> Result<Vec<FailingCommand>> {
    let (cond, params) = selection.condition();
    let no_match_programs = polylogue_model::command::NO_MATCH_PROGRAMS
        .iter()
        .map(|program| format!("'{program}'"))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "WITH runs AS (
           SELECT coalesce(s.repo_root, 'cwd:' || s.cwd) AS repository, s.repo AS repo,
                  t.command_norm AS command, t.outcome, t.exit_code, coalesce(t.at, m.at) AS at,
                  t.pk AS call, s.id AS session
           FROM tool_call t JOIN block b ON b.pk = t.call_block_pk
           JOIN message m ON m.pk = b.message_pk JOIN session s ON s.pk = t.session_pk
           WHERE t.command_norm IS NOT NULL AND t.outcome IN ('ok', 'error')
             AND NOT (t.exit_code IS 1 AND (
                 substr(t.command_norm, 1, instr(t.command_norm || ' ', ' ') - 1)
                   IN ({no_match_programs}) OR t.command_norm = 'git grep'))
             AND coalesce(s.repo_root, s.cwd) IS NOT NULL AND {cond}),
         ranked AS (
           SELECT *, row_number() OVER (PARTITION BY repository, command
                                         ORDER BY at DESC NULLS LAST, call DESC) AS rn,
                  max(CASE WHEN outcome = 'ok' THEN at END)
                    OVER (PARTITION BY repository, command) AS last_ok
           FROM runs)
         SELECT repository, repo, command, at, session, exit_code, last_ok,
                (SELECT count(*) FROM runs r WHERE r.repository = ranked.repository
                   AND r.command = ranked.command AND r.outcome = 'error'
                   AND (ranked.last_ok IS NULL OR r.at > ranked.last_ok))
         FROM ranked WHERE rn = 1 AND outcome = 'error'
         ORDER BY at DESC NULLS LAST, repository, command"
    );
    let mut stmt = reader.conn()?.prepare(&sql)?;
    let rows = stmt.query_map(rusqlite::params_from_iter(params), |r| {
        Ok(FailingCommand {
            repository: r.get(0)?,
            repo: r.get(1)?,
            command: r.get(2)?,
            at: r.get(3)?,
            session: r.get(4)?,
            exit_code: r.get(5)?,
            last_ok_at: r.get(6)?,
            failures_since_ok: r.get::<_, i64>(7)? as u64,
        })
    })?;
    Ok(rows.collect::<rusqlite::Result<_>>()?)
}
