//! Deterministic profile facts. Usage and cost remain in their existing readers.
use super::{Derivation, Version};
use crate::{
    Result,
    measure::profiles::{CLUSTER_GAP_MS, Facts, RULE_VERSION, latency},
};
use rusqlite::{OptionalExtension, Transaction};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "profile",
    version: Version::Rules(RULE_VERSION),
    after: &[
        "duplicates",
        "session",
        "repository",
        "tool_calls",
        "title",
        "session_survival",
    ],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

fn run(tx: &Transaction<'_>, pk: i64) -> Result<()> {
    let mut statement=tx.prepare_cached("SELECT pk,id,role,written_by,at,words,stop_reason,duration_ms FROM message
        WHERE session_pk=?1 AND duplicate_of IS NULL ORDER BY order_input_id,seq,record_position,id")?;
    let messages = statement
        .query_map([pk], |r| {
            Ok(Message {
                role: r.get(2)?,
                author: r.get(3)?,
                at: r.get(4)?,
                words: r.get::<_, i64>(5)? as u64,
                duration: r.get(7)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let times: Vec<i64> = messages.iter().filter_map(|m| m.at).collect();
    let (first_at, last_at, source) = crate::measure::profiles::bounds(tx, pk)?;
    let event_times = if times.is_empty() {
        tx.prepare_cached(
            "SELECT at FROM event WHERE session_pk=?1 AND at IS NOT NULL ORDER BY at,id",
        )?
        .query_map([pk], |r| r.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?
    } else {
        vec![]
    };
    let clustered = clustered(if times.is_empty() {
        &event_times
    } else {
        &times
    });
    let mut thinking_messages = BTreeSet::new();
    let mut kinds = tx.prepare_cached("SELECT DISTINCT b.message_pk,b.kind FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.session_pk=?1 AND m.duplicate_of IS NULL")?;
    for row in kinds.query_map([pk], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
        let (message, kind) = row?;
        if kind == "thinking" {
            thinking_messages.insert(message);
        }
    }
    // Authored prose has stored word counts; no block body is needed here.
    let substantive_count: u64 = tx.query_row("SELECT count(*) FROM message m
        WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND m.role IN ('user','assistant')
        AND m.written_by<>'harness' AND m.words>2
        AND NOT EXISTS(SELECT 1 FROM block b WHERE b.message_pk=m.pk AND b.kind IN ('tool_call','tool_result','thinking'))", [pk], |r| r.get::<_,i64>(0).map(|n|n as u64))?;
    let last_text: Option<(String,String,Option<String>)> = tx.prepare_cached("SELECT id,role,stop_reason FROM message
        WHERE session_pk=?1 AND duplicate_of IS NULL AND role IN ('user','assistant') AND written_by<>'harness' AND words>0
        ORDER BY order_input_id DESC,seq DESC,record_position DESC,id DESC LIMIT 1")?.query_row([pk],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).optional()?;
    let mut actions = BTreeMap::<String, u64>::new();
    let mut tools = BTreeSet::new();
    let mut durations = vec![];
    let mut pending = 0;
    let mut unpaired_past_gap = 0;
    let mut final_error = false;
    let mut calls=tx.prepare_cached("SELECT t.tool,t.action,t.outcome,t.result_block_pk,coalesce(t.at,cm.at),rm.at
        FROM tool_call t JOIN block cb ON cb.pk=t.call_block_pk JOIN message cm ON cm.pk=cb.message_pk
        LEFT JOIN block rb ON rb.pk=t.result_block_pk LEFT JOIN message rm ON rm.pk=rb.message_pk
        WHERE t.session_pk=?1 AND cm.duplicate_of IS NULL
        ORDER BY cm.order_input_id,cm.seq,cm.record_position,cm.id,cb.idx,t.pk")?;
    let mut rows = calls.query([pk])?;
    while let Some(row) = rows.next()? {
        tools.insert(row.get::<_, String>(0)?);
        let action: Option<String> = row.get(1)?;
        *actions
            .entry(action.unwrap_or_else(|| "other".into()))
            .or_default() += 1;
        match row.get::<_, String>(2)?.as_str() {
            "error" => final_error = true,
            "ok" => final_error = false,
            _ => {}
        }
        let paired = row.get::<_, Option<i64>>(3)?.is_some();
        let a: Option<i64> = row.get(4)?;
        let b: Option<i64> = row.get(5)?;
        if paired {
            if let (Some(a), Some(b)) = (a, b) {
                durations.push((b - a).max(0));
            }
        } else {
            pending += 1;
            if a.zip(last_at).is_some_and(|(a, b)| b - a > CLUSTER_GAP_MS) {
                unpaired_past_gap += 1;
            }
        }
    }
    let mut agent_gaps = vec![];
    let mut user_gaps = vec![];
    let eligible: Vec<_> = messages
        .iter()
        .filter(|m| m.at.is_some() && m.eligible())
        .collect();
    for pair in eligible.windows(2) {
        let delta = (pair[1].at.unwrap() - pair[0].at.unwrap()).max(0);
        if pair[0].role == "user" && pair[1].role == "assistant" {
            agent_gaps.push(delta);
        }
        if pair[0].role == "assistant" && pair[1].role == "user" && delta <= 1_800_000 {
            user_gaps.push(delta);
        }
    }
    let users = messages
        .iter()
        .filter(|m| m.role == "user" && m.eligible())
        .count();
    let assistants = messages
        .iter()
        .filter(|m| m.role == "assistant" && m.eligible())
        .count();
    let n = messages.len();
    let call_count: u64 = actions.values().sum();
    let ratio = (call_count as f64 / n.max(1) as f64 * 10_000.0).round() / 10_000.0;
    let count = |names: &[&str]| {
        names
            .iter()
            .map(|name| actions.get(*name).copied().unwrap_or(0))
            .sum::<u64>()
    };
    let reads = count(&["file_read", "search", "web"]);
    let edits = count(&["file_write", "file_edit"]);
    let runs = count(&["shell", "git"]);
    let dispatch = count(&["subagent"]);
    let shape = if dispatch > 0 {
        "subagent_dispatch"
    } else if ratio < 0.1 && users + assistants > 0 {
        "chat"
    } else if reads >= 3 && edits == 0 && users <= 2 {
        "batch_review"
    } else if (edits > 0 || runs >= 3) && ratio >= 0.2 && n >= 10 {
        "agentic_loop"
    } else if reads > 0 || ratio >= 0.1 {
        "exploratory"
    } else {
        "unknown"
    };
    let (state, method) = if pending > 0 {
        ("tool_left", "pending_tool_blocks")
    } else if last_text.as_ref().is_some_and(|m| m.1 == "user") {
        ("question_left", "last_message_role")
    } else if final_error {
        ("error_left", "action_outcome")
    } else if last_text
        .as_ref()
        .is_some_and(|m| m.1 == "assistant" && m.2.as_deref() == Some("refusal"))
    {
        ("refused", "stop_reason")
    } else if last_text
        .as_ref()
        .is_some_and(|m| m.1 == "assistant" && m.2.as_deref() == Some("max_tokens"))
    {
        ("truncated", "stop_reason")
    } else {
        ("unknown", "no_signal")
    };
    let compaction_count: u64 = tx.query_row(
        "SELECT count(*) FROM event WHERE session_pk=?1 AND kind='compaction'",
        [pk],
        |r| r.get::<_, i64>(0).map(|n| n as u64),
    )?;
    let tool_active = durations.iter().sum();
    let facts = Facts {
        rule_version: RULE_VERSION,
        cluster_gap_ms: CLUSTER_GAP_MS,
        first_at,
        last_at,
        timestamp_source: source,
        timestamped_messages: times.len() as u64,
        message_count: n as u64,
        word_count: messages.iter().map(|m| m.words).sum(),
        substantive_count,
        compaction_count,
        total_duration_ms: messages
            .iter()
            .filter_map(|m| m.duration)
            .reduce(|a, b| a.saturating_add(b)),
        thinking_count: thinking_messages.len() as u64,
        wall_duration_ms: first_at.zip(last_at).map(|(a, b)| (b - a).max(0)),
        message_clustered_duration_ms: clustered,
        tool_active_duration_ms: tool_active,
        tool_latency: latency(durations),
        agent_response_latency: latency(agent_gaps),
        user_response_latency: latency(user_gaps),
        unpaired_tools_past_gap: unpaired_past_gap,
        tools: tools.into_iter().collect(),
        tool_call_count: call_count,
        actions,
        workflow_shape: shape.into(),
        workflow_features: [
            ("message_count", n as f64),
            ("compaction_count", compaction_count as f64),
            ("tool_call_count", call_count as f64),
            ("tool_ratio", ratio),
            ("read_count", reads as f64),
            ("edit_count", edits as f64),
            ("run_count", runs as f64),
            ("dispatch_count", dispatch as f64),
            ("user_turn_count", users as f64),
            ("assistant_turn_count", assistants as f64),
            (
                "thinking_ratio",
                (thinking_messages.len() as f64 / n.max(1) as f64 * 10_000.0).round() / 10_000.0,
            ),
            ("tool_active_duration_ms", tool_active as f64),
        ]
        .into_iter()
        .map(|(k, v)| (k.into(), v))
        .collect(),
        final_observed_state: state.into(),
        final_observed_state_method: method.into(),
        final_message: last_text.map(|m| m.0),
    };
    crate::measure::profiles::write_facts(tx, pk, &facts)?;
    tx.prepare_cached("DELETE FROM session_facet WHERE session_pk=?1")?
        .execute([pk])?;
    for (family, sql) in [
        (
            "roles",
            "SELECT DISTINCT role FROM message WHERE session_pk=?1 AND duplicate_of IS NULL",
        ),
        (
            "written_by",
            "SELECT DISTINCT written_by FROM message WHERE session_pk=?1 AND duplicate_of IS NULL",
        ),
        (
            "block_kinds",
            "SELECT DISTINCT b.kind FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.session_pk=?1 AND m.duplicate_of IS NULL",
        ),
        (
            "actions",
            "SELECT DISTINCT t.action FROM tool_call t JOIN block b ON b.pk=t.call_block_pk JOIN message m ON m.pk=b.message_pk WHERE t.session_pk=?1 AND m.duplicate_of IS NULL AND t.action IS NOT NULL",
        ),
        (
            "flags",
            "SELECT DISTINCT CASE WHEN b.kind='thinking' THEN 'thinking' WHEN b.kind='tool_call' THEN 'tool_call' ELSE 'attachment' END FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.session_pk=?1 AND m.duplicate_of IS NULL AND b.kind IN ('thinking','tool_call','image','file')",
        ),
    ] {
        tx.prepare_cached(&format!("WITH values_for_session(value) AS ({sql}) INSERT INTO session_facet SELECT ?1,?2,value FROM values_for_session"))?.execute(rusqlite::params![pk,family])?;
    }
    Ok(())
}

struct Message {
    role: String,
    author: String,
    at: Option<i64>,
    words: u64,
    duration: Option<i64>,
}
impl Message {
    fn eligible(&self) -> bool {
        matches!(self.role.as_str(), "user" | "assistant") && self.author != "harness"
    }
}

fn clustered(times: &[i64]) -> i64 {
    let mut total = 0;
    let mut interval = None::<(i64, i64, i64)>;
    for &time in times {
        interval = Some(match interval {
            Some((min, max, previous)) if time.abs_diff(previous) <= CLUSTER_GAP_MS as u64 => {
                (min.min(time), max.max(time), time)
            }
            Some((min, max, _)) => {
                total += max - min;
                (time, time, time)
            }
            None => (time, time, time),
        });
    }
    total + interval.map_or(0, |(min, max, _)| max - min)
}

#[cfg(test)]
mod tests;
