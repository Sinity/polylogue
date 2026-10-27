//! Deterministic profile facts. Usage and cost remain in their existing readers.
use super::{Derivation, Version};
use crate::{
    Result,
    measure::profiles::{CLUSTER_GAP_MS, Facts, RULE_VERSION, latency},
};
use rusqlite::{Transaction, params};
use std::collections::{BTreeMap, BTreeSet};

pub(super) const DERIVATION: Derivation = Derivation {
    name: "profile",
    version: Version::Rules(RULE_VERSION),
    after: &["duplicates", "session", "repository", "tool_calls"],
    run: |tx, pk, _| run(tx, pk),
    finish: None,
};

fn run(tx: &Transaction<'_>, pk: i64) -> Result<()> {
    let mut statement=tx.prepare_cached("SELECT pk,id,role,written_by,prompt_kind,at,words,stop_reason,duration_ms FROM message
        WHERE session_pk=?1 AND duplicate_of IS NULL ORDER BY order_input_id,seq,record_position,id")?;
    let messages = statement
        .query_map([pk], |r| {
            Ok(Message {
                pk: r.get(0)?,
                id: r.get(1)?,
                role: r.get(2)?,
                author: r.get(3)?,
                prompt: r.get(4)?,
                at: r.get(5)?,
                words: r.get::<_, i64>(6)? as u64,
                stop: r.get(7)?,
                duration: r.get(8)?,
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let times: Vec<i64> = messages.iter().filter_map(|m| m.at).collect();
    let (start, last, title, origin, repo): (
        Option<i64>,
        Option<i64>,
        Option<String>,
        String,
        Option<String>,
    ) = tx.query_row(
        "SELECT started_at,last_activity_at,title,origin,repo FROM session WHERE pk=?1",
        [pk],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
    )?;
    let event_times = tx
        .prepare_cached(
            "SELECT at FROM event WHERE session_pk=?1 AND at IS NOT NULL ORDER BY at,id",
        )?
        .query_map([pk], |r| r.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let (first_at, last_at, source) = if !times.is_empty() {
        (
            times.iter().min().copied(),
            times.iter().max().copied(),
            "provider_supplied",
        )
    } else if !event_times.is_empty() {
        (
            event_times.first().copied(),
            event_times.last().copied(),
            "session_events_fallback",
        )
    } else if start.is_some() || last.is_some() {
        (start.or(last), last.or(start), "session_timestamp_fallback")
    } else {
        (None, None, "absent")
    };
    let clustered = clustered(if times.is_empty() {
        &event_times
    } else {
        &times
    });
    let authors: BTreeMap<i64, &Message> = messages.iter().map(|m| (m.pk, m)).collect();
    let mut prompt_text = BTreeMap::<i64, String>::new();
    let mut tool_messages = BTreeSet::new();
    let mut thinking_messages = BTreeSet::new();
    let mut kinds = tx.prepare_cached("SELECT DISTINCT b.message_pk,b.kind FROM block b JOIN message m ON m.pk=b.message_pk WHERE b.session_pk=?1 AND m.duplicate_of IS NULL")?;
    for row in kinds.query_map([pk], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))? {
        let (message, kind) = row?;
        if kind == "thinking" {
            thinking_messages.insert(message);
        }
        if matches!(kind.as_str(), "tool_call" | "tool_result") {
            tool_messages.insert(message);
        }
    }
    let mut topic = None;
    let mut last_text = None;
    let mut masker = polylogue_model::masking::MessageMasker::default();
    // Unmarked tool output cannot change masking state. Read only authored
    // prose and messages whose PEM markers can affect the following prose.
    let mut text_blocks=tx.prepare_cached("SELECT b.pk,b.message_pk FROM message m JOIN block b ON b.message_pk=m.pk
        WHERE m.session_pk=?1 AND m.duplicate_of IS NULL AND
          (m.pem<>0 OR (b.kind='text' AND m.role IN ('user','assistant') AND m.written_by<>'harness'))
        ORDER BY m.order_input_id,m.seq,m.record_position,m.id,b.idx")?;
    let mut text_rows = text_blocks.query([pk])?;
    while let Some(row) = text_rows.next()? {
        let message_pk: i64 = row.get(1)?;
        let mut block = crate::ordered_blocks::one(tx, row.get(0)?)?;
        masker.block(&mut block);
        if let polylogue_model::Block::Text { text } = &block
            && let Some(m) = authors.get(&message_pk)
            && !text.trim().is_empty()
            && m.eligible()
        {
            last_text = Some(*m);
            prompt_text.entry(message_pk).or_default().push_str(text);
        }
    }
    let substantive = |m: &&Message| {
        m.eligible()
            && !tool_messages.contains(&m.pk)
            && !thinking_messages.contains(&m.pk)
            && prompt_text
                .get(&m.pk)
                .is_some_and(|s| s.trim().chars().count() > 10)
    };
    let substantive_count = messages.iter().filter(substantive).count() as u64;
    let prompts = || {
        messages
            .iter()
            .filter(|m| m.role == "user" && m.eligible() && m.prompt.as_deref() != Some("command"))
    };
    for m in prompts().filter(substantive).chain(prompts()) {
        if let Some(text) = prompt_text.get(&m.pk)
            && let Some(label) = clean_topic(text, 120)
        {
            topic = Some(label);
            break;
        }
    }
    let cleaned_title = title
        .as_deref()
        .and_then(|s| clean_topic(&polylogue_model::masking::mask(s), 120));
    let topic_source = if topic.is_some() {
        "first_substantive_user_turn"
    } else if cleaned_title.is_some() {
        "cleaned_title"
    } else {
        "absent"
    };
    if topic.is_none() {
        topic = cleaned_title;
    }
    let repo = crate::measure::profiles::repositories(tx, pk)?
        .0
        .into_iter()
        .find_map(|r| r.label)
        .or(repo);
    let mut topic_source = topic_source.to_owned();
    if origin == "codex"
        && topic_source == "first_substantive_user_turn"
        && let (Some(text), Some(repo)) = (&topic, &repo)
    {
        let repo = repo
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or(repo)
            .trim_end_matches(".git");
        topic = Some(format!(
            "{repo}: {}",
            text.chars().take(60).collect::<String>().trim_end()
        ));
        topic_source = "repo_plus_first_user_turn".into();
    }
    let mut actions = BTreeMap::<String, u64>::new();
    let mut tools = BTreeSet::new();
    let mut durations = vec![];
    let mut pending = 0;
    let mut stuck = 0;
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
                stuck += 1;
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
    } else if last_text.is_some_and(|m| m.role == "user") {
        ("question_left", "last_message_role")
    } else if final_error {
        ("error_left", "action_outcome")
    } else if last_text
        .is_some_and(|m| m.role == "assistant" && m.stop.as_deref() == Some("refusal"))
    {
        ("refused", "stop_reason")
    } else if last_text
        .is_some_and(|m| m.role == "assistant" && m.stop.as_deref() == Some("max_tokens"))
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
        topic,
        topic_source,
        first_at,
        last_at,
        timestamp_source: source.into(),
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
        engaged_duration_ms: clustered,
        tool_active_duration_ms: tool_active,
        tool_latency: latency(durations),
        agent_response_latency: latency(agent_gaps),
        user_response_latency: latency(user_gaps),
        stuck_tool_count: stuck,
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
        terminal_state: state.into(),
        terminal_state_method: method.into(),
        terminal_message: last_text.map(|m| m.id.clone()),
    };
    tx.execute("INSERT INTO session_profile(session_pk,facts) VALUES (?1,?2) ON CONFLICT(session_pk) DO UPDATE SET facts=?2",
        params![pk,serde_json::to_string(&facts)?])?;
    Ok(())
}

struct Message {
    pk: i64,
    id: String,
    role: String,
    author: String,
    prompt: Option<String>,
    at: Option<i64>,
    words: u64,
    stop: Option<String>,
    duration: Option<i64>,
}
impl Message {
    fn eligible(&self) -> bool {
        matches!(self.role.as_str(), "user" | "assistant") && self.author != "harness"
    }
}

fn clean_topic(text: &str, width: usize) -> Option<String> {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let prefixes = [
        "Caveat: The messages below were generated by the user while running local commands.",
        "Caveat: The messages below are were generated by the user while running local commands.",
        "[SUGGESTION MODE: Suggest what the user might naturally type next into Claude Code.",
        "This session is being continued from a previous session that ran out of context.",
        "Base directory for this skill:",
    ];
    let stripped = prefixes
        .iter()
        .find_map(|p| text.strip_prefix(p))
        .unwrap_or(&text)
        .trim_start_matches([' ', ':', '-']);
    let value = stripped.chars().take(width).collect::<String>();
    (!value.is_empty()).then_some(value)
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
