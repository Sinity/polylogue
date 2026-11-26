//! Hand-written facts for each golden case: `facts.json` beside the input,
//! written by a person from the input file and the docs alone, never from
//! the parser. It holds what a reader of the archive sees (sessions and
//! their links, messages in order with role, time, parent and text, tool
//! calls with their results) and nothing a person cannot work out (hashes,
//! offsets, counts), so a refresh of `expected.json` cannot change it. The
//! format is described in `docs/testing.md` (Goldens).

use super::golden::{cases, parse_case};
use polylogue_model::{Block, IdKind, Output, Session, Timestamp};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::fs;

#[test]
fn each_case_matches_its_hand_written_facts() {
    let mut failures = Vec::new();
    let mut compared = 0;
    for (origin, case) in cases() {
        let facts_path = case.join("facts.json");
        compared += 1;
        let text = match fs::read_to_string(&facts_path) {
            Ok(text) => text,
            Err(error) => {
                failures.push(format!("{}: {error}", facts_path.display()));
                continue;
            }
        };
        let facts: Value = match serde_json::from_str(&text) {
            Ok(facts) => facts,
            Err(error) => {
                failures.push(format!("{}: {error}", facts_path.display()));
                continue;
            }
        };
        let facts = match normalized(facts) {
            Ok(facts) => facts,
            Err(error) => {
                failures.push(format!("{}: {error}", facts_path.display()));
                continue;
            }
        };
        let step = match parse_case(&origin, &case) {
            Ok(step) => step,
            Err(error) => {
                failures.push(format!("{}: {error}", case.display()));
                continue;
            }
        };
        let parsed = projection(&step.output, facts.get("events").is_some());
        let mut differences = Vec::new();
        differ("", &facts, &parsed, &mut differences);
        if !differences.is_empty() {
            failures.push(format!(
                "{}: the parse differs from the facts\n{}",
                facts_path.display(),
                differences.join("\n")
            ));
        }
    }
    assert!(compared > 0, "no golden cases found");
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// The parse in the shape of `facts.json`. Events are part of it only when
/// the facts list them.
fn projection(output: &Output, with_events: bool) -> Value {
    let mut sessions: BTreeMap<String, Session> = BTreeMap::new();
    for session in &output.sessions {
        sessions
            .entry(session.id.to_string())
            .and_modify(|known| known.merge(session))
            .or_insert_with(|| session.clone());
    }
    // Messages by session in output order; a message sent again replaces its
    // earlier row in place, as the writer does.
    let mut messages: BTreeMap<String, Vec<(&polylogue_model::MessageRow, bool)>> = BTreeMap::new();
    for row in &output.messages {
        let rows = messages.entry(row.message.session.to_string()).or_default();
        let content_named = row.id_kind == IdKind::Content;
        match rows
            .iter_mut()
            .find(|(known, _)| known.message.id == row.message.id)
        {
            Some(known) => *known = (row, content_named),
            None => rows.push((row, content_named)),
        }
    }
    let mut ids: Vec<String> = sessions.keys().cloned().collect();
    ids.extend(
        messages
            .keys()
            .filter(|id| !sessions.contains_key(*id))
            .cloned(),
    );
    ids.sort();

    let mut session_values = Vec::new();
    for id in ids {
        let mut value = Map::new();
        value.insert("id".into(), json!(id));
        let session = sessions.get(&id);
        value.insert(
            "kind".into(),
            session.map_or(Value::Null, |s| json!(s.kind.as_str())),
        );
        if let Some(title) = session.and_then(|s| s.title.as_ref()) {
            value.insert("title".into(), json!(title.text));
        }
        let rows = messages.get(&id).map(Vec::as_slice).unwrap_or_default();
        let ended: BTreeMap<&str, &polylogue_model::ToolOutcome> = output
            .background
            .iter()
            .filter(|update| update.session.to_string() == id)
            .filter_map(|update| Some((update.call_id.as_deref()?, &update.outcome)))
            .collect();
        value.insert(
            "messages".into(),
            Value::Array(session_messages(rows, &ended)),
        );
        session_values.push(Value::Object(value));
    }

    let mut facts = Map::new();
    facts.insert("sessions".into(), Value::Array(session_values));
    let mut links: Vec<Value> = output
        .links
        .iter()
        .map(|link| {
            let mut value = Map::new();
            value.insert("kind".into(), json!(link.kind.as_str()));
            value.insert("parent".into(), json!(link.parent.to_string()));
            value.insert("child".into(), json!(link.child.to_string()));
            if let Some(how) = link.how {
                value.insert("how".into(), json!(how.as_str()));
            }
            Value::Object(value)
        })
        .collect();
    if !links.is_empty() {
        links.sort_by_key(link_order);
        facts.insert("links".into(), Value::Array(links));
    }
    let documents: Vec<Value> = output
        .documents
        .iter()
        .map(|row| {
            let mut value = Map::new();
            if let Some(title) = &row.document.title {
                value.insert("title".into(), json!(title));
            }
            value.insert("text".into(), json!(joined_text(&row.document.blocks)));
            Value::Object(value)
        })
        .collect();
    if !documents.is_empty() {
        facts.insert("documents".into(), Value::Array(documents));
    }
    if with_events {
        let events = output
            .events
            .iter()
            .map(|event| {
                json!({
                    "session": event.session.as_ref().map(ToString::to_string),
                    "kind": event.kind,
                    "at": event.at.map(|at| at.to_string()),
                })
            })
            .collect();
        facts.insert("events".into(), Value::Array(events));
    }
    let lookups: Vec<Value> = output
        .repo_lookups
        .iter()
        .map(|lookup| {
            json!({
                "path": lookup.path,
                "remote": lookup.remote,
                "root_commits": lookup.root_commits,
            })
        })
        .collect();
    if !lookups.is_empty() {
        facts.insert("repo_lookups".into(), Value::Array(lookups));
    }
    Value::Object(facts)
}

/// One session's messages, with each call's result found by its call ID
/// among all the session's results. A call that ran on after its result was
/// written (a background task) shows the outcome it ended with.
fn session_messages(
    rows: &[(&polylogue_model::MessageRow, bool)],
    ended: &BTreeMap<&str, &polylogue_model::ToolOutcome>,
) -> Vec<Value> {
    let mut results: BTreeMap<&str, Vec<Value>> = BTreeMap::new();
    let mut call_ids = Vec::new();
    for (row, _) in rows {
        for block in &row.message.blocks {
            match block {
                Block::ToolResult {
                    call_id: Some(call_id),
                    outcome,
                    output,
                    ..
                } => results.entry(call_id).or_default().push(result_value(
                    ended.get(call_id.as_str()).copied().unwrap_or(outcome),
                    output.as_deref(),
                )),
                Block::ToolCall {
                    call_id: Some(call_id),
                    ..
                } => call_ids.push(call_id.as_str()),
                _ => {}
            }
        }
    }
    let position = |id: &polylogue_model::MessageId| {
        rows.iter()
            .position(|(row, content_named)| *content_named && &row.message.id == id)
    };
    rows.iter()
        .map(|(row, content_named)| {
            let message = &row.message;
            let mut value = Map::new();
            value.insert(
                "id".into(),
                if *content_named {
                    Value::Null
                } else {
                    json!(message.id.to_string())
                },
            );
            value.insert("role".into(), json!(message.role.as_str()));
            value.insert(
                "at".into(),
                message.at.map_or(Value::Null, |at| json!(at.to_string())),
            );
            value.insert(
                "parent".into(),
                match &message.parent {
                    None => Value::Null,
                    Some(parent) => match position(parent) {
                        Some(index) => json!(index + 1),
                        None => json!(parent.to_string()),
                    },
                },
            );
            value.insert("text".into(), json!(joined_text(&message.blocks)));
            let thinking: Vec<&str> = message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    Block::Thinking {
                        text: Some(text), ..
                    } if !text.is_empty() => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            if !thinking.is_empty() {
                value.insert("thinking".into(), json!(thinking.join("\n\n")));
            }
            let calls: Vec<Value> = message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    Block::ToolCall {
                        call_id,
                        tool,
                        command,
                        path,
                        ..
                    } => {
                        let mut call = Map::new();
                        call.insert("name".into(), json!(tool));
                        if let Some(command) = command {
                            call.insert("command".into(), json!(command));
                        }
                        if let Some(path) = path {
                            call.insert("path".into(), json!(path));
                        }
                        let found = call_id
                            .as_deref()
                            .and_then(|id| results.get(id))
                            .map(Vec::as_slice)
                            .unwrap_or_default();
                        call.insert(
                            "result".into(),
                            match found {
                                [] => Value::Null,
                                [one] => one.clone(),
                                several => Value::Array(several.to_vec()),
                            },
                        );
                        Some(Value::Object(call))
                    }
                    _ => None,
                })
                .collect();
            if !calls.is_empty() {
                value.insert("calls".into(), Value::Array(calls));
            }
            let unpaired: Vec<Value> = message
                .blocks
                .iter()
                .filter_map(|block| match block {
                    Block::ToolResult {
                        call_id,
                        outcome,
                        output,
                        ..
                    } if call_id.as_deref().is_none_or(|id| !call_ids.contains(&id)) => {
                        let outcome = call_id
                            .as_deref()
                            .and_then(|id| ended.get(id).copied())
                            .unwrap_or(outcome);
                        Some(result_value(outcome, output.as_deref()))
                    }
                    _ => None,
                })
                .collect();
            if !unpaired.is_empty() {
                value.insert("results".into(), Value::Array(unpaired));
            }
            Value::Object(value)
        })
        .collect()
}

fn result_value(outcome: &polylogue_model::ToolOutcome, output: Option<&str>) -> Value {
    let outcome = serde_json::to_value(outcome).expect("an outcome serializes");
    json!({"outcome": outcome["kind"], "text": output})
}

/// The text parts of a message or document, in order, empty ones left out,
/// joined by a blank line.
fn joined_text(blocks: &[Block]) -> String {
    blocks
        .iter()
        .filter_map(|block| match block {
            Block::Text { text } if !text.is_empty() => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn link_order(link: &Value) -> (String, String, String) {
    let field = |name: &str| link[name].as_str().unwrap_or_default().to_owned();
    (field("parent"), field("child"), field("kind"))
}

/// The facts with every time as the instant the parse renders and sessions
/// and links in the order the projection gives them.
fn normalized(mut facts: Value) -> Result<Value, String> {
    let sessions = facts
        .get_mut("sessions")
        .and_then(Value::as_array_mut)
        .ok_or("facts have no sessions list")?;
    for session in sessions.iter_mut() {
        for message in session
            .get_mut("messages")
            .and_then(Value::as_array_mut)
            .ok_or("a session has no messages list")?
        {
            normalize_time(message)?;
        }
    }
    sessions.sort_by_key(|session| session["id"].as_str().unwrap_or_default().to_owned());
    if let Some(links) = facts.get_mut("links").and_then(Value::as_array_mut) {
        links.sort_by_key(link_order);
    }
    if let Some(events) = facts.get_mut("events").and_then(Value::as_array_mut) {
        for event in events {
            normalize_time(event)?;
        }
    }
    Ok(facts)
}

fn normalize_time(holder: &mut Value) -> Result<(), String> {
    let Some(at) = holder.get_mut("at") else {
        return Err(format!("no `at` in {holder}"));
    };
    if !at.is_null() {
        let instant = written_time(at).ok_or_else(|| format!("unreadable time {at}"))?;
        *at = json!(instant.to_string());
    }
    Ok(())
}

/// A time as the facts write it: RFC 3339 text with an offset, or a number
/// of seconds since 1970, or of milliseconds when it is 10^11 or more.
/// Decimal digits are read exactly, not through a float.
fn written_time(at: &Value) -> Option<Timestamp> {
    match at {
        Value::String(text) => Timestamp::parse_rfc3339(text),
        Value::Number(number) => {
            let text = number.to_string();
            let (whole, fraction) = text.split_once('.').unwrap_or((&text, ""));
            let whole: i128 = whole.parse().ok()?;
            let digits = format!("{fraction:0<9}");
            let fraction: i128 = digits.get(..9)?.parse().ok()?;
            let fraction = if whole < 0 { -fraction } else { fraction };
            let nanos = if whole.abs() >= 100_000_000_000 {
                whole * 1_000_000 + fraction / 1_000
            } else {
                whole * 1_000_000_000 + fraction
            };
            Timestamp::from_unix_ns(nanos)
        }
        _ => None,
    }
}

/// Every place where the facts and the parse differ, by JSON path.
fn differ(path: &str, facts: &Value, parsed: &Value, out: &mut Vec<String>) {
    match (facts, parsed) {
        (Value::Object(want), Value::Object(got)) => {
            let mut keys: Vec<&String> = want.keys().chain(got.keys()).collect();
            keys.sort();
            keys.dedup();
            for key in keys {
                let path = format!("{path}.{key}");
                match (want.get(key), got.get(key)) {
                    (Some(want), Some(got)) => differ(&path, want, got, out),
                    (Some(want), None) => out.push(format!("{path}: facts {want}, parse has none")),
                    (None, Some(got)) => out.push(format!("{path}: facts have none, parse {got}")),
                    (None, None) => {}
                }
            }
        }
        (Value::Array(want), Value::Array(got)) => {
            for index in 0..want.len().max(got.len()) {
                let path = format!("{path}[{index}]");
                match (want.get(index), got.get(index)) {
                    (Some(want), Some(got)) => differ(&path, want, got, out),
                    (Some(want), None) => out.push(format!("{path}: facts {want}, parse has none")),
                    (None, Some(got)) => out.push(format!("{path}: facts have none, parse {got}")),
                    (None, None) => {}
                }
            }
        }
        (want, got) if want != got => out.push(format!("{path}: facts {want}, parse {got}")),
        _ => {}
    }
}
