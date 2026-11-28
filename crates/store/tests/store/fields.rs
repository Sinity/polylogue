//! Each query field reads its own value: selected as a column over Morrow,
//! next to fields whose values differ from it on the same rows.

use crate::{program, rows, submit};
use polylogue_model::user::{Delivery, UserWrite};
use polylogue_model::{DeliveryEvent, DeliveryOutcome, DeliverySender};
use polylogue_query::ast::Unit;
use polylogue_query::fields::{FIELDS, Kind};
use polylogue_testkit::{Morrow, Set, session, short};
use serde_json::{Value, json};

/// `rows` as tuples of the named fields, Morrow's session IDs as `#n`,
/// sorted: the reply's multiset of values where the docs promise no order.
fn tuples(rows: &[Value], fields: &[&str]) -> Vec<Vec<Value>> {
    let mut out: Vec<Vec<Value>> = rows
        .iter()
        .map(|row| {
            fields
                .iter()
                .map(|field| match &row[*field] {
                    Value::String(text) => json!(short(text)),
                    value => value.clone(),
                })
                .collect()
        })
        .collect();
    out.sort_by_key(|tuple| serde_json::to_string(tuple).unwrap());
    out
}

/// Every declared field of the four units with stored rows runs as a
/// selected column over an archive the engine built.
#[test]
fn every_field_of_sessions_messages_tool_calls_and_deliveries_runs_as_a_column() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    let mut refused = Vec::new();
    for field in FIELDS.iter().filter(|field| {
        matches!(
            field.unit,
            Unit::Sessions | Unit::Messages | Unit::ToolCalls | Unit::Deliveries
        ) && field.kind != Kind::Special
    }) {
        let text = field.unit.name();
        let answer =
            program(&reader, text, &[field.name]).and_then(|program| reader.query_rows(&program));
        if let Err(error) = answer {
            refused.push(format!("{text}.{}: {error}", field.name));
        }
    }
    assert_eq!(refused, Vec::<String>::new());
}

/// The command as typed and its normalized form for Morrow's 18 shell
/// calls (FACTS.md, tool calls of each session; the commands are in the
/// Claude Code and Codex files). docs/measures.md defines `command_norm`
/// as the command normalized for comparison across sessions; the rule
/// written here is the one `polylogue_model::command::normalize`
/// documents: the last stage's program and subcommand, arguments dropped.
#[test]
fn shell_calls_show_the_typed_command_beside_its_normalized_form() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    let fields = ["session", "command", "command_norm"];
    let got = tuples(
        &rows(&reader, "tool_calls where action:shell", &fields),
        &fields,
    );
    let mut want: Vec<Vec<Value>> = [
        ("#1", "cargo init --lib", "cargo init"),
        ("#1", "cargo build", "cargo build"),
        ("#1", "ls -R src", "ls"),
        ("#2", "cargo test", "cargo test"),
        ("#2", "cargo test", "cargo test"),
        ("#2", "cargo test", "cargo test"),
        ("#2", "cargo clippy", "cargo clippy"),
        ("#2", "cargo build", "cargo build"),
        ("#4", "cargo test", "cargo test"),
        ("#4", "cargo clippy -- -D warnings", "cargo clippy"),
        ("#4", "cargo clippy -- -D warnings", "cargo clippy"),
        ("#4", "cargo test", "cargo test"),
        (
            "#4",
            "git add -A && git commit -m \"Add parse_absolute() for ISO dates\"",
            "git commit",
        ),
        ("#5", "sed -n 1,80p src/parse.rs", "sed"),
        ("#5", "cargo test", "cargo test"),
        ("#5", "date -d 'next friday'", "date"),
        ("#6", "cargo test", "cargo test"),
        ("#6", "cargo clippy", "cargo clippy"),
    ]
    .into_iter()
    .map(|(session, command, norm)| vec![json!(session), json!(command), json!(norm)])
    .collect();
    want.sort_by_key(|tuple| serde_json::to_string(tuple).unwrap());
    assert_eq!(got, want);
}

/// Each physical session's own messages, prose prompts and tool calls
/// (FACTS.md, sessions table and story): three counts that differ on
/// every session.
#[test]
fn session_counts_each_count_their_own_rows() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    let fields = ["id", "messages", "human_prompts", "tool_calls"];
    let got = tuples(&rows(&reader, "kind:*", &fields), &fields);
    let want: Vec<Vec<Value>> = [
        ("#1", 12, 3, 6),
        ("#2", 18, 3, 13),
        ("#3", 6, 0, 4),
        ("#4", 11, 3, 8),
        ("#5", 13, 2, 3),
        ("#6", 10, 1, 3),
        ("#7", 6, 3, 0),
        ("#8", 4, 2, 0),
    ]
    .into_iter()
    .map(|(id, messages, prompts, calls)| {
        vec![json!(id), json!(messages), json!(prompts), json!(calls)]
    })
    .collect();
    assert_eq!(got, want);
}

/// #4's own messages in time order: who the source says spoke beside who
/// wrote it, where tool results and the compaction summary differ
/// (the file's records after the copied prefix of #2; FACTS.md, #4).
#[test]
fn a_message_shows_its_role_and_its_writer_apart() {
    let morrow = Morrow::build(Set::Main);
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    let got = rows(
        &reader,
        "messages where session.id:claude-code:7f3a2c10-0004-4000-8000-000000000004 | sort by at",
        &["at", "role", "written_by"],
    );
    let want: Vec<Value> = [
        ("2026-10-03T14:00:00+02:00", "user", "human"),
        ("2026-10-03T14:09:00+02:00", "assistant", "model"),
        ("2026-10-03T14:18:00+02:00", "user", "tool"),
        ("2026-10-03T14:27:00+02:00", "assistant", "model"),
        ("2026-10-03T14:36:00+02:00", "user", "tool"),
        ("2026-10-03T14:40:41+02:00", "user", "harness"),
        ("2026-10-03T14:42:00+02:00", "user", "human"),
        ("2026-10-03T14:49:00+02:00", "assistant", "model"),
        ("2026-10-03T14:56:00+02:00", "user", "tool"),
        ("2026-10-03T15:03:00+02:00", "assistant", "model"),
        ("2026-10-03T15:10:00+02:00", "user", "human"),
    ]
    .into_iter()
    .map(|(at, role, written_by)| json!({"at": at, "role": role, "written_by": written_by}))
    .collect();
    assert_eq!(got, want);
}

/// Morrow's delivery and two more with different recipients, events,
/// senders, references and outcomes.
#[test]
fn deliveries_show_their_recipient_event_sender_and_outcome() {
    let morrow = Morrow::build(Set::Main);
    let (engine, _lock) = morrow.engine();
    let writer = engine.writer();
    let deliver = |id: &str, recipient: u8, sender, event, about: u8, outcome| {
        let delivery = submit(
            &writer,
            id,
            UserWrite::Deliver(Delivery {
                recipient_kind: "session".into(),
                recipient_id: Some(session(recipient).to_string()),
                sender,
                event,
                cwd: Some("/home/op/morrow".into()),
                text: format!("context about #{about}"),
                refs: vec![format!("session:{}", session(about))],
                left_out: vec![],
                prepared_position: None,
            }),
        );
        submit(
            &writer,
            &format!("{id}-outcome"),
            UserWrite::DeliveryEvent {
                delivery: delivery.clone(),
                outcome,
                detail: None,
            },
        );
        delivery
    };
    let first = deliver(
        "0199d8a0-0000-7000-8000-000000000d01",
        4,
        DeliverySender::Hook,
        DeliveryEvent::SessionStart,
        2,
        DeliveryOutcome::Delivered,
    );
    let second = deliver(
        "0199d8a0-0000-7000-8000-000000000d02",
        2,
        DeliverySender::Mcp,
        DeliveryEvent::SessionBrief,
        1,
        DeliveryOutcome::Failed,
    );
    let pool = morrow.reads();
    let reader = pool.reader().unwrap();
    let fields = ["id", "recipient", "event", "sender", "outcome"];
    let want_first = vec![
        json!(first),
        json!("#4"),
        json!("session_start"),
        json!("hook"),
        json!("delivered"),
    ];
    let want_second = vec![
        json!(second),
        json!("#2"),
        json!("session_brief"),
        json!("mcp"),
        json!("failed"),
    ];
    let want_morrow = vec![
        json!("01a101c9-1dc0-7000-8000-000000000d01"),
        json!("#4"),
        json!("session_brief"),
        json!("hook"),
        json!("delivered"),
    ];
    let mut want = vec![want_first.clone(), want_second, want_morrow.clone()];
    want.sort_by_key(|tuple| serde_json::to_string(tuple).unwrap());
    assert_eq!(tuples(&rows(&reader, "deliveries", &fields), &fields), want);
    assert_eq!(
        tuples(
            &rows(
                &reader,
                "deliveries where outcome:delivered refs:claude-code:7f3a2c10-0002",
                &fields
            ),
            &fields
        ),
        [want_first, want_morrow]
    );
}
