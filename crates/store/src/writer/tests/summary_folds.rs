//! Session summaries are a function of the session's rows
//! (docs/measures.md, convergence): whatever steps the rows arrived in, and
//! after a re-parse of the whole input, every stored summary equals what
//! the same rows written in one step give.
use crate::fixture::writer::{WriterArchive, message, sid};
use crate::writer::{ChainParse, InputKey, ParseStep};
use polylogue_model::{
    Block, ErrorReason, Event, Json, MessageRow, Output, RecordCounts, RecordOutcome, Role,
    Session, SessionKind, Timestamp, ToolAction, ToolOutcome,
};
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

const INPUT: &str = "input-s";
const PATH: &str = "/p/s";

/// One record of the generated session.
#[derive(Debug, Clone)]
enum Record {
    /// A person's prompt.
    Prompt,
    /// A model reply that ends its turn, naming one of two models.
    Reply { model: u8, thinking: bool },
    /// A model's shell call, whose command is one of [`COMMANDS`]; one a
    /// script ran changes which code calls the session counts.
    Call { command: u8, script: bool },
    /// A model's code call.
    Code,
    /// The result of the latest call without one, failed or not.
    Result { failed: bool },
    /// A compaction boundary.
    Compaction,
    /// A hook line; `late` places its time before the record before it.
    Hook { kind: u8, late: bool },
}

const COMMANDS: &[(&str, &str)] = &[
    ("cargo test", "ok"),
    ("git commit -m one", "[main abc1234] one\n 1 file changed"),
    (
        "gh pr create --fill",
        "https://github.com/owner/repo/pull/7\n",
    ),
    ("ls", "a b"),
];

const HOOKS: &[&str] = &[
    "hook_permission_request",
    "hook_stop",
    "hook_notification",
    "hook_user_prompt_submit",
];

fn record() -> impl Strategy<Value = Record> {
    prop_oneof![
        3 => Just(Record::Prompt),
        3 => (0u8..2, any::<bool>()).prop_map(|(model, thinking)| Record::Reply { model, thinking }),
        3 => (0..COMMANDS.len() as u8, prop::bool::weighted(0.1))
            .prop_map(|(command, script)| Record::Call { command, script }),
        1 => Just(Record::Code),
        3 => any::<bool>().prop_map(|failed| Record::Result { failed }),
        1 => Just(Record::Compaction),
        2 => (0..HOOKS.len() as u8, any::<bool>()).prop_map(|(kind, late)| Record::Hook { kind, late }),
    ]
}

/// The records and where the steps divide them.
fn session() -> impl Strategy<Value = (Vec<Record>, Vec<usize>)> {
    prop::collection::vec(record(), 1..24).prop_flat_map(|records| {
        let n = records.len();
        (Just(records), prop::collection::vec(0..=n, 0..6))
    })
}

/// What one record is parsed into: a message or an event.
enum Parsed {
    Message(Box<polylogue_model::Message>),
    Event(Box<Event>),
}

/// The parsed form of every record, record `i` at time `1_000 * (i + 1)`.
fn parsed(records: &[Record]) -> Vec<Parsed> {
    let mut open: Vec<String> = Vec::new();
    records
        .iter()
        .enumerate()
        .map(|(i, record)| {
            let at = 1_000 * (i as i64 + 1);
            let id = format!("m{i}");
            match record {
                Record::Prompt => Parsed::Message(Box::new(message(
                    &id,
                    "s",
                    Role::User,
                    at,
                    vec![Block::Text {
                        text: format!("prompt {i}\nmore"),
                    }],
                ))),
                Record::Reply { model, thinking } => {
                    let mut blocks = vec![Block::Text {
                        text: format!("reply {i}"),
                    }];
                    if *thinking {
                        blocks.insert(
                            0,
                            Block::Thinking {
                                text: Some("thought".into()),
                                signature: None,
                                redacted: false,
                            },
                        );
                    }
                    let mut m = message(&id, "s", Role::Assistant, at, blocks);
                    m.model = Some(format!("model-{model}"));
                    m.stop_reason = Some("end_turn".into());
                    Parsed::Message(Box::new(m))
                }
                Record::Code => {
                    let call_id = format!("t{i}");
                    open.push(call_id.clone());
                    let mut m = message(
                        &id,
                        "s",
                        Role::Assistant,
                        at,
                        vec![Block::ToolCall {
                            written_text: None,
                            run_by_script: false,
                            call_id: Some(call_id),
                            tool: Some("run".into()),
                            action: ToolAction::Code,
                            input: Json::Null,
                            command: None,
                            path: None,
                        }],
                    );
                    m.model = Some("model-1".into());
                    m.stop_reason = Some("tool_use".into());
                    Parsed::Message(Box::new(m))
                }
                Record::Call { command, script } => {
                    let call_id = format!("t{i}");
                    open.push(call_id.clone());
                    let mut m = message(
                        &id,
                        "s",
                        Role::Assistant,
                        at,
                        vec![Block::ToolCall {
                            written_text: None,
                            run_by_script: *script,
                            call_id: Some(call_id),
                            tool: Some("Bash".into()),
                            action: ToolAction::Shell,
                            input: Json::Null,
                            command: Some(COMMANDS[*command as usize].0.into()),
                            path: None,
                        }],
                    );
                    m.model = Some("model-0".into());
                    m.stop_reason = Some("tool_use".into());
                    Parsed::Message(Box::new(m))
                }
                Record::Result { failed } => {
                    let call = open.pop();
                    let output = call
                        .as_deref()
                        .and_then(|c| c.strip_prefix('t'))
                        .and_then(|n| n.parse::<usize>().ok())
                        .and_then(|n| match &records[n] {
                            Record::Call { command, .. } => Some(COMMANDS[*command as usize].1),
                            _ => None,
                        })
                        .unwrap_or("none");
                    let mut m = message(
                        &id,
                        "s",
                        Role::User,
                        at,
                        vec![Block::ToolResult {
                            reported_duration_ms: None,
                            call_id: call.or(Some("unknown".into())),
                            outcome: if *failed {
                                ToolOutcome::Error {
                                    reason: Some(ErrorReason::ExitCode(1)),
                                }
                            } else {
                                ToolOutcome::Ok
                            },
                            output: Some(output.into()),
                            data: None,
                            attachments: vec![],
                        }],
                    );
                    m.prompt_kind = None;
                    Parsed::Message(Box::new(m))
                }
                Record::Compaction => Parsed::Event(Box::new(event(
                    "compaction",
                    at,
                    serde_json::json!({"trigger": "auto", "pre_tokens": 1000 + i}),
                ))),
                Record::Hook { kind, late } => Parsed::Event(Box::new(event(
                    HOOKS[*kind as usize],
                    if *late { at - 1_500 } else { at },
                    serde_json::json!({"notification_type": "permission_prompt", "n": i}),
                ))),
            }
        })
        .collect()
}

fn event(kind: &str, at: i64, data: Json) -> Event {
    Event {
        at_quality: polylogue_model::TimeQuality::Intrinsic,
        at_precision: Some(polylogue_model::Precision::Instant),
        at_until: None,
        session: Some(sid("s")),
        kind: kind.into(),
        at: Timestamp::from_unix_ms(at),
        after: None,
        range: None,
        data,
    }
}

/// The step's output: record `i` of `records`, for each `(i, offset)` of
/// `at`, at that byte offset.
fn output(records: &[Record], at: &[(usize, u64)]) -> Output {
    let mut output = Output {
        sessions: vec![Session::new(sid("s"), SessionKind::Main)],
        ..Output::default()
    };
    let mut parsed: Vec<Option<Parsed>> = parsed(records).into_iter().map(Some).collect();
    for &(i, offset) in at {
        match parsed[i].take().expect("each record once") {
            Parsed::Message(message) => output.messages.push(MessageRow {
                response: None,
                page_text: false,
                id_kind: polylogue_model::IdKind::Native,
                offset,
                end: offset + 1,
                message: *message,
            }),
            Parsed::Event(event) => {
                let mut event = *event;
                event.range = Some((offset, offset + 1));
                output.events.push(event);
            }
        }
    }
    output
}

/// Writes records `range` of `records` as one appended step, adding where
/// each was stored to `stored`.
fn write(
    w: &mut WriterArchive,
    records: &[Record],
    range: std::ops::Range<usize>,
    stored: &mut Vec<(usize, u64)>,
) {
    let start = w.offset();
    let at: Vec<(usize, u64)> = range
        .clone()
        .map(|i| (i, start + (i - range.start) as u64))
        .collect();
    w.write_output(
        INPUT,
        "claude-code",
        PATH,
        output(records, &at),
        range.len() as u64,
    );
    stored.extend(at);
}

/// Parses the stored input again, whole, into the same records.
fn reparse(w: &WriterArchive, records: &[Record], stored: &[(usize, u64)]) {
    let length = w.offset();
    let mut counts = RecordCounts {
        records: records.len() as u64,
        ..RecordCounts::default()
    };
    for _ in records {
        counts.add("record", &RecordOutcome::Used);
    }
    let mut step = ParseStep::new(output(records, stored), counts, 0..length, true);
    step.parse_version = "synthetic=1".into();
    let mut reading = ChainParse::new().unwrap();
    reading.push(&mut step).unwrap();
    w.writer
        .replace_chain(
            InputKey {
                id: INPUT.into(),
                format: "claude-code".into(),
                route: polylogue_model::Route::Live,
            },
            1,
            length,
            reading,
        )
        .unwrap();
    w.writer.derive().unwrap();
}

/// Every stored summary of the archive's sessions, by natural keys.
fn summaries(w: &WriterArchive) -> Json {
    let mut out = serde_json::Map::new();
    for (name, sql) in [
        (
            "session",
            "SELECT id, title, derived_title, kind, started_at, last_activity_at, message_count,
                    human_prompt_count, human_command_count, word_count, human_word_count,
                    model_word_count, thinking_count, compaction_count, tool_call_count,
                    error_count, live_facts, live_last_at, stub
             FROM session ORDER BY id",
        ),
        (
            "span",
            "SELECT s.id, first_at, last_at, time_source, wall_ms, clustered_ms, tool_sum_ms,
                    tool_union_ms, tool_overlap_ms, tool_untimed, tool_no_result,
                    tool_same_timestamp, tool_reversed, timed_messages, untimed_messages,
                    unordered_gaps, first_failure_at, prompts_to_failure, prompted, ends_with,
                    ends_with_message, last_stop_reason
             FROM session_span p JOIN session s ON s.pk = p.session_pk ORDER BY s.id",
        ),
        (
            "time",
            "SELECT s.id, party, key, ms FROM session_time t JOIN session s ON s.pk = t.session_pk
             ORDER BY s.id, party, key",
        ),
        (
            "values",
            "SELECT s.id, field, value, count FROM session_value v JOIN session s ON s.pk = v.session_pk
             WHERE n > 0 ORDER BY s.id, field, value",
        ),
        (
            "first_prompt",
            "SELECT s.id, m.id AS message FROM session_first_prompt f
             JOIN session s ON s.pk = f.session_pk LEFT JOIN message m ON m.pk = f.message_pk
             ORDER BY s.id",
        ),
        (
            "calls",
            "SELECT call_id, outcome, outcome_reason, command_norm FROM tool_call ORDER BY call_id",
        ),
        (
            "commits",
            "SELECT s.id, l.commit_hash, l.evidence, t.call_id FROM commit_link l
             JOIN session s ON s.pk = l.session_pk
             LEFT JOIN tool_call t ON t.call_block_pk = l.evidence_block_pk
             ORDER BY s.id, commit_hash",
        ),
        (
            "pulls",
            "SELECT s.id, l.repo, l.number, l.url, l.evidence FROM pull_request_link l
             JOIN session s ON s.pk = l.session_pk ORDER BY s.id, l.repo, l.number",
        ),
        (
            "intervals",
            "SELECT s.id, aspect, value, started_at, ended_at, closed_by FROM live_interval i
             JOIN session s ON s.pk = i.session_pk ORDER BY s.id, aspect, started_at",
        ),
        (
            "lines",
            "SELECT s.id, i.position, hex(i.unit) AS unit, hex(i.entry) AS entry, i.class, i.call_id,
                    hex(i.counts) AS counts, m.id AS message, e.id AS event
             FROM transcript_line i JOIN session s ON s.pk = i.session_pk
             LEFT JOIN message m ON m.pk = i.message_pk LEFT JOIN event e ON e.pk = i.event_pk
             ORDER BY s.id, i.position",
        ),
    ] {
        out.insert(name.into(), Json::Array(w.rows(sql)));
    }
    // The line count and the counts over every line; the counters follow
    // the division into steps (an append moves one, a rewrite the other).
    let index = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let sessions: Vec<(String, i64)> = index
        .prepare("SELECT id, pk FROM session ORDER BY id")
        .unwrap()
        .query_map([], |r| <(String, i64)>::try_from(r))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    let totals = sessions
        .into_iter()
        .filter_map(|(id, pk)| {
            let state = crate::summary::lines::state(&index, pk).unwrap()?;
            Some(serde_json::json!({
                "id": id,
                "entries": state.entries,
                "counts": state.counts,
            }))
        })
        .collect();
    out.insert("line_totals".into(), Json::Array(totals));
    Json::Object(out)
}

/// The records written in one step.
fn whole(records: &[Record]) -> Json {
    let mut w = WriterArchive::new();
    write(&mut w, records, 0..records.len(), &mut Vec::new());
    summaries(&w)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
        ..ProptestConfig::default()
    })]

    /// Any division of a session's records into appended steps, and a
    /// re-parse of the input after them, gives the summaries the records
    /// written in one step give. Hook lines arrive out of time order.
    #[test]
    fn summaries_after_any_appends_equal_those_from_the_final_rows(
        (records, mut cuts) in session(),
    ) {
        let expected = whole(&records);
        cuts.push(0);
        cuts.push(records.len());
        cuts.sort_unstable();
        cuts.dedup();
        let mut w = WriterArchive::new();
        let mut stored = Vec::new();
        for step in cuts.windows(2) {
            write(&mut w, &records, step[0]..step[1], &mut stored);
        }
        prop_assert_eq!(&summaries(&w), &expected, "after the steps {:?}", cuts);
        reparse(&w, &records, &stored);
        prop_assert_eq!(&summaries(&w), &expected, "after a re-parse");
    }
}
