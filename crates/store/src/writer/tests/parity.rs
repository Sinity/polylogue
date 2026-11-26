//! The index is a function of the record (docs/architecture.md, one path):
//! the same readings give the same replies through the reader whatever order
//! their inputs arrive in and however their records are divided into steps.
//! The reference reads every input whole in one transaction, which composes
//! every session from all its readings at once.
use super::*;
use proptest::prelude::*;
use proptest::test_runner::FileFailurePersistence;

/// What one record of a generated input states.
#[derive(Debug, Clone)]
enum Row {
    /// Message `n`: its session follows from `n`, and `version` picks what
    /// this input says about it (text, time, completion).
    Message {
        n: usize,
        version: u8,
        parent: Option<usize>,
    },
    /// A message of the copy session, named by its content.
    Copy { n: usize, version: u8 },
    /// An event of session `session` following message `after`.
    Event {
        session: usize,
        after: usize,
        version: u8,
    },
    /// A quota reading in session `session`: every session of the origin
    /// states it under one ID, and `version` picks its value.
    Quota { session: usize, version: u8 },
    /// A link from session `parent` to `child`; the child may never arrive.
    Link { parent: usize, child: usize },
    /// A record that is not a message but stands for message `target`.
    Alias { name: usize, target: usize },
    /// Session `session` records the copy session's ID as its alias.
    Records { session: usize },
}

#[derive(Debug, Clone)]
struct Input {
    route: Route,
    records: Vec<Row>,
    /// A later chain of the input: the file rewritten with these records.
    rewrite: Option<Vec<Row>>,
}

impl Input {
    fn chain(&self, chain: usize) -> &[Row] {
        match chain {
            0 => &self.records,
            _ => self.rewrite.as_deref().unwrap_or_default(),
        }
    }

    fn chains(&self) -> usize {
        1 + usize::from(self.rewrite.is_some())
    }
}

const SESSIONS: usize = 3;

fn session_name(n: usize) -> String {
    format!("s{n}")
}

/// The copy session: its messages are content-named, so a session that
/// records its ID shows them under its own.
const COPY: &str = "copy";

fn message_session(n: usize) -> usize {
    n % SESSIONS
}

fn row() -> impl Strategy<Value = Row> {
    prop_oneof![
        6 => (0usize..6, 0u8..3, prop::option::of(0usize..8))
            .prop_map(|(n, version, parent)| Row::Message { n, version, parent }),
        2 => (0usize..3, 0u8..2).prop_map(|(n, version)| Row::Copy { n, version }),
        2 => (0usize..SESSIONS, 0usize..8, 0u8..2)
            .prop_map(|(session, after, version)| Row::Event { session, after, version }),
        1 => (0usize..SESSIONS, 0u8..2).prop_map(|(session, version)| Row::Quota { session, version }),
        1 => (0usize..SESSIONS, 0usize..SESSIONS + 1)
            .prop_map(|(parent, child)| Row::Link { parent, child }),
        1 => (0usize..3, 0usize..6).prop_map(|(name, target)| Row::Alias { name, target }),
        1 => (0usize..SESSIONS).prop_map(|session| Row::Records { session }),
    ]
}

fn route() -> impl Strategy<Value = Route> {
    prop_oneof![Just(Route::Live), Just(Route::Export), Just(Route::Capture)]
}

fn input() -> impl Strategy<Value = Input> {
    (
        route(),
        prop::collection::vec(row(), 1..6),
        prop::option::weighted(0.3, prop::collection::vec(row(), 1..4)),
    )
        .prop_map(|(route, records, rewrite)| Input {
            route,
            // An imported file is one commit of one chain.
            rewrite: rewrite.filter(|_| !route.is_imported()),
            records,
        })
}

/// Each record is this many bytes of its input.
const RECORD: u64 = 10;

/// An input brought in by import is named by its bytes, and is read whole.
fn input_id(n: usize, input: &Input) -> String {
    if input.route.is_imported() {
        polylogue_model::BlobHash::of(&input_bytes(n, input.records.len())).to_string()
    } else {
        format!("input-{n}")
    }
}

fn input_bytes(n: usize, records: usize) -> Vec<u8> {
    chain_bytes(n, 0, records)
}

/// A chain's bytes; a rewrite differs from the first chain at its first byte.
fn chain_bytes(n: usize, chain: usize, records: usize) -> Vec<u8> {
    let n = n + 13 * chain;
    (0..records as u64 * RECORD)
        .map(|i| b'a' + ((i as usize + n) % 26) as u8)
        .collect()
}

fn timed(
    ms: Option<i64>,
) -> (
    Option<Timestamp>,
    polylogue_model::TimeQuality,
    Option<polylogue_model::Precision>,
) {
    match ms {
        Some(ms) => (
            Timestamp::from_unix_ms(ms),
            polylogue_model::TimeQuality::Intrinsic,
            Some(polylogue_model::Precision::Instant),
        ),
        None => (None, polylogue_model::TimeQuality::Unknown, None),
    }
}

fn message(id: String, session: &str, n: usize, version: u8, parent: Option<String>) -> Message {
    let role = if n.is_multiple_of(2) {
        Role::User
    } else {
        Role::Assistant
    };
    let mut m = text_message("x", session, role, &format!("{id} version {version}"), None);
    m.id = MessageId::parse(&id).unwrap();
    m.parent = parent.map(|p| MessageId::parse(&p).unwrap());
    // Version 0 has no time; 1 and 2 disagree about it.
    let at = match version {
        0 => None,
        v => Some(10_000 * n as i64 + 1_000 * i64::from(v)),
    };
    (m.at, m.at_quality, m.at_precision) = timed(at);
    if role == Role::Assistant && version != 1 {
        m.stop_reason = Some("end_turn".into());
    }
    if version == 2 {
        m.model = Some(format!("model-{n}"));
    }
    // An assistant message calls a tool and the user message after it
    // answers, with structured data beside the output: the build carries
    // blocks and their data apart from the rest of a message.
    if role == Role::Assistant {
        m.blocks.push(Block::ToolCall {
            call_id: Some(format!("call-{n}")),
            tool: Some("Bash".into()),
            action: polylogue_model::ToolAction::Shell,
            input: serde_json::json!({ "command": format!("make {n}") }),
            written_text: None,
            command: Some(format!("make {n}")),
            path: None,
            run_by_script: false,
        });
    } else if n > 0 {
        m.blocks.push(Block::ToolResult {
            reported_duration_ms: None,
            call_id: Some(format!("call-{}", n - 1)),
            outcome: if version == 1 {
                polylogue_model::ToolOutcome::Error { reason: None }
            } else {
                polylogue_model::ToolOutcome::Ok
            },
            output: Some(format!("output {n} version {version}")),
            data: Some(serde_json::json!({ "exit_code": version })),
            attachments: Vec::new(),
        });
    }
    m
}

/// The rows of records `from..to` of an input's chain, as one parse step.
fn step(n: usize, input: &Input, chain: usize, from: usize, to: usize) -> Batch {
    let id = input_id(n, input);
    let mut output = Output::default();
    let mut sessions = std::collections::BTreeMap::<String, Session>::new();
    fn state(sessions: &mut std::collections::BTreeMap<String, Session>, name: &str) -> SessionId {
        sessions
            .entry(name.to_owned())
            .or_insert_with(|| Session::new(sid(name), SessionKind::Main))
            .id
            .clone()
    }
    for (i, row) in input.chain(chain).iter().enumerate().take(to).skip(from) {
        let offset = i as u64 * RECORD;
        let end = offset + RECORD;
        match row {
            Row::Message { n, version, parent } => {
                let name = session_name(message_session(*n));
                state(&mut sessions, &name);
                let mut message = message(
                    format!("claude-code:m{n}"),
                    &name,
                    *n,
                    *version,
                    parent.map(|p| {
                        if p < 6 {
                            format!("claude-code:m{p}")
                        } else {
                            format!("claude-code:a{}", p - 6)
                        }
                    }),
                );
                message.session = sid(&name);
                output.messages.push(MessageRow {
                    response: None,
                    page_text: false,
                    id_kind: polylogue_model::IdKind::Native,
                    offset,
                    end,
                    message,
                });
            }
            Row::Copy { n, version } => {
                state(&mut sessions, COPY);
                output.messages.push(MessageRow {
                    response: None,
                    page_text: false,
                    id_kind: polylogue_model::IdKind::Content,
                    offset,
                    end,
                    message: {
                        // A copy's text only: its tool calls would repeat
                        // the call IDs of the session it is shown in.
                        let mut copy =
                            message(format!("claude-code:{COPY}~{n}"), COPY, *n, *version, None);
                        copy.blocks.truncate(1);
                        copy
                    },
                });
            }
            Row::Event {
                session: s,
                after,
                version,
            } => {
                let name = session_name(*s);
                let id = state(&mut sessions, &name);
                let (at, at_quality, at_precision) = timed(Some(5_000 * *s as i64 + 100));
                output.events.push(Event {
                    session: Some(id),
                    kind: "marker".into(),
                    at,
                    at_quality,
                    at_precision,
                    at_until: None,
                    after: Some(MessageId::parse(&format!("claude-code:m{after}")).unwrap()),
                    range: Some((offset, end)),
                    data: serde_json::json!({ "version": version }),
                });
            }
            Row::Quota {
                session: s,
                version,
            } => {
                let id = state(&mut sessions, &session_name(*s));
                let (at, at_quality, at_precision) = timed(Some(7_000));
                output.events.push(Event {
                    session: Some(id),
                    kind: polylogue_model::reports::QUOTA_REPORT.into(),
                    at,
                    at_quality,
                    at_precision,
                    at_until: None,
                    after: None,
                    range: Some((offset, end)),
                    data: serde_json::json!({ "report_id": "quota", "limit": version }),
                });
            }
            Row::Link { parent, child } => {
                let parent = state(&mut sessions, &session_name(*parent));
                output.links.push(Link {
                    kind: LinkKind::Spawn,
                    copies_prefix: false,
                    how: None,
                    parent,
                    child: sid(&session_name(*child)),
                    branch_point: None,
                    call_id: None,
                    evidence: format!("{id}.spawn"),
                });
            }
            Row::Alias { name, target } => output.aliases.push(polylogue_model::Alias {
                offset,
                end,
                name: MessageId::parse(&format!("claude-code:a{name}")).unwrap(),
                target: Some(MessageId::parse(&format!("claude-code:m{target}")).unwrap()),
            }),
            Row::Records { session: s } => {
                let name = session_name(*s);
                state(&mut sessions, &name);
                sessions
                    .get_mut(&name)
                    .expect("just stated")
                    .aliases
                    .push(sid(COPY));
            }
        }
    }
    output.sessions = sessions.into_values().collect();
    let records = (to - from) as u64;
    let mut step = ParseStep::new(
        output,
        counts(records),
        from as u64 * RECORD..to as u64 * RECORD,
        true,
    );
    step.parse_version = "synthetic=1".into();
    let mut batch = batch(&id, input.route, from as u64 * RECORD, vec![], &[]);
    batch.parse = vec![step];
    batch.length = to as u64 * RECORD;
    if chain > 0 && from == 0 {
        batch.change = Change::Rewritten;
    }
    batch
}

fn stored_step(
    w: &TestArchive,
    n: usize,
    input: &Input,
    chain: usize,
    from: usize,
    to: usize,
) -> crate::staging::PreparedBatch {
    let mut batch = step(n, input, chain, from, to);
    let bytes = chain_bytes(n, chain, input.chain(chain).len());
    let range = &bytes[from * RECORD as usize..to * RECORD as usize];
    let (blob, file) = w.archive.prepare_blob(range).unwrap();
    batch.new_blob = Some(blob);
    crate::staging::PreparedBatch {
        batch,
        files: file.into_iter().collect(),
    }
}

/// Each content is stored once and named by what shows it: every content
/// group is named by a text or a reading, every shown block names a piece of
/// a message reading's content, and every shown event names an event
/// reading's piece (docs/archive.md, content).
fn assert_content_named_once(w: &TestArchive) {
    let conn = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let count = |sql: &str| -> i64 { conn.query_row(sql, [], |r| r.get(0)).unwrap() };
    assert_eq!(
        count(
            "SELECT count(*) FROM content_group g WHERE NOT EXISTS(SELECT 1 FROM text WHERE group_id=g.id)
             AND NOT EXISTS(SELECT 1 FROM message_parse WHERE content_group=g.id)
             AND NOT EXISTS(SELECT 1 FROM event_parse WHERE content_group=g.id)"
        ),
        0,
        "content groups nothing names"
    );
    assert_eq!(
        count(
            "SELECT count(*) FROM block b WHERE NOT EXISTS(SELECT 1 FROM message_parse r
             WHERE r.content_group=b.content_group AND b.content_start>=r.content_start
             AND b.content_start+b.content_length+coalesce(b.data_length,0)<=r.content_start+r.content_length)"
        ),
        0,
        "shown blocks outside every message reading's content"
    );
    assert_eq!(
        count(
            "SELECT count(*) FROM event e WHERE NOT EXISTS(SELECT 1 FROM event_parse r
             WHERE r.content_group=e.content_group AND r.content_start=e.content_start
             AND r.content_length=e.content_length AND r.data_length=e.data_length)"
        ),
        0,
        "shown events naming no event reading's content"
    );
}

/// Every row the index schema says names another names one that exists. A
/// build loads its tables with references unenforced, so the installed file
/// is where they are first checked.
fn assert_references_hold(w: &TestArchive) {
    let conn = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let dangling: Vec<(String, Option<i64>, String)> = conn
        .prepare("SELECT \"table\", rowid, parent FROM pragma_foreign_key_check")
        .unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap()
        .collect::<rusqlite::Result<_>>()
        .unwrap();
    assert_eq!(dangling, [], "rows naming a row that does not exist");
}

/// Everything the reader answers about every session, by natural keys.
fn replies(w: &TestArchive) -> Vec<String> {
    assert_content_named_once(w);
    assert_references_hold(w);
    let pool = ReadPool::open(&w.archive).unwrap();
    let r = pool.reader().unwrap();
    let mut out = Vec::new();
    let ids = r.query("SELECT id FROM session ORDER BY id", &[]).unwrap();
    for row in ids {
        let id = SessionId::parse(row["id"].as_str().unwrap()).unwrap();
        out.push(format!("session {id}: {:?}", r.session(&id).unwrap()));
        for message in r.messages(&id, false).unwrap() {
            out.push(format!("  message {message:?}"));
            // Each call with the result paired to it, and each result.
            for (ordinal, block) in message.blocks.iter().enumerate() {
                if matches!(block, Block::ToolCall { .. } | Block::ToolResult { .. }) {
                    let pair = r.message_tool_pair(&message.id, ordinal as u64).unwrap();
                    out.push(format!("    pair {ordinal} {pair:?}"));
                }
            }
        }
        // The session's entries in reading order, events placed by the
        // messages they follow. A session left with no rows reads the same
        // with an empty index or none.
        let items = r
            .item_index(&id)
            .ok()
            .flatten()
            .map(|items| items.entries)
            .unwrap_or_default();
        out.push(format!("  items {items:?}"));
        let mut events = Vec::new();
        r.visit_event_records(&id, 0, |mut event| {
            // When its record was committed belongs to the record's arrival,
            // which the two paths differ in by construction.
            event.commit = 0;
            events.push(format!("  event {event:?}"));
            std::ops::ControlFlow::Continue(())
        })
        .unwrap();
        out.extend(events);
        for link in r.child_links(&id).unwrap() {
            out.push(format!("  child {link:?}"));
        }
        for link in r.parent_links(&id).unwrap() {
            out.push(format!("  parent {link:?}"));
        }
    }
    for row in r
        .query(
            "SELECT m.id AS message, m.session_order AS place, m.in_progress AS open FROM message m ORDER BY m.id",
            &[],
        )
        .unwrap()
    {
        out.push(format!("place {row}"));
    }
    out
}

/// How a capture's steps are read.
#[derive(Clone, Copy)]
enum Reading {
    /// Bytes only.
    Never,
    /// Each step's rows with its bytes (upkeep).
    Each,
    /// Bytes only for the first steps; then the index is built from the
    /// record, and each later step's rows come with its bytes.
    BuildAfter(usize),
}

/// Stores the inputs' records divided at `cuts`, in the order `picks`
/// chooses, each step its own transaction, read as `reading` says.
/// Returns the archive's chain of each input's chain.
fn capture(
    mut w: TestArchive,
    inputs: &[Input],
    cuts: &[bool],
    picks: &[u32],
    reading: Reading,
) -> (TestArchive, std::collections::BTreeMap<(usize, usize), i64>) {
    let mut steps: Vec<std::collections::VecDeque<(usize, usize, usize)>> = inputs
        .iter()
        .enumerate()
        .map(|(n, input)| {
            let mut ranges = std::collections::VecDeque::new();
            for chain in 0..input.chains() {
                let records = input.chain(chain).len();
                let mut from = 0;
                for i in 1..=records {
                    let cut = i == records
                        || (!input.route.is_imported()
                            && cuts[(n * 7 + chain * 3 + i) % cuts.len()]);
                    if cut {
                        ranges.push_back((chain, from, i));
                        from = i;
                    }
                }
            }
            ranges
        })
        .collect();
    let mut chains = std::collections::BTreeMap::new();
    let mut stored_records = std::collections::BTreeMap::new();
    let mut pick = picks.iter().cycle();
    let mut taken = 0;
    while steps.iter().any(|ranges| !ranges.is_empty()) {
        if matches!(reading, Reading::BuildAfter(k) if k == taken) && taken > 0 {
            w = build_from_record(w, inputs, &chains, &stored_records);
        }
        let parse = match reading {
            Reading::Never => false,
            Reading::Each => true,
            Reading::BuildAfter(k) => taken >= k,
        };
        taken += 1;
        let waiting: Vec<usize> = (0..steps.len()).filter(|&n| !steps[n].is_empty()).collect();
        let n = waiting[*pick.next().unwrap() as usize % waiting.len()];
        let (chain, from, to) = steps[n].pop_front().unwrap();
        let mut step = stored_step(&w, n, &inputs[n], chain, from, to);
        if !parse {
            step.batch.parse.clear();
        }
        let results = w.writer.store_group(vec![step]).unwrap();
        let [Ok(stored)] = &results[..] else {
            panic!("{results:?}");
        };
        chains.insert((n, chain), stored.chain);
        stored_records.insert((n, chain), to);
        w.writer.flush().unwrap();
    }
    if matches!(reading, Reading::BuildAfter(k) if k >= taken) && taken > 0 {
        w = build_from_record(w, inputs, &chains, &stored_records);
    }
    (w, chains)
}

/// Opens the archive again without its index, so a build is due, and
/// builds it from the stored record: each chain read whole, as the record
/// holds it, into one of two runs, then loaded, composed and installed.
fn build_from_record(
    w: TestArchive,
    inputs: &[Input],
    chains: &std::collections::BTreeMap<(usize, usize), i64>,
    stored_records: &std::collections::BTreeMap<(usize, usize), usize>,
) -> TestArchive {
    let w = w.reopened_without_index();
    let sender = w.writer.sender();
    let mut runs = [
        crate::writer::build::BuildWorker::create(&w.archive, 0, 32 << 20).unwrap(),
        crate::writer::build::BuildWorker::create(&w.archive, 1, 32 << 20).unwrap(),
    ];
    for (i, (&(n, chain), &stored)) in chains.iter().enumerate() {
        let records = stored_records[&(n, chain)];
        let mut batch = step(n, &inputs[n], chain, 0, records);
        let mut open = runs[i % 2]
            .begin(&sender, &batch.input, stored, batch.length)
            .unwrap();
        for step in &mut batch.parse {
            crate::writer::ChainSink::step(&mut open, step).unwrap();
        }
        open.finish().unwrap();
    }
    let paths = runs
        .into_iter()
        .flat_map(|run| run.finish().unwrap())
        .collect();
    let phases = w.writer.build(paths).unwrap();
    assert_eq!(
        phases.iter().map(|p| p.name.as_str()).collect::<Vec<_>>(),
        [
            "sorting texts",
            "writing tables",
            "indexing",
            "assembling sessions",
            "indexing sessions",
            "joining sessions across files",
            "finishing indexes",
            "search index",
            "installing"
        ]
    );
    assert!(w.archive.index_build().is_none());
    w
}

/// The same captures, then every chain read whole in one transaction, which
/// composes every session from all its readings at once.
fn reference(inputs: &[Input], cuts: &[bool], picks: &[u32]) -> Vec<String> {
    let (w, chains) = capture(test_archive(), inputs, cuts, picks, Reading::Never);
    let group = chains
        .iter()
        .map(|(&(n, chain), &stored)| {
            let input = &inputs[n];
            let mut batch = step(n, input, chain, 0, input.chain(chain).len());
            batch.change = Change::Reparse { chain: stored };
            batch.observation = None;
            crate::staging::PreparedBatch {
                batch,
                files: Vec::new(),
            }
        })
        .collect();
    let results = w.writer.store_group(group).unwrap();
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    w.writer.flush().unwrap();
    replies(&w)
}

/// The inputs' records divided at `cuts` and sent in the order `picks`
/// chooses, each step its own transaction.
fn upkeep(inputs: &[Input], cuts: &[bool], picks: &[u32]) -> Vec<String> {
    let (w, _) = capture(test_archive(), inputs, cuts, picks, Reading::Each);
    replies(&w)
}

/// The index built from the record after the first `built` steps, the
/// rest read by upkeep.
fn built_then_upkeep(inputs: &[Input], cuts: &[bool], picks: &[u32], built: usize) -> Vec<String> {
    let (w, _) = capture(
        test_archive(),
        inputs,
        cuts,
        picks,
        Reading::BuildAfter(built),
    );
    replies(&w)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 24,
        failure_persistence: Some(Box::new(FileFailurePersistence::Off)),
        ..ProptestConfig::default()
    })]
    #[test]
    fn any_order_and_division_gives_the_same_replies(
        inputs in prop::collection::vec(input(), 1..4),
        cuts in prop::collection::vec(any::<bool>(), 8),
        picks in prop::collection::vec(any::<u32>(), 6),
    ) {
        let expected = reference(&inputs, &cuts, &picks);
        prop_assert_eq!(upkeep(&inputs, &cuts, &picks), expected);
    }

    /// A build reads the record at one position and upkeep reads what
    /// arrived after it; together they give the replies upkeep alone gives
    /// (docs/architecture.md, one path).
    #[test]
    fn a_build_equals_upkeep_for_the_same_record(
        inputs in prop::collection::vec(input(), 1..4),
        cuts in prop::collection::vec(any::<bool>(), 8),
        picks in prop::collection::vec(any::<u32>(), 6),
        built in 1usize..12,
    ) {
        let expected = reference(&inputs, &cuts, &picks);
        prop_assert_eq!(built_then_upkeep(&inputs, &cuts, &picks, built), expected);
    }
}

/// Two sessions state one quota reading: the build shows it once, from both
/// sessions' readings, as upkeep does.
#[test]
fn a_reading_two_sessions_state_is_built_once() {
    let input = |session: usize, version: u8| Input {
        route: Route::Live,
        records: vec![
            Row::Message {
                n: session,
                version: 1,
                parent: None,
            },
            Row::Quota { session, version },
        ],
        rewrite: None,
    };
    let inputs = vec![input(0, 0), input(1, 1)];
    let expected = reference(&inputs, &[true; 8], &[0; 6]);
    assert_eq!(
        expected
            .iter()
            .filter(|line| line.starts_with("  event") && line.contains("quota_report"))
            .count(),
        1
    );
    assert_eq!(
        built_then_upkeep(&inputs, &[true; 8], &[0; 6], 12),
        expected
    );
}

/// Two Drive revisions of one AI Studio prompt record the same copy name:
/// the name names neither, so the copy shows as itself, with its links,
/// whether the prompts arrive together or apart.
#[test]
fn both_prompts_in_one_batch_move_the_copy_and_its_links_back() {
    let copy = Input {
        route: Route::Export,
        records: vec![
            Row::Copy { n: 0, version: 1 },
            Row::Copy { n: 1, version: 1 },
        ],
        rewrite: None,
    };
    let first = Input {
        route: Route::Capture,
        records: vec![
            Row::Records { session: 0 },
            Row::Message {
                n: 0,
                version: 1,
                parent: None,
            },
        ],
        rewrite: None,
    };
    let second = Input {
        route: Route::Capture,
        records: vec![
            Row::Records { session: 1 },
            Row::Message {
                n: 1,
                version: 1,
                parent: None,
            },
        ],
        rewrite: None,
    };
    let link = Input {
        route: Route::Live,
        records: vec![Row::Link {
            parent: 2,
            child: 0,
        }],
        rewrite: None,
    };
    let inputs = vec![copy, first, second, link];
    let together = reference(&inputs, &[true; 8], &[0, 1, 0, 0, 0, 0]);
    // The copy is its own session: two sessions record its name.
    assert!(
        together
            .iter()
            .any(|line| line.starts_with("session claude-code:copy:"))
    );
    assert!(
        together
            .iter()
            .any(|line| line.contains("claude-code:copy~0"))
    );
    // One prompt alone takes the copy in; the second arriving gives it back.
    let apart = upkeep(&inputs, &[true; 8], &[0, 1, 0, 0, 0, 0]);
    assert_eq!(apart, together);
    let single = reference(&inputs[..2], &[true; 8], &[0; 6]);
    assert!(
        single
            .iter()
            .all(|line| !line.starts_with("session claude-code:copy:"))
    );
    assert!(single.iter().any(|line| line.contains("claude-code:s0~0")));
}

/// A message naming a parent the archive does not hold reads it unresolved;
/// when the parent arrives, the child's row is not written again.
#[test]
fn an_absent_parent_reads_unresolved_and_its_arrival_writes_nothing_into_the_child() {
    let child = Input {
        route: Route::Live,
        records: vec![Row::Message {
            n: 1,
            version: 1,
            parent: Some(0),
        }],
        rewrite: None,
    };
    let parent = Input {
        route: Route::Live,
        records: vec![Row::Message {
            n: 0,
            version: 1,
            parent: None,
        }],
        rewrite: None,
    };
    let w = test_archive();
    w.writer
        .store_group(vec![stored_step(&w, 0, &child, 0, 0, 1)])
        .unwrap();
    w.writer.flush().unwrap();
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let row = |conn: &rusqlite::Connection| -> (i64, String, i64) {
        conn.query_row(
            "SELECT m.pk, m.parent_id, s.changed FROM message m JOIN session s ON s.pk=m.session_pk WHERE m.id='claude-code:m1'",
            [],
            |r| <(i64, String, i64)>::try_from(r),
        )
        .unwrap()
    };
    let before = row(&conn);
    assert_eq!(before.1, "claude-code:m0");
    let pool = ReadPool::open(&w.archive).unwrap();
    assert_eq!(
        pool.reader().unwrap().messages(&sid("s1"), false).unwrap()[0].parent,
        Some(MessageId::parse("claude-code:m0").unwrap()),
    );
    // m0 is in session s0; its arrival changes no row of s1.
    w.writer
        .store_group(vec![stored_step(&w, 1, &parent, 0, 0, 1)])
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(row(&conn), before);
}

/// A link is shown while a reading states it. The session it names leaving
/// the index (its one reading replaced by a reading that no longer states
/// it) takes the session's rows and leaves the link another input states,
/// as a build from the same readings shows it.
#[test]
fn a_link_stays_while_a_reading_states_it_when_the_session_it_names_leaves() {
    let input = |records| Input {
        route: Route::Live,
        records,
        rewrite: None,
    };
    let message = |n| Row::Message {
        n,
        version: 1,
        parent: None,
    };
    let linking = input(vec![Row::Link {
        parent: 0,
        child: 1,
    }]);
    let w = test_archive();
    w.writer
        .store_group(vec![
            stored_step(&w, 0, &linking, 0, 0, 1),
            stored_step(&w, 1, &input(vec![message(1)]), 0, 0, 1),
        ])
        .unwrap();
    w.writer.flush().unwrap();
    let conn = rusqlite::Connection::open(w.archive.index_db()).unwrap();
    let rows = |sql: &str| -> Vec<String> {
        conn.prepare(sql)
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    };
    const LINKS: &str = "SELECT parent_id || ' > ' || child_id FROM link";
    const SESSIONS: &str = "SELECT id FROM session ORDER BY id";
    assert_eq!(rows(LINKS), ["claude-code:s0 > claude-code:s1"]);
    assert_eq!(rows(SESSIONS), ["claude-code:s0", "claude-code:s1"]);
    // The second input's chain read again, now stating a message of s2.
    let mut batch = step(1, &input(vec![message(2)]), 0, 0, 1);
    batch.change = Change::Reparse { chain: 1 };
    batch.observation = None;
    let results = w
        .writer
        .store_group(vec![crate::staging::PreparedBatch {
            batch,
            files: Vec::new(),
        }])
        .unwrap();
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    w.writer.flush().unwrap();
    assert_eq!(rows(SESSIONS), ["claude-code:s0", "claude-code:s2"]);
    assert_eq!(rows(LINKS), ["claude-code:s0 > claude-code:s1"]);
}

/// A text leaves the index with the last reading row that holds it, and
/// its search entries with it.
#[test]
fn a_text_leaves_with_its_last_row_and_its_search_entries() {
    let w = test_archive();
    let one = Input {
        route: Route::Live,
        records: vec![Row::Message {
            n: 0,
            version: 1,
            parent: None,
        }],
        rewrite: None,
    };
    w.writer
        .store_group(vec![stored_step(&w, 0, &one, 0, 0, 1)])
        .unwrap();
    w.writer.flush().unwrap();
    let conn = crate::archive::open_read(&w.archive.index_db()).unwrap();
    let text = "claude-code:m0 version 1";
    let hash = blake3::hash(text.as_bytes());
    let held = |conn: &rusqlite::Connection| -> (i64, i64) {
        conn.query_row(
            &format!(
                "SELECT (SELECT count(*) FROM text WHERE hash=?1), (SELECT count(*) FROM ({}))",
                crate::search::matching_texts(crate::search::SnippetIndex::Prose, "'\"version\"'")
            ),
            [hash.as_bytes().as_slice()],
            |r| <(i64, i64)>::try_from(r),
        )
        .unwrap()
    };
    assert_eq!(held(&conn), (1, 1));
    // The chain read again from its start without the message: its reading
    // is replaced, and the text has no holder left.
    let mut empty = step(
        0,
        &Input {
            route: Route::Live,
            records: vec![],
            rewrite: None,
        },
        0,
        0,
        0,
    );
    empty.change = Change::Reparse { chain: 1 };
    empty.length = RECORD;
    let empty_step = empty.parse.first_mut().unwrap();
    empty_step.parsed_upto = RECORD;
    empty_step.reset = true;
    empty_step.counts = counts(1);
    w.writer.send(empty).unwrap();
    w.writer.flush().unwrap();
    assert_eq!(held(&conn), (0, 0));
    assert!(texts(&w, "s0").is_empty());
    // The message's content leaves with its reading, and its text's group
    // with the text.
    let groups: i64 = conn
        .query_row("SELECT count(*) FROM content_group", [], |r| r.get(0))
        .unwrap();
    assert_eq!(groups, 0);
}
