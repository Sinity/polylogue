use super::*;
use crate::parse::{Parse, Step};
use polylogue_model::{Block, Message, Role, Session};
use proptest::prelude::*;
use std::path::{Path, PathBuf};

/// Settles a run of steps over one chain in order, as the engine does.
pub(crate) fn settled(steps: &[Step]) -> Vec<Step> {
    let mut names = Formed::new(NoEarlierRows);
    steps
        .iter()
        .cloned()
        .map(|mut step| {
            if step.repeats_whole_conversations {
                settle(&mut step.output, &mut NoEarlierRows).unwrap();
            } else {
                names.settle(&mut step.output).unwrap();
            }
            step
        })
        .collect()
}

/// What a run of settled steps leaves in the index for one chain, compared
/// as values: a row at an offset replaces the one there before, session
/// facts merge in order, links and copies are a set.
#[derive(Debug, PartialEq)]
struct Reading {
    messages: BTreeMap<(u64, String), Json>,
    events: BTreeMap<(u64, u64, String), Json>,
    aliases: BTreeMap<String, Json>,
    sessions: BTreeMap<String, Json>,
    links: std::collections::BTreeSet<String>,
    copies: std::collections::BTreeSet<String>,
    background: Vec<Json>,
}

fn reading(steps: &[Step]) -> Reading {
    let mut sessions: BTreeMap<String, Session> = BTreeMap::new();
    let mut reading = Reading {
        messages: BTreeMap::new(),
        events: BTreeMap::new(),
        aliases: BTreeMap::new(),
        sessions: BTreeMap::new(),
        links: Default::default(),
        copies: Default::default(),
        background: Vec::new(),
    };
    for step in steps {
        let out = &step.output;
        // A message sent again at its offset, longer, replaces the earlier
        // row: keyed by its start, not its ID.
        for row in &out.messages {
            reading
                .messages
                .retain(|(offset, _), _| *offset != row.offset);
            reading.messages.insert(
                (row.offset, row.message.id.to_string()),
                serde_json::to_value(row).unwrap(),
            );
        }
        for event in &out.events {
            let (start, end) = event.range.unwrap_or_default();
            reading.events.insert(
                (start, end, event.kind.clone()),
                serde_json::to_value(event).unwrap(),
            );
        }
        for alias in &out.aliases {
            reading.aliases.insert(
                alias.name.to_string(),
                serde_json::to_value(&alias.target).unwrap(),
            );
        }
        for session in &out.sessions {
            match sessions.get_mut(&session.id.to_string()) {
                Some(stated) => stated.merge(session),
                None => {
                    sessions.insert(session.id.to_string(), session.clone());
                }
            }
        }
        for link in &out.links {
            reading.links.insert(serde_json::to_string(link).unwrap());
        }
        for copy in &out.prefix_copies {
            reading.copies.insert(serde_json::to_string(copy).unwrap());
        }
        for update in &out.background {
            reading
                .background
                .push(serde_json::to_value(update).unwrap());
        }
    }
    reading.sessions = sessions
        .into_iter()
        .map(|(id, s)| (id, serde_json::to_value(s).unwrap()))
        .collect();
    reading
}

/// Parses `bytes` in steps ending at `cuts`, carrying the parser state as
/// JSON between them, as appends to a growing file are parsed.
fn parse_in_steps(parser: &Parse, bytes: &[u8], cuts: &[usize]) -> Vec<Step> {
    let mut steps = Vec::new();
    let mut state: Option<Json> = None;
    let mut done = 0usize;
    for &cut in cuts.iter().chain(std::iter::once(&bytes.len())) {
        if cut < done {
            continue;
        }
        let (step, next) = parser
            .step(state.as_ref(), &bytes[done..cut], done as u64)
            .unwrap();
        done += step.consumed as usize;
        state = Some(serde_json::from_slice(&serde_json::to_vec(&next).unwrap()).unwrap());
        steps.push(step);
    }
    steps
}

/// The golden inputs of every origin read one record per line.
fn line_goldens() -> Vec<(String, PathBuf, Vec<u8>)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/sources");
    let mut cases = Vec::new();
    for origin in std::fs::read_dir(&root).unwrap() {
        let origin = origin.unwrap().path();
        let name = origin.file_name().unwrap().to_string_lossy().into_owned();
        let Some(spec) = crate::origin::by_name(&name) else {
            continue;
        };
        for case in std::fs::read_dir(&origin).unwrap() {
            let case = case.unwrap().path();
            let Ok(path) = std::fs::read_to_string(case.join("path")) else {
                continue;
            };
            let path = PathBuf::from(path.trim());
            let Some(input) = std::fs::read_dir(&case)
                .unwrap()
                .filter_map(Result::ok)
                .map(|e| e.path())
                .find(|p| p.file_stem().is_some_and(|s| s == "input"))
            else {
                continue;
            };
            if matches!((spec.open)(&path), Some(Parse::Lines(_))) {
                cases.push((name.clone(), path, std::fs::read(input).unwrap()));
            }
        }
    }
    cases.sort();
    assert!(
        cases.iter().any(|(origin, ..)| origin == "claude-code")
            && cases.iter().any(|(origin, ..)| origin == "codex")
            && cases.iter().any(|(origin, ..)| origin == "grok-cli"),
        "the goldens of the line origins are found"
    );
    cases
}

fn open(origin: &str, path: &Path) -> Parse {
    (crate::origin::by_name(origin).unwrap().open)(path).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// docs/architecture.md, Parsers: a chain parsed in pieces gives the
    /// reading it gives parsed at once, for every line origin's goldens.
    #[test]
    fn any_division_into_steps_gives_the_one_step_reading(
        cuts in proptest::collection::vec(0.0f64..1.0, 0..6),
    ) {
        for (origin, path, bytes) in line_goldens() {
            let parser = open(&origin, &path);
            let whole = reading(&settled(&parse_in_steps(&parser, &bytes, &[])));
            let mut cuts: Vec<usize> =
                cuts.iter().map(|f| (f * bytes.len() as f64) as usize).collect();
            cuts.sort_unstable();
            let divided = reading(&settled(&parse_in_steps(&parser, &bytes, &cuts)));
            prop_assert_eq!(&divided, &whole, "{} {} cut at {:?}", origin, path.display(), cuts);
        }
    }

    /// docs/archive.md, Readings: what a parser makes of a record depends
    /// only on the bytes up to the end of that record.
    #[test]
    fn a_prefix_reading_is_the_rows_of_the_whole_reading_that_end_within_it(
        cut in 0.0f64..1.0,
    ) {
        for (origin, path, bytes) in line_goldens() {
            let parser = open(&origin, &path);
            let (whole, _) = parser.step(None, &bytes, 0).unwrap();
            let cut = (cut * bytes.len() as f64) as usize;
            let (prefix, _) = parser.step(None, &bytes[..cut], 0).unwrap();
            let within = |step: &Step| ranged_rows(step, prefix.consumed);
            prop_assert_eq!(
                within(&prefix),
                within(&whole),
                "{} {} prefix of {} bytes",
                origin,
                path.display(),
                cut
            );
        }
    }
}

/// The rows a step holds that end within `end`, as values.
fn ranged_rows(step: &Step, end: u64) -> Vec<Json> {
    let out = &step.output;
    let mut rows = Vec::new();
    rows.extend(
        out.messages
            .iter()
            .filter(|row| row.end <= end)
            .map(|row| serde_json::to_value(row).unwrap()),
    );
    rows.extend(
        out.events
            .iter()
            .filter(|event| event.range.is_some_and(|(_, e)| e <= end))
            .map(|event| serde_json::to_value(event).unwrap()),
    );
    rows.extend(
        out.aliases
            .iter()
            .filter(|alias| alias.end <= end)
            .map(|alias| serde_json::to_value(alias).unwrap()),
    );
    rows.extend(
        out.documents
            .iter()
            .filter(|row| row.end <= end)
            .map(|row| serde_json::to_value(row).unwrap()),
    );
    rows
}

/// docs/architecture.md, Parsers: a parent, a placement, a link's branch
/// point and an alias's target are written by the name the source used, or
/// for a content-named message by its content and position: never by
/// something a parser looked up or remembered.
#[test]
fn rows_carry_names_as_the_source_writes_them() {
    for (origin, path, bytes) in line_goldens() {
        let parser = open(&origin, &path);
        let (step, _) = parser.step(None, &bytes, 0).unwrap();
        let out = &step.output;
        let text = String::from_utf8_lossy(&bytes);
        let named = out.aliases.iter().map(|alias| &alias.name);
        let written = out
            .messages
            .iter()
            .filter_map(|row| row.message.parent.as_ref())
            .chain(out.events.iter().filter_map(|event| event.after.as_ref()))
            .chain(
                out.links
                    .iter()
                    .filter_map(|link| link.branch_point.as_ref()),
            )
            .chain(out.aliases.iter().filter_map(|alias| alias.target.as_ref()))
            .chain(named);
        for id in written {
            let native = id.native().as_str();
            assert!(
                content_ids::as_written(id).is_some() || text.contains(native),
                "{origin} {}: {id} is not a name the input writes",
                path.display()
            );
        }
    }
}

fn session() -> SessionId {
    SessionId::parse("grok-cli:s").unwrap()
}

fn content_row(text: &str, position: u64) -> MessageRow {
    let blocks = vec![Block::Text { text: text.into() }];
    let id = content_ids::named(&session(), Role::User, &blocks, position).unwrap();
    let message = Message {
        id,
        session: session(),
        parent: None,
        role: Role::User,
        at: None,
        at_quality: polylogue_model::TimeQuality::Unknown,
        at_precision: None,
        at_until: None,
        model: None,
        prompt_kind: None,
        blocks,
        usage: Default::default(),
        stop_reason: None,
        reported_cost_usd: None,
        duration_ms: None,
        production: Default::default(),
        attrs: Default::default(),
    };
    MessageRow {
        offset: position,
        end: position + 1,
        id_kind: polylogue_model::IdKind::Content,
        page_text: false,
        response: None,
        message,
    }
}

fn step_of(rows: Vec<MessageRow>) -> Step {
    Step {
        output: Output {
            messages: rows,
            ..Output::default()
        },
        counts: Default::default(),
        consumed: 0,
        start: 0,
        repeats_whole_conversations: false,
    }
}

fn ids(steps: &[Step]) -> Vec<String> {
    settled(steps)
        .iter()
        .flat_map(|step| &step.output.messages)
        .map(|row| row.message.id.native().as_str().to_owned())
        .collect()
}

/// docs/archive.md, Readings: equal messages are numbered in order across
/// the chain, however its records divide into steps, and removing an
/// earlier copy gives the later one the lower number (`A X A` → `X A`).
#[test]
fn equal_content_named_messages_are_numbered_in_chain_order() {
    let at_once = ids(&[step_of(vec![
        content_row("A", 0),
        content_row("X", 1),
        content_row("A", 2),
    ])]);
    let in_steps = ids(&[
        step_of(vec![content_row("A", 0)]),
        step_of(vec![content_row("X", 1), content_row("A", 2)]),
    ]);
    assert_eq!(at_once, in_steps);
    assert!(at_once[0].ends_with("-0") && at_once[2].ends_with("-1"));
    assert_eq!(
        at_once[0][..at_once[0].len() - 2],
        at_once[2][..at_once[2].len() - 2]
    );
    let rewritten = ids(&[step_of(vec![content_row("X", 0), content_row("A", 1)])]);
    assert_eq!(rewritten, [at_once[1].clone(), at_once[0].clone()]);
}

/// A reference written as a content name and position names the message
/// settled from it, in its own step or an earlier one.
#[test]
fn a_reference_to_a_content_named_message_follows_its_number() {
    let first = content_row("A", 0);
    let second = content_row("A", 5);
    let mut child = content_row("B", 9);
    child.message.parent = Some(second.message.id.clone());
    let steps = settled(&[step_of(vec![first, second]), step_of(vec![child])]);
    let parent = steps[1].output.messages[0].message.parent.clone().unwrap();
    assert_eq!(parent, steps[0].output.messages[1].message.id);
    assert!(parent.native().as_str().ends_with("-1"));
}

#[test]
fn claude_sidechains_keep_their_sessions_when_parsed_in_pieces() {
    let bytes =
        include_bytes!("../../../../examples/sources/claude-code/two-sidechains/input.jsonl");
    let parser = open(
        "claude-code",
        Path::new(".claude/projects/p/two-sidechains.jsonl"),
    );
    let whole = reading(&settled(&parse_in_steps(&parser, bytes, &[])));
    for cut in 0..=bytes.len() {
        let divided = reading(&settled(&parse_in_steps(&parser, bytes, &[cut])));
        assert_eq!(divided, whole, "cut at {cut}");
    }
}
