use super::*;
use crate::test_daemon::{self, Call};
use polylogue_model::{
    Reference, Timestamp,
    assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink},
    read::*,
    user::UserWrite,
};
use serde_json::json;

fn write(d: &Daemon, id: &str, assertion: NewAssertion) {
    let outcome = d
        .writes
        .as_ref()
        .unwrap()
        .user(polylogue_store::fixture::authored::operator_request(
            id.into(),
            UserWrite::Assert {
                assertions: vec![assertion],
            },
        ))
        .unwrap();
    assert_eq!(outcome.id, id);
}
fn note(text: &str, targets: &[&str]) -> NewAssertion {
    NewAssertion {
        kind: AssertionKind::Note,
        key: None,
        value: Some(json!(text)),
        valid_until: None,
        links: targets
            .iter()
            .map(|target| NewAssertionLink {
                role: LinkRole::About,
                reference: Reference::parse(target).unwrap(),
            })
            .collect(),
    }
}
fn read(d: &Daemon, target: &str, hide: bool) -> ReadPage {
    let words = [
        target,
        "read",
        "--budget",
        "100000",
        if hide { "--hide" } else { "--show" },
        if hide {
            "annotations"
        } else {
            "annotations:header"
        },
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    let parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    assert_eq!(
        parsed.selection,
        Some(crate::cli::line::SelectionWord::Text(target.to_owned()))
    );
    let steps = parsed
        .steps
        .into_iter()
        .map(|step| step.step)
        .collect::<Vec<_>>();
    crate::test_answers::read(
        Call::new("rows", json!({"selection":target,"steps":steps})).answer(d),
    )
}

fn annotations(page: &ReadPage, message: &str) -> Vec<Annotation> {
    let items: Vec<_> = page
        .document
        .items
        .iter()
        .filter_map(|item| {
            if item.source.reference == format!("message:{message}")
                && let ItemKind::MessageHeader { header } = &item.kind
            {
                Some(header.assertions.clone())
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        items.len(),
        1,
        "one header for {message}: {:?}",
        page.document.items
    );
    items.into_iter().next().unwrap()
}

#[test]
fn message_annotations_include_original_spans_once_and_keep_current_user_facts() {
    let mut d = test_daemon::start_with_sources_and_workers(
        "neutral target text",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/neighbor.jsonl"),
                polylogue_testkit::claude_user("u10", "s1", "neutral neighbor text"),
            )
            .unwrap();
        },
        1,
        2,
    );
    d.pause_signals();
    let message = "message:claude-code:u1";
    let span = "span:message:claude-code:u1#c=0-6";
    let block_span = "span:block:claude-code:u1#0#c=7-13";
    write(&d.daemon, "direct-note", note("direct note", &[message]));
    write(&d.daemon, "span-old", note("superseded span note", &[span]));
    let mut revised = note("current span note", &[span]);
    revised.links.push(NewAssertionLink {
        role: LinkRole::Revises,
        reference: Reference::parse("annotation:span-old@span-old").unwrap(),
    });
    write(&d.daemon, "span-new", revised);
    write(
        &d.daemon,
        "block-note",
        note("block span note", &[block_span]),
    );
    write(
        &d.daemon,
        "same-note",
        note("same note via two links", &[message, span]),
    );
    write(
        &d.daemon,
        "neighbor-note",
        note("neighbor note", &["span:message:claude-code:u10#c=0-6"]),
    );
    let mut expired = note("expired note", &[span]);
    expired.valid_until = Some(Timestamp::from_unix_ms(2).unwrap());
    write(&d.daemon, "expired-note", expired);
    let at = d.daemon.reads.reader().unwrap().change_position().unwrap();
    write(
        &d.daemon,
        "internal-mark",
        NewAssertion {
            kind: AssertionKind::Seen,
            key: Some(message.into()),
            value: Some(json!({"position":{"item":null,"at":at.to_string()}})),
            valid_until: None,
            links: vec![],
        },
    );
    let page = read(&d.daemon, message, false);
    let shown = annotations(&page, "claude-code:u1");
    assert_eq!(
        shown
            .iter()
            .map(|annotation| annotation.reference.as_str())
            .collect::<Vec<_>>(),
        [
            "annotation:block-note@block-note",
            "annotation:direct-note@direct-note",
            "annotation:same-note@same-note",
            "annotation:span-old@span-new"
        ]
    );
    assert!(shown.iter().all(|annotation| annotation.kind == "note"));
    assert_eq!(shown.last().unwrap().value, json!("current span note"));
    for format in [Format::Terminal, Format::Markdown, Format::Html] {
        let mut bytes = Vec::new();
        polylogue_render::encode_document(&page.document, format, Default::default(), &mut bytes)
            .unwrap();
        let output = String::from_utf8(bytes).unwrap();
        for text in [
            "direct note",
            "current span note",
            "block span note",
            "same note via two links",
        ] {
            assert_eq!(output.matches(text).count(), 1, "{format:?}: {output}");
        }
        for text in [
            "neighbor note",
            "expired note",
            "superseded span note",
            "internal-mark",
        ] {
            assert!(!output.contains(text), "{format:?}: {output}");
        }
    }
    assert!(annotations(&read(&d.daemon, message, true), "claude-code:u1").is_empty());
    let session = read(&d.daemon, "session:claude-code:s1", false);
    assert_eq!(annotations(&session, "claude-code:u1"), shown);
    assert_eq!(annotations(&session, "claude-code:u10").len(), 1);
    d.restart();
    d.pause_signals();
    assert_eq!(
        annotations(&read(&d.daemon, message, false), "claude-code:u1"),
        shown
    );
}

fn historical_call(target: &str, position: &polylogue_model::ChangePosition, budget: u64) -> Call {
    let words = [
        target.to_string(),
        "read".into(),
        "--at".into(),
        position.to_string(),
        "--budget".into(),
        budget.to_string(),
    ];
    let parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    let steps = parsed
        .steps
        .into_iter()
        .map(|step| step.step)
        .collect::<Vec<_>>();
    Call::new("rows", json!({"selection":target,"steps":steps}))
}

#[test]
fn ordinary_session_reads_keep_message_and_span_revisions_at_two_positions() {
    let mut d = test_daemon::start("neutral target text");
    d.pause_signals();
    let target = "session:claude-code:s1";
    let message = "message:claude-code:u1";
    let span = "span:block:claude-code:u1#0#c=0-6";
    let before = d.daemon.reads.reader().unwrap().change_position().unwrap();
    write(
        &d.daemon,
        "message-original",
        note("original message note", &[message]),
    );
    write(
        &d.daemon,
        "span-original",
        note("original span note", &[span]),
    );
    let first = d.daemon.reads.reader().unwrap().change_position().unwrap();
    let original_call = historical_call(target, &first, 100000);
    let original = crate::test_answers::read(original_call.clone().answer(&d.daemon));
    let original_notes = annotations(&original, "claude-code:u1");
    assert_eq!(
        original_notes
            .iter()
            .map(|note| note.value.clone())
            .collect::<Vec<_>>(),
        [json!("original message note"), json!("original span note")]
    );

    // Issue both edits through the ordinary line executor, then retry the
    // accepted old-position read after the predicate's visible facts changed.
    for (root, text) in [
        ("message-original", "edited message note"),
        ("span-original", "edited span note"),
    ] {
        let answer = Call::new("rows", json!({"selection":format!("annotation:{root}@{root}"),"steps":[{"verb":"edit","text":text}]})).answer(&d.daemon);
        let _ = crate::test_answers::step(answer);
    }
    let second = d.daemon.reads.reader().unwrap().change_position().unwrap();
    assert!(second.archive > first.archive);
    let edited_call = historical_call(target, &second, 100000);
    let edited = crate::test_answers::read(edited_call.clone().answer(&d.daemon));
    let edited_notes = annotations(&edited, "claude-code:u1");
    assert_eq!(
        edited_notes
            .iter()
            .map(|note| note.value.clone())
            .collect::<Vec<_>>(),
        [json!("edited message note"), json!("edited span note")]
    );
    assert_eq!(
        annotations(
            &crate::test_answers::read(original_call.clone().answer(&d.daemon)),
            "claude-code:u1"
        ),
        original_notes
    );
    assert!(
        annotations(
            &crate::test_answers::read(historical_call(target, &before, 100000).answer(&d.daemon)),
            "claude-code:u1"
        )
        .is_empty()
    );

    // A later withdrawal must not erase either past revision, and both
    // revisions remain addressable through the ordinary annotation query.
    let _ = crate::test_answers::step(Call::new("rows", json!({"selection":"annotation:message-original","steps":[{"verb":"withdraw","reason":"no longer current"}]})).answer(&d.daemon));
    assert_eq!(
        annotations(&read(&d.daemon, target, false), "claude-code:u1").len(),
        1
    );
    for root in ["message-original", "span-original"] {
        let history = crate::test_answers::listing(Call::new("rows", json!({"selection":format!("annotations where root:annotation:{root} withdrawn:*")})).answer(&d.daemon));
        assert_eq!(history.value().unwrap()["count"], 2, "{history:?}");
        let raw_root = crate::test_answers::listing(
            Call::new(
                "rows",
                json!({"selection":format!("annotations where root:{root} withdrawn:*")}),
            )
            .answer(&d.daemon),
        );
        assert_eq!(raw_root.value().unwrap()["count"], 2, "{raw_root:?}");
        assert_eq!(
            history.value().unwrap()["rows"],
            raw_root.value().unwrap()["rows"]
        );
    }
    d.restart();
    d.pause_signals();
    for (call, expected) in [(original_call, original_notes), (edited_call, edited_notes)] {
        let page = crate::test_answers::read(call.answer(&d.daemon));
        assert_eq!(annotations(&page, "claude-code:u1"), expected);
    }
}

#[test]
fn historical_session_continuations_keep_the_position_after_new_revisions_and_restart() {
    let mut d = test_daemon::start("neutral target text");
    d.pause_signals();
    write(
        &d.daemon,
        "paged-note",
        note("the original paged note", &["message:claude-code:u1"]),
    );
    let position = d.daemon.reads.reader().unwrap().change_position().unwrap();
    let first = crate::test_answers::read(
        historical_call("session:claude-code:s1", &position, 1).answer(&d.daemon),
    );
    let resume = first
        .document
        .footer
        .next
        .clone()
        .expect("the small page continues");
    let _ = crate::test_answers::step(Call::new("rows", json!({"selection":"annotation:paged-note","steps":[{"verb":"edit","text":"a later paged note"}]})).answer(&d.daemon));
    d.restart();
    d.pause_signals();
    let mut next = Some(resume);
    let mut seen = std::collections::BTreeSet::new();
    let mut items = first.document.items;
    while let Some(ready) = next {
        assert_eq!(
            ready.request.arguments["steps"][0]["at"],
            position.to_string()
        );
        assert!(
            seen.insert(serde_json::to_string(&ready.request).unwrap()),
            "continuation must advance"
        );
        let page = crate::test_answers::read(Call::from_request(&ready.request).answer(&d.daemon));
        let repeated =
            crate::test_answers::read(Call::from_request(&ready.request).answer(&d.daemon));
        assert_eq!(page.document.items, repeated.document.items);
        next = page.document.footer.next;
        items.extend(page.document.items);
    }
    let notes = items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::MessageHeader { header } => Some(&header.assertions),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].value, json!("the original paged note"));
}

#[test]
fn historical_session_positions_refuse_another_generation_and_future_counters() {
    let d = test_daemon::start("neutral target text");
    let current = d.daemon.reads.reader().unwrap().change_position().unwrap();
    let mut foreign = current.clone();
    foreign.generation = "another-generation".into();
    let mut future_index = current.clone();
    future_index.index += 1000;
    let mut future_archive = current;
    future_archive.archive += 1000;
    for position in [foreign, future_index, future_archive] {
        let refusal = crate::test_answers::refusal(
            historical_call("session:claude-code:s1", &position, 100000).answer(&d.daemon),
        );
        assert_eq!(refusal.code.as_deref(), Some("bad_request"));
    }
}

#[test]
fn context_read_at_keeps_opaque_numeric_message_coordinates() {
    let mut d = test_daemon::start_with_sources_and_workers(
        "first prompt",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/numeric.jsonl"),
                polylogue_testkit::claude_user("2:3", "numeric", "numeric message"),
            )
            .unwrap();
        },
        1,
        2,
    );
    d.pause_signals();
    let words = [
        "session:claude-code:numeric",
        "read",
        "--view",
        "context",
        "--at",
        "claude-code:2:3",
        "--budget",
        "100000",
    ]
    .map(str::to_owned);
    let parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    let page = crate::test_answers::read(Call::new("rows", json!({"selection":"session:claude-code:numeric", "steps":parsed.steps.into_iter().map(|step|step.step).collect::<Vec<_>>()})).answer(&d.daemon));
    assert!(
        page.document
            .items
            .iter()
            .any(|item| item.source.reference == "message:claude-code:2:3")
    );
}
