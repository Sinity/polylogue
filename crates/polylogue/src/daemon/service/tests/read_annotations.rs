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
    crate::test_answers::read(Call::new("rows", json!({
        "selection":target,
        "steps":[{"verb":"read","budget":100_000,"hide":if hide { vec!["assertions"] } else { vec![] }}]
    })).answer(d))
}
fn annotations(page: &ReadPage, message: &str) -> Vec<Annotation> {
    let items: Vec<_> = page
        .document
        .items
        .iter()
        .filter_map(|item| {
            if item
                .source
                .message
                .as_ref()
                .is_some_and(|id| id.to_string() == message)
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
            links: vec![NewAssertionLink {
                role: LinkRole::About,
                reference: Reference::parse(message).unwrap(),
            }],
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
