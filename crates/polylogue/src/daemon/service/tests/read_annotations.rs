use super::*;
use crate::test_daemon::{self, Call};
use polylogue_model::{
    Reference, Timestamp,
    annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink},
    read::*,
    user::UserWrite,
};
use serde_json::json;

fn write(d: &Daemon, id: &str, annotation: NewAnnotation) {
    let outcome = d
        .writes
        .as_ref()
        .unwrap()
        .archive()
        .user(polylogue_store::fixture::authored::operator_request(
            id.into(),
            UserWrite::Assert {
                annotations: vec![annotation],
            },
        ))
        .unwrap();
    assert_eq!(outcome.id, id);
}
fn note(text: &str, targets: &[&str]) -> NewAnnotation {
    NewAnnotation {
        kind: AnnotationKind::Note,
        key: None,
        value: Some(json!(text)),
        valid_until: None,
        links: targets
            .iter()
            .map(|target| NewAnnotationLink {
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
    let parsed =
        crate::cli::line::parse(&words, &crate::cli::line::SavedNames::default(), false).unwrap();
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

fn annotations(page: &ReadPage, message: &str) -> Vec<AnnotationMark> {
    let items: Vec<_> = page
        .document
        .items
        .iter()
        .filter_map(|item| {
            if item.source.reference == format!("message:{message}")
                && let ItemKind::MessageHeader { header } = &item.kind
            {
                Some(header.annotations.clone())
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
    revised.links.push(NewAnnotationLink {
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
        NewAnnotation {
            kind: AnnotationKind::Seen,
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
    assert_eq!(
        annotations(&session, "claude-code:u10")
            .iter()
            .map(|annotation| annotation.value.clone())
            .collect::<Vec<_>>(),
        [json!("neighbor note")]
    );
    d.restart();
    d.pause_signals();
    assert_eq!(
        annotations(&read(&d.daemon, message, false), "claude-code:u1"),
        shown
    );
}

fn historical_call(
    target: &str,
    position: &impl std::fmt::Display,
    budget: u64,
    view: View,
) -> Call {
    let words = [
        target.to_string(),
        "read".into(),
        "--at".into(),
        position.to_string(),
        "--view".into(),
        view.name().into(),
        "--budget".into(),
        budget.to_string(),
    ];
    let parsed =
        crate::cli::line::parse(&words, &crate::cli::line::SavedNames::default(), false).unwrap();
    let steps = parsed
        .steps
        .into_iter()
        .map(|step| step.step)
        .collect::<Vec<_>>();
    Call::new("rows", json!({"selection":target,"steps":steps}))
}

#[test]
fn ordinary_session_reads_keep_message_and_span_revisions_at_two_positions() {
    for view in [View::Transcript, View::Context] {
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
        let original_call = historical_call(target, &first, 100000, view);
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
        if view == View::Context {
            for position in [&first, &second] {
                let expected = crate::test_answers::read(
                    historical_call(target, position, 100000, view).answer(&d.daemon),
                );
                let (items, _, _) = context_stream(
                    &d.daemon,
                    target,
                    ReadOptions {
                        view,
                        at: Some(ReadAt::try_from(position.to_string()).unwrap()),
                        ..Default::default()
                    },
                );
                assert_eq!(items, expected.document.items);
            }
        }
        let edited_call = historical_call(target, &second, 100000, view);
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
                &crate::test_answers::read(
                    historical_call(target, &before, 100000, view).answer(&d.daemon)
                ),
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
        historical_call("session:claude-code:s1", &position, 1, View::Transcript).answer(&d.daemon),
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
            ItemKind::MessageHeader { header } => Some(&header.annotations),
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
    foreign.index_instance = "another-generation".into();
    let mut future_index = current.clone();
    future_index.index += 1000;
    let mut future_archive = current;
    future_archive.archive += 1000;
    for position in [foreign, future_index, future_archive] {
        let refusal = crate::test_answers::refusal(
            historical_call(
                "session:claude-code:s1",
                &position,
                100000,
                View::Transcript,
            )
            .answer(&d.daemon),
        );
        assert_eq!(refusal.code.as_deref(), Some("bad_request"));
    }
}

#[test]
fn a_selected_message_with_an_opaque_numeric_id_is_its_context_anchor() {
    let mut d = test_daemon::start_with_sources_and_workers(
        "first prompt",
        &|home| {
            std::fs::write(
                home.join(".claude/projects/-p/numeric.jsonl"),
                ["2:3", "02:03"]
                    .iter()
                    .map(|id| polylogue_testkit::claude_user(id, "numeric", "numeric message"))
                    .collect::<String>(),
            )
            .unwrap();
        },
        2,
        2,
    );
    d.pause_signals();
    for message in ["claude-code:2:3", "claude-code:02:03"] {
        // The selection names the message; a numeric ID that also reads as
        // a position stays a message reference.
        let page = crate::test_answers::read(
            Call::new(
                "rows",
                json!({
                    "selection": format!("message:{message}"),
                    "steps": [{"verb": "read", "view": "context", "budget": 100000}],
                }),
            )
            .answer(&d.daemon),
        );
        assert!(
            page.document
                .items
                .iter()
                .any(|item| item.source.reference == format!("message:{message}"))
        );
        // The context at the first message does not include the later one.
        if message == "claude-code:2:3" {
            assert!(
                !page
                    .document
                    .items
                    .iter()
                    .any(|item| item.source.reference == "message:claude-code:02:03")
            );
        }
    }
}

fn time(text: &str) -> Timestamp {
    Timestamp::parse_rfc3339(text).unwrap()
}

fn write_at(d: &Daemon, id: &str, annotation: NewAnnotation, at: Timestamp) {
    let mut request = polylogue_store::fixture::authored::operator_request(
        id.into(),
        UserWrite::Assert {
            annotations: vec![annotation],
        },
    );
    request.at = at;
    d.writes.as_ref().unwrap().archive().user(request).unwrap();
}

#[test]
fn ordinary_session_annotation_times_keep_old_new_withdrawn_and_expired_versions() {
    for view in [View::Transcript, View::Context] {
        let now = time("2026-11-21T12:00:00Z");
        let mut d = test_daemon::start_with_clock_and_day(
            "neutral target text",
            polylogue_model::WallClock::new(move || now.into()),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::UTC,
                "00:00".parse().unwrap(),
            )
            .unwrap(),
        );
        d.pause_signals();
        let first = time("2026-11-21T08:00:00Z");
        let second = time("2026-11-21T09:00:00Z");
        let withdrawn = time("2026-11-21T10:00:00Z");
        let expired = time("2026-11-21T11:00:00Z");
        let target = "session:claude-code:s1";
        write_at(
            &d.daemon,
            "time-message",
            note("old message", &["message:claude-code:u1"]),
            first,
        );
        let mut span = note("old span", &["span:block:claude-code:u1#0#c=0-6"]);
        span.valid_until = Some(expired);
        write_at(&d.daemon, "time-span", span, first);

        // Restore an actual undated authored record through the product writer.
        // Its receipt time must never become the fact's authored time.
        let mut unknown = d
            .daemon
            .reads
            .reader()
            .unwrap()
            .annotation(&polylogue_model::RevisionReference::exact(
                "time-message".into(),
                "time-message".into(),
            ))
            .unwrap()
            .unwrap()
            .fact;
        unknown.id = "time-unknown".into();
        unknown.root = unknown.id.clone();
        unknown.created_at = None;
        unknown.value = Some(json!("undated note"));
        d.daemon
            .writes
            .as_ref()
            .unwrap()
            .archive()
            .restore_annotation(unknown, None)
            .unwrap();

        for (root, text) in [("time-message", "new message"), ("time-span", "new span")] {
            let _ = crate::test_answers::step(Call::new("rows", json!({"selection":format!("annotation:{root}@{root}"),"steps":[{"verb":"edit","text":text}]})).at(second).answer(&d.daemon));
        }
        let _ = crate::test_answers::step(Call::new("rows", json!({"selection":"annotation:time-message","steps":[{"verb":"withdraw","reason":"no longer current"}]})).at(withdrawn).answer(&d.daemon));
        // Current reads still expose unknown authored time, and apply current expiry.
        assert_eq!(
            annotations(&read(&d.daemon, target, false), "claude-code:u1")[0].value,
            json!("undated note")
        );
        let cases = [
            (first, vec![json!("old message"), json!("old span")]),
            (second, vec![json!("new message"), json!("new span")]),
            (withdrawn, vec![json!("new span")]),
            (expired, vec![]),
        ];
        for restart in [false, true] {
            if restart {
                d.restart();
                d.pause_signals();
            }
            for (at, expected) in &cases {
                let call = historical_call(target, at, 100000, view);
                if view == View::Context {
                    let expected = crate::test_answers::read(call.clone().answer(&d.daemon));
                    let (items, _, _) = context_stream(
                        &d.daemon,
                        target,
                        ReadOptions {
                            view,
                            at: Some(ReadAt::Time(*at)),
                            ..Default::default()
                        },
                    );
                    assert_eq!(items, expected.document.items);
                }
                for page in [call.clone().answer(&d.daemon), call.answer(&d.daemon)] {
                    let page = crate::test_answers::read(page);
                    assert_eq!(
                        annotations(&page, "claude-code:u1")
                            .iter()
                            .map(|a| a.value.clone())
                            .collect::<Vec<_>>(),
                        *expected
                    );
                    let notice = format!(
                        "Annotations as of {at}, by authored time; approximate across machines. Unknown authored times are omitted. Expiry uses this time."
                    );
                    assert!(
                        page.document.items.iter().any(
                            |item| matches!(&item.kind, ItemKind::Text { text } if *text == notice)
                        ),
                        "{at}"
                    );
                }
            }
        }
        // Include the local spellings that also resemble three-part positions.
        for at in ["08:00", "08:00:00", "2026-11-21T08:00:00"] {
            let page = crate::test_answers::read(
                historical_call(target, &at, 100000, view).answer(&d.daemon),
            );
            assert_eq!(
                annotations(&page, "claude-code:u1")
                    .iter()
                    .map(|a| a.value.clone())
                    .collect::<Vec<_>>(),
                cases[0].1
            );
        }
        // A written second and its fractional instant choose the same boundary.
        // The immediately preceding/following milliseconds straddle a real edit.
        for (at, expected) in [
            ("2026-11-21T08:59:59.999Z", &cases[0].1),
            ("2026-11-21T09:00:00Z", &cases[1].1),
            ("2026-11-21T09:00:00.000Z", &cases[1].1),
            ("2026-11-21T09:00:00.001Z", &cases[1].1),
            ("2026-11-21T10:59:59.999Z", &cases[2].1),
            ("2026-11-21T11:00:00.000Z", &cases[3].1),
        ] {
            let page = crate::test_answers::read(
                historical_call(target, &at, 100000, view).answer(&d.daemon),
            );
            assert_eq!(
                &annotations(&page, "claude-code:u1")
                    .iter()
                    .map(|a| a.value.clone())
                    .collect::<Vec<_>>(),
                expected,
                "{at}"
            );
        }
        // Calendar periods use the same lower-bound rule, in the configured day.
        // A note authored the day before is the only one the period start admits.
        write_at(
            &d.daemon,
            "time-early",
            note("early note", &["message:claude-code:u1"]),
            time("2026-11-20T12:00:00Z"),
        );
        for at in ["today", "2026-11-21"] {
            let page = crate::test_answers::read(
                historical_call(target, &at, 100000, view).answer(&d.daemon),
            );
            assert_eq!(
                annotations(&page, "claude-code:u1")
                    .iter()
                    .map(|a| a.value.clone())
                    .collect::<Vec<_>>(),
                [json!("early note")],
                "{at}"
            );
        }
    }
}

#[test]
fn session_annotation_time_continuations_freeze_relative_time_through_edit_and_restart() {
    use std::sync::{
        Arc,
        atomic::{AtomicI64, Ordering},
    };
    let original_now = time("2026-11-21T12:00:00Z");
    let clock = Arc::new(AtomicI64::new(original_now.unix_ms()));
    let read_clock = clock.clone();
    let mut d = test_daemon::start_with_clock(
        "neutral target text",
        polylogue_model::WallClock::new(move || {
            jiff::Timestamp::from_millisecond(read_clock.load(Ordering::SeqCst)).unwrap()
        }),
    );
    d.pause_signals();
    write_at(
        &d.daemon,
        "time-page",
        note("the original timed note", &["message:claude-code:u1"]),
        time("2026-11-21T10:00:00Z"),
    );
    let first = crate::test_answers::read(
        historical_call("session:claude-code:s1", &"1h", 1, View::Transcript).answer(&d.daemon),
    );
    let resume = first
        .document
        .footer
        .next
        .clone()
        .expect("the small page continues");
    let _ = crate::test_answers::step(Call::new("rows", json!({"selection":"annotation:time-page","steps":[{"verb":"edit","text":"a later timed note"}]})).answer(&d.daemon));
    clock.store(time("2026-11-22T12:00:00Z").unix_ms(), Ordering::SeqCst);
    d.restart();
    d.pause_signals();
    let mut next = Some(resume);
    let mut previous = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut items = first.document.items;
    while let Some(ready) = next {
        assert_eq!(
            ready.request.arguments["steps"][0]["at"],
            "2026-11-21T11:00:00Z"
        );
        assert!(seen.insert(serde_json::to_string(&ready.request).unwrap()));
        let page = crate::test_answers::read(Call::from_request(&ready.request).answer(&d.daemon));
        let repeated =
            crate::test_answers::read(Call::from_request(&ready.request).answer(&d.daemon));
        assert_eq!(page.document.items, repeated.document.items);
        previous = page.document.footer.previous.or(previous);
        next = page.document.footer.next;
        items.extend(page.document.items);
    }
    let previous = previous.expect("a later page has an executable previous page");
    assert_eq!(
        previous.request.arguments["steps"][0]["at"],
        "2026-11-21T11:00:00Z"
    );
    let page = crate::test_answers::read(Call::from_request(&previous.request).answer(&d.daemon));
    let repeated =
        crate::test_answers::read(Call::from_request(&previous.request).answer(&d.daemon));
    assert_eq!(page.document.items, repeated.document.items);
    assert!(
        !serde_json::to_string(&page.document.items)
            .unwrap()
            .contains("a later timed note")
    );
    let notes = items
        .iter()
        .filter_map(|item| match &item.kind {
            ItemKind::MessageHeader { header } => Some(&header.annotations),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].value, json!("the original timed note"));
}

#[test]
fn annotation_time_selection_preserves_invalid_time_refusals() {
    let d = test_daemon::start("neutral target text");
    for at in ["not-a-time", "last 0h", "2026-13-01"] {
        let refusal = crate::test_answers::refusal(
            historical_call("session:claude-code:s1", &at, 100000, View::Transcript)
                .answer(&d.daemon),
        );
        assert_eq!(refusal.code.as_deref(), Some("bad_request"), "{refusal:?}");
    }
}

#[test]
fn historical_context_pages_bind_the_message_and_cutoff_through_append_and_reopen() {
    use std::io::Write;
    for (by_time, physical) in [(false, false), (true, false), (true, true)] {
        let now = time("2026-11-21T12:00:00Z");
        let mut d = test_daemon::start_with_sources_and_workers(
            "neutral prior text",
            &|home| {
                std::fs::OpenOptions::new()
                    .append(true)
                    .open(home.join(".claude/projects/-p/s1.jsonl"))
                    .unwrap()
                    .write_all(
                        polylogue_testkit::claude_assistant(
                            "a1",
                            "s1",
                            "the accepted context anchor",
                        )
                        .as_bytes(),
                    )
                    .unwrap();
            },
            1,
            2,
        );
        let path = d.home.join(".claude/projects/-p/s1.jsonl");
        d.pause_signals();
        write_at(
            &d.daemon,
            "context-note",
            note("the original context note", &["message:claude-code:u1"]),
            time("2026-11-21T10:00:00Z"),
        );
        write_at(
            &d.daemon,
            "context-span",
            note(
                "the original context span",
                &["span:block:claude-code:u1#0#c=0-6"],
            ),
            time("2026-11-21T10:00:00Z"),
        );
        let position = d.daemon.reads.reader().unwrap().change_position().unwrap();
        let at = if by_time {
            "1h".into()
        } else {
            position.to_string()
        };
        let expected_at = if by_time {
            "2026-11-21T11:00:00Z".into()
        } else {
            position.to_string()
        };
        if !by_time {
            let range = Reference::MessageRange {
                first: polylogue_model::MessageId::parse("claude-code:u1").unwrap(),
                last: polylogue_model::MessageId::parse("claude-code:a1").unwrap(),
            }
            .to_string();
            let error = crate::test_answers::refusal(
                historical_call(&range, &at, 100000, View::Context).answer(&d.daemon),
            );
            assert_eq!(error.code.as_deref(), Some("bad_request"));
            assert!(
                error
                    .message
                    .starts_with("historical context takes one message or a session")
            );
        }
        let target = if physical {
            "root:claude-code:s1"
        } else {
            "session:claude-code:s1"
        };
        let mut call = historical_call(target, &at, 1, View::Context);
        call.arguments["steps"][0]["physical"] = json!(physical);
        let first = crate::test_answers::read(call.at(now).answer(&d.daemon));
        assert_eq!(
            first.snapshot.query, target,
            "the acknowledged reading retains its original selector"
        );
        assert_eq!(first.document.header.target, target);
        let resume = first
            .document
            .footer
            .next
            .clone()
            .expect("the context needs more than one tiny page");
        if !physical {
            let mut forged = resume.clone();
            forged.request.arguments["steps"][0]["physical"] = json!(true);
            let error =
                crate::test_answers::refusal(Call::from_request(&forged.request).answer(&d.daemon));
            assert_eq!(error.code.as_deref(), Some("invalid_query"));
            assert_eq!(
                error.message,
                "the read cursor belongs to another reading: the reading options differ"
            );
        }
        let _ = crate::test_answers::step(Call::new("rows", json!({"selection":"annotation:context-note","steps":[{"verb":"edit","text":"a later context note"}]})).at(now).answer(&d.daemon));
        std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap()
            .write_all(
                polylogue_testkit::claude_user(
                    "u2",
                    "s1",
                    "a later message outside the accepted context",
                )
                .as_bytes(),
            )
            .unwrap();
        d.restart();
        d.wait_for(|status| {
            status["store"]["messages"] == 3 && status["store"]["search_pending"] == 0
        });
        d.pause_signals();
        let mut next = Some(resume);
        let mut previous = None;
        let mut items = first.document.items;
        let mut seen = std::collections::BTreeSet::new();
        while let Some(ready) = next {
            assert_eq!(
                ready.request.arguments["references"],
                json!(["message:claude-code:a1"])
            );
            assert_eq!(ready.request.arguments["steps"][0]["at"], expected_at);
            assert_eq!(ready.request.arguments["steps"][0]["view"], "context");
            let rows: crate::protocol::line::Rows =
                serde_json::from_value(ready.request.arguments.clone()).unwrap();
            let [crate::protocol::line::Step::Read(step)] = rows.steps.as_slice() else {
                panic!("one bound read")
            };
            assert_eq!(step.options.physical, physical);
            assert!(
                seen.insert(serde_json::to_string(&ready.request).unwrap()),
                "the context continuation advances"
            );
            let page =
                crate::test_answers::read(Call::from_request(&ready.request).answer(&d.daemon));
            let repeat =
                crate::test_answers::read(Call::from_request(&ready.request).answer(&d.daemon));
            assert_eq!(page.document.items, repeat.document.items);
            previous = page.document.footer.previous.or(previous);
            next = page.document.footer.next;
            items.extend(page.document.items);
        }
        let headers = items
            .iter()
            .filter_map(|item| {
                matches!(&item.kind, ItemKind::MessageHeader { .. })
                    .then_some(item.source.reference.as_str())
            })
            .collect::<Vec<_>>();
        assert_eq!(
            headers,
            ["message:claude-code:u1", "message:claude-code:a1"]
        );
        let notes = items
            .iter()
            .filter_map(|item| match &item.kind {
                ItemKind::MessageHeader { header } => Some(&header.annotations),
                _ => None,
            })
            .flatten()
            .map(|note| note.value.clone())
            .collect::<Vec<_>>();
        assert_eq!(
            notes,
            [
                json!("the original context note"),
                json!("the original context span")
            ]
        );
        let previous = previous.expect("the accepted context has an executable previous page");
        assert_eq!(
            previous.request.arguments["references"],
            json!(["message:claude-code:a1"])
        );
        assert_eq!(previous.request.arguments["steps"][0]["at"], expected_at);
        let page =
            crate::test_answers::read(Call::from_request(&previous.request).answer(&d.daemon));
        assert!(
            !serde_json::to_string(&page.document.items)
                .unwrap()
                .contains("a later context note")
        );
        let fresh = crate::test_answers::read(
            historical_call(target, &at, 100000, View::Context).answer(&d.daemon),
        );
        assert!(
            fresh
                .document
                .items
                .iter()
                .any(|item| item.source.reference == "message:claude-code:u2"),
            "a new request may choose the now-latest context"
        );
    }
}

/// Decode the finite read protocol, preserving original frames independently
/// from presentation so a hidden block cannot pass a layout-only annotation.
pub(super) fn context_stream(
    d: &Daemon,
    target: &str,
    options: ReadOptions,
) -> (
    Vec<ReadItem>,
    Vec<polylogue_model::bundle::Record>,
    ReadFooter,
) {
    let request = polylogue_reading::ReadRequest {
        start: Default::default(),
        references: None,
        target: target.into(),
        options,
        delivery: ReadDelivery::Stream,
        format: polylogue_model::read::Format::Jsonl,
        budget: polylogue_model::read::ReadBudget::Default,
        alternates: Vec::new(),
        cursor: None,
    };
    let actor = polylogue_model::authorship::Authorship {
        author: polylogue_model::Author::Operator,
        via: polylogue_model::authorship::Channel::Cli,
    };
    let operation = d.operation_context(&actor).unwrap();
    let reader = d.reads.reader().unwrap();
    let mut bytes = Vec::new();
    let complete = polylogue_reading::stream(&reader, d, &request, &operation, &mut bytes).unwrap();
    let frames = command_surface::SseDecoder::new().push(&bytes).unwrap();
    assert_eq!(frames.first().unwrap().event, "begin");
    assert_eq!(frames.last().unwrap().event, "complete");
    assert_eq!(
        frames.iter().filter(|frame| frame.event == "begin").count(),
        1
    );
    assert_eq!(
        frames
            .iter()
            .filter(|frame| frame.event == "complete")
            .count(),
        1
    );
    let mut items = Vec::new();
    let mut originals = Vec::new();
    for frame in &frames[1..frames.len() - 1] {
        assert_eq!(frame.event, "record");
        let record: ReadRecord = serde_json::from_str(&frame.data).unwrap();
        if let Some(original) = record.record() {
            originals.push(original.clone());
        }
        for part in record.parts() {
            if let ReadPart::Item(item) = part {
                items.push(item.clone());
            }
        }
    }
    assert_eq!(complete.shown, items.len() as u64);
    assert_eq!(
        serde_json::from_str::<ReadFooter>(&frames.last().unwrap().data).unwrap(),
        complete
    );
    let refs: std::collections::HashSet<_> =
        originals.iter().map(|record| record.reference()).collect();
    assert_eq!(refs.len(), originals.len(), "each original is emitted once");
    (items, originals, complete)
}

#[test]
fn historical_context_refuses_around_and_fresh_physical_message_options() {
    let d = test_daemon::start("neutral target text");
    let mut around = historical_call(
        "message:claude-code:u1",
        &"2026-11-21T10:00:00Z",
        100000,
        View::Context,
    );
    around.arguments["steps"][0]["around"] = json!(1);
    let error = crate::test_answers::refusal(around.answer(&d.daemon));
    assert_eq!(error.code.as_deref(), Some("bad_request"));
    assert_eq!(
        error.message,
        "historical context does not accept around; select its context message"
    );
    let mut physical = historical_call(
        "message:claude-code:u1",
        &"2026-11-21T10:00:00Z",
        100000,
        View::Context,
    );
    physical.arguments["steps"][0]["physical"] = json!(true);
    let error = crate::test_answers::refusal(physical.answer(&d.daemon));
    assert_eq!(error.code.as_deref(), Some("bad_request"));
    assert_eq!(
        error.message,
        "--physical applies to root:REF; a session reference is already one physical session"
    );
}

#[test]
fn current_and_historical_context_streams_share_the_page_projection() {
    let d = test_daemon::start("neutral target text");
    for at in [None, Some("2026-11-21T10:00:00Z")] {
        let call = match at {
            Some(at) => historical_call("session:claude-code:s1", &at, 100000, View::Context),
            None => Call::new(
                "rows",
                json!({"selection":"session:claude-code:s1","steps":[{"verb":"read","view":"context","budget":100000}]}),
            ),
        };
        let page = crate::test_answers::read(call.answer(&d.daemon));
        let (items, _, complete) = context_stream(
            &d.daemon,
            "session:claude-code:s1",
            ReadOptions {
                view: View::Context,
                at: at.map(|at| ReadAt::try_from(at.to_owned()).unwrap()),
                ..Default::default()
            },
        );
        assert_eq!(items, page.document.items);
        assert_eq!(complete.counts.sessions, 1);
        assert_eq!(complete.counts.messages, 1);
    }
}

#[test]
fn context_stream_stops_on_scope_reset_and_cancellation_before_source_frames() {
    use std::io::Write;
    struct Interrupt<'a> {
        bytes: Vec<u8>,
        action: Option<Box<dyn FnOnce() + 'a>>,
    }
    impl Write for Interrupt<'_> {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.bytes.extend_from_slice(bytes);
            if let Some(action) = self.action.take() {
                action();
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    for scope_reset in [false, true] {
        let mut d = test_daemon::start("neutral context text");
        d.pause_signals();
        let actor = polylogue_model::authorship::Authorship {
            author: polylogue_model::Author::Operator,
            via: polylogue_model::authorship::Channel::Cli,
        };
        let operation = d.daemon.operation_context(&actor).unwrap();
        let limit = polylogue_store::read::limit::ReadLimit::new(
            crate::daemon::execution::default_read_deadline(),
            Default::default(),
        );
        let cancellation = limit.cancellation.clone();
        let archive = &d.daemon.archive;
        let mut output = Interrupt {
            bytes: vec![],
            action: Some(Box::new(move || {
                if scope_reset {
                    archive.advance_generation_for_test().unwrap();
                } else {
                    cancellation.cancel();
                }
            })),
        };
        let result = limit.scope(|| {
            let reader = d.daemon.reads.reader()?;
            polylogue_reading::stream(
                &reader,
                d.daemon.as_ref(),
                &polylogue_reading::ReadRequest {
                    start: Default::default(),
                    references: None,
                    target: "session:claude-code:s1".into(),
                    options: ReadOptions {
                        view: View::Context,
                        ..Default::default()
                    },
                    delivery: ReadDelivery::Stream,
                    format: polylogue_model::read::Format::Jsonl,
                    budget: polylogue_model::read::ReadBudget::Default,
                    alternates: Vec::new(),
                    cursor: None,
                },
                &operation,
                &mut output,
            )
        });
        assert_eq!(
            result.unwrap_err().code(),
            if scope_reset {
                polylogue_model::Code::ScopeChanged
            } else {
                polylogue_model::Code::Cancelled
            }
        );
        let frames = command_surface::SseDecoder::new()
            .push(&output.bytes)
            .unwrap();
        assert_eq!(
            frames
                .iter()
                .map(|frame| frame.event.as_str())
                .collect::<Vec<_>>(),
            ["begin"]
        );
    }
}
