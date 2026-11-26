use super::*;
use crate::test_daemon::{self, Call};
use polylogue_model::read::{ReadAt, ReadOptions, View};
use proptest::prelude::*;
use serde_json::json;

fn refusal(format: &str) -> (String, String) {
    let step = ReadStep {
        format: Some(format.into()),
        ..ReadStep::default()
    };
    let failure = request_for(step, "sessions".into()).expect_err("format is refused");
    (failure.code().to_owned(), failure.error.message.clone())
}

/// A typo is a bad value naming the formats, not a planned feature.
#[test]
fn an_unknown_format_is_a_bad_value_naming_the_formats() {
    assert_eq!(
        refusal("markdwon"),
        (
            "bad_value".to_owned(),
            "unknown format `markdwon`; formats are terminal, markdown, html, json and jsonl"
                .to_owned()
        )
    );
}

/// A documented format whose encoder is not built is reported as such.
#[test]
fn a_documented_format_without_an_encoder_is_not_built() {
    assert_eq!(
        refusal("citation"),
        (
            "not_built".to_owned(),
            "read --format citation is not built yet".to_owned()
        )
    );
}

#[test]
fn a_known_format_is_accepted() {
    let step = ReadStep {
        format: Some("jsonl".into()),
        ..ReadStep::default()
    };
    let read = request_for(step, "sessions".into()).unwrap();
    assert_eq!(read.format, Format::Jsonl);
}

/// Every built option, as a caller writes it on any surface.
fn every_option() -> ReadStep {
    serde_json::from_value(json!({
        "view": "context",
        "show": ["tools:5", "thinking", "turns"],
        "hide": ["annotations"],
        "from": "seen",
        "around": 2,
        "diff": "compaction:1",
        "alternate": "translation:pl",
        "physical": true,
        "neutral_names": true,
        "at": "2026-11-21T10:00:00Z",
        "format": "markdown",
        "budget": 0,
    }))
    .unwrap()
}

/// The step a reading continues with is the step it was asked with, with
/// the next cursor: no option is lost or renamed on the way through the
/// reading's request.
#[test]
fn a_continuation_carries_every_option_of_its_step() {
    let step = every_option();
    let request = request_for(step.clone(), "root:claude-code:s1".into()).unwrap();
    assert_eq!(request.options, step.options);
    assert_eq!(request.budget, ReadBudget::Tokens(0));
    assert_eq!(request.format, Format::Markdown);
    let next = step_from(&request, Some("sealed".into()));
    assert_eq!(
        next,
        ReadStep {
            cursor: Some("sealed".into()),
            ..step
        }
    );
    // The line owns its read cursor in the typed read step.
    let words = [
        "session:test:one",
        "read",
        "--view",
        "dialogue",
        "--cursor",
        "sealed",
    ]
    .map(str::to_owned);
    let parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    let call = crate::cli::run_line::request(&parsed).unwrap();
    assert!(call.cursor.is_none());
    let rows: Rows = serde_json::from_value(call.arguments).unwrap();
    let [Step::Read(step)] = rows.steps.as_slice() else {
        panic!("one read")
    };
    assert_eq!(step.options.view, View::Dialogue);
    assert_eq!(step.cursor.as_deref(), Some("sealed"));
}

/// The built options a session reading takes, in every combination the
/// strategy draws: views, show and hide words, the start, physical parts,
/// neutral names, the alternate, a time to read annotations at, the format.
fn built_step() -> impl Strategy<Value = ReadStep> {
    let words = prop::sample::subsequence(
        vec![
            "tools:5",
            "tools:full",
            "thinking",
            "harness:line",
            "turns",
            "context",
            "inherited",
        ],
        0..4,
    );
    let hidden = prop::sample::subsequence(vec!["annotations", "spawns", "thinking"], 0..3);
    (
        prop::sample::select(vec![
            View::Transcript,
            View::Dialogue,
            View::Context,
            View::Raw,
        ]),
        words,
        hidden,
        any::<bool>(),
        any::<bool>(),
        any::<bool>(),
        prop::option::of(Just("none")),
        prop::option::of(Just("2026-11-21T10:00:00Z")),
        prop::sample::select(vec!["json", "markdown", "html"]),
    )
        .prop_map(
            |(view, show, hide, seen, physical, neutral_names, alternate, at, format)| ReadStep {
                options: ReadOptions {
                    view,
                    show: show.into_iter().map(str::to_owned).collect(),
                    hide: hide.into_iter().map(str::to_owned).collect(),
                    from: seen.then(|| "seen".to_owned()),
                    alternate: alternate.map(str::to_owned),
                    physical,
                    neutral_names,
                    at: at.map(|at| ReadAt::try_from(at.to_owned()).unwrap()),
                    ..ReadOptions::default()
                },
                format: Some(format.into()),
                budget: ReadBudget::Tokens(1),
                ..ReadStep::default()
            },
        )
        .prop_filter("the view takes the words", |step| {
            step.options.validate().is_ok()
        })
}

/// A read step with every built option set reads its first page; following
/// `next` asks for the same step with the next cursor, through the daemon.
#[test]
fn read_step_round_trips_every_option() {
    let d = test_daemon::start("neutral target text");
    let mut runner = proptest::test_runner::TestRunner::new(proptest::test_runner::Config {
        cases: 24,
        ..Default::default()
    });
    runner
        .run(&built_step(), |step| {
            let line = json!({
                "selection": "root:claude-code:s1",
                "steps": [serde_json::to_value(Step::Read(Box::new(step.clone()))).unwrap()],
            });
            let page = crate::test_answers::read(Call::new("rows", line).answer(&d.daemon));
            let next = page
                .document
                .footer
                .next
                .expect("a one-token page has a next page");
            let rows: Rows = serde_json::from_value(next.request.arguments).unwrap();
            // A named reference continues as the selection, or as the row
            // the line resolved when it resolved one.
            let target = rows
                .selection
                .clone()
                .or_else(|| rows.references.first().cloned());
            if step.options.view == View::Context && step.options.at.is_some() {
                // A historical context continues bound to the message it
                // was read at (docs/reading.md, Ranges).
                prop_assert!(
                    target
                        .as_deref()
                        .is_some_and(|target| target.starts_with("message:claude-code:")),
                    "{:?}",
                    target
                );
            } else {
                prop_assert_eq!(target.as_deref(), Some("root:claude-code:s1"));
            }
            let [Step::Read(continued)] = rows.steps.as_slice() else {
                panic!("one read continuation: {rows:?}")
            };
            prop_assert!(continued.cursor.is_some());
            prop_assert_eq!(
                &ReadStep {
                    cursor: None,
                    ..(**continued).clone()
                },
                &step
            );
            Ok(())
        })
        .unwrap();
}

/// The selection, not `--at`, names the message a context is read at.
#[test]
fn context_at_message_names_the_selection_line() {
    let d = test_daemon::start("neutral target text");
    let error = crate::test_answers::refusal(
        Call::new(
            "rows",
            json!({
                "selection": "session:claude-code:s1",
                "steps": [{"verb": "read", "view": "context", "at": "claude-code:u1"}],
            }),
        )
        .answer(&d.daemon),
    );
    assert_eq!(error.code.as_deref(), Some("bad_request"));
    assert_eq!(
        error.message,
        "--at takes a time or a position, not the message claude-code:u1; select the message: `polylogue message:claude-code:u1 read --view context`"
    );
}

/// A cursor whose options digest an earlier build wrote is stale, and the
/// refusal names the reading to start again, as a line and a ready call.
#[test]
fn cursor_from_before_the_change_is_stale_with_rerun() {
    use polylogue_model::read::{CursorSeal, PageCursor};
    let d = test_daemon::start("neutral target text");
    let first = json!({
        "selection": "session:claude-code:s1",
        "steps": [{"verb": "read", "view": "dialogue", "budget": 1}],
    });
    let page = crate::test_answers::read(Call::new("rows", first).answer(&d.daemon));
    let next = page.document.footer.next.expect("a next page");
    let mut rows: Rows = serde_json::from_value(next.request.arguments).unwrap();
    let Step::Read(step) = &mut rows.steps[0] else {
        panic!("a read continuation")
    };
    let reader = d.daemon.reads.reader().unwrap();
    let seal = CursorSeal {
        key: reader.cursor_key().unwrap(),
        generation: reader.change_position().unwrap().generation,
    };
    drop(reader);
    let mut cursor = PageCursor::open(step.cursor.as_deref().unwrap(), &seal).unwrap();
    // The previous build's digest: the bare hash of the options it held.
    cursor.digest = blake3::hash(b"{\"view\":\"dialogue\",\"output_lines\":100}")
        .to_hex()
        .to_string();
    step.cursor = Some(cursor.seal(&seal));
    let error = crate::test_answers::refusal(
        Call::new("rows", serde_json::to_value(&rows).unwrap()).answer(&d.daemon),
    );
    assert_eq!(error.code.as_deref(), Some("stale_cursor"));
    assert_eq!(
        error.message,
        "Polylogue was updated since this page was issued; run `polylogue session:claude-code:s1 read --view dialogue` again to start over"
    );
    let Some(command_surface::Suggestion::Request(again)) = error.suggestion() else {
        panic!("a ready call: {error:?}")
    };
    let again: Rows = serde_json::from_value(again.arguments.clone()).unwrap();
    assert_eq!(again.selection.as_deref(), Some("session:claude-code:s1"));
    let [Step::Read(again)] = again.steps.as_slice() else {
        panic!("one read")
    };
    assert_eq!(again.cursor, None);
    assert_eq!(again.options.view, View::Dialogue);
}
