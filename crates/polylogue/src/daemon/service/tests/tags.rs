use crate::protocol::line::{LineAnswer, StepAnswer, StepOutcome};
use crate::test_daemon::{self, Call, TestDaemon};
use polylogue_model::{
    Author, Reference,
    authorship::{Authorship, Channel},
    user::{AssertQueryOutcome, UserWrite},
};
use serde_json::json;

fn command(id: &str, references: &[String], step: serde_json::Value) -> Call {
    Call::new(
        "rows",
        json!({"references":references,"all":true,"steps":[step]}),
    )
    .id(id)
    .authorship(Authorship {
        author: Author::Operator,
        via: Channel::Cli,
    })
}

fn wrote(d: &TestDaemon, request: Call) -> (AssertQueryOutcome, String) {
    let answer = request.answer(&d.daemon);
    assert_eq!(answer.exit_code(), 0, "{answer:?}");
    let line: LineAnswer = serde_json::from_value(
        answer
            .value()
            .unwrap_or_else(|| panic!("{answer:?}"))
            .clone(),
    )
    .unwrap();
    let printed = crate::cli::print::line(&line, command_surface::TextStyle::Plain);
    let LineAnswer::Steps(mut steps) = line else {
        panic!("expected steps")
    };
    assert_eq!(steps.steps.len(), 1);
    let StepOutcome::Done {
        reply: StepAnswer::WroteEach(report),
        ..
    } = steps.steps.remove(0).outcome
    else {
        panic!("expected a typed authored result")
    };
    (report, printed)
}

fn tag(id: &str, references: &[String], name: &str) -> Call {
    command(id, references, json!({"verb":"tag","name":name}))
}

fn note(d: &TestDaemon, id: &str) -> String {
    let (report, _) = wrote(
        d,
        command(
            id,
            &["session:claude-code:s1".into()],
            json!({"verb":"note","text":id}),
        ),
    );
    let id = report.id.unwrap();
    format!("annotation:{id}@{id}")
}

#[test]
fn tag_value_maps_to_key_and_value() {
    let d = test_daemon::start("synthetic tag target");
    let targets = vec!["session:claude-code:s1".into()];
    for (id, name, key, value) in [
        (
            "tag-colons",
            "topic:first:second",
            "topic",
            Some(json!({"value":"first:second"})),
        ),
        ("tag-empty", "topic:", "topic", Some(json!({"value":""}))),
        ("tag-bare", "reviewed", "reviewed", None),
    ] {
        let (report, _) = wrote(&d, tag(id, &targets, name));
        assert_eq!((report.targets, report.written), (1, 1));
        assert!(report.already.is_empty());
        let request = d
            .daemon
            .reads
            .reader()
            .unwrap()
            .user_request(&format!("{id}-1"))
            .unwrap()
            .unwrap();
        let UserWrite::AssertQuery {
            annotation,
            resolved,
            ..
        } = request.write
        else {
            panic!("expected query annotation")
        };
        assert_eq!(annotation.key.as_deref(), Some(key));
        assert_eq!(annotation.value, value);
        assert_eq!(resolved, vec![Reference::parse(&targets[0]).unwrap()]);
    }
    for (id, name) in [("tag-empty-key", ":value"), ("tag-blank-key", "   ")] {
        let refused = tag(id, &targets, name).answer(&d.daemon);
        assert_eq!(
            crate::test_answers::refusal(refused).code.as_deref(),
            Some("bad_request")
        );
        assert!(
            d.daemon
                .reads
                .reader()
                .unwrap()
                .user_request(&format!("{id}-1"))
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn all_tagged_rows_create_no_step_request_and_retries_keep_the_original_result() {
    let d = test_daemon::start("synthetic tag target");
    let targets = vec!["session:claude-code:s1".into()];
    let original = tag("tag-original", &targets, "status:ready");
    let (first, _) = wrote(&d, original.clone());
    let (skipped, text) = wrote(&d, tag("tag-already", &targets, "status:ready"));
    assert_eq!(skipped.id, None);
    assert_eq!((skipped.targets, skipped.written), (1, 0));
    assert_eq!(skipped.already, targets);
    assert_eq!(
        text,
        "tagged 0 rows\nalready tagged: session:claude-code:s1\n"
    );
    assert!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .user_request("tag-already-1")
            .unwrap()
            .is_none()
    );
    let (repeated, text) = wrote(&d, original);
    assert!(repeated.repeat);
    assert_eq!(repeated.id, first.id);
    assert_eq!(repeated.written, 1);
    assert!(repeated.already.is_empty());
    let written = first.id.as_ref().unwrap();
    let (report_lines, undo) = text.split_once("undo: ").expect("an undo line");
    assert_eq!(report_lines, "tagged 1 row\nrepeat; no new writes\n");
    assert!(
        undo.contains(&format!("'annotation:{written}@{written}'")),
        "{undo}"
    );
}

#[test]
fn tag_skips_rows_already_tagged_and_reports_the_accepted_population() {
    let d = test_daemon::start("synthetic tag target");
    let first = note(&d, "note-first");
    let second = note(&d, "note-second");
    let (seed, _) = wrote(
        &d,
        tag("tag-seed", std::slice::from_ref(&first), "status:ready"),
    );
    let targets = vec![first.clone(), second.clone()];
    let intended = tag("tag-bulk", &targets, "status:ready");
    let (report, text) = wrote(&d, intended.clone());
    assert_eq!((report.targets, report.written), (2, 1));
    assert_eq!(report.already, vec![first.clone()]);
    let (report_lines, undo) = text.split_once("undo: ").expect("an undo line");
    assert_eq!(
        report_lines,
        format!("tagged 1 row\nalready tagged: {first}\n")
    );
    // The undo withdraws what this write added, not the row already tagged.
    assert!(undo.contains("'annotation:"), "{undo}");
    assert!(!undo.contains(&first), "{undo}");
    let request = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .user_request("tag-bulk-1")
        .unwrap()
        .unwrap();
    let UserWrite::AssertQuery { resolved, .. } = request.write else {
        panic!("expected query annotation")
    };
    assert_eq!(
        resolved,
        targets
            .iter()
            .map(|text| Reference::parse(text).unwrap())
            .collect::<Vec<_>>()
    );
    let seed = seed.id.unwrap();
    let withdrawal = command(
        "withdraw-seed",
        &[format!("annotation:{seed}@{seed}")],
        json!({"verb":"withdraw"}),
    )
    .answer(&d.daemon);
    assert!(withdrawal.failure().is_none(), "{withdrawal:?}");
    let (repeated, _) = wrote(&d, intended);
    assert!(repeated.repeat);
    assert_eq!(repeated.already, report.already);
    assert_eq!(repeated.written, report.written);
    let refs = targets
        .iter()
        .map(|text| Reference::parse(text).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .tagged(&refs, "status", Some(&json!("ready")), &Author::Operator)
            .unwrap(),
        [Reference::parse(&second).unwrap()].into_iter().collect()
    );
    let changed = tag("tag-bulk", &targets, "status:different").answer(&d.daemon);
    assert_eq!(
        crate::test_answers::refusal(changed).code.as_deref(),
        Some("conflict")
    );
}

#[test]
fn a_tag_over_no_matching_rows_keeps_the_empty_outcome_without_a_step_write() {
    let d = test_daemon::start("synthetic tag target");
    let answer = Call::new(
        "rows",
        json!({"selection":"sessions where 0 = 1", "all":true,
            "steps":[{"verb":"tag","name":"status:ready"}]}),
    )
    .id("tag-no-matches")
    .answer(&d.daemon);
    assert_eq!(answer.exit_code(), 1, "{answer:?}");
    let line: LineAnswer = serde_json::from_value(crate::test_answers::empty(answer)).unwrap();
    let LineAnswer::Steps(steps) = line else {
        panic!("expected steps")
    };
    let StepOutcome::Done {
        reply: StepAnswer::WroteEach(report),
        undo,
    } = &steps.steps[0].outcome
    else {
        panic!("expected successful empty tag")
    };
    assert_eq!((report.targets, report.written), (0, 0));
    assert_eq!(report.id, None);
    assert!(report.already.is_empty() && undo.is_empty());
    assert!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .user_request("tag-no-matches-1")
            .unwrap()
            .is_none()
    );
}

/// The refusals docs/annotations.md names reach the caller through the
/// write path with their reason's code and sentence.
#[test]
fn annotation_refusals_carry_their_documented_reason() {
    use polylogue_model::Reason;
    let d = test_daemon::start("the first synthetic session");
    let revisions = |d: &TestDaemon| {
        Call::new("status", json!({}))
            .answer(&d.daemon)
            .value()
            .unwrap()["store"]["revisions"]
            .as_u64()
            .unwrap()
    };
    let before = revisions(&d);
    std::fs::write(
        d.home.join(".claude/projects/-p/s2.jsonl"),
        polylogue_testkit::claude_user("u2", "s2", "the second synthetic session"),
    )
    .unwrap();
    d.wait_for(|status| {
        status["store"]["revisions"].as_u64().unwrap() > before
            && status["store"]["search_pending"] == 0
    });
    let both: Vec<String> = vec![
        "session:claude-code:s1".into(),
        "session:claude-code:s2".into(),
    ];
    let bulk = command(
        "refused-bulk",
        &both,
        json!({"verb":"note","as":"title","text":"one title"}),
    )
    .answer(&d.daemon);
    // The answer as `--json` prints it and the socket, HTTP and MCP carry
    // it: the line answers, and its one step failed with the typed reason.
    let wire = serde_json::to_value(&bulk).unwrap();
    assert_eq!(
        wire["value"]["reply"]["steps"][0]["outcome"]["error"]["reason"],
        json!({"name": "single_valued_bulk", "fields": {"kind": "title", "rows": 2}}),
        "{wire}"
    );
    let bulk = crate::test_answers::refusal(bulk);
    let reason = Reason::SingleValuedBulk {
        kind: "title".into(),
        rows: 2,
    };
    assert_eq!(
        (bulk.code.as_deref(), bulk.message.as_str()),
        (Some(reason.code().as_str()), reason.to_string().as_str())
    );
    let shape = crate::test_answers::refusal(
        command(
            "refused-shape",
            &both[..1],
            json!({"verb":"note","as":"outcome","text":"great"}),
        )
        .answer(&d.daemon),
    );
    assert_eq!(shape.code.as_deref(), Some("bad_value"), "{shape:?}");
    assert!(
        shape
            .message
            .starts_with("the outcome value is malformed: "),
        "{shape:?}"
    );
}
