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
            assertion,
            resolved,
            ..
        } = request.write
        else {
            panic!("expected query assertion")
        };
        assert_eq!(assertion.key.as_deref(), Some(key));
        assert_eq!(assertion.value, value);
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
    assert!(text.contains("repeat; no new writes"));
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
    assert!(
        text.starts_with("tagged 1 row\nalready tagged: annotation:"),
        "{text}"
    );
    let request = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .user_request("tag-bulk-1")
        .unwrap()
        .unwrap();
    let UserWrite::AssertQuery { resolved, .. } = request.write else {
        panic!("expected query assertion")
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
