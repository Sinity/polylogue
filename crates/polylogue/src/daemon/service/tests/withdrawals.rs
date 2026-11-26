use super::*;
use crate::protocol::line::{LineAnswer, SaveReport, StepAnswer, StepOutcome, StepsAnswer};
use polylogue_model::{
    Author, Reference, RevisionReference,
    authorship::{Authorship, Channel},
    user::UserWrite,
};
use serde_json::{Value, json};

fn call(id: &str, arguments: Value) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new("rows", arguments)
        .id(id)
        .authorship(Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        })
}
fn steps(answer: command_surface::Answer<Value>) -> StepsAnswer {
    let value = crate::test_answers::ok(answer);
    let LineAnswer::Steps(steps) = serde_json::from_value(value).unwrap() else {
        panic!("expected steps")
    };
    steps
}
fn save(d: &test_daemon::TestDaemon, id: &str, name: &str, text: &str, watch: bool) -> SaveReport {
    match crate::test_answers::step(
        call(
            id,
            json!({"selection":text,"all":true,
        "steps":[{"verb":"save","name":name,"watch":watch}]}),
        )
        .answer(&d.daemon),
    ) {
        StepAnswer::Saved(report) => report,
        other => panic!("expected Save, got {other:?}"),
    }
}
fn note(d: &test_daemon::TestDaemon, id: &str) {
    assert_eq!(
        call(
            id,
            json!({"references":["session:claude-code:s1"],
        "steps":[{"verb":"note","text":id}]})
        )
        .answer(&d.daemon)
        .exit_code(),
        0
    );
}
fn count(d: &test_daemon::TestDaemon, selection: &str) -> u64 {
    let answer = call("count-annotations", json!({"selection":selection})).answer(&d.daemon);
    assert!(
        matches!(
            &answer.outcome,
            command_surface::Outcome::Ok(_) | command_surface::Outcome::Empty(_)
        ),
        "{answer:?}"
    );
    let LineAnswer::Listing(listing) =
        serde_json::from_value(answer.value().unwrap().clone()).unwrap()
    else {
        panic!("expected listing")
    };
    serde_json::to_value(listing).unwrap()["count"]
        .as_u64()
        .unwrap()
}
fn request(d: &test_daemon::TestDaemon, id: &str) -> polylogue_model::authored::PreparedRequest {
    let reader = d.daemon.reads.reader().unwrap();
    let mut found = None;
    reader
        .visit_authored(&mut |record| {
            if record.request.id == id {
                found = Some(record.clone());
            }
            Ok(())
        })
        .unwrap();
    found.unwrap()
}
fn command(d: &test_daemon::TestDaemon, text: &str) -> StepsAnswer {
    let (words, input) = crate::cli::line::tests::shell_arguments(text);
    let mut parsed = crate::cli::line::parse(
        &words,
        &crate::cli::line::FirstWordCatalog::default(),
        false,
    )
    .unwrap();
    crate::cli::run_line::read_stdin(&mut parsed, &mut std::io::Cursor::new(input)).unwrap();
    let request = crate::cli::run_line::request(&parsed).unwrap();
    steps(
        crate::test_daemon::Call::from_request(&request)
            .authorship(Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            })
            .answer(&d.daemon),
    )
}

#[test]
fn withdraw_saved_query_by_name_pins_the_family_and_undo_restores_its_watch() {
    let d = test_daemon::start("synthetic query target");
    note(&d, "first-note");
    note(&d, "second-note");
    let first = save(
        &d,
        "save-first",
        "focus",
        "annotations where kind:note",
        true,
    );
    let second = save(
        &d,
        "save-second",
        "focus",
        "annotations where kind:note | sort by id",
        true,
    );
    assert_ne!(
        first.saved.as_ref().unwrap().id,
        second.saved.as_ref().unwrap().id
    );
    let before = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .saved_query_record("focus")
        .unwrap()
        .unwrap();
    // The definition is one selected record, even though its results contain two notes.
    let withdrawn = command(
        &d,
        "polylogue --request-id withdraw-focus query:focus withdraw --reason finished",
    );
    let StepOutcome::Done {
        reply: StepAnswer::Wrote(outcome),
        undo,
    } = &withdrawn.steps[0].outcome
    else {
        panic!("expected withdrawal")
    };
    assert!(!outcome.repeat && !outcome.pending);
    assert_eq!(undo.len(), 1);
    let original = request(&d, "withdraw-focus-1");
    let root = Reference::Annotation(RevisionReference {
        root: first.saved.unwrap().id,
        revision: None,
    });
    assert!(
        matches!(&original.request.write, UserWrite::WithdrawAnnotations { targets, reason }
        if targets == &vec![root.clone()] && reason.as_deref() == Some("finished"))
    );
    assert_eq!(
        original.effect.annotations.len(),
        1,
        "definition and watch share one withdrawal"
    );
    let watch_root = Reference::Annotation(RevisionReference {
        root: first.watch.unwrap().id,
        revision: None,
    });
    let targets = original.effect.annotations[0]
        .links
        .iter()
        .filter(|link| link.role == polylogue_model::annotation::LinkRole::Withdraws)
        .map(|link| link.reference.clone())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(targets, [root.clone(), watch_root].into_iter().collect());
    assert!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .saved_query_target("focus")
            .unwrap()
            .is_none()
    );
    assert!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .saved_query_record("focus")
            .unwrap()
            .unwrap()
            .watch
            .is_none()
    );
    assert_eq!(count(&d, "annotations where kind:note"), 2);
    // A retry sends the same request: the command line's spelling of it.
    let repeated = command(
        &d,
        "polylogue --request-id withdraw-focus query:focus withdraw --reason finished",
    );
    assert!(
        matches!(&repeated.steps[0].outcome, StepOutcome::Done { reply: StepAnswer::Wrote(report), undo: again }
        if report.repeat && report.id == outcome.id && again == undo)
    );
    assert_eq!(request(&d, "withdraw-focus-1"), original);
    command(&d, &undo[0]);
    assert_eq!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .saved_query_target("focus")
            .unwrap(),
        Some(root)
    );
    let restored = d
        .daemon
        .reads
        .reader()
        .unwrap()
        .saved_query_record("focus")
        .unwrap()
        .unwrap();
    assert_eq!(restored.revisions, before.revisions);
    assert_eq!(restored.kept, before.kept);
    assert_eq!(restored.watch, before.watch);
}

#[test]
fn withdrawal_retry_does_not_retarget_a_recreated_saved_name() {
    let d = test_daemon::start("synthetic query target");
    let first = save(&d, "first-definition", "focus", "sessions", false);
    let arguments = json!({"selection":"query:focus","steps":[{"verb":"withdraw"}]});
    steps(call("old-withdrawal", arguments.clone()).answer(&d.daemon));
    let original = request(&d, "old-withdrawal-1");
    let replacement = save(&d, "replacement-definition", "focus", "messages", false)
        .saved
        .unwrap();
    assert_ne!(first.saved.unwrap().id, replacement.id);
    let retry = steps(call("old-withdrawal", arguments).answer(&d.daemon));
    assert!(
        matches!(&retry.steps[0].outcome, StepOutcome::Done { reply: StepAnswer::Wrote(report), .. } if report.repeat)
    );
    assert_eq!(request(&d, "old-withdrawal-1"), original);
    assert_eq!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .saved_query_target("focus")
            .unwrap(),
        Some(Reference::Annotation(RevisionReference {
            root: replacement.id,
            revision: None
        }))
    );
}

#[test]
fn a_saved_query_expression_withdraws_results_instead_of_the_definition() {
    for selection in [
        "query:selected-notes | sort by id",
        "query:selected-notes where kind:note",
        "query:selected-notes@1",
    ] {
        let d = test_daemon::start("synthetic query target");
        note(&d, "result-note");
        let saved = save(
            &d,
            "result-query",
            "selected-notes",
            "annotations where kind:note",
            false,
        )
        .saved
        .unwrap();
        steps(
            call(
                "withdraw-results",
                json!({"selection":selection, "all":true,
        "steps":[{"verb":"withdraw"}]}),
            )
            .answer(&d.daemon),
        );
        let original = request(&d, "withdraw-results-1");
        let note = Reference::Annotation(RevisionReference::exact(
            "result-note-1".into(),
            "result-note-1".into(),
        ));
        assert!(
            matches!(&original.request.write, UserWrite::WithdrawAnnotations { targets, .. } if targets == &vec![note])
        );
        assert_eq!(count(&d, "annotations where kind:note"), 0);
        assert_eq!(
            d.daemon
                .reads
                .reader()
                .unwrap()
                .saved_query_target("selected-notes")
                .unwrap(),
            Some(Reference::Annotation(RevisionReference {
                root: saved.id,
                revision: None
            }))
        );
    }
}

#[test]
fn withdraw_on_sessions_offers_an_executable_annotation_selection() {
    let d = test_daemon::start("synthetic query target");
    note(&d, "about-session");
    let reply = steps(
        call(
            "wrong-kind",
            json!({"references":["session:claude-code:s1"],
        "steps":[{"verb":"withdraw","reason":"reviewed"}]}),
        )
        .answer(&d.daemon),
    );
    let StepOutcome::Failed { error } = &reply.steps[0].outcome else {
        panic!("expected row refusal")
    };
    assert_eq!(error.code.as_deref(), Some("bad_request"));
    assert!(
        error
            .message
            .contains("annotations where about:\"session:claude-code:s1\""),
        "{error:?}"
    );
    let Some(command_surface::Suggestion::Request(next)) = error.suggestion() else {
        panic!("expected executable correction")
    };
    assert_eq!(
        next.arguments["selection"],
        "annotations where about:\"session:claude-code:s1\""
    );
    assert_eq!(next.arguments["steps"][0]["reason"], "reviewed");
    assert!(
        d.daemon
            .reads
            .reader()
            .unwrap()
            .user_request("wrong-kind-1")
            .unwrap()
            .is_none()
    );
    let corrected = steps(
        crate::test_daemon::Call::from_request(next)
            .authorship(Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            })
            .answer(&d.daemon),
    );
    assert!(matches!(
        corrected.steps[0].outcome,
        StepOutcome::Done {
            reply: StepAnswer::Wrote(_),
            ..
        }
    ));
}

#[test]
fn the_withdrawal_correction_quotes_opaque_addresses_and_keeps_other_notes() {
    let d = test_daemon::start("synthetic query target");
    let address = "url:https://example.invalid/\" OR kind:note | count\u{1b}[31m";
    note(&d, "unrelated-note");
    steps(
        call(
            "opaque-note",
            json!({"references":[address],
        "steps":[{"verb":"note","text":"about an opaque address"}]}),
        )
        .answer(&d.daemon),
    );
    let refused = steps(
        call(
            "opaque-refusal",
            json!({"references":[address],
        "steps":[{"verb":"withdraw"}]}),
        )
        .answer(&d.daemon),
    );
    let printed = crate::cli::print::line(
        &LineAnswer::Steps(refused.clone()),
        command_surface::TextStyle::Plain,
    );
    assert!(!printed.contains('\u{1b}'), "{printed:?}");
    let StepOutcome::Failed { error } = &refused.steps[0].outcome else {
        panic!("expected row refusal")
    };
    let Some(command_surface::Suggestion::Request(next)) = error.suggestion() else {
        panic!("expected executable correction")
    };
    steps(
        crate::test_daemon::Call::from_request(next)
            .authorship(Authorship {
                author: Author::Operator,
                via: Channel::Cli,
            })
            .answer(&d.daemon),
    );
    assert_eq!(count(&d, "annotations where kind:note"), 1);
    let remaining = request(&d, &format!("{}-1", next.request_id));
    assert!(
        matches!(&remaining.request.write, UserWrite::WithdrawAnnotations { targets, .. }
        if targets == &vec![Reference::Annotation(RevisionReference::exact("opaque-note-1".into(), "opaque-note-1".into()))])
    );
}

#[test]
fn shipped_or_absent_saved_names_refuse_without_a_write() {
    let d = test_daemon::start("synthetic query target");
    for (index, name) in ["recent", "missing-name"].iter().enumerate() {
        let id = format!("not-authored-{index}");
        let error = crate::test_answers::refusal(
            call(
                &id,
                json!({"selection":format!("query:{name}"),
            "steps":[{"verb":"withdraw"}]}),
            )
            .answer(&d.daemon),
        );
        assert_eq!(error.code.as_deref(), Some("not_found"));
        assert!(
            d.daemon
                .reads
                .reader()
                .unwrap()
                .user_request(&id)
                .unwrap()
                .is_none()
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
fn withdrawing_an_existing_literal_saved_key_does_not_run_its_results() {
    use polylogue_model::annotation::{AnnotationPatch, Change};
    for name in ["legacy_focus", "Legacy Focus"] {
        let d = test_daemon::start("synthetic query target");
        note(&d, "result-note");
        let saved = save(
            &d,
            "first-definition",
            "initial-name",
            "annotations where kind:note",
            false,
        )
        .saved
        .unwrap();
        // Existing authored facts may carry names outside new Save's spelling
        // rule. Use the ordinary revision writer rather than editing stored rows.
        let invocation = d.daemon.invocation(&Authorship {
            author: Author::Operator,
            via: Channel::Cli,
        });
        crate::test_answers::ok(crate::daemon::service::write(
            &d.daemon,
            &invocation,
            "rename-existing-definition".into(),
            UserWrite::ReviseAnnotations {
                bases: vec![RevisionReference::exact(saved.id.clone(), saved.id.clone())],
                patch: AnnotationPatch {
                    key: Change::Set(name.into()),
                    ..Default::default()
                },
            },
        ));
        let selection = format!("query:{}", polylogue_query::quoted(name));
        steps(
            call(
                "withdraw-existing",
                json!({"selection":selection,
            "steps":[{"verb":"withdraw"}]}),
            )
            .answer(&d.daemon),
        );
        let original = request(&d, "withdraw-existing-1");
        assert!(
            matches!(&original.request.write, UserWrite::WithdrawAnnotations { targets, .. }
            if targets == &vec![Reference::Annotation(RevisionReference { root:saved.id.clone(), revision:None })])
        );
        assert!(
            d.daemon
                .reads
                .reader()
                .unwrap()
                .saved_query_target(name)
                .unwrap()
                .is_none()
        );
        assert_eq!(count(&d, "annotations where kind:note"), 1);
    }
}
