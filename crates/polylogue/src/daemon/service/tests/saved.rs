use super::*;
use crate::protocol::line::{LineAnswer, SaveReport, StepAnswer, StepOutcome};
use serde_json::{Value, json};

fn save(id: &str, name: Option<&str>, selection: &str) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new(
        "rows",
        json!({"selection":selection,"steps":[{"verb":"save","name":name}]}),
    )
    .id(id)
}

#[track_caller]
fn report(answer: command_surface::Answer<Value>) -> SaveReport {
    let value = crate::test_answers::ok(answer);
    let LineAnswer::Steps(mut reports) = serde_json::from_value(value).unwrap() else {
        panic!("expected Save steps")
    };
    assert_eq!(reports.steps.len(), 1);
    match reports.steps.remove(0).outcome {
        StepOutcome::Done {
            reply: StepAnswer::Saved(report),
            ..
        } => report,
        other => panic!("expected completed Save: {other:?}"),
    }
}

fn query(d: &test_daemon::TestDaemon, selection: &str) -> command_surface::Answer<Value> {
    crate::test_daemon::Call::new("rows", json!({"selection":selection}))
        .answer(&d.daemon)
        .map(|value| {
            let LineAnswer::Listing(reply) = serde_json::from_value(value).unwrap() else {
                panic!("expected listing")
            };
            serde_json::to_value(reply).unwrap()
        })
}

#[test]
fn a_saved_query_runs_by_name_and_saving_again_revises_it() {
    let d = test_daemon::start("first prompt");
    let first = report(save("r1", Some("everything"), "kind:*").answer(&d.daemon))
        .saved
        .unwrap();
    let value = crate::test_answers::ok(query(&d, "query:everything"));
    assert_eq!(value["count"], 1, "{value}");
    assert_eq!(value["definitions"][0]["revision"], first.id);
    let second = report(save("r2", Some("everything"), "kind:subagent").answer(&d.daemon))
        .saved
        .unwrap();
    let value = crate::test_answers::empty(query(&d, "query:everything"));
    assert_eq!(value["count"], 0, "{value}");
    assert_eq!(value["definitions"][0]["revision"], second.id);
    let annotations = crate::test_answers::ok(query(&d, "annotations where kind:saved_query"));
    assert_eq!(
        annotations["count"], 1,
        "one family, revised: {annotations}"
    );
}

#[test]
fn save_retries_keep_the_first_revision_after_the_definition_changes() {
    let mut d = test_daemon::start("first prompt");
    report(save("first", Some("everything"), "sessions").answer(&d.daemon));
    report(save("second", Some("everything"), "messages").answer(&d.daemon));
    d.restart();
    for (id, text) in [("first", "sessions"), ("second", "messages")] {
        let repeated = report(save(id, Some("everything"), text).answer(&d.daemon));
        assert!(repeated.repeat);
        assert_eq!(repeated.saved.unwrap().id, format!("{id}-1"));
    }
    d.wait_for(|status| status["caught_up"] == true);
    let current = crate::test_answers::ok(query(&d, "query:everything"));
    assert_eq!(current["unit"], "messages");
    assert_eq!(current["definitions"][0]["revision"], "second-1");
    for request in [
        save("first", Some("everything"), "documents"),
        save("first", Some("other"), "sessions"),
    ] {
        let failure = crate::test_answers::refusal(request.answer(&d.daemon));
        assert_eq!(failure.code.as_deref(), Some("conflict"));
    }
}

#[test]
fn kept_rows_retry_after_membership_changes() {
    let mut d = test_daemon::start("first prompt");
    let first = report(save("kept", Some("before"), "annotations").answer(&d.daemon));
    report(save("saved", Some("everything"), "sessions").answer(&d.daemon));
    d.restart();
    let repeated = report(save("kept", Some("before"), "annotations").answer(&d.daemon));
    assert!(repeated.repeat);
    assert_eq!(repeated.saved_rows, first.saved_rows);
    d.wait_for(|_| true);
    let rows = crate::test_answers::empty(query(&d, &first.saved_rows.result));
    assert_eq!(rows["count"], 0);
    for request in [
        save("kept", Some("before"), "sessions"),
        save("kept", Some("after"), "annotations"),
    ] {
        assert_eq!(
            crate::test_answers::refusal(request.answer(&d.daemon))
                .code
                .as_deref(),
            Some("conflict")
        );
    }
}

#[test]
fn a_save_that_query_names_could_not_say_is_refused() {
    let d = test_daemon::start("first prompt");
    for (index, (name, text)) in [
        ("two words", "kind:*"),
        ("", "kind:*"),
        ("broken", "\"unclosed"),
    ]
    .into_iter()
    .enumerate()
    {
        let failure = crate::test_answers::refusal(
            save(&format!("bad-{index}"), Some(name), text).answer(&d.daemon),
        );
        assert_eq!(
            failure.kind,
            if name == "broken" {
                command_surface::FailureKind::InvalidQuery
            } else {
                command_surface::FailureKind::BadValue
            },
            "{failure:?}"
        );
    }
}

fn watch(id: &str, selection: &str, hold: Option<&str>) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new("rows",json!({"selection":selection,"steps":[{"verb":"save","name":"my-watch","watch":true,"hold":hold}]})).id(id)
}

#[test]
fn a_watch_is_stored_as_a_signal_rule_with_its_query_hold_and_interrupt() {
    let d = test_daemon::start("first prompt");
    assert!(
        report(watch("w1", "sessions where live:waiting", Some("20m")).answer(&d.daemon))
            .watch
            .is_some()
    );
    let rules = crate::test_answers::ok(query(&d, "annotations where kind:signal"));
    assert_eq!(rules["count"], 1, "{rules}");
    let value = &rules["rows"][0]["value"];
    assert_eq!(value["query"], "query:my-watch@1");
    assert_eq!(value["for"], "20m");
    assert_eq!(value["interrupt"], false);
    for (request, kind) in [
        (
            watch("w2", "sessions where nonsense_field:1", None),
            command_surface::FailureKind::UnknownField,
        ),
        (
            watch("w3", "kind:*", Some("soon")),
            command_surface::FailureKind::BadRequest,
        ),
    ] {
        let failure = crate::test_answers::refusal(request.answer(&d.daemon));
        assert_eq!(failure.kind, kind, "{failure:?}");
    }
}

#[test]
fn a_pipe_in_a_quoted_value_keeps_the_match_rows() {
    let d = test_daemon::start("synthetic|prompt");
    let kept = report(
        save(
            "quoted-pipe",
            None,
            "sessions where exists messages(\"synthetic|prompt\")",
        )
        .answer(&d.daemon),
    )
    .saved_rows;
    assert_eq!(
        crate::test_answers::ok(query(&d, &format!("{}.matches", kept.result)))["count"],
        1
    );
}

#[test]
fn accepting_caller_is_preserved_on_retry_and_another_actor_cannot_reuse_the_id() {
    use polylogue_model::{
        Author,
        authorship::{Authorship, Channel},
    };
    let d = test_daemon::start("first prompt");
    let caller = Authorship {
        author: Author::AgentOrigin(polylogue_model::Origin::new("codex").unwrap()),
        via: Channel::Mcp,
    };
    let limit = polylogue_store::read::limit::ReadLimit::new(
        crate::daemon::execution::default_read_deadline(),
        Default::default(),
    );
    let call = |authorship: &Authorship| {
        save("caller-request", Some("caller-query"), "sessions")
            .within(&limit.clone())
            .authorship(authorship.clone())
            .answer(&d.daemon)
    };
    let saved = report(call(&caller)).saved.unwrap();
    let reads = d.daemon.reads.reader().unwrap();
    let original = reads.user_request("caller-request-1").unwrap().unwrap();
    assert_eq!(original.authorship, caller);
    assert_eq!(
        reads
            .annotation(&polylogue_model::RevisionReference::exact(
                saved.id.clone(),
                saved.id
            ))
            .unwrap()
            .unwrap()
            .fact
            .author,
        caller.author
    );
    drop(reads);
    assert!(report(call(&caller)).repeat);
    for changed in [
        Authorship {
            author: Author::Operator,
            via: caller.via,
        },
        Authorship {
            author: caller.author.clone(),
            via: Channel::Cli,
        },
    ] {
        assert_eq!(
            crate::test_answers::refusal(call(&changed)).code.as_deref(),
            Some("conflict")
        );
    }
    let reads = d.daemon.reads.reader().unwrap();
    assert_eq!(
        reads.user_request("caller-request-1").unwrap(),
        Some(original)
    );
}
