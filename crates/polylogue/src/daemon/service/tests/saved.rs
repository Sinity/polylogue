use super::*;

fn answer(
    d: &test_daemon::TestDaemon,
    request: crate::test_daemon::Call,
) -> command_surface::Answer<serde_json::Value> {
    (request).answer(&d.daemon)
}

#[track_caller]
fn ok(d: &test_daemon::TestDaemon, request: crate::test_daemon::Call) -> serde_json::Value {
    crate::test_answers::ok(answer(d, request))
}

#[track_caller]
fn empty(d: &test_daemon::TestDaemon, request: crate::test_daemon::Call) -> serde_json::Value {
    crate::test_answers::empty(answer(d, request))
}

fn save(id: &str, name: &str, text: &str) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new("save", serde_json::json!({"name": name, "query": text})).id(id)
}

fn query(text: &str) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new("query", serde_json::json!({"query": text}))
}

#[test]
fn a_saved_query_runs_by_name_and_saving_again_revises_it() {
    let d = test_daemon::start("first prompt");
    let first = ok(&d, save("r1", "everything", "kind:*"));
    let v = ok(&d, query("query:everything"));
    assert_eq!(v["count"], 1, "{v}");
    assert_eq!(v["definitions"][0]["revision"], first["id"]);
    let second = ok(&d, save("r2", "everything", "kind:subagent"));
    let v = empty(&d, query("query:everything"));
    assert_eq!(v["count"], 0, "the newer revision runs: {v}");
    assert_eq!(v["definitions"][0]["revision"], second["id"]);
    let assertions = ok(&d, query("assertions where kind:saved_query"));
    assert_eq!(assertions["count"], 1, "one family, revised: {assertions}");
}

#[test]
fn save_retries_keep_the_first_revision_after_the_definition_changes() {
    let mut d = test_daemon::start("first prompt");
    ok(&d, save("first", "everything", "sessions"));
    ok(&d, save("second", "everything", "messages"));
    d.restart();
    for (id, text) in [("first", "sessions"), ("second", "messages")] {
        let repeated = ok(&d, save(id, "everything", text));
        assert_eq!(repeated["id"], id);
        assert_eq!(repeated["repeat"], true);
    }
    let current = ok(&d, query("query:everything"));
    assert_eq!(current["unit"], "messages");
    assert_eq!(current["definitions"][0]["revision"], "second");
    for request in [
        save("first", "everything", "documents"),
        save("first", "other", "sessions"),
    ] {
        assert!(
            matches!(answer(&d, request), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("conflict"))
        );
    }
}

#[test]
fn keep_retries_return_the_original_rows_after_membership_changes() {
    let mut d = test_daemon::start("first prompt");
    let keep = |id: &str, text: &str, name: &str| {
        crate::test_daemon::Call::new(
            "keep",
            serde_json::json!({"query": text, "name": Some(name)}),
        )
        .id(id)
    };
    ok(&d, keep("kept", "assertions", "before"));
    ok(&d, save("saved", "everything", "sessions"));
    d.restart();
    let repeated = ok(&d, keep("kept", "assertions", "before"));
    assert_eq!(repeated["repeat"], true);
    let rows = empty(&d, query("result:kept"));
    assert_eq!(rows["count"], 0);
    for request in [
        keep("kept", "sessions", "before"),
        keep("kept", "assertions", "after"),
    ] {
        assert!(
            matches!(answer(&d, request), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("conflict"))
        );
    }
}

#[test]
fn a_save_that_query_names_could_not_say_is_refused() {
    let d = test_daemon::start("first prompt");
    for (name, text) in [
        ("two words", "kind:*"),
        ("", "kind:*"),
        ("broken", "\"unclosed"),
    ] {
        match answer(&d, save("r", name, text)) {
            command_surface::Answer {
                outcome: command_surface::Outcome::Error(_),
                ..
            } => {}
            other => panic!("{name:?} {text:?}: {other:?}"),
        }
    }
}

fn watch(id: &str, query: &str, hold: Option<&str>) -> crate::test_daemon::Call {
    crate::test_daemon::Call::new("watch", serde_json::json!({"query": query, "name": Some("my watch"), "for": hold.map(str::to_owned)})).id(id)
}

#[test]
fn a_watch_is_a_signal_rule_delivering_notices() {
    let d = test_daemon::start("first prompt");
    ok(&d, watch("w1", "sessions where live:waiting", Some("20m")));
    let rules = ok(&d, query("assertions where kind:signal"));
    assert_eq!(rules["count"], 1, "{rules}");
    let value = &rules["rows"][0]["value"];
    assert_eq!(value["query"], "sessions where live:waiting");
    assert_eq!(value["for"], "20m");
    assert_eq!(value["interrupt"], false);
    for bad in [
        watch("w2", "sessions where nonsense_field:1", None),
        watch("w3", "kind:*", Some("soon")),
    ] {
        match answer(&d, bad) {
            command_surface::Answer {
                outcome: command_surface::Outcome::Error(_),
                ..
            } => {}
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn a_pipe_in_a_quoted_value_keeps_the_match_rows() {
    let d = test_daemon::start("synthetic|prompt");
    ok(
        &d,
        crate::test_daemon::Call::new(
            "keep",
            serde_json::json!({"query": "sessions where exists messages(\"synthetic|prompt\")"}),
        )
        .id("quoted-pipe"),
    );
    assert_eq!(ok(&d, query("result:quoted-pipe.matches"))["count"], 1);
}

#[test]
fn accepting_caller_is_preserved_on_retry_and_another_actor_cannot_reuse_the_id() {
    use polylogue_model::Author;
    use polylogue_model::authorship::{Authorship, Channel};
    let d = test_daemon::start("first prompt");
    let caller = Authorship {
        author: Author::AgentOrigin(polylogue_model::Origin::new("codex").unwrap()),
        via: Channel::Mcp,
    };
    let limit = crate::daemon::execution::limit(None, Default::default());
    let call = |authorship: &Authorship| {
        (save("caller-request", "caller-query", "sessions"))
            .limit(limit.clone())
            .authorship((authorship).clone())
            .answer(&d.daemon)
    };
    crate::test_answers::ok(call(&caller));
    let reads = d.daemon.reads.reader().unwrap();
    let original = reads.user_request("caller-request").unwrap().unwrap();
    assert_eq!(original.authorship, caller);
    assert_eq!(
        reads
            .assertion(&polylogue_model::RevisionReference::exact(
                "caller-request".into(),
                "caller-request".into()
            ))
            .unwrap()
            .unwrap()
            .fact
            .author,
        caller.author
    );
    drop(reads);
    let Some(data) = call(&caller).value().cloned() else {
        panic!("retry failed")
    };
    assert_eq!(data["repeat"], true);
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
        assert!(
            matches!(call(&changed), command_surface::Answer { outcome: command_surface::Outcome::Error(ref error), .. } if error.code.as_deref() == Some("conflict"))
        );
    }
    let reads = d.daemon.reads.reader().unwrap();
    assert_eq!(
        reads.user_request("caller-request").unwrap(),
        Some(original)
    );
}
