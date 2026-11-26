use super::*;
use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};

/// A synthetic archive: one session with a title, a repository, a working
/// directory and a model, a saved query and a kept result.
fn populated(root: &std::path::Path) -> Fixture {
    let mut fixture = Fixture::new(root).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "codex:abc123",
            title: Some("A neutral test title"),
            repo: Some("two words"),
            cwd: Some("/home/op/project/sub"),
            last: Some(100),
            ..Default::default()
        })
        .unwrap();
    for id in ["codex:m1", "codex:m2"] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "assistant",
                    written_by: "model",
                    model: Some("alpha"),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    fixture.derive().unwrap();
    let plan = polylogue_query::plan::prepare(
        "tool_calls | group by action | count",
        &polylogue_query::Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    fixture.saved_query("by-action", &plan).unwrap();
    fixture
        .kept_result(
            "kept-id",
            "sessions",
            r#"{"unit":"sessions","columns":[],"rows":[]}"#,
        )
        .unwrap();
    fixture
}

fn rendered(fixture: &Fixture, text: &str) -> Vec<String> {
    let reads = fixture.reads().unwrap();
    complete(&reads.reader().unwrap(), text, text.len())
        .unwrap()
        .into_iter()
        .map(|candidate| {
            format!(
                "{}{}{}",
                &text[..candidate.replace.start],
                candidate.value,
                &text[candidate.replace.end..]
            )
        })
        .collect()
}

fn candidates(fixture: &Fixture, text: &str, position: usize) -> Vec<Completion> {
    let reads = fixture.reads().unwrap();
    complete(&reads.reader().unwrap(), text, position).unwrap()
}

#[test]
fn field_values_come_from_the_archive_with_their_counts() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = populated(&directory.path().join("archive"));
    assert!(rendered(&fixture, "repo:two").contains(&"repo:\"two words\"".into()));
    let model = candidates(&fixture, "messages where model:al", 23)
        .into_iter()
        .find(|candidate| candidate.value == "alpha")
        .unwrap();
    assert_eq!(model.help, "2 occurrences");
    // A working directory also offers its parent directories.
    let directories = rendered(&fixture, "cwd:/home/op/pro");
    for expected in ["cwd:/home/op/project", "cwd:/home/op/project/sub"] {
        assert!(
            directories.iter().any(|text| text == expected),
            "{directories:?}"
        );
    }
    assert!(rendered(&fixture, "repo:unobserved").is_empty());
}

#[test]
fn saved_query_fields_stay_scoped_to_their_output() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = populated(&directory.path().join("archive"));
    let fields = rendered(&fixture, "query:by-action | where co");
    assert!(
        fields.iter().any(|text| text.ends_with("count")),
        "{fields:?}"
    );
    assert!(
        !rendered(&fixture, "query:by-action | where ti")
            .iter()
            .any(|text| text.ends_with("title"))
    );
    let names = candidates(&fixture, "query:by", 8);
    let saved = names
        .iter()
        .find(|candidate| candidate.value == "by-action")
        .unwrap();
    assert_eq!(saved.help, "tool_calls | group by action | count");
    // A shipped query is offered with its one line.
    let shipped = polylogue_model::saved_queries::SHIPPED_QUERIES
        .iter()
        .find(|query| query.name == "waiting")
        .unwrap();
    let waiting = candidates(&fixture, "query:wai", 9)
        .into_iter()
        .find(|candidate| candidate.value == "waiting")
        .unwrap();
    assert_eq!(waiting.help, shipped.help);
}

#[test]
fn title_and_reference_completions_share_candidates() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = populated(&directory.path().join("archive"));
    for text in ["codex:a", "codex:abc", "session:codex:abc", "abc", "ntrl"] {
        let found = candidates(&fixture, text, text.len());
        assert!(
            found
                .iter()
                .any(|candidate| candidate.help.contains("A neutral test title")
                    && candidate.help.contains("two words")),
            "{text}: {found:?}"
        );
    }
    // After the origin, the native ID's shortest unique prefix replaces
    // only what follows the colon.
    let native = candidates(&fixture, "codex:", 6)
        .into_iter()
        .find(|candidate| candidate.value == "a")
        .unwrap();
    assert_eq!((native.replace.start, native.replace.end), (6, 6));
}

#[test]
fn quoted_middle_token_keeps_utf8_offsets_and_the_query_suffix() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = populated(&directory.path().join("archive"));
    let text = "messages where repo:\"two old\" | count";
    let position = text.find(" old").unwrap();
    let candidate = candidates(&fixture, text, position)
        .into_iter()
        .find(|candidate| candidate.value == "\"two words\"")
        .unwrap();
    assert_eq!(
        format!(
            "{}{}{}",
            &text[..candidate.replace.start],
            candidate.value,
            &text[candidate.replace.end..]
        ),
        "messages where repo:\"two words\" | count"
    );
    assert!(candidates(&fixture, "repo:λ", 6).is_empty());
}

#[test]
fn saved_and_kept_names_are_runnable_source_suggestions_at_the_first_word() {
    let directory = tempfile::tempdir().unwrap();
    let fixture = populated(&directory.path().join("archive"));
    assert!(rendered(&fixture, "by-").contains(&"query:by-action".into()));
    let catalog = base_catalog();
    let query = rendered(&fixture, "kept")
        .iter()
        .map(|text| query_language::parse::parse(text, &catalog).unwrap())
        .find(|query| {
            matches!(&query.source,
            query_language::ast::Source::Result { reference, part: ResultPart::Rows, .. }
                if reference == "result:kept-id")
        })
        .expect("the kept result suggests its runnable stored result source");
    let formatted = query_language::format::format_query(&query, &catalog).unwrap();
    assert_eq!(
        query_language::parse::parse(&formatted, &catalog)
            .unwrap()
            .source,
        query.source
    );
    let kept = candidates(&fixture, "result:kep", 10)
        .into_iter()
        .find(|candidate| candidate.value == "kept-id")
        .unwrap();
    assert_eq!(kept.help, "kept-id · sessions · 0 rows");
}

#[test]
fn terminal_controls_never_enter_candidates() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&directory.path().join("archive")).unwrap();
    let hostile = "\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\u{202e}";
    fixture
        .session(SessionFacts {
            id: "claude-code:abc",
            title: Some(hostile),
            repo: Some("path\\part \"λ\""),
            last: Some(100),
            ..Default::default()
        })
        .unwrap();
    fixture
        .session(SessionFacts {
            id: "claude-code:def",
            repo: Some(hostile),
            last: Some(90),
            ..Default::default()
        })
        .unwrap();
    let text = rendered(&fixture, "repo:path").into_iter().next().unwrap();
    let parsed = polylogue_query::parse(&text).unwrap();
    let polylogue_query::ast::Source::Unit {
        condition: Some(condition),
        ..
    } = parsed.source
    else {
        panic!("{text}")
    };
    let polylogue_query::ast::ConditionKind::Match {
        value: polylogue_query::ast::Value::Scalar { text, .. },
        ..
    } = &condition.kind
    else {
        panic!()
    };
    assert_eq!(text, "path\\part \"λ\"");
    let repositories = candidates(&fixture, "repo:", 5);
    assert!(
        repositories
            .iter()
            .all(|candidate| !candidate.value.chars().any(char::is_control)),
        "{repositories:?}"
    );
    let sessions = candidates(&fixture, "claude-code:", 12);
    assert!(sessions.iter().any(|candidate| candidate.value == "a"));
    assert!(
        sessions
            .iter()
            .all(|candidate| !candidate.help.contains('\x1b')
                && !candidate.help.contains('\u{202e}'))
    );
}

#[test]
fn qualified_saved_queries_complete_their_match_fields_and_values() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&directory.path().join("archive")).unwrap();
    let mut bindings = polylogue_query::Bindings::default();
    for (name, text) in [
        ("qualified", "sessions where exists tool_calls()"),
        ("nested", "query:qualified"),
        ("calls", "tool_calls where outcome:error"),
        ("nested-calls", "query:calls"),
    ] {
        let plan = polylogue_query::plan::prepare(text, &bindings, |_| Ok(())).unwrap();
        fixture.saved_query(name, &plan).unwrap();
        bindings
            .add_saved(
                name,
                &plan,
                &format!("{name}-revision"),
                &format!("{name}-root"),
            )
            .unwrap();
    }
    for name in ["qualified", "nested"] {
        let fields = rendered(
            &fixture,
            &format!("query:{name} | matches | where step.out"),
        );
        assert!(
            fields.iter().any(|text| text.ends_with("outcome")),
            "{fields:?}"
        );
        let values = rendered(
            &fixture,
            &format!("query:{name} | matches | where step.outcome:e"),
        );
        assert!(
            values.iter().any(|text| text.ends_with("error")),
            "{values:?}"
        );
        let parent = rendered(&fixture, &format!("query:{name} | matches | where rep"));
        assert!(
            !parent.iter().any(|text| text.ends_with("repo")),
            "{parent:?}"
        );
    }
    for name in ["calls", "nested-calls"] {
        let fields = rendered(&fixture, &format!("query:{name} | where session.rep"));
        assert!(
            fields.iter().any(|text| text.ends_with("session.repo")),
            "{fields:?}"
        );
    }
    assert!(
        !rendered(&fixture, "query:absent | matches | where rep")
            .iter()
            .any(|text| text.ends_with("repo"))
    );
    let text = "sessions where exists tool_calls() | matches | where step.out";
    assert!(
        grammar(text, text.len())
            .iter()
            .any(|candidate| candidate.value == "step.outcome")
    );
}

#[test]
fn settings_offer_declared_keys_and_choices_and_current_values_when_known() {
    let directory = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(&directory.path().join("archive")).unwrap();
    fixture.setting("browser.actions", "on").unwrap();
    let reads = fixture.reads().unwrap();
    let reader = reads.reader().unwrap();
    let authored = reader.completion_setting_keys().unwrap();
    let current = |key: &str| reader.setting_current(key, "fixture-machine").ok();
    let keys = settings(None, "browser.", &authored, current);
    let actions = keys
        .iter()
        .find(|candidate| candidate.value == "browser.actions")
        .unwrap();
    assert_eq!(actions.help, "current: on");
    let values: Vec<_> = settings(Some("browser.actions"), "o", &authored, current)
        .into_iter()
        .map(|candidate| candidate.value)
        .collect();
    assert_eq!(values, ["on", "off"]);
    // Without the daemon, the declarations alone.
    let offline = settings(None, "browser.", &[], |_| None);
    assert!(
        offline
            .iter()
            .any(|candidate| candidate.value == "browser.actions" && candidate.help.is_empty())
    );
}

#[test]
fn completion_handler_answers_from_the_daemons_reader() {
    let daemon = crate::test_daemon::start("A synthetic completion prompt");
    let text = "tool_calls where outcome:err";
    let response = crate::test_daemon::Call::new(
        "internal complete",
        serde_json::json!({"text": text, "position": text.len()}),
    )
    .answer(&daemon.daemon);
    let Some(data) = response.value().cloned() else {
        panic!("{response:?}")
    };
    // The outcomes are ok, error, pending and unknown; only `error` starts with `err`.
    let values: Vec<&str> = data
        .as_array()
        .unwrap()
        .iter()
        .map(|candidate| candidate["value"].as_str().unwrap())
        .collect();
    assert_eq!(values, ["error"]);
    daemon.set("signal.interrupts_per_hour", "2000");
    let response = crate::test_daemon::Call::new(
        "internal complete",
        serde_json::json!({"text": "", "setting": true, "key": "signal.interrupts_per_hour"}),
    )
    .answer(&daemon.daemon);
    let values = response.value().cloned().unwrap();
    assert_eq!(values[0]["value"], "2000", "{values}");
}
