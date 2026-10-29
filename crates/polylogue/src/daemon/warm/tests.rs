use super::*;
use serde_json::json;

#[test]
fn recent_keeps_the_most_recently_used() {
    let mut r = Recent::new(2);
    r.put("a".into(), 1);
    r.put("b".into(), 2);
    assert_eq!(r.get("a"), Some(1));
    r.put("c".into(), 3);
    assert_eq!(r.get("b"), None, "b was the least recently used");
    assert_eq!(r.get("a"), Some(1));
    assert_eq!(r.get("c"), Some(3));
}

#[test]
fn an_answer_is_reused_only_at_its_read_position() {
    let warm = Warm::default();
    warm.keep_answer("q".into(), "g:1:1".into(), false, &json!({"rows": [1]}));
    assert_eq!(warm.answer("q", "g:1:1"), Some(json!({"rows": [1]})));
    assert_eq!(
        warm.answer("q", "g:2:1"),
        None,
        "a write moved the position"
    );
    assert_eq!(warm.answer("other", "g:1:1"), None);
}

#[test]
fn a_first_page_is_reused_only_briefly() {
    let warm = Warm::default();
    warm.keep_answer("q".into(), "p".into(), true, &json!(1));
    assert!(warm.answer("q", "p").is_some());
    if let Ok(mut answers) = warm.answers.lock()
        && let Some(entry) = answers.entries.back_mut()
    {
        entry.1.made -= FIRST_PAGE_FOR;
    }
    assert!(warm.answer("q", "p").is_none(), "the clock moved on");
}

#[test]
fn completions_cover_fields_values_and_sessions() {
    let data = Completion {
        values: BTreeMap::from([("repo".into(), vec!["polylogue".into(), "sinex".into()])]),
        sessions: vec![("claude-code:abc".into(), "a title".into())],
        ..Completion::default()
    };
    let values = |word: &str| -> Vec<String> {
        completions(&data, word)
            .into_iter()
            .map(|(v, _)| v)
            .collect()
    };
    assert!(values("re").contains(&"repo:".into()));
    assert_eq!(values("repo:p"), vec!["repo:polylogue"]);
    assert!(values("ses").contains(&"sessions".into()));
    assert!(values("claude").contains(&"claude-code:abc".into()));
    assert_eq!(values("claude-code:a"), vec!["claude-code:abc"]);
}

#[test]
fn values_with_spaces_are_quoted() {
    let data = Completion {
        values: BTreeMap::from([("repo".into(), vec!["two words".into()])]),
        ..Default::default()
    };
    assert!(
        completions(&data, "repo:two")
            .iter()
            .any(|(value, _)| value == "repo:\"two words\"")
    );
}

#[test]
fn count_failures_are_visible_and_a_success_clears_them() {
    let daemon = crate::test_daemon::start("count worker failure");
    let warm = Warm::default();
    warm.refresh_counts(&daemon.daemon.reads);
    assert!(warm.count_error().is_none());
    let before = warm.archive_counts.lock().unwrap().clone().unwrap();
    let closed = polylogue_store::ReadPool::open(&daemon.daemon.archive).unwrap();
    closed.close();
    warm.refresh_counts(&closed);
    assert!(warm.count_error().is_some());
    assert_eq!(
        warm.archive_counts.lock().unwrap().as_ref().unwrap().status,
        before.status
    );
    warm.archive_counts.lock().unwrap().as_mut().unwrap().took = Duration::ZERO;
    warm.recount_wanted.store(false, Ordering::Release);
    let reader = daemon.daemon.reads.reader().unwrap();
    let cached = warm.counts(&reader, "later position").unwrap();
    assert_eq!(cached.position, before.position);
    assert!(warm.count_error().is_some());
    assert!(warm.recount_wanted.load(Ordering::Acquire));
    drop(reader);
    warm.refresh_counts(&daemon.daemon.reads);
    assert!(warm.count_error().is_none());
}

#[test]
fn completions_escape_help_and_omit_values_with_controls() {
    let hostile = "\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\u{202e}";
    let mut tags = vec!["ordinary".into()];
    for code in 0..=159 {
        let c = char::from_u32(code).unwrap();
        if c.is_control() {
            tags.push(c.to_string());
        }
    }
    let data = Completion {
        values: BTreeMap::from([("tag".into(), tags), ("query".into(), vec![hostile.into()])]),
        sessions: vec![("claude-code:abc".into(), hostile.into())],
        ..Completion::default()
    };
    let tags = completions(&data, "tag:");
    assert!(tags.iter().any(|(value, _)| value == "tag:ordinary"));
    assert!(
        tags.iter()
            .all(|(value, _)| ["tag:ordinary", "tag:unknown", "tag:*"].contains(&value.as_str()))
    );
    assert!(
        completions(&data, "query:")
            .iter()
            .all(|(value, _)| !value.contains(hostile))
    );
    assert_eq!(
        completions(&data, "claude-code:"),
        vec![(
            "claude-code:abc".into(),
            polylogue_render::terminal::escape(hostile)
        )]
    );
}

#[test]
fn completion_finds_quoted_literal_prefixes_and_preserves_backslashes() {
    let data = Completion {
        values: BTreeMap::from([(
            "repo".into(),
            vec!["two words".into(), "path\\part \"λ\"".into()],
        )]),
        ..Completion::default()
    };
    assert_eq!(completions(&data, "repo:two")[0].0, "repo:\"two words\"");
    assert_eq!(completions(&data, "repo:\"two")[0].0, "repo:\"two words\"");
    let value = &completions(&data, "repo:path")[0].0;
    let parsed = polylogue_query::parse(value).unwrap();
    let polylogue_query::ast::Source::Unit {
        condition: Some(cond),
        ..
    } = parsed.source
    else {
        panic!()
    };
    let polylogue_query::ast::ConditionKind::Match {
        value: polylogue_query::ast::Value::Scalar { text, .. },
        ..
    } = cond.kind
    else {
        panic!()
    };
    assert_eq!(text, "path\\part \"λ\"");
}

#[test]
fn completion_reads_declared_values_and_refreshes_observed_models() {
    let d = crate::test_daemon::start("synthetic prompt");
    let values = |word: &str| match crate::daemon::service::handle(
        crate::daemon::Request::Complete {
            text: word.into(),
            position: word.len(),
        },
        &d.daemon,
    ) {
        crate::daemon::Response::Ok { data, .. } => data,
        other => panic!("{other:?}"),
    };
    assert!(
        values("live:")
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["value"] == "waiting"
                && v["replace"]["start"] == 5
                && v["replace"]["end"] == 5)
    );
    assert!(
        !values("model:")
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value["value"] == "\"model with spaces\"")
    );
    let mut message: serde_json::Value = serde_json::from_str(
        &polylogue_testkit::claude_assistant("a1", "s1", "synthetic answer"),
    )
    .unwrap();
    message["message"]["model"] = json!("model with spaces");
    std::fs::write(
        d.home.join(".claude/projects/-p/s1.jsonl"),
        format!(
            "{}{}\n",
            polylogue_testkit::claude_user("u1", "s1", "synthetic prompt"),
            message
        ),
    )
    .unwrap();
    d.wait_for(|s| s["store"]["messages"] == 2 && values("model:model").as_array().unwrap().iter().any(|value|value["value"]=="\"model with spaces\""));
    assert!(
        values("model:model")
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["value"] == "\"model with spaces\"")
    );
}

#[test]
fn completion_request_preserves_full_text_offsets_and_reports_invalid_boundaries() {
    let daemon = crate::test_daemon::start("synthetic prompt");
    let text = "tool_calls where outcome:err | count";
    let position = text.find(" | ").unwrap();
    match crate::daemon::service::handle(
        crate::daemon::Request::Complete {
            text: text.into(),
            position,
        },
        &daemon.daemon,
    ) {
        crate::daemon::Response::Ok { data, .. } => {
            let candidate = data
                .as_array()
                .unwrap()
                .iter()
                .find(|candidate| candidate["value"] == "error")
                .unwrap();
            let start = candidate["replace"]["start"].as_u64().unwrap() as usize;
            let end = candidate["replace"]["end"].as_u64().unwrap() as usize;
            assert_eq!(candidate["kind"], "value");
            assert_eq!(
                format!(
                    "{}{}{}",
                    &text[..start],
                    candidate["value"].as_str().unwrap(),
                    &text[end..]
                ),
                "tool_calls where outcome:error | count"
            );
        }
        other => panic!("{other:?}"),
    }
    for position in [6, 8] {
        match crate::daemon::service::handle(
            crate::daemon::Request::Complete {
                text: "repo:λ".into(),
                position,
            },
            &daemon.daemon,
        ) {
            crate::daemon::Response::Error { kind, .. } => assert_eq!(kind, "bad_request"),
            other => panic!("{other:?}"),
        }
    }
}
