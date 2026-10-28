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
        units: vec!["sessions".into(), "messages".into()],
        fields: vec!["repo:".into(), "origin:".into()],
        values: vec!["repo:polylogue".into(), "repo:sinex".into()],
        sessions: vec![("claude-code:abc".into(), "a title".into())],
    };
    let values = |word: &str| -> Vec<String> {
        completions(&data, word)
            .into_iter()
            .map(|(v, _)| v)
            .collect()
    };
    assert_eq!(values("re"), vec!["repo:"]);
    assert_eq!(values("repo:p"), vec!["repo:polylogue"]);
    assert_eq!(values("ses"), vec!["sessions"]);
    assert_eq!(values("claude"), vec!["claude-code:abc"]);
    assert_eq!(values("claude-code:a"), vec!["claude-code:abc"]);
}

#[test]
fn values_with_spaces_are_quoted() {
    assert_eq!(quoted("polylogue"), "polylogue");
    assert_eq!(quoted("two words"), "\"two words\"");
}

#[test]
fn completions_escape_help_and_omit_values_with_controls() {
    let hostile = "\x1b]8;;https://example.test\x1b\\link\x1b]8;;\x1b\\\x7f\u{202e}";
    let mut values = vec!["tag:ordinary".into(), format!("query:{hostile}")];
    for code in 0..=159 {
        let c = char::from_u32(code).unwrap();
        if c.is_control() {
            values.push(format!("tag:{c}"));
        }
    }
    let data = Completion {
        values,
        sessions: vec![("claude-code:abc".into(), hostile.into())],
        ..Completion::default()
    };
    assert_eq!(
        completions(&data, "tag:"),
        vec![("tag:ordinary".into(), "".into())]
    );
    assert!(completions(&data, "query:").is_empty());
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
        values: vec![
            format!("repo:{}", quoted("two words")),
            format!("repo:{}", quoted("path\\part \"λ\"")),
        ],
        ..Completion::default()
    };
    assert_eq!(completions(&data, "repo:two")[0].0, "repo:\"two words\"");
    assert_eq!(completions(&data, "repo:\"two")[0].0, "repo:\"two words\"");
    let value = &completions(&data, "repo:path")[0].0;
    let parsed = polylogue_query::parse(value).unwrap();
    let polylogue_query::ast::Source::Sessions { cond } = parsed.source else {
        panic!()
    };
    let polylogue_query::ast::CondKind::Match {
        value: polylogue_query::ast::Value::One { text, .. },
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
        crate::daemon::Request::Complete { word: word.into() },
        &d.daemon,
    ) {
        crate::daemon::Response::Ok { data } => data,
        other => panic!("{other:?}"),
    };
    assert!(
        values("live:")
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["value"] == "live:waiting")
    );
    assert!(values("model:").as_array().unwrap().is_empty());
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
    d.wait_for(|s| s["store"]["messages"] == 2);
    assert!(
        values("model:model")
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v["value"] == "model:\"model with spaces\"")
    );
}
