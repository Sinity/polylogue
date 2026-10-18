use super::*;

fn words(query: &str) -> Vec<TextTerm> {
    bind(query).unwrap().text
}

#[test]
fn binds_supported_forms_and_rejects_the_rest() {
    let plan = bind(
        "sessions where origin:claude-code kind:* text:\"weekday offset\" refactor* \
         | sort by last_active desc | limit 3",
    )
    .unwrap();
    assert!(plan.all_kinds && plan.recent);
    assert_eq!(plan.limit, Some(3));
    assert_eq!(plan.text.len(), 2);
    assert_eq!(words("find list"), vec![TextTerm::Word("list".into())]);
    assert_eq!(
        words("find messages"),
        vec![TextTerm::Word("messages".into())]
    );
    for query in [
        "find",
        "find token OR other",
        "near:chrono",
        "messages chrono",
        "chrono OR parser",
        "chrono | limit 1 | where x",
        "origin:",
        "\"unfinished",
        "chrono AND",
        "chrono | sort by cost desc",
        "chrono | limit 3 | sort by last_active desc",
        "--reveal",
        "reop:morrow",
    ] {
        assert!(bind(query).is_err(), "{query}");
    }
}

#[test]
fn quoted_keywords_are_search_text() {
    for query in [
        "\"where\"",
        "sessions \"where\"",
        "sessions where \"where\"",
    ] {
        assert_eq!(
            words(query),
            vec![TextTerm::Phrase("where".into())],
            "{query}"
        );
    }
    assert_eq!(
        words("\"sessions\""),
        vec![TextTerm::Phrase("sessions".into())]
    );
    for query in ["where chrono", "sessions where", "sessions where | limit 0"] {
        assert!(bind(query).is_err(), "{query}");
    }
}

#[test]
fn quoted_stage_words_are_not_syntax() {
    for query in [
        "chrono | \"limit\" 0",
        "chrono | limit \"0\"",
        "chrono | \"sort\" by last_active desc",
        "chrono | sort \"by\" last_active desc",
        "chrono | sort by \"last_active\" desc",
        "chrono | sort by last_active \"desc\"",
        "chrono | limit 1 \"|\" limit 0",
    ] {
        assert!(bind(query).is_err(), "{query}");
    }
}

#[test]
fn quoted_and_is_text_and_quoted_pipe_is_rejected() {
    assert_eq!(words("\"AND\""), vec![TextTerm::Phrase("AND".into())]);
    assert_eq!(
        words("chrono AND \"AND\""),
        vec![
            TextTerm::Word("chrono".into()),
            TextTerm::Phrase("AND".into())
        ]
    );
    for query in ["\"|\"", "chrono AND \"|\""] {
        let error = bind(query).unwrap_err();
        assert!(error.message.contains("empty phrase"), "{query}: {error:?}");
    }
}

#[test]
fn tool_queries_need_one_session_and_a_limit_in_range() {
    let plan = bind_tools("find tool_calls where session:codex:a text:grep | limit 5")
        .unwrap()
        .unwrap();
    assert!(plan.calls_only);
    assert_eq!(plan.session, "codex:a");
    assert_eq!(plan.text.as_deref(), Some("grep"));
    assert_eq!(plan.limit, 5);
    assert_eq!(
        bind_tools("tool_items session:x").unwrap().unwrap().limit,
        100
    );
    assert!(bind_tools("chrono").unwrap().is_none());
    for query in [
        "tool_calls",
        "tool_calls session:a session:b",
        "tool_calls session:a | limit 0",
        "tool_calls session:a | limit 4097",
        "tool_calls session:a | sort by last_active desc",
        "tool_calls session:a origin:codex",
    ] {
        assert!(bind_tools(query).is_err(), "{query}");
    }
}
