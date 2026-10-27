use super::*;

#[test]
fn facets_preserve_source_expressions_and_pipeline_text() {
    for text in [
        r#"sessions where (repo:morrow OR repo:atlas) near:"broken | parser" | sort by last_active desc"#,
        r#"messages where ("parse error" OR "export") AND -role:system"#,
        r#"("a | b" OR "λ") -tag:closed"#,
    ] {
        let result = filtered(text, "origin", "claude-code").unwrap();
        let old = polylogue_query::parse(text).unwrap();
        let new = polylogue_query::parse(&result).unwrap();
        assert_eq!(
            old.stages.iter().map(|s| &s.kind).collect::<Vec<_>>(),
            new.stages.iter().map(|s| &s.kind).collect::<Vec<_>>()
        );
        assert!(result.contains(r#"AND origin:"claude-code""#), "{result}");
        assert!(result.contains("OR"));
    }
    for text in [
        "sessions",
        "session",
        " sessions ",
        " SESSIONS ",
        " session ",
        "sessions|sort by last_active desc",
        "",
        "sessions | sort by last_active desc",
    ] {
        assert!(filtered(text, "origin", "claude-code").is_ok(), "{text}");
    }
}

#[test]
fn facet_literals_cannot_add_query_operations() {
    let value = "line\n\t\rmorrow\" OR origin:* | count \\ λ <script>";
    let text = filtered("sessions", "repo", value).unwrap();
    let parsed = polylogue_query::parse(&text).unwrap();
    assert!(parsed.stages.is_empty());
    let Source::Unit {
        cond: Some(cond), ..
    } = parsed.source
    else {
        panic!("{text}")
    };
    let polylogue_query::ast::CondKind::Match {
        value: polylogue_query::ast::Value::One { text, .. },
        ..
    } = cond.kind
    else {
        panic!()
    };
    assert_eq!(text, value);
    assert!(filtered("sessions", "origin:* | count", "x").is_err());
    assert!(filtered("documents", "live", "waiting").is_err());
}

#[test]
fn unsupported_queries_remain_unedited_and_counts_drop_only_ordering() {
    for text in ["", " "] {
        assert_eq!(
            grouped(text, "origin").unwrap(),
            "sessions | group by origin | count"
        );
    }
    for text in [
        "sessions | union (sessions where repo:morrow)",
        "sessions | limit 2",
        "sessions | messages",
        "sessions with messages(first 2)",
        "sessions | group by origin | count",
        "result:\"synthetic:opaque\"",
    ] {
        assert!(filtered(text, "origin", "claude-code").is_err(), "{text}");
        assert!(grouped(text, "origin").is_err(), "{text}");
    }
    assert_eq!(
        grouped(
            r#"sessions where title:"a | b" | sort by last_active desc"#,
            "origin"
        )
        .unwrap(),
        r#"sessions where title:"a | b" | group by origin | count"#
    );
    assert!(grouped("sessions", "tag").is_err());
}
