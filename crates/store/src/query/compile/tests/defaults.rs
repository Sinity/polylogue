use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};

#[test]
fn session_rows_keep_source_meaning_and_retained_identity() {
    let options = opts(&[("work", "sessions where repo:morrow")]);
    for query in [
        "sessions | sort by id | limit 1",
        "query:work | limit 1",
        "sessions | union (sessions where repo:morrow)",
        "sessions | fuse (sessions where physical:true)",
    ] {
        assert_eq!(
            compile(query, &options).unwrap().session_rows,
            SessionRows::Logical,
            "{query}"
        );
    }
    for query in [
        "sessions where physical:true",
        "sessions where id:claude-code:one",
        "sessions where kind:main",
        "sessions where physical:true | fuse (sessions)",
    ] {
        assert_eq!(
            compile(query, &options).unwrap().session_rows,
            SessionRows::Physical,
            "{query}"
        );
    }
    for query in [
        "sessions | union (sessions where physical:true)",
        "sessions where physical:true | union (sessions)",
        "sessions | union (sessions where physical:true) | fields id | sort by id | limit 1",
    ] {
        assert!(
            matches!(
                compile(query, &options).unwrap().session_rows,
                SessionRows::PerRow(_)
            ),
            "{query}"
        );
    }
    let mut options = options;
    options.kept.insert(
        "captured".into(),
        Kept {
            output_metadata: Default::default(),
            shape: Some(Shape::Rows),
            row_identity: true,
            unit: "sessions".into(),
            columns: c("sessions").columns,
            match_columns: vec![],
        },
    );
    assert_eq!(
        compile("result:captured", &options).unwrap().session_rows,
        SessionRows::Physical
    );
}

#[test]
fn mixed_session_coordinates_survive_projection_nesting_and_keyset_pages() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    for (id, cwd) in [("a", "/a"), ("a-tail", "/a"), ("b", "/b"), ("b-tail", "/b")] {
        fixture
            .session(SessionFacts {
                id: &format!("claude-code:{id}"),
                kind: "main",
                cwd: Some(cwd),
                ..Default::default()
            })
            .unwrap();
    }
    fixture
        .link("claude-code:a", "claude-code:a-tail", "continuation", None)
        .unwrap();
    fixture
        .link("claude-code:b", "claude-code:b-tail", "continuation", None)
        .unwrap();
    let options = Options {
        page_size: Some(1),
        ..opts(&[])
    };
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let source = "sessions where cwd:/a | union (sessions where id:claude-code:b)";
    for query in [
        format!("{source} | fields id | sort by id | limit 2"),
        format!(
            "{source} | fields id | union (sessions where id:claude-code:a | fields id) | sort by id | limit 2"
        ),
        format!("{source} | intersect (sessions where physical:true) | sort by id"),
    ] {
        let program = compile(&query, &options).unwrap();
        let mut selected = Vec::new();
        let mut after = None;
        loop {
            let page = reader.query_page(&program, after.as_deref()).unwrap();
            for row in &page.rows {
                let reference = crate::selection::row_reference(&program, row).unwrap();
                selected.push((
                    reference.to_string(),
                    program.session_rows.logical(row).unwrap(),
                ));
            }
            if !page.more {
                break;
            }
            let last = page.rows.last().unwrap();
            after = Some(
                program
                    .order
                    .iter()
                    .map(|key| last[&key.column].clone())
                    .collect::<Vec<_>>(),
            );
        }
        assert_eq!(
            selected,
            [
                ("session:claude-code:a".into(), true),
                ("session:claude-code:b".into(), false)
            ],
            "{query}"
        );
    }
    let grouped = compile(
        &format!("{source} | group by origin | count | fields origin, count"),
        &options,
    )
    .unwrap();
    assert_eq!(reader.query_rows(&grouped).unwrap().len(), 1);
}

fn ids(fixture: &Fixture, options: &Options, text: &str) -> Vec<String> {
    let program = compile(text, options).unwrap_or_else(|error| panic!("{text}: {error}"));
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut ids: Vec<String> = reader
        .query_rows(&program)
        .unwrap_or_else(|error| panic!("{text}: {error}"))
        .into_iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    ids.sort();
    ids
}

#[test]
fn active_unknown_matches() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "claude-code:paths",
            ..Default::default()
        })
        .unwrap();
    for (id, on_active_path) in [
        ("claude-code:on", Some(true)),
        ("claude-code:off", Some(false)),
        ("claude-code:unknown", None),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "assistant",
                    written_by: "model",
                    on_active_path,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let options = opts(&[]);
    let ids = |text| ids(&fixture, &options, text);
    // The default keeps the active path and the messages whose path is unknown.
    assert_eq!(ids("messages"), ["claude-code:on", "claude-code:unknown"]);
    assert_eq!(ids("messages where active:true"), ["claude-code:on"]);
    assert_eq!(ids("messages where active:false"), ["claude-code:off"]);
    assert_eq!(
        ids("messages where active:unknown"),
        ["claude-code:unknown"]
    );
    // Unknown matches neither a condition nor its NOT.
    assert_eq!(
        ids("messages where active:* NOT active:true"),
        ["claude-code:off"]
    );
    assert_eq!(
        ids("messages where active:*"),
        ["claude-code:off", "claude-code:on", "claude-code:unknown"]
    );
}

#[test]
fn saved_query_hidden_condition_cannot_widen() {
    let dir = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(dir.path()).unwrap();
    for id in ["claude-code:shown", "claude-code:hidden"] {
        fixture
            .session(SessionFacts {
                id,
                repo: Some("morrow"),
                ..Default::default()
            })
            .unwrap();
    }
    fixture
        .note("hidden", None, None, "session:claude-code:hidden")
        .unwrap();
    let options = opts(&[
        ("morrow", "sessions where repo:morrow"),
        ("hidden-morrow", "sessions where repo:morrow hidden:true"),
        ("over-morrow", "query:morrow"),
    ]);
    let ids = |text| ids(&fixture, &options, text);
    // Inline, a top-level condition on `hidden` lifts the default.
    assert_eq!(
        ids("sessions where repo:morrow hidden:true"),
        ["claude-code:hidden"]
    );
    // A saved relation bound its own defaults; the caller only narrows it,
    // as the source and as a top-level condition alike.
    assert_eq!(ids("query:morrow"), ["claude-code:shown"]);
    assert!(ids("query:morrow hidden:true").is_empty());
    assert_eq!(ids("query:morrow hidden:*"), ["claude-code:shown"]);
    assert!(ids("sessions where query:morrow hidden:true").is_empty());
    assert!(ids("query:over-morrow hidden:true").is_empty());
    // A saved relation that lifts a default itself keeps what it lifted.
    assert_eq!(ids("query:hidden-morrow"), ["claude-code:hidden"]);
    assert_eq!(
        ids("sessions where query:hidden-morrow"),
        ["claude-code:hidden"]
    );
    // Under OR it is membership in its own population; the outer defaults
    // still apply around it.
    assert_eq!(
        ids("sessions where query:morrow OR repo:none"),
        ["claude-code:shown"]
    );
    assert!(ids("sessions where query:hidden-morrow OR repo:none").is_empty());
}
