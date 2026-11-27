use super::*;
use crate::query::ast::Unit;
use polylogue_model::{
    Json, Reference,
    annotation::{AnnotationKind, LinkRole, NewAnnotation, NewAnnotationLink},
    user::UserWrite,
};

fn all_columns_options() -> crate::query::Options {
    let mut opts = super::options();
    opts.fields.clear();
    opts
}

fn rows(f: &Fixture, text: &str, fields: &[&str]) -> Vec<Value> {
    let mut opts = all_columns_options();
    opts.fields = fields.iter().map(|s| (*s).into()).collect();
    opts.page_size = Some(100);
    let program = crate::query::compile(text, &opts).unwrap_or_else(|e| panic!("{text}: {e}"));
    let mut rows = f
        .reads()
        .unwrap()
        .reader()
        .unwrap()
        .query_rows(&program)
        .unwrap_or_else(|e| panic!("{text}: {e}"));
    for row in &mut rows {
        present(&program, row).unwrap();
    }
    rows
}
fn statement(kind: AnnotationKind, value: &str, links: &[(LinkRole, String)]) -> NewAnnotation {
    NewAnnotation {
        kind,
        key: None,
        value: Some(Json::String(value.into())),
        valid_until: None,
        links: links
            .iter()
            .map(|(role, r)| NewAnnotationLink {
                role: *role,
                reference: Reference::parse(r).unwrap(),
            })
            .collect(),
    }
}
fn session(f: &mut Fixture, id: &str) -> i64 {
    f.session(SessionFacts {
        id,
        kind: "main",
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn request_field_matches_every_row_of_one_write() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let request = f
        .user(UserWrite::Assert {
            annotations: vec![
                statement(
                    AnnotationKind::Note,
                    "first",
                    &[(LinkRole::About, "session:test:work".into())],
                ),
                statement(
                    AnnotationKind::Note,
                    "second",
                    &[(LinkRole::About, "session:test:work".into())],
                ),
            ],
        })
        .unwrap();
    let got = rows(
        &f,
        &format!("annotations where request:{request} | sort by id"),
        &["id", "value"],
    );
    assert_eq!(
        got.iter()
            .map(|x| x["value"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["first", "second"]
    );
    assert!(
        rows(
            &f,
            &format!("annotations where request:{request}x"),
            &["id"]
        )
        .is_empty()
    );
}

#[test]
fn links_to_matches_supports_links_and_ignores_about_links() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let target = f
        .note("note", None, Some("original"), "session:test:work")
        .unwrap();
    let linked = f
        .user(UserWrite::Assert {
            annotations: vec![statement(
                AnnotationKind::Note,
                "support",
                &[
                    (LinkRole::About, "session:test:work".into()),
                    (LinkRole::Supports, format!("annotation:{target}@{target}")),
                ],
            )],
        })
        .unwrap();
    assert_eq!(
        rows(
            &f,
            &format!("annotations where links_to:annotation:{target}"),
            &["id"]
        )[0]["id"],
        linked
    );
    assert!(
        rows(&f, "annotations where links_to:session:test:work", &["id"]).is_empty(),
        "about links do not satisfy links_to"
    );
}

#[test]
fn a_session_counts_open_items_and_corrections_about_anything_inside_it() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let pk = session(&mut f, "test:work");
    session(&mut f, "test:other");
    // One reference of each kind that lies in the session, and one outside it.
    let inside = [
        "session:test:work",
        "message:test:work-m",
        "block:test:work-m#0",
        "span:message:test:work-m#b=0-2",
        "span:block:test:work-m#0#c=0-2",
        "tool:test:work#call-1",
    ];
    for kind in [AnnotationKind::Question, AnnotationKind::Correction] {
        for target in inside.into_iter().chain(["session:test:other"]) {
            f.user(UserWrite::Assert {
                annotations: vec![statement(kind, target, &[(LinkRole::About, target.into())])],
            })
            .unwrap();
        }
    }
    // The message arrives after the annotations that name it and its parts.
    f.message(
        pk,
        MessageFacts {
            id: "test:work-m",
            role: "assistant",
            written_by: "model",
            text: Some("work"),
            ..Default::default()
        },
    )
    .unwrap();
    let found = rows(&f, "sessions", &["id", "open_items", "corrections"]);
    let counts: Vec<_> = found
        .iter()
        .map(|row| {
            (
                row["id"].as_str().unwrap(),
                row["open_items"].as_i64().unwrap(),
                row["corrections"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(counts.len(), 2);
    assert!(counts.contains(&("test:work", 6, 6)), "{counts:?}");
    assert!(counts.contains(&("test:other", 1, 1)), "{counts:?}");
}

#[test]
fn answered_and_open_follow_links() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let question = f
        .user(UserWrite::Assert {
            annotations: vec![statement(
                AnnotationKind::Question,
                "why?",
                &[(LinkRole::About, "session:test:work".into())],
            )],
        })
        .unwrap();
    assert_eq!(
        rows(&f, "annotations where kind:question open:true", &["id"])[0]["id"],
        question
    );
    assert_eq!(
        rows(
            &f,
            "annotations where kind:question answered:false",
            &["id"]
        )[0]["id"],
        question
    );
    f.user(UserWrite::Assert {
        annotations: vec![statement(
            AnnotationKind::Note,
            "because",
            &[
                (LinkRole::About, format!("annotation:{question}@{question}")),
                (
                    LinkRole::RepliesTo,
                    format!("annotation:{question}@{question}"),
                ),
            ],
        )],
    })
    .unwrap();
    assert!(rows(&f, "annotations where kind:question open:true", &["id"]).is_empty());
    assert_eq!(
        rows(&f, "annotations where kind:question answered:true", &["id"])[0]["id"],
        question
    );
    let blocker = f
        .user(UserWrite::Assert {
            annotations: vec![statement(
                AnnotationKind::Problem,
                "blocked",
                &[
                    (LinkRole::About, "session:test:work".into()),
                    (LinkRole::About, format!("annotation:{question}@{question}")),
                    (
                        LinkRole::Blocks,
                        format!("annotation:{question}@{question}"),
                    ),
                ],
            )],
        })
        .unwrap();
    assert_eq!(
        rows(
            &f,
            &format!("annotations where id:{question} open:true"),
            &["id"]
        )[0]["id"],
        question
    );
    f.user(UserWrite::WithdrawAnnotations {
        targets: vec![Reference::parse(&format!("annotation:{blocker}@{blocker}")).unwrap()],
        reason: None,
    })
    .unwrap();
    assert!(
        rows(
            &f,
            &format!("annotations where id:{question} open:true"),
            &["id"]
        )
        .is_empty()
    );
}

#[test]
fn judged_unknown_includes_an_unjudged_revision() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let note = f
        .note("note", None, Some("unreviewed"), "session:test:work")
        .unwrap();
    assert_eq!(
        rows(&f, "annotations where judged:unknown", &["id"])[0]["id"],
        note
    );
}

#[test]
fn ends_open_each_fact() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    for id in ["prompt", "tool", "todo", "partial", "handoff", "done"] {
        let pk = session(&mut f, &format!("test:{id}"));
        let message = f
            .message(
                pk,
                MessageFacts {
                    id: &format!("test:{id}-m"),
                    role: if id == "prompt" { "user" } else { "assistant" },
                    written_by: if id == "prompt" { "human" } else { "model" },
                    text: Some("recorded"),
                    ..Default::default()
                },
            )
            .unwrap();
        match id {
            "tool" => {
                f.tool_call(
                    pk,
                    message,
                    CallFacts {
                        tool: "Bash",
                        action: "shell",
                        command: Some("cargo test"),
                        outcome: "unknown",
                        ..Default::default()
                    },
                )
                .unwrap();
            }
            "todo" => {
                f.live_facts(
                    pk,
                    &polylogue_model::live::LiveFacts {
                        todo: Some("next item".into()),
                        ..Default::default()
                    },
                )
                .unwrap();
            }
            "partial" => {
                f.note("outcome", None, Some("partial"), "session:test:partial")
                    .unwrap();
            }
            "handoff" => {
                f.note("handoff", None, Some("continue"), "session:test:handoff")
                    .unwrap();
            }
            _ => {}
        }
    }
    let got = rows(&f, "sessions where ends_open:true | sort by id", &["id"]);
    assert_eq!(
        got.iter()
            .map(|x| x["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        [
            "test:handoff",
            "test:partial",
            "test:prompt",
            "test:todo",
            "test:tool"
        ]
    );
}

/// A failure is recovered by a later passing run of the same command
/// (docs/measures.md, tool outcomes); a later failure is not recovery.
#[test]
fn a_failure_is_recovered_only_by_a_later_success() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let pk = f
        .session(SessionFacts {
            id: "test:work",
            kind: "main",
            last: Some(0),
            ..Default::default()
        })
        .unwrap();
    let m = f
        .message(
            pk,
            MessageFacts {
                id: "test:work-m",
                role: "assistant",
                written_by: "model",
                text: Some("work"),
                at: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    for (command, outcome, at) in [
        ("cargo test", "error", 1),
        ("cargo test", "error", 2),
        ("cargo build", "error", 3),
        ("cargo build", "ok", 4),
    ] {
        f.tool_call(
            pk,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some(command),
                outcome,
                at: Some(at),
                ..Default::default()
            },
        )
        .unwrap();
    }
    let got = rows(&f, "sessions where id:test:work", &["unrecovered_failures"]);
    assert_eq!(got[0]["unrecovered_failures"], 1);
}

#[test]
fn triage_score_sums_named_facts() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let pk = f
        .session(SessionFacts {
            id: "test:work",
            kind: "main",
            last: Some(0),
            ..Default::default()
        })
        .unwrap();
    let m = f
        .message(
            pk,
            MessageFacts {
                id: "test:work-m",
                role: "assistant",
                written_by: "model",
                text: Some("work"),
                at: Some(0),
                ..Default::default()
            },
        )
        .unwrap();
    for i in 0..3 {
        f.tool_call(
            pk,
            m,
            CallFacts {
                tool: "Bash",
                action: "shell",
                command: Some("cargo test"),
                outcome: "error",
                at: Some(i),
                output: Some("failure"),
                ..Default::default()
            },
        )
        .unwrap();
    }
    for _ in 0..3 {
        f.note("correction", None, Some("correction"), "session:test:work")
            .unwrap();
    }
    f.note("question", None, Some("open question"), "session:test:work")
        .unwrap();
    let got = rows(
        &f,
        "sessions where id:test:work",
        &[
            "unrecovered_failures",
            "failure_streak",
            "outlier",
            "corrections",
            "open_items",
            "triage_score",
        ],
    );
    assert_eq!(got[0]["unrecovered_failures"], 1);
    assert_eq!(got[0]["failure_streak"], 3);
    assert_eq!(got[0]["outlier"], false);
    assert_eq!(got[0]["corrections"], 3);
    assert_eq!(got[0]["open_items"], 1);
    assert_eq!(got[0]["triage_score"], 4);
    f.note("tag", Some("triaged"), None, "session:test:work")
        .unwrap();
    assert_eq!(
        rows(&f, "sessions where id:test:work", &["triage_score"])[0]["triage_score"],
        0
    );
    f.message_time(
        m,
        Some(polylogue_model::Timestamp::from_unix_ms(1000).unwrap()),
        polylogue_model::TimeAccuracy::intrinsic(polylogue_model::Precision::Instant),
    )
    .unwrap();
    assert_eq!(
        rows(&f, "sessions where id:test:work", &["triage_score"])[0]["triage_score"],
        4,
        "activity after triage exposes the remaining work again"
    );
}

#[test]
fn month_day_matches_across_years() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let pk = session(&mut f, "test:work");
    for (id, at) in [
        ("test:old", "2025-11-09T12:00:00Z"),
        ("test:new", "2026-11-09T12:00:00Z"),
        ("test:other", "2026-11-10T12:00:00Z"),
    ] {
        let at = at.parse::<jiff::Timestamp>().unwrap().as_millisecond();
        f.message(
            pk,
            MessageFacts {
                id,
                at: Some(at),
                role: "user",
                written_by: "human",
                text: Some("date"),
                ..Default::default()
            },
        )
        .unwrap();
    }
    assert_eq!(
        rows(&f, "messages where month_day:11-09 | sort by id", &["id"])
            .iter()
            .map(|x| x["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["test:new", "test:old"]
    );
}

#[test]
fn cost_with_parents_includes_prefix() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    for (id, values) in [
        ("a", vec![2_000_000_000, 3_000_000_000, 100_000_000_000]),
        ("b", vec![5_000_000_000]),
        ("c", vec![7_000_000_000]),
    ] {
        let pk = session(&mut f, &format!("test:{id}"));
        for (i, cost) in values.into_iter().enumerate() {
            f.message(
                pk,
                MessageFacts {
                    id: &format!("test:{id}-{i}"),
                    role: "assistant",
                    written_by: "model",
                    text: Some("priced"),
                    cost_nusd: Some(cost),
                    ..Default::default()
                },
            )
            .unwrap();
        }
    }
    f.link("test:a", "test:b", "fork", None).unwrap();
    f.link("test:b", "test:c", "fork", None).unwrap();
    let conn =
        crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP).unwrap();
    conn.execute(
        "UPDATE session_link SET branch_point='test:a-1' WHERE child_id='test:b'",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE session_link SET branch_point='test:a-0' WHERE child_id='test:c'",
        [],
    )
    .unwrap();
    drop(conn);
    assert_eq!(
        rows(&f, "sessions where id:test:b", &["cost_with_parents"])[0]["cost_with_parents"],
        10
    );
    assert_eq!(
        rows(&f, "sessions where id:test:c", &["cost_with_parents"])[0]["cost_with_parents"],
        9,
        "a branch point in the inherited prefix excludes b's own message and a's tail"
    );
}

#[test]
fn alternate_true_admits_alternate_text() {
    use crate::fixture::{
        authored::operator_request,
        writer::{WriterArchive, message},
    };
    use polylogue_model::{
        AlternateKind, AlternateRelation, Block, NewAlternate, NewAlternatePart, Role,
    };
    let mut archive = WriterArchive::new();
    archive.write(
        "work",
        vec![message(
            "original",
            "work",
            Role::Assistant,
            1,
            vec![Block::Text {
                text: "original text".into(),
            }],
        )],
    );
    let source = "message:claude-code:work-original";
    let part = "block:claude-code:work-original#0";
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let source_hash = reader
        .reference_hash(&Reference::parse(source).unwrap())
        .unwrap()
        .unwrap();
    let of_hash = reader
        .reference_hash(&Reference::parse(part).unwrap())
        .unwrap()
        .unwrap();
    drop(reader);
    drop(pool);
    archive
        .writer
        .archive()
        .user(operator_request(
            "translated".into(),
            UserWrite::Alternate {
                alternate: NewAlternate {
                    base: None,
                    source: source.into(),
                    source_hash,
                    kind: AlternateKind::Translation,
                    from_lang: Some("en".into()),
                    to_lang: Some("pl".into()),
                    method: None,
                    parts: vec![NewAlternatePart {
                        of: part.into(),
                        of_hash,
                        relation: AlternateRelation::Translates,
                        body: "unikalny".into(),
                    }],
                },
            },
        ))
        .unwrap();
    archive.writer.flush_rows().unwrap();
    let read = |text: &str| {
        let mut opts = all_columns_options();
        opts.origins = vec!["claude-code".into()];
        opts.fields = vec!["id".into()];
        let program = crate::query::compile(text, &opts).unwrap();
        let pool = crate::ReadPool::open(&archive.archive).unwrap();
        let mut rows = pool.reader().unwrap().query_rows(&program).unwrap();
        for row in &mut rows {
            present(&program, row).unwrap();
        }
        rows
    };
    assert!(read("messages where unikalny").is_empty());
    assert_eq!(
        read("messages where alternate:true unikalny")[0]["id"],
        "claude-code:work-original"
    );
    assert_eq!(
        read("messages where lang:pl unikalny")[0]["id"],
        "claude-code:work-original"
    );
    assert!(read("messages where lang:de unikalny").is_empty());
    assert_eq!(read("alternates where to_lang:pl").len(), 1);
    let assert_head = |expected: Option<&str>| {
        let pool = crate::ReadPool::open(&archive.archive).unwrap();
        let reader = pool.reader().unwrap();
        let chosen = reader
            .selected_alternate(source, source_hash, &["translation:pl".into()], false)
            .unwrap();
        assert_eq!(chosen.as_ref().map(|v| v.id.as_str()), expected);
        let expected_rows = expected
            .into_iter()
            .map(|id| serde_json::json!({"id":id}))
            .collect::<Vec<_>>();
        assert_eq!(read("alternates where to_lang:pl"), expected_rows);
    };
    let revise = |id: &str, base: &str, body: &str| {
        archive
            .writer
            .archive()
            .user(operator_request(
                id.into(),
                UserWrite::Alternate {
                    alternate: NewAlternate {
                        base: Some(base.into()),
                        source: source.into(),
                        source_hash,
                        kind: AlternateKind::Translation,
                        from_lang: Some("en".into()),
                        to_lang: Some("pl".into()),
                        method: None,
                        parts: vec![NewAlternatePart {
                            of: part.into(),
                            of_hash,
                            relation: AlternateRelation::Translates,
                            body: body.into(),
                        }],
                    },
                },
            ))
            .unwrap();
        archive.writer.flush_rows().unwrap();
    };
    let withdraw = |id: &str, target: &str| {
        archive
            .writer
            .archive()
            .user(operator_request(
                id.into(),
                UserWrite::WithdrawAnnotations {
                    targets: vec![Reference::parse(target).unwrap()],
                    reason: None,
                },
            ))
            .unwrap();
        archive.writer.flush_rows().unwrap();
    };
    assert_head(Some("translated"));
    revise("translated-next", "translated", "drugi");
    revise("translated-last", "translated-next", "trzeci");
    assert_head(Some("translated-last"));
    assert!(read("messages where alternate:true unikalny").is_empty());
    assert_eq!(read("messages where lang:pl trzeci").len(), 1);
    assert_eq!(read("alternates where current:false to_lang:pl").len(), 2);
    // A withdrawn intermediate revision remains transparent to its successor.
    withdraw("withdraw-middle", "alternate:translated@translated-next");
    assert_head(Some("translated-last"));
    assert!(read("messages where alternate:true unikalny").is_empty());
    assert_eq!(read("alternates where withdrawn:true to_lang:pl").len(), 1);
    withdraw("withdraw-last", "alternate:translated@translated-last");
    assert_head(Some("translated"));
    assert_eq!(read("messages where lang:pl unikalny").len(), 1);
    withdraw("withdraw-root", "alternate:translated");
    assert_head(None);
    assert!(read("messages where alternate:true unikalny").is_empty());
    assert!(read("messages where lang:pl unikalny").is_empty());
    withdraw("restore-root", "annotation:withdraw-root@withdraw-root");
    assert_head(Some("translated"));
    assert_eq!(read("messages where alternate:true unikalny").len(), 1);
}

#[test]
fn queries_runs_results_units_list_rows() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let plan = polylogue_query::plan::prepare(
        "sessions",
        &polylogue_query::Bindings {
            origins: vec!["test".into()],
            ..Default::default()
        },
        |_| Ok(()),
    )
    .unwrap();
    f.saved_query("example", &plan).unwrap();
    f.kept_result("018f0000-0000-7000-8000-000000000001","sessions",r#"{"unit":"sessions","columns":[{"name":"id","kind":"text"}],"rows":[{"id":"test:work"}]}"#).unwrap();
    let named = rows(&f, "queries where name:*", &["name", "text", "query"]);
    assert_eq!(named.len(), 1);
    assert_eq!(named[0]["name"], "example");
    assert_eq!(named[0]["text"], "sessions");
    assert_eq!(named[0]["query"], format!("query:{}", plan.question_id()));
    let kept = rows(&f, "results", &["id", "query", "rows"]);
    assert_eq!(kept.len(), 1);
    assert_eq!(kept[0]["id"], "result:018f0000-0000-7000-8000-000000000001");
    assert_eq!(kept[0]["rows"], 1);
    assert!(
        rows(&f, "runs", &["id", "query", "at"]).is_empty(),
        "the fixture has no executed run; listing must not invent one"
    );
}

#[test]
fn last_is_a_search_word() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let first = session(&mut f, "test:first");
    let second = session(&mut f, "test:second");
    f.message(
        first,
        MessageFacts {
            id: "test:first-m",
            role: "user",
            written_by: "human",
            text: Some("last word"),
            ..Default::default()
        },
    )
    .unwrap();
    f.message(
        second,
        MessageFacts {
            id: "test:second-m",
            role: "user",
            written_by: "human",
            text: Some("different"),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rows(&f, "last", &["id"])[0]["id"], "test:first");
}

#[test]
fn recorded_unit_objects_execute_with_original_hash_and_body() {
    let fixture: Value = serde_json::from_str(include_str!(
        "../../../../../query/src/plan/tests/fixtures/recorded-unit-objects.json"
    ))
    .unwrap();
    assert_eq!(
        fixture["source_revision"],
        "a548c4e667ac2dfe4bf601fedfc9b6fb6299e26d"
    );
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:parent");
    f.note(
        "note",
        None,
        Some("retained interpretation"),
        "session:test:parent",
    )
    .unwrap();
    for (n, record) in fixture["objects"].as_array().unwrap().iter().enumerate() {
        let original = record["original_json"].as_str().unwrap();
        let object: polylogue_model::query_objects::QueryObject =
            serde_json::from_str(original).unwrap();
        assert_eq!(object.id.to_string(), record["hash"].as_str().unwrap());
        let plan =
            polylogue_query::plan::replay(&object, polylogue_query::plan::catalog_fingerprint())
                .unwrap();
        assert_eq!(plan.query_object().unwrap(), object);
        f.saved_query(&format!("recorded-{n}"), &plan).unwrap();
        let pool = f.reads().unwrap();
        let reader = pool.reader().unwrap();
        let restored = reader
            .query_object(&object.id.to_string())
            .unwrap()
            .unwrap();
        assert_eq!(
            serde_json::to_string(&restored).unwrap(),
            original,
            "original sealed body bytes survive persistence"
        );
        let restored_plan =
            polylogue_query::plan::replay(&restored, polylogue_query::plan::catalog_fingerprint())
                .unwrap();
        let program = crate::query::lower(&restored_plan, &all_columns_options()).unwrap();
        let mut got = reader.query_rows(&program).unwrap();
        for row in &mut got {
            present(&program, row).unwrap();
        }
        let named = reader
            .prepare_query_plan(&format!("query:recorded-{n}"), &all_columns_options())
            .unwrap();
        assert!(
            named
                .definitions()
                .iter()
                .any(|d| d.question_id == object.id)
        );
        let named_program = reader.lower_plan(&named, &all_columns_options()).unwrap();
        let mut named_rows = reader.query_rows(&named_program).unwrap();
        for row in &mut named_rows {
            present(&named_program, row).unwrap();
        }
        assert_eq!(
            named_rows, got,
            "current named reader executes the original object"
        );
        assert_eq!(got.len(), 1);
        assert_eq!(got[0]["value"], "retained interpretation");
        assert_eq!(restored_plan.question_id(), object.id);
    }
}

#[test]
fn here_matches_worktrees_and_repo_root() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    for (id, cwd, root) in [
        ("test:main", "/workspace/main/src", None),
        ("test:worktree", "/workspace/branch/src", None),
        ("test:clone", "/elsewhere/clone", Some("shared-root")),
        ("test:other", "/workspace/mainly", None),
    ] {
        let pk = f
            .session(SessionFacts {
                id,
                kind: "main",
                cwd: Some(cwd),
                ..Default::default()
            })
            .unwrap();
        if let Some(root) = root {
            f.live(pk, "session_start", 1, None, Some(root)).unwrap();
        }
    }
    let mut opts = all_columns_options();
    opts.page_size = Some(20);
    opts.fields = vec!["id".into()];
    opts.here = Some(polylogue_model::query_objects::HereScope {
        paths: vec!["/workspace/main".into(), "/workspace/branch".into()],
        repo_roots: vec!["shared-root".into()],
    });
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    for (text, expected) in [
        (
            "here | sort by id",
            vec!["test:clone", "test:main", "test:worktree"],
        ),
        (
            "repo:here | sort by id",
            vec!["test:clone", "test:main", "test:worktree"],
        ),
        ("cwd:here | sort by id", vec!["test:main", "test:worktree"]),
        ("NOT repo:here | sort by id", vec!["test:other"]),
    ] {
        let p = crate::query::compile(text, &opts).unwrap();
        let got = reader.query_rows(&p).unwrap();
        assert_eq!(
            got.iter()
                .map(|r| r["id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            expected,
            "{text}"
        );
    }
}

#[test]
fn here_without_cwd_refused() {
    for text in ["here", "repo:here", "cwd:here"] {
        let error = crate::query::compile(text, &all_columns_options())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("here names the caller's repository")
                && error.contains("no working directory"),
            "{text}: {error}"
        );
    }
}

#[test]
fn goal_stays_open_until_a_current_reply_is_accepted() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let goal = f
        .user(UserWrite::Assert {
            annotations: vec![statement(
                AnnotationKind::Goal,
                "finish",
                &[(LinkRole::About, "session:test:work".into())],
            )],
        })
        .unwrap();
    let reply = f
        .user(UserWrite::Assert {
            annotations: vec![statement(
                AnnotationKind::Note,
                "done",
                &[
                    (LinkRole::About, format!("annotation:{goal}@{goal}")),
                    (LinkRole::RepliesTo, format!("annotation:{goal}@{goal}")),
                ],
            )],
        })
        .unwrap();
    assert_eq!(
        rows(&f, "annotations where kind:goal open:true", &["id"])[0]["id"],
        goal
    );
    f.user(UserWrite::Assert {
        annotations: vec![NewAnnotation {
            kind: AnnotationKind::Judgment,
            key: Some("accepted".into()),
            value: None,
            valid_until: None,
            links: vec![NewAnnotationLink {
                role: LinkRole::About,
                reference: Reference::parse(&format!("annotation:{reply}@{reply}")).unwrap(),
            }],
        }],
    })
    .unwrap();
    assert!(rows(&f, "annotations where kind:goal open:true", &["id"]).is_empty());
}

#[test]
fn outlier_uses_only_the_repositorys_preceding_thirty_days() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let now = "2026-10-29T12:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    for (id, repo, at, cost) in [
        ("p1", "morrow", now - 86400000, 2_000_000_000),
        ("p2", "morrow", now - 2 * 86400000, 3_000_000_000),
        ("p3", "morrow", now - 3 * 86400000, 4_000_000_000),
        ("future", "morrow", now + 86400000, 1_000_000_000_000),
        ("old", "morrow", now - 31 * 86400000, 1_000_000_000_000),
        ("other", "elsewhere", now - 86400000, 1_000_000_000_000),
        ("current", "morrow", now, 9_000_000_000),
        // Its window is p1..p3 and current: median (3e9+4e9)/2 = 3.5e9, so the
        // outlier threshold is 10.5e9 and 10e9 stays below it.
        ("modest", "morrow", now + 1000, 10_000_000_000),
    ] {
        let name = format!("test:{id}");
        let pk = f
            .session(SessionFacts {
                id: &name,
                kind: "main",
                repo: Some(repo),
                last: Some(at),
                ..Default::default()
            })
            .unwrap();
        f.message(
            pk,
            MessageFacts {
                id: &format!("test:{id}-m"),
                role: "assistant",
                written_by: "model",
                at: Some(at),
                text: Some("priced"),
                cost_nusd: Some(cost),
                ..Default::default()
            },
        )
        .unwrap();
    }
    assert_eq!(
        rows(&f, "sessions where id:test:current", &["outlier"])[0]["outlier"],
        true
    );
    assert_eq!(
        rows(&f, "sessions where id:test:modest", &["outlier"])[0]["outlier"],
        false
    );
}

#[test]
fn since_seen_tracks_the_computed_author_and_new_source_or_authored_work() {
    use crate::fixture::{
        authored::operator_request,
        writer::{WriterArchive, message},
    };
    use polylogue_model::{
        Role,
        annotation::{AnnotationPatch, Change},
        user::SeenMark,
    };
    let mut archive = WriterArchive::new();
    let first = message("first", "reader", Role::User, 1, vec![]);
    archive.write("reader", vec![first.clone()]);
    let note = archive
        .writer
        .archive()
        .user(operator_request(
            "note-for-reader".into(),
            UserWrite::Assert {
                annotations: vec![statement(
                    AnnotationKind::Note,
                    "original",
                    &[(LinkRole::About, "session:claude-code:reader".into())],
                )],
            },
        ))
        .unwrap()
        .id;
    let count = |archive: &WriterArchive, author: &str, query: &str| {
        let mut opts = all_columns_options();
        opts.origins = vec!["claude-code".into()];
        opts.seen_marks_of = Some(author.into());
        let program = crate::query::compile(query, &opts).unwrap();
        crate::ReadPool::open(&archive.archive)
            .unwrap()
            .reader()
            .unwrap()
            .query_rows(&program)
            .unwrap()
            .len()
    };
    let sessions = "sessions where since:seen";
    let annotations = "annotations where kind:note since:seen";
    assert_eq!(count(&archive, "operator", sessions), 1);
    assert_eq!(count(&archive, "operator", annotations), 1);
    let at = crate::ReadPool::open(&archive.archive)
        .unwrap()
        .reader()
        .unwrap()
        .change_position()
        .unwrap();
    archive
        .writer
        .archive()
        .user(operator_request(
            "mark-reader".into(),
            UserWrite::MarkSeen {
                marks: vec![
                    SeenMark {
                        target: "session:claude-code:reader".into(),
                        item: None,
                        at: at.clone(),
                    },
                    SeenMark {
                        target: format!("annotation:{note}"),
                        item: None,
                        at,
                    },
                ],
            },
        ))
        .unwrap();
    assert_eq!(count(&archive, "operator", sessions), 0);
    assert_eq!(count(&archive, "operator", annotations), 0);
    assert_eq!(count(&archive, "operator", "annotations"), 1);
    assert_eq!(
        count(&archive, "operator", "annotations where kind:seen"),
        2
    );
    let feed_program = crate::query::compile("feed", &all_columns_options()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let feed_reader = pool.reader().unwrap();
    let feed = feed_reader.query_rows(&feed_program).unwrap();
    assert!(feed.iter().all(|row| row["annotation_kind"] != "seen"));
    drop(feed_reader);
    drop(pool);
    assert_eq!(count(&archive, "other-reader", sessions), 1);
    assert_eq!(count(&archive, "other-reader", annotations), 1);
    archive.write(
        "reader",
        vec![
            first,
            message("later", "reader", Role::Assistant, 2, vec![]),
        ],
    );
    assert_eq!(count(&archive, "operator", sessions), 1);
    assert_eq!(count(&archive, "operator", annotations), 0);
    archive
        .writer
        .archive()
        .user(operator_request(
            "revise-reader-note".into(),
            UserWrite::ReviseAnnotations {
                bases: vec![polylogue_model::RevisionReference::exact(
                    note.clone(),
                    note,
                )],
                patch: AnnotationPatch {
                    value: Change::Set(Json::String("new authored content".into())),
                    ..Default::default()
                },
            },
        ))
        .unwrap();
    assert_eq!(count(&archive, "operator", annotations), 1);
}

#[test]
fn runs_lists_the_native_begin_completion_and_reopened_record() {
    use crate::fixture::writer::{WriterArchive, message};
    use polylogue_model::Role;
    use polylogue_model::{
        Author, RunId, Timestamp,
        authorship::{Authorship, Channel},
        query_objects::{
            ExecutionBinding, ExecutionPresentation, QueryRun, QueryRunOutcome, ReplayBinding,
            RunBinding, RunOutcome,
        },
    };
    use std::collections::BTreeMap;
    let mut archive = WriterArchive::new();
    let opts = all_columns_options();
    archive.write(
        "run-before",
        vec![message(
            "before",
            "run-before",
            Role::User,
            opts.clock.now.as_millisecond() - 3600000,
            vec![],
        )],
    );
    let plan = polylogue_query::plan::prepare(
        "sessions where since:1d",
        &polylogue_query::Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    let object = plan.query_object().unwrap();
    archive
        .writer
        .archive()
        .query_object(object.clone(), None)
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let position = pool.reader().unwrap().change_position().unwrap();
    drop(pool);
    let expected = QueryRun {
        id: RunId::parse("018f0000-0000-7000-8000-000000000011").unwrap(),
        question: Some(object.id),
        at: opts.clock.now.into(),
        read_at: position.clone(),
        machine: "synthetic-machine".into(),
        authorship: Authorship {
            author: Author::Operator,
            via: Channel::Mcp,
        },
        binding: RunBinding::Original {
            execution: Box::new(ExecutionBinding {
                query_options: crate::query::execute::options_snapshot(&opts, false, None).unwrap(),
                presentation: ExecutionPresentation::Query,
                accepted_request: Some(
                    serde_json::json!({"id":"accepted-run","operation":"rows","arguments":{"selection":"sessions where since:1d"}}),
                ),
                replay: ReplayBinding {
                    bound_ast: object.query.clone(),
                    source_outputs: object.source_outputs.clone(),
                    definitions: vec![],
                    context: query_language::cursor::CursorContext {
                        question: object.id,
                        clock: opts.clock.wire(),
                        resolved_times: vec![],
                        random_seed: 7,
                        read_positions: BTreeMap::from([
                            ("generation".into(), position.generation.clone()),
                            ("index".into(), position.index.to_string()),
                            ("archive".into(), position.archive.to_string()),
                        ]),
                        versions: BTreeMap::from([("compiler".into(), "shared-query-v1".into())]),
                    },
                    scope_fingerprint: None,
                    original_text: "sessions where since:1d".into(),
                },
            }),
        },
    };
    let read = |archive: &WriterArchive, text: &str| {
        let program = crate::query::compile(text, &opts).unwrap();
        let pool = crate::ReadPool::open(&archive.archive).unwrap();
        let reader = pool.reader().unwrap();
        let mut rows = reader.query_rows(&program).unwrap();
        for row in &mut rows {
            present(&program, row).unwrap();
        }
        rows
    };
    archive
        .writer
        .archive()
        .begin_query_run(expected.clone(), None)
        .unwrap();
    let begun = read(&archive, "runs");
    assert_eq!(begun.len(), 1);
    assert_eq!(begun[0]["id"], format!("run:{}", expected.id));
    assert_eq!(begun[0]["query"], format!("query:{}", object.id));
    assert_eq!(begun[0]["author"], "operator");
    assert_eq!(begun[0]["via"], "mcp");
    assert_eq!(begun[0]["read_at"], position.to_string());
    assert_eq!(
        serde_json::from_value::<Timestamp>(begun[0]["at"].clone()).unwrap(),
        expected.at
    );
    assert!(begun[0]["count"].is_null());
    assert!(begun[0]["outcome"].is_null());
    let completed = QueryRunOutcome {
        run: expected.id.clone(),
        binding_hash: expected.binding_hash().unwrap(),
        run_hash: expected.record_hash().unwrap(),
        at: expected.at,
        outcome: RunOutcome::Ok,
        count: Some(7),
        exact: true,
        next: None,
    };
    drop(
        archive
            .writer
            .archive()
            .finish_query_run(completed.clone())
            .unwrap(),
    );
    assert!(archive.writer.flush_rows().unwrap().errors.is_empty());
    let finished = read(&archive, "runs where outcome:ok");
    assert_eq!(finished.len(), 1);
    assert_eq!(finished[0]["count"], 7);
    assert_eq!(finished[0]["exact"], true);
    assert_eq!(finished[0]["outcome"], serde_json::json!({"state":"ok"}));
    assert!(read(&archive, "runs where outcome:error").is_empty());
    archive
        .writer
        .archive()
        .begin_query_run(expected.clone(), None)
        .unwrap();
    drop(
        archive
            .writer
            .archive()
            .finish_query_run(completed)
            .unwrap(),
    );
    assert!(archive.writer.flush_rows().unwrap().errors.is_empty());
    assert_eq!(
        read(&archive, "runs"),
        finished,
        "retry creates no duplicate row"
    );
    archive.restart();
    assert_eq!(
        read(&archive, "runs"),
        finished,
        "native reopen retains the same projected row"
    );
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    assert_eq!(
        pool.reader().unwrap().query_run(&expected.id).unwrap(),
        Some(expected.clone())
    );
    drop(pool);
    let later_clock = query_language::time::Clock::new(
        "2026-10-31T12:00:00Z".parse().unwrap(),
        opts.clock.day.clone(),
    );
    archive.write(
        "run-after",
        vec![message(
            "after",
            "run-after",
            Role::User,
            later_clock.now.as_millisecond() - 3600000,
            vec![],
        )],
    );
    let mut current = opts.clone();
    current.clock = later_clock;
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let selected = reader
        .prepare_selection(&format!("run:{}", expected.id), &current)
        .unwrap();
    assert_eq!(selected.question_id(), object.id);
    let mut rerun = reader.query_rows(selected.program()).unwrap();
    for row in &mut rerun {
        present(selected.program(), row).unwrap();
    }
    assert_eq!(
        rerun
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["claude-code:run-after"]
    );
    assert_eq!(
        reader.query_run(&expected.id).unwrap(),
        Some(expected.clone()),
        "running the question again does not rewrite the original run"
    );
    assert_eq!(
        reader
            .prepare_query_selection("run:018f0000-0000-7000-8000-000000000099", &current)
            .err()
            .expect("unknown run must fail")
            .code(),
        polylogue_model::Code::NotFound
    );
    drop(selected);
    drop(reader);
    drop(pool);
    let mut object_run = expected.clone();
    object_run.id = RunId::parse("018f0000-0000-7000-8000-000000000012").unwrap();
    object_run.question = None;
    object_run.binding = RunBinding::OriginalObject {
        execution: Box::new(polylogue_model::query_objects::ObjectExecutionBinding {
            reference: Reference::Request("synthetic-original-read".into()),
            content_hash: polylogue_model::BlobHash::of(b"synthetic original object"),
            scope_fingerprint: None,
            versions: Default::default(),
            presentation: ExecutionPresentation::Read {
                options: Default::default(),
                delivery: polylogue_model::read::ReadDelivery::Stream,
                format: polylogue_model::read::Format::Jsonl,
                page_budget: None,
                direct_source: None,
            },
            accepted_request: None,
        }),
    };
    archive
        .writer
        .archive()
        .begin_query_run(object_run.clone(), None)
        .unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let no_question = reader
        .prepare_query_selection(&format!("run:{}", object_run.id), &current)
        .err()
        .expect("an original-object read has no question");
    assert_eq!(no_question.code(), polylogue_model::Code::BadRequest);
    assert!(no_question.to_string().contains("has no query question"));
    assert_eq!(reader.query_run(&object_run.id).unwrap(), Some(object_run));
}

#[test]
fn original_kept_bodies_retain_their_recorded_units_and_reference_columns() {
    let original: Value = serde_json::from_str(include_str!(
        "../../../../../query/src/plan/tests/fixtures/recorded-unit-objects.json"
    ))
    .unwrap();
    let d = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(d.path()).unwrap();
    for record in original["kept"].as_array().unwrap() {
        let id = record["id"].as_str().unwrap();
        let query = record["query"].as_str().unwrap();
        let bytes = record["rows_json"].as_str().unwrap();
        let body: Value = serde_json::from_str(bytes).unwrap();
        fixture.kept_result(id, query, bytes).unwrap();
        let pool = fixture.reads().unwrap();
        let reader = pool.reader().unwrap();
        // The query surface exposes decoded rows, not the immutable stored body.
        let stored: String = reader
            .conn()
            .unwrap()
            .query_row(
                "SELECT rows_json FROM source_saved_rows WHERE id=?1",
                [id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            stored, bytes,
            "original kept bytes and ID survive admission"
        );
        use sha2::{Digest, Sha256};
        assert_eq!(
            format!("{:x}", Sha256::digest(stored.as_bytes())),
            record["rows_sha256"].as_str().unwrap()
        );
        let prepared = reader
            .prepare_query_selection(&format!("result:{id}"), &all_columns_options())
            .unwrap();
        let program = prepared.program();
        let mut values = reader.query_rows(program).unwrap();
        for row in &mut values {
            present(program, row).unwrap();
        }
        assert_eq!(values, body["rows"].as_array().unwrap().clone(), "{id}");
        let schema = prepared.plan().output_schema().unwrap();
        assert_eq!(schema.unit.as_deref(), Some("annotations"));
        assert!(
            matches!(
                &program.row_kind,
                crate::query::compile::RowKind::UnitRows {
                    unit: Unit::Annotations,
                    ..
                }
            ),
            "recorded annotations keep native authored identity"
        );
        let reference = program
            .columns
            .iter()
            .find(|c| c.name == "reference")
            .unwrap();
        assert_eq!(
            reference.reference,
            Some(polylogue_query::ColumnReference::Annotation)
        );
        assert_eq!(body["columns"][1]["reference"], "annotation");
        assert_eq!(values[0]["reference"], "annotation:f4-note-1@f4-note-1");
    }
}

/// The Store crate has no testkit dependency (testkit itself depends on Store).
/// This demo population exercises the Reader on current live states, local and
/// other repositories, open authored work, and a persisted saved question.
#[test]
fn every_shipped_query_runs_on_demo_archive() {
    use polylogue_model::{query_objects::HereScope, saved_queries::SHIPPED_QUERIES};
    use std::collections::{BTreeMap, BTreeSet};
    let d = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(d.path()).unwrap();
    let mut opts = all_columns_options();
    opts.page_size = Some(100);
    opts.here = Some(HereScope {
        paths: vec!["/workspace/morrow".into()],
        repo_roots: vec!["demo-root".into()],
    });
    let now = opts.clock.now.as_millisecond();
    for (id, cwd, live, waiting) in [
        ("test:running", "/workspace/morrow", true, None),
        (
            "test:waiting",
            "/workspace/morrow",
            true,
            Some("permission"),
        ),
        ("test:unfinished", "/workspace/morrow/src", false, None),
        ("test:elsewhere", "/workspace/other", false, None),
    ] {
        let pk = fixture
            .session(SessionFacts {
                id,
                kind: "main",
                cwd: Some(cwd),
                started: Some(now - 1000),
                last: Some(now),
                ..Default::default()
            })
            .unwrap();
        fixture
            .message(
                pk,
                MessageFacts {
                    id: &format!("{id}-message"),
                    role: "user",
                    written_by: "human",
                    at: Some(now),
                    text: Some("A synthetic open task"),
                    ..Default::default()
                },
            )
            .unwrap();
        if live {
            fixture
                .live(
                    pk,
                    "hook_user_prompt_submit",
                    now,
                    waiting,
                    Some("demo-root"),
                )
                .unwrap();
        }
    }
    fixture
        .note(
            "question",
            None,
            Some("What remains?"),
            "session:test:unfinished",
        )
        .unwrap();
    let saved = polylogue_query::plan::prepare(
        "sessions where id:test:unfinished",
        &polylogue_query::Bindings {
            origins: vec!["test".into()],
            ..Default::default()
        },
        |_| Ok(()),
    )
    .unwrap();
    fixture.saved_query("demo-saved", &saved).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut counts = BTreeMap::new();
    for shipped in SHIPPED_QUERIES {
        let selection = format!("query:{}", shipped.name);
        let prepared = reader
            .prepare_query_selection(&selection, &opts)
            .unwrap_or_else(|e| panic!("{}: {e}", shipped.name));
        assert!(
            prepared
                .plan()
                .definitions()
                .iter()
                .any(|d| d.name == shipped.name),
            "{} resolves through the current saved-name reader",
            shipped.name
        );
        let mut values = reader
            .query_rows(prepared.program())
            .unwrap_or_else(|e| panic!("{}: {e}", shipped.name));
        for row in &mut values {
            present(prepared.program(), row).unwrap();
        }
        if shipped.name == "unfinished" {
            assert!(values.iter().any(|row| row["id"] == "test:unfinished"));
            assert!(
                values.iter().all(|row| row["id"] != "test:elsewhere"),
                "here excludes the other repository"
            );
        }
        counts.insert(shipped.name, values.len());
    }
    assert_eq!(
        counts.keys().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([
            "running",
            "waiting",
            "recent",
            "unfinished",
            "notices",
            "review",
            "questions",
            "triage",
            "saved",
            "open",
        ])
    );
    for nonempty in [
        "running",
        "waiting",
        "recent",
        "unfinished",
        "questions",
        "triage",
        "saved",
        "open",
    ] {
        assert!(
            counts[nonempty] > 0,
            "{nonempty} must execute against matching facts"
        );
    }
    // This fixture has neither signal deliveries nor non-operator annotations.
    assert_eq!(counts["notices"], 0);
    assert_eq!(counts["review"], 0);
}

#[test]
fn resolved_result_name_is_disclosed_with_its_immutable_id() {
    let d = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(d.path()).unwrap();
    let id = "018f0000-0000-7000-8000-000000000021";
    fixture.kept_result(id,"sessions",r#"{"unit":"sessions","columns":[{"name":"id","kind":"text"}],"rows":[{"id":"test:fixed"}]}"#).unwrap();
    let pool = fixture.reads().unwrap();
    let reader = pool.reader().unwrap();
    let opts = all_columns_options();
    let mut bindings = reader
        .selection_definitions(&format!("result:{id}"), &opts.origins)
        .unwrap();
    // The name resolver owns this mapping; lowering must carry its decision
    // into the existing reply field without putting the alias in question identity.
    bindings
        .kept_names
        .insert("latest-report".into(), id.into());
    let named =
        polylogue_query::plan::prepare("result:latest-report", &bindings, |_| Ok(())).unwrap();
    let direct =
        polylogue_query::plan::prepare(&format!("result:{id}"), &bindings, |_| Ok(())).unwrap();
    assert_eq!(named.question_id(), direct.question_id());
    let program = reader.lower_plan(&named, &opts).unwrap();
    assert!(program.definitions.contains(&polylogue_query::Definition {
        name: "result:latest-report".into(),
        revision: format!("result:{id}"),
    }));
    let mut rows = reader.query_rows(&program).unwrap();
    for row in &mut rows {
        present(&program, row).unwrap();
    }
    assert_eq!(rows, [serde_json::json!({"id":"test:fixed"})]);
}

#[test]
fn running_keeps_a_silent_session_whose_root_is_alive() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let pk = session(&mut f, "test:silent");
    f.live_facts(
        pk,
        &polylogue_model::live::LiveFacts {
            root: Some(polylogue_model::live::ProcessIdentity {
                boot: "boot".into(),
                pid: 1,
                start: 1,
            }),
            last_at: Some(1_000),
            ..Default::default()
        },
    )
    .unwrap();
    // A live root does not age out: three days of silence is still running.
    let now = 1_000 + 3 * 86_400_000;
    let running = f
        .reads()
        .unwrap()
        .reader()
        .unwrap()
        .running_candidates(now)
        .unwrap()
        .at(now);
    let ids: Vec<&str> = running.iter().map(|r| r.session.as_str()).collect();
    assert_eq!(ids, ["test:silent"]);
}

#[test]
fn since_seen_returns_a_session_whose_live_or_derived_counter_passed_the_mark() {
    use crate::fixture::{
        authored::operator_request,
        writer::{WriterArchive, message},
    };
    use polylogue_model::{Role, user::SeenMark};
    let mut archive = WriterArchive::new();
    archive.write("live", vec![message("one", "live", Role::User, 1, vec![])]);
    archive.write(
        "derived",
        vec![message("one", "derived", Role::User, 1, vec![])],
    );
    let ids = |archive: &WriterArchive| {
        let mut opts = all_columns_options();
        opts.origins = vec!["claude-code".into()];
        opts.seen_marks_of = Some("operator".into());
        let program =
            crate::query::compile("sessions where since:seen | fields id", &opts).unwrap();
        let mut ids: Vec<String> = crate::ReadPool::open(&archive.archive)
            .unwrap()
            .reader()
            .unwrap()
            .query_rows(&program)
            .unwrap()
            .iter_mut()
            .map(|row| {
                present(&program, row).unwrap();
                row["id"].as_str().unwrap().to_owned()
            })
            .collect();
        ids.sort();
        ids
    };
    let at = crate::ReadPool::open(&archive.archive)
        .unwrap()
        .reader()
        .unwrap()
        .change_position()
        .unwrap();
    archive
        .writer
        .archive()
        .user(operator_request(
            "mark-both".into(),
            UserWrite::MarkSeen {
                marks: ["live", "derived"]
                    .map(|name| SeenMark {
                        target: format!("session:claude-code:{name}"),
                        item: None,
                        at: at.clone(),
                    })
                    .to_vec(),
            },
        ))
        .unwrap();
    assert!(ids(&archive).is_empty());
    // A live-state change or a derive pass after the mark makes the session
    // changed again (docs/live-sessions.md: live-state changes are session
    // changes on the change feed).
    let index = crate::archive::open_write(&archive.archive.index_db(), 16).unwrap();
    for (column, name) in [("live_changed", "live"), ("summary_changed", "derived")] {
        index
            .execute(
                &format!("UPDATE session SET {column} = ?1 WHERE id = ?2"),
                rusqlite::params![at.index + 5, format!("claude-code:{name}")],
            )
            .unwrap();
    }
    drop(index);
    assert_eq!(
        ids(&archive),
        vec![
            "claude-code:derived".to_owned(),
            "claude-code:live".to_owned()
        ]
    );
}

/// A session's tree is its own responses and those of every session below it
/// through `spawn` links; a fork is not part of the tree, nor is what the
/// fork spawned (docs/measures.md, session cost).
#[test]
fn tree_cost_follows_spawn_links_only() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    for (id, cost) in [
        ("parent", 1_000_000_000),
        ("spawned", 2_000_000_000),
        ("grandchild", 4_000_000_000),
        ("fork", 8_000_000_000),
        ("fork-spawned", 16_000_000_000),
    ] {
        let pk = session(&mut f, &format!("test:{id}"));
        f.message(
            pk,
            MessageFacts {
                id: &format!("test:{id}-0"),
                role: "assistant",
                written_by: "model",
                text: Some("priced"),
                cost_nusd: Some(cost),
                ..Default::default()
            },
        )
        .unwrap();
    }
    f.link("test:parent", "test:spawned", "spawn", None)
        .unwrap();
    f.link("test:spawned", "test:grandchild", "spawn", None)
        .unwrap();
    f.link("test:parent", "test:fork", "fork", None).unwrap();
    f.link("test:fork", "test:fork-spawned", "spawn", None)
        .unwrap();
    let costs = |fixture: &Fixture, id: &str| {
        let row = rows(
            fixture,
            &format!("sessions where id:test:{id}"),
            &["tree_cost", "subagent_cost"],
        )
        .remove(0);
        (row["tree_cost"].clone(), row["subagent_cost"].clone())
    };
    // Costs are shown in USD.
    let usd = |whole: i64| serde_json::json!(whole);
    assert_eq!(costs(&f, "parent"), (usd(7), usd(6)));
    assert_eq!(costs(&f, "fork"), (usd(24), usd(16)));
    assert_eq!(costs(&f, "grandchild"), (usd(4), usd(0)));
    f.link("test:grandchild", "test:missing", "spawn", None)
        .unwrap();
    assert_eq!(costs(&f, "grandchild"), (usd(4), serde_json::Value::Null));
}
