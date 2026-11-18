use super::*;
use crate::query::ast::Unit;
use polylogue_model::{
    Json, Reference,
    assertion::{AssertionKind, LinkRole, NewAssertion, NewAssertionLink},
    user::UserWrite,
};

fn rows(f: &Fixture, text: &str, fields: &[&str]) -> Vec<Value> {
    let mut opts = options();
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
fn statement(kind: AssertionKind, value: &str, links: &[(LinkRole, String)]) -> NewAssertion {
    NewAssertion {
        kind,
        key: None,
        value: Some(Json::String(value.into())),
        valid_until: None,
        links: links
            .iter()
            .map(|(role, r)| NewAssertionLink {
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
            assertions: vec![
                statement(
                    AssertionKind::Note,
                    "first",
                    &[(LinkRole::About, "session:test:work".into())],
                ),
                statement(
                    AssertionKind::Note,
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
    assert_eq!(
        rows(
            &f,
            &format!("annotations where request:{request}x"),
            &["id"]
        ),
        []
    );
}

#[test]
fn links_to_matches_any_revision() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let target = f
        .note("note", None, Some("original"), "session:test:work")
        .unwrap();
    let linked = f
        .user(UserWrite::Assert {
            assertions: vec![statement(
                AssertionKind::Note,
                "support",
                &[
                    (LinkRole::About, "session:test:work".into()),
                    (LinkRole::Supports, format!("assertion:{target}@{target}")),
                ],
            )],
        })
        .unwrap();
    assert_eq!(
        rows(
            &f,
            &format!("annotations where links_to:assertion:{target}"),
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
fn answered_and_open_follow_links() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    session(&mut f, "test:work");
    let question = f
        .user(UserWrite::Assert {
            assertions: vec![statement(
                AssertionKind::Question,
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
        assertions: vec![statement(
            AssertionKind::Note,
            "because",
            &[
                (LinkRole::About, format!("assertion:{question}")),
                (LinkRole::Answers, format!("assertion:{question}")),
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
            assertions: vec![statement(
                AssertionKind::Problem,
                "blocked",
                &[
                    (LinkRole::About, "session:test:work".into()),
                    (LinkRole::Blocks, format!("assertion:{question}")),
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
    f.user(UserWrite::WithdrawAssertions {
        targets: vec![Reference::parse(&format!("assertion:{blocker}@{blocker}")).unwrap()],
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

#[test]
fn triage_score_sums_named_facts() {
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let pk = session(&mut f, "test:work");
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
        "UPDATE link SET branch_point='test:a-1' WHERE child_id='test:b'",
        [],
    )
    .unwrap();
    conn.execute(
        "UPDATE link SET branch_point='test:a-0' WHERE child_id='test:c'",
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
    use polylogue_model::{BlobHash, NewVariant, NewVariantPart, VariantKind, VariantRelation};
    let d = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(d.path()).unwrap();
    let pk = session(&mut f, "test:work");
    f.message(
        pk,
        MessageFacts {
            id: "test:original",
            role: "assistant",
            written_by: "model",
            text: Some("original text"),
            ..Default::default()
        },
    )
    .unwrap();
    f.user(UserWrite::Variant {
        variant: NewVariant {
            base: None,
            source: "message:test:original".into(),
            source_hash: BlobHash([1; 32]),
            kind: VariantKind::Translation,
            from_lang: Some("en".into()),
            to_lang: Some("pl".into()),
            method: None,
            parts: vec![NewVariantPart {
                of: "block:test:original#0".into(),
                of_hash: BlobHash([2; 32]),
                relation: VariantRelation::Translates,
                body: "unikalny".into(),
            }],
        },
    })
    .unwrap();
    assert!(rows(&f, "messages where unikalny", &["id"]).is_empty());
    assert_eq!(
        rows(&f, "messages where alternate:true unikalny", &["id"])[0]["id"],
        "test:original"
    );
    assert_eq!(
        rows(&f, "messages where lang:pl unikalny", &["id"])[0]["id"],
        "test:original"
    );
    assert!(rows(&f, "messages where lang:de unikalny", &["id"]).is_empty());
    assert_eq!(
        rows(&f, "alternates where lang:pl unikalny", &["to_lang"])[0]["to_lang"],
        "pl"
    );
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
    for old in ["assertions", "links", "changes"] {
        assert!(Unit::from_name(old).is_none());
    }
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
    session(&mut f, "test:child");
    f.link("test:parent", "test:child", "spawn", None).unwrap();
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
            .query_object(&object.id.to_string(), None)
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
        let program = crate::query::lower(&restored_plan, &options()).unwrap();
        let mut got = reader.query_rows(&program).unwrap();
        for row in &mut got {
            present(&program, row).unwrap();
        }
        let named = reader
            .prepare_query_plan(&format!("query:recorded-{n}"), &options())
            .unwrap();
        assert!(
            named
                .definitions()
                .iter()
                .any(|d| d.question_id == object.id)
        );
        let named_program = reader.lower_plan(&named, &options()).unwrap();
        let mut named_rows = reader.query_rows(&named_program).unwrap();
        for row in &mut named_rows {
            present(&named_program, row).unwrap();
        }
        assert_eq!(
            named_rows, got,
            "current named reader executes the original object"
        );
        match object.query.source {
            query_language::ast::Source::Unit { ref unit, .. } if unit == "links" => {
                assert_eq!(got.len(), 1);
                assert_eq!(got[0]["parent"], "test:parent");
                assert_eq!(got[0]["child"], "test:child");
            }
            query_language::ast::Source::Unit { ref unit, .. } if unit == "assertions" => {
                assert_eq!(got.len(), 1);
                assert_eq!(got[0]["value"], "retained interpretation");
            }
            query_language::ast::Source::Unit { ref unit, .. }
                if unit == "changes" && object.query.transforms.is_empty() =>
            {
                assert!(got.iter().any(|r| r["assertion_kind"] == "note"));
            }
            query_language::ast::Source::Unit { ref unit, .. } if unit == "changes" => {
                assert_eq!(got.len(), 1);
                assert_eq!(got[0]["assertion_kind"], "note");
                assert_eq!(got[0]["count"], 1);
            }
            _ => panic!("fixture is not an original renamed unit"),
        }
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
    let mut opts = options();
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
        let error = crate::query::compile(text, &options())
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
            assertions: vec![statement(
                AssertionKind::Goal,
                "finish",
                &[(LinkRole::About, "session:test:work".into())],
            )],
        })
        .unwrap();
    let reply = f
        .user(UserWrite::Assert {
            assertions: vec![statement(
                AssertionKind::Note,
                "done",
                &[
                    (LinkRole::About, format!("assertion:{goal}")),
                    (LinkRole::Answers, format!("assertion:{goal}")),
                ],
            )],
        })
        .unwrap();
    assert_eq!(
        rows(&f, "annotations where kind:goal open:true", &["id"])[0]["id"],
        goal
    );
    f.user(UserWrite::Assert {
        assertions: vec![NewAssertion {
            kind: AssertionKind::Judgment,
            key: Some("accepted".into()),
            value: None,
            valid_until: None,
            links: vec![NewAssertionLink {
                role: LinkRole::About,
                reference: Reference::parse(&format!("assertion:{reply}@{reply}")).unwrap(),
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
}

#[test]
fn since_seen_tracks_the_computed_author_and_new_source_or_authored_work() {
    use crate::fixture::{
        authored::operator_request,
        writer::{WriterArchive, message},
    };
    use polylogue_model::{
        Role,
        assertion::{AssertionPatch, Change},
        user::SeenMark,
    };
    let mut archive = WriterArchive::new();
    let first = message("first", "reader", Role::User, 1, vec![]);
    archive.write("reader", vec![first.clone()]);
    let note = archive
        .writer
        .user(operator_request(
            "note-for-reader".into(),
            UserWrite::Assert {
                assertions: vec![statement(
                    AssertionKind::Note,
                    "original",
                    &[(LinkRole::About, "session:claude-code:reader".into())],
                )],
            },
        ))
        .unwrap()
        .id;
    let count = |archive: &WriterArchive, author: &str, query: &str| {
        let mut opts = options();
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
                        target: format!("assertion:{note}"),
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
    let feed_program = crate::query::compile("feed", &options()).unwrap();
    let pool = crate::ReadPool::open(&archive.archive).unwrap();
    let feed_reader = pool.reader().unwrap();
    let feed = feed_reader.query_rows(&feed_program).unwrap();
    assert!(feed.iter().all(|row| row["assertion_kind"] != "seen"));
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
        .user(operator_request(
            "revise-reader-note".into(),
            UserWrite::ReviseAssertions {
                bases: vec![polylogue_model::RevisionReference::exact(
                    note.clone(),
                    note,
                )],
                patch: AssertionPatch {
                    value: Change::Set(Json::String("new authored content".into())),
                    ..Default::default()
                },
            },
        ))
        .unwrap();
    assert_eq!(count(&archive, "operator", annotations), 1);
}
