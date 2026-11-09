use super::*;

#[test]
fn explicit_money_projection_keeps_the_declared_dollar_unit() {
    let archive = morrow();
    let fields = vec!["id".into(), "cost".into()];
    let answer = run_q(
        &archive,
        Ask {
            fields: &fields,
            ..ask("sessions where id:claude-code:7f3a2c10-0001-4000-8000-000000000001")
        },
    )
    .unwrap();
    assert_eq!(answer["count"], 1);
    assert_eq!(answer["rows"][0]["cost"], 1.26);
    assert_eq!(answer["columns"][1]["kind"], "money");
}

#[test]
fn empty_exists_and_scoped_matches_use_the_real_child_projection() {
    let mut archive = morrow();
    archive
        .fx
        .session(SessionFacts {
            id: "codex:empty-exists",
            ..Default::default()
        })
        .unwrap();
    let session = "sessions where id:claude-code:7f3a2c10-0001-4000-8000-000000000001";
    for (target, present) in [
        (session, true),
        ("sessions where id:codex:empty-exists", false),
    ] {
        assert_eq!(
            q(
                &archive,
                &format!("{target} AND exists tool_calls() | count")
            )["rows"][0]["count"],
            u64::from(present)
        );
        assert_eq!(
            q(
                &archive,
                &format!("{target} AND NOT exists tool_calls() | count")
            )["rows"][0]["count"],
            u64::from(!present)
        );
    }
    assert_eq!(
        q(
            &archive,
            "sessions where id:codex:empty-exists AND exists tool_calls() | matches"
        )["count"],
        0
    );
    let matched = q(
        &archive,
        &format!("{session} AND exists tool_calls() | matches"),
    );
    assert_eq!(matched["count"], 6);
    assert!(
        matched["rows"]
            .as_array()
            .unwrap()
            .iter()
            .all(|row| row["step.session"] == "claude-code:7f3a2c10-0001-4000-8000-000000000001")
    );
    let scoped = q(
        &archive,
        &format!("{session} | messages where exists tool_calls(action:shell) | matches | count"),
    );
    assert_eq!(scoped["rows"][0]["count"], 6);
}

#[test]
fn grouped_aggregate_keeps_implicit_count_partial_evidence_and_nullable_keyset_order() {
    let directory = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&directory.path().join("archive")).unwrap();
    for (id, repo, cost) in [
        ("a", Some("east"), Some(1_000_000_000)),
        ("b", Some("east"), None),
        ("c", Some("west"), Some(5_000_000_000)),
        ("d", None, Some(2_000_000_000)),
    ] {
        let session = fx
            .session(SessionFacts {
                id: &format!("claude-code:{id}"),
                repo,
                ..Default::default()
            })
            .unwrap();
        fx.message(
            session,
            MessageFacts {
                cost_nusd: cost,
                ..msg(
                    &format!("{id}-priced"),
                    "assistant",
                    "model",
                    "synthetic",
                    at(1, 1),
                )
            },
        )
        .unwrap();
        if id == "a" {
            fx.message(
                session,
                msg(
                    "a-unpriced",
                    "assistant",
                    "model",
                    "synthetic unpriced",
                    at(1, 2),
                ),
            )
            .unwrap();
        }
    }
    let archive = TestArchive {
        _dir: directory,
        fx,
    };
    let implicit = q(
        &archive,
        "sessions | group by repo | where count > 1 | sort by count",
    );
    assert_eq!(implicit["rows"].as_array().unwrap().len(), 1);
    assert_eq!(implicit["rows"][0]["repo"], "east");
    assert_eq!(implicit["rows"][0]["count"], 2);
    for direction in ["asc", "desc"] {
        let text = format!("sessions | group by repo | agg sum(cost) | sort by repo {direction}");
        let whole = q(&archive, &text);
        let expected = whole["rows"].as_array().unwrap();
        assert_eq!(expected.len(), 3);
        assert!(
            expected.last().unwrap()["repo"].is_null(),
            "unknown group keys stay last in both directions"
        );
        for row in expected {
            assert_eq!(
                row.as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect::<std::collections::BTreeSet<_>>(),
                ["repo", "count", "sum_cost", "unknown_cost", "partial_cost"]
                    .map(str::to_owned)
                    .into_iter()
                    .collect()
            );
        }
        let east = expected.iter().find(|row| row["repo"] == "east").unwrap();
        assert_eq!(east["count"], 2);
        assert_eq!(east["sum_cost"], 1.0);
        assert_eq!(east["unknown_cost"], 1);
        assert_eq!(east["partial_cost"], 1);
        for page_size in 1..=3 {
            let mut rows = Vec::new();
            let mut cursor: Option<String> = None;
            loop {
                let page = run_q(
                    &archive,
                    Ask {
                        page_size: Some(page_size),
                        cursor: cursor.as_deref(),
                        ..ask(&text)
                    },
                )
                .unwrap();
                rows.extend(page["rows"].as_array().unwrap().iter().cloned());
                cursor = page["next"].as_str().map(str::to_owned);
                if cursor.is_none() {
                    break;
                }
            }
            assert_eq!(&rows, expected, "{direction} page size {page_size}");
        }
    }
}
