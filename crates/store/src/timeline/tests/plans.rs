use super::*;

fn plan(
    reader: &Reader<'_>,
    selection: &Selection<'_>,
    bounds: Option<(i64, i64)>,
    after: Option<&Position>,
) -> Vec<String> {
    row_queries(reader, selection, bounds, after, 3, false)
        .unwrap()
        .into_iter()
        .map(|(sql, params)| {
            serde_json::to_string(
                &reader
                    .query(&format!("EXPLAIN QUERY PLAN {sql}"), &params)
                    .unwrap(),
            )
            .unwrap()
        })
        .collect()
}

#[test]
fn dated_pages_seek_time_session_and_model_indexes_without_sorting_the_selection() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(temp.path()).unwrap();
    let mut sessions = Vec::new();
    for id in ["codex:early", "codex:middle", "codex:late"] {
        let session = f
            .session(SessionFacts {
                id,
                kind: "main",
                ..Default::default()
            })
            .unwrap();
        add(
            &mut f,
            session,
            &format!("{id}:response"),
            Some(at("2024-03-31T12:00:00Z")),
        );
        sessions.push(session);
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let day = LogicalDay::named("UTC", 240).unwrap();
    let bounds = Some(day.day("2024-03-31".parse().unwrap()));
    let plans = plan(&r, &all(None, false, false), bounds, None);
    assert_eq!(plans.len(), 3);
    for (plan, index) in plans.iter().zip(["message_at", "event_at"]) {
        assert!(plan.contains(index), "{plan}");
        assert!(!plan.contains("TEMP B-TREE"), "{plan}");
    }
    let after = Position {
        undated: false,
        at: Some(at("2024-03-31T12:00:00Z")),
        kind: "message".into(),
        id: "codex:early:response".into(),
    };
    for (selected, session_led) in [
        (None, false),
        (Some(sessions.clone()), false),
        (Some(vec![sessions[2]]), true),
        (Some(vec![sessions[2], sessions[1], sessions[2]]), true),
    ] {
        let usage = crate::measure::Selection {
            sessions: selected,
            ..Default::default()
        };
        for model in [None, Some("synthetic")] {
            let selection = Selection {
                responses: Some(&usage),
                model,
                ..all(None, false, false)
            };
            let index = if session_led && model.is_some() {
                "message_session_model_at"
            } else if session_led {
                "message_session_at"
            } else if model.is_some() {
                "message_model_at"
            } else {
                "message_at"
            };
            for cursor in [None, Some(&after)] {
                let plans = plan(&r, &selection, bounds, cursor);
                assert!(!plans.is_empty());
                for plan in plans {
                    assert!(plan.contains(index), "wanted {index}: {plan}");
                    assert!(!plan.contains("TEMP B-TREE"), "{plan}");
                }
            }
        }
    }
    for (selected, index) in [
        (None, "message_model_at"),
        (Some(vec![sessions[2]]), "message_session_model_at"),
    ] {
        let usage = crate::measure::Selection {
            sessions: selected,
            ..Default::default()
        };
        let unknown = Selection {
            responses: Some(&usage),
            unknown_model: true,
            ..all(None, false, false)
        };
        let plans = plan(&r, &unknown, None, None);
        assert_eq!(plans.len(), 1);
        for plan in plans {
            assert!(plan.contains(index), "{plan}");
            assert!(!plan.contains("TEMP B-TREE"), "{plan}");
        }
    }
}

#[test]
fn selected_session_pages_merge_equal_times_and_continue_through_undated_rows() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(temp.path()).unwrap();
    let mut sessions = Vec::new();
    for id in ["codex:a", "codex:b", "claude-code:outside"] {
        let session = f
            .session(SessionFacts {
                id,
                kind: "main",
                ..Default::default()
            })
            .unwrap();
        for (suffix, time) in [
            ("early", Some(at("2024-03-30T12:00:00Z"))),
            ("tie", Some(at("2024-03-31T12:00:00Z"))),
            ("undated", None),
            ("unrepresentable", Some(i64::MAX)),
        ] {
            add(&mut f, session, &format!("{id}:{suffix}"), time);
        }
        sessions.push(session);
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let day = LogicalDay::named("UTC", 240).unwrap();
    for usage in [
        crate::measure::Selection {
            sessions: Some(vec![sessions[1], sessions[0], sessions[1]]),
            ..Default::default()
        },
        crate::measure::Selection {
            origins: Some(vec!["codex".into()]),
            ..Default::default()
        },
        crate::measure::Selection {
            sessions: Some(sessions.clone()),
            origins: Some(vec!["codex".into()]),
            ..Default::default()
        },
    ] {
        let selection = Selection {
            responses: Some(&usage),
            model: Some("synthetic"),
            ..all(None, false, false)
        };
        let mut cursor = None;
        let mut ids = Vec::new();
        loop {
            let result = page(&r, &day, &selection, cursor.as_ref(), 1).unwrap();
            ids.extend(
                result["rows"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|row| row["id"].as_str().unwrap().to_owned()),
            );
            cursor = result
                .get("after")
                .filter(|p| !p.is_null())
                .map(|p| serde_json::from_value(p.clone()).unwrap());
            if cursor.is_none() {
                break;
            }
        }
        assert_eq!(
            ids,
            [
                "codex:a:early",
                "codex:b:early",
                "codex:a:tie",
                "codex:b:tie",
                "codex:a:undated",
                "codex:a:unrepresentable",
                "codex:b:undated",
                "codex:b:unrepresentable"
            ]
        );
    }
    for sessions in [vec![], vec![i64::MAX]] {
        let usage = crate::measure::Selection {
            sessions: Some(sessions),
            ..Default::default()
        };
        let selection = Selection {
            responses: Some(&usage),
            ..all(None, false, false)
        };
        assert!(
            page(&r, &day, &selection, None, 1).unwrap()["rows"]
                .as_array()
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn combined_response_pages_work_while_the_index_is_deferred() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(temp.path()).unwrap();
    let mut sessions = Vec::new();
    for id in ["codex:selected", "codex:outside"] {
        let session = f
            .session(SessionFacts {
                id,
                ..Default::default()
            })
            .unwrap();
        add(
            &mut f,
            session,
            &format!("{id}:response"),
            Some(at("2024-03-31T12:00:00Z")),
        );
        sessions.push(session);
    }
    crate::archive::open_write(&f.archive.index_db(), crate::archive::cache::SETUP)
        .unwrap()
        .execute_batch("DROP INDEX message_session_model_at;")
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let usage = crate::measure::Selection {
        sessions: Some(vec![sessions[0]]),
        ..Default::default()
    };
    let selection = Selection {
        responses: Some(&usage),
        model: Some("synthetic"),
        ..all(None, false, false)
    };
    let day = LogicalDay::named("UTC", 240).unwrap();
    let result = page(&r, &day, &selection, None, 50).unwrap();
    assert_eq!(result["rows"].as_array().unwrap().len(), 1);
    assert_eq!(result["rows"][0]["id"], "codex:selected:response");
}
