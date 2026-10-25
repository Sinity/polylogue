use super::*;

#[test]
fn session_query_uses_the_derived_title_without_replacing_recorded_values() {
    let dir = tempfile::tempdir().unwrap();
    let mut fx = Fixture::new(&dir.path().join("archive")).unwrap();
    for (id, title) in [
        ("claude-code:derived", None),
        ("claude-code:recorded", Some("Provider title")),
        ("claude-code:empty", Some("")),
    ] {
        let session = fx
            .session(SessionFacts {
                id,
                title,
                ..SessionFacts::default()
            })
            .unwrap();
        fx.message(
            session,
            msg(
                &format!("{id}-prompt"),
                "user",
                "human",
                "First human question\nextra detail",
                at(1, 1),
            ),
        )
        .unwrap();
    }
    fx.session(SessionFacts {
        id: "claude-code:unknown",
        ..SessionFacts::default()
    })
    .unwrap();
    fx.derive().unwrap();
    let w = TestArchive { _dir: dir, fx };
    let rows = q(&w, "sessions")["rows"].as_array().unwrap().clone();
    let row = |id: &str| rows.iter().find(|r| r["id"] == id).unwrap();
    assert_eq!(row("claude-code:derived")["title"], "First human question");
    assert_eq!(row("claude-code:recorded")["title"], "Provider title");
    assert_eq!(row("claude-code:empty")["title"], "");
    assert!(row("claude-code:unknown")["title"].is_null());
    let filtered = q(&w, r#"sessions where title:"First human question""#);
    assert_eq!(filtered["count"], 1);
    assert_eq!(filtered["rows"][0]["id"], "claude-code:derived");
    let fields = vec!["id".into(), "title".into()];
    let selected = run_q(
        &w,
        Ask {
            fields: &fields,
            ..ask("sessions")
        },
    )
    .unwrap();
    for shown in selected["rows"].as_array().unwrap() {
        assert_eq!(shown["title"], row(shown["id"].as_str().unwrap())["title"]);
        assert_eq!(shown.as_object().unwrap().len(), 2);
    }
    let pool = w.fx.reads().unwrap();
    let reader = pool.reader().unwrap();
    let recorded = reader
        .query(
            "SELECT title, derived_title FROM session WHERE id='claude-code:derived'",
            &[],
        )
        .unwrap();
    for shown in &rows {
        let id = polylogue_model::SessionId::parse(shown["id"].as_str().unwrap()).unwrap();
        let title = reader.display_title(&id).unwrap();
        assert_eq!(serde_json::to_value(&title).unwrap(), shown["title"]);
        let source = reader.session(&id).unwrap().unwrap();
        let read = polylogue_render::session(
            &source,
            title.as_deref(),
            vec![],
            polylogue_render::View::Full,
            polylogue_render::Format::Json,
            &[],
        )
        .unwrap();
        assert_eq!(read["display_title"], shown["title"]);
        assert_eq!(read["session"], serde_json::to_value(&source).unwrap());
    }
    assert!(recorded[0]["title"].is_null());
    assert_eq!(recorded[0]["derived_title"], "First human question");
}
