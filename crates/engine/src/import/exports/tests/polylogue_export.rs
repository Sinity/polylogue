use super::*;
use polylogue_model::{
    Session, SessionId, SessionKind, TitleSource,
    export::{Document, SessionExport},
};

fn exported_session(name: &str) -> SessionExport {
    let sid = SessionId::parse(&format!("claude-code:{name}")).unwrap();
    let mut session = Session::new(sid.clone(), SessionKind::Main);
    session.offer_title(&format!("Recorded {name}"), TitleSource::Custom);
    let messages = (0..2)
        .map(|n| {
            serde_json::from_value(json!({
                "id":format!("claude-code:{name}-m{n}"),"session":sid,
                "parent":(n>0).then(|| format!("claude-code:{name}-m0")),
                "role":if n==0 {"user"} else {"assistant"},
                "blocks":[{"type":"text","text":format!("{name} text {n}")}],
                "usage":{"input":12,"output":3},"stop_reason":"end_turn",
                "production":{"effort":"medium"}
            }))
            .unwrap()
        })
        .collect();
    let events = vec![
        serde_json::from_value(json!({
            "session":sid,"kind":"compaction","after":format!("claude-code:{name}-m0"),
            "range":[10,20],"data":{"label":"stored event"}
        }))
        .unwrap(),
    ];
    SessionExport {
        session,
        messages,
        events,
        links: vec![],
    }
}

#[test]
fn exported_sessions_keep_ids_fields_links_and_exact_spans_after_source_deletion() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("export.json");
    let first = exported_session("first");
    let mut second = exported_session("second");
    second.links.push(
        serde_json::from_value(json!({
            "kind":"continuation","parent":first.session.id,"child":second.session.id,
            "at":"claude-code:first-m1","evidence":"explicit export link"
        }))
        .unwrap(),
    );
    let document = Document::new(vec![first, second]);
    // The version follows sessions, while recognition's format remains first.
    let bytes = format!(
        "{{\"format\":\"polylogue-export\",\"sessions\":{},\"version\":1}}",
        serde_json::to_string(&document.sessions).unwrap()
    );
    fs::write(&path, &bytes).unwrap();
    let mut before = None;
    for rebuilt in [false, true] {
        if rebuilt {
            fs::remove_file(&path).unwrap();
            polylogue_testkit::delete_index(&w.root);
        }
        let (mut engine, _lock) = engine(&w, 8);
        if rebuilt {
            let pass = engine.converge().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        } else {
            let report = engine.import(std::slice::from_ref(&path)).unwrap();
            assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        }
        let reader = engine.reads.reader().unwrap();
        let mut shown = Vec::new();
        for session in &document.sessions {
            let actual = reader.session_export(&session.session.id).unwrap().unwrap();
            assert_eq!(actual.messages, session.messages);
            assert_eq!(actual.session.title, session.session.title);
            assert_eq!(actual.events.len(), 1);
            assert_eq!(actual.events[0].data, session.events[0].data);
            assert_eq!(actual.events[0].after, session.events[0].after);
            let (start, end) = actual.events[0].range.unwrap();
            let source: SessionExport =
                serde_json::from_slice(&bytes.as_bytes()[start as usize..end as usize]).unwrap();
            assert_eq!(&source, session);
            shown.push(actual);
        }
        assert!(
            shown[1]
                .links
                .iter()
                .any(|l| l == &document.sessions[1].links[0])
        );
        let ranges = reader
            .query("SELECT source_start,source_end FROM message_source", &[])
            .unwrap();
        assert_eq!(ranges.len(), 4);
        for range in ranges {
            let start = range["source_start"].as_u64().unwrap() as usize;
            let end = range["source_end"].as_u64().unwrap() as usize;
            let _: SessionExport = serde_json::from_slice(&bytes.as_bytes()[start..end]).unwrap();
        }
        let view = serde_json::to_value(shown).unwrap();
        if let Some(expected) = before {
            assert_eq!(view, expected);
        }
        before = Some(view);
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 2);
    }
}

#[test]
fn invalid_export_session_or_suffix_does_not_commit_the_valid_prefix() {
    let session = serde_json::to_string(&exported_session("prefix")).unwrap();
    for ending in [
        ",{\"session\":false}],\"version\":1}".to_owned(),
        "],\"version\":1} false".to_owned(),
        "],\"version\":99,\"version\":1}".to_owned(),
    ] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("export.json");
        fs::write(
            &path,
            format!("{{\"format\":\"polylogue-export\",\"sessions\":[{session}{ending}"),
        )
        .unwrap();
        let (mut engine, _lock) = engine(&w, 8);
        let report = engine.import(&[path]).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 0);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM session"), 0);
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 1);
    }
}
