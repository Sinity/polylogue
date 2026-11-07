use super::*;
use polylogue_testkit::ClaudeHome;

#[test]
fn unknown_import_refuses_and_explain_stores_nothing() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("unrelated.jsonl");
    std::fs::write(&path, b"{\"type\":\"arbitrary\"}\n").unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert_eq!(report.refused.len(), 1);
    assert!(!report.refused[0].structures_tried.is_empty());
    assert!(report.imported.is_empty());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
    std::fs::write(
        &path,
        br#"[{"conversation_id":"example","current_node":null,"mapping":{}}]"#,
    )
    .unwrap();
    let report = e
        .import_with_options(
            std::slice::from_ref(&path),
            &ImportOptions {
                format: None,
                explain: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(report.members.len(), 1);
    assert_eq!(report.members[0].origin.as_deref(), Some("chatgpt"));
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.input"), 0);
}

#[test]
fn reimport_changes_nothing_in_archive_rows() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    std::fs::write(
        &path,
        br#"[{"conversation_id":"example","current_node":null,"mapping":{}}]"#,
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 64);
    assert_eq!(
        e.import(std::slice::from_ref(&path))
            .unwrap()
            .imported
            .len(),
        1
    );
    let before = ["input", "revision", "extent", "observation"]
        .map(|table| w.count(&format!("SELECT COUNT(*) AS n FROM a.{table}")));
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert_eq!(report.already_stored, std::slice::from_ref(&path));
    let after = ["input", "revision", "extent", "observation"]
        .map(|table| w.count(&format!("SELECT COUNT(*) AS n FROM a.{table}")));
    assert_eq!(before, after);
}

#[test]
fn declared_native_export_signatures_reach_their_scanners_without_whole_json_shape() {
    for (name, raw, origin, complete) in [
        (
            "conversations.json",
            "[{\"id\":\"synthetic\",\"mapping\":{},\"current_node\":null},",
            "chatgpt",
            false,
        ),
        (
            "OpenRouter Chat synthetic.json",
            "{\"version\":1,\"messages\":{}} false",
            "openrouter",
            false,
        ),
        (
            "traces.jsonl",
            "{\"resourceSpans\":[]}\n{\"resourceSpans\":[]}\n",
            "otel-genai",
            true,
        ),
    ] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join(name);
        std::fs::write(&path, raw).unwrap();
        let (mut e, _lock) = engine(&w, 64);
        let report = e.import(std::slice::from_ref(&path)).unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        assert_eq!(report.imported, [path]);
        assert_eq!(report.members.len(), 1);
        assert_eq!(report.members[0].origin.as_deref(), Some(origin));
        assert_eq!(report.members[0].class, "parsed");
        assert_eq!(
            w.count("SELECT sum(length) AS n FROM a.extent"),
            raw.len() as i64
        );
        assert_eq!(
            w.count("SELECT parsed_upto AS n FROM parse_state"),
            if complete { raw.len() as i64 } else { 0 }
        );
        if !complete {
            assert_eq!(w.count("SELECT sum(unreadable) AS n FROM parse_state"), 1);
            assert_eq!(
                w.count("SELECT count(*) AS n FROM parse_state WHERE checkpoint IS NOT NULL"),
                0
            );
        }
    }
}

#[test]
fn native_export_names_without_a_provider_signature_stay_refused() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    std::fs::write(&path, br#"[{"arbitrary":true}]"#).unwrap();
    let (mut e, _lock) = engine(&w, 64);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert_eq!(report.refused.len(), 1, "{report:?}");
    assert!(report.imported.is_empty(), "{report:?}");
    assert_eq!(report.members[0].class, "refused");
    assert_eq!(w.count("SELECT count(*) AS n FROM a.input"), 0);
}
