use super::*;
use crate::{ExciseRequest, ExciseScope};

fn events(w: &ClaudeHome) -> Vec<serde_json::Value> {
    w.reads().reader().unwrap().query(
        "SELECT e.id,e.kind,e.source_start,e.source_end,pl_block(e.data) AS data,i.id AS input
         FROM current_event e JOIN input_ref i ON i.pk=e.input_pk ORDER BY e.source_start,e.kind,e.id", &[]
    ).unwrap()
}

#[test]
fn native_input_event_import_rebuild_and_input_excision_preserve_exact_scope() {
    for chunk in [1, 7, 64] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("synthetic.atof.jsonl");
        let raw = include_bytes!("../../../../../../examples/sources/hermes/atof/input.jsonl");
        fs::write(&path, raw).unwrap();
        let first = {
            let (mut e, _lock) = engine(&w, chunk);
            let imported = e.import(std::slice::from_ref(&path)).unwrap();
            assert!(
                imported.pass.errors.is_empty(),
                "{:?}",
                imported.pass.errors
            );
            assert_eq!(w.count("SELECT COUNT(*) AS n FROM session"), 0);
            assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 0);
            let rows = events(&w);
            assert_eq!(rows.len(), 4);
            for row in &rows {
                let range = row["source_start"].as_u64().unwrap() as usize
                    ..row["source_end"].as_u64().unwrap() as usize;
                let source: serde_json::Value = serde_json::from_slice(&raw[range]).unwrap();
                let data: serde_json::Value =
                    serde_json::from_str(row["data"].as_str().unwrap()).unwrap();
                if row["kind"] != "hermes_import_diagnostic" {
                    assert_eq!(data, source);
                }
            }
            rows
        };
        fs::remove_file(&path).unwrap();
        polylogue_testkit::delete_index(&w.root);
        {
            let (mut e, _lock) = engine(&w, chunk);
            assert!(e.converge().unwrap().errors.is_empty());
            assert_eq!(events(&w), first);
            let removed = e
                .excise(ExciseRequest {
                    id: "remove-observer-input".into(),
                    scope: ExciseScope::Input(first[0]["input"].as_str().unwrap().into()),
                    reason: None,
                    dry_run: false,
                })
                .unwrap();
            assert!(removed.sessions.is_empty());
            assert!(events(&w).is_empty());
            assert_eq!(
                w.count("SELECT COUNT(*) AS n FROM event"),
                0,
                "no orphan payload"
            );
            assert_eq!(w.count("SELECT COUNT(*) AS n FROM event_source"), 0);
        }
        polylogue_testkit::delete_index(&w.root);
        let (mut e, _lock) = engine(&w, chunk);
        assert!(e.converge().unwrap().errors.is_empty());
        assert!(events(&w).is_empty());
    }
}

#[test]
fn malformed_observer_suffix_never_publishes_a_valid_prefix() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("synthetic.atof.jsonl");
    fs::write(&path,b"{\"atof_version\":\"0.1\",\"kind\":\"mark\",\"uuid\":\"synthetic\",\"name\":\"neutral\"}\n{").unwrap();
    let (mut e, _lock) = engine(&w, 7);
    let _ = e.import(&[path]);
    assert!(events(&w).is_empty());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM session"), 0);
}

#[test]
fn input_event_text_excision_replays_only_the_overwritten_native_observation() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("synthetic.atof.jsonl");
    let secret = "synthetic-erased-observation";
    let raw = format!(
        "{}\n",
        json!({"atof_version":"0.1","kind":"mark","uuid":"synthetic","name":"neutral","timestamp":"2026-01-02T03:04:05Z","data":{"text":secret,"approved":false}})
    );
    fs::write(&path, &raw).unwrap();
    {
        let (mut e, _lock) = engine(&w, 7);
        assert!(
            e.import(std::slice::from_ref(&path))
                .unwrap()
                .pass
                .errors
                .is_empty()
        );
        assert!(serde_json::to_string(&events(&w)).unwrap().contains(secret));
        let removed = e
            .excise(ExciseRequest {
                id: "erase-observation-text".into(),
                scope: ExciseScope::Text(secret.into()),
                reason: None,
                dry_run: false,
            })
            .unwrap();
        assert_eq!(removed.inputs_reparsed.len(), 1);
        let rows = events(&w);
        assert_eq!(rows.len(), 1);
        assert!(!serde_json::to_string(&rows).unwrap().contains(secret));
        let data: serde_json::Value =
            serde_json::from_str(rows[0]["data"].as_str().unwrap()).unwrap();
        assert_eq!(data["data"]["approved"], false);
    }
    fs::remove_file(&path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 7);
    assert!(e.converge().unwrap().errors.is_empty());
    assert_eq!(events(&w).len(), 1);
    assert!(!serde_json::to_string(&events(&w)).unwrap().contains(secret));
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM session"), 0);
}
