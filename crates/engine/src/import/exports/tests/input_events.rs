use super::*;

fn events(w: &ClaudeHome) -> Vec<serde_json::Value> {
    w.reads()
        .reader()
        .unwrap()
        .query(
            "SELECT e.id,e.kind,e.source_start,e.source_end,pl_block(e.data) AS data,i.id AS input
         FROM event e JOIN input_ref i ON i.pk=e.input_pk ORDER BY e.source_start,e.kind,e.id",
            &[],
        )
        .unwrap()
}

#[test]
fn native_input_event_import_and_rebuild_preserve_exact_scope() {
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
            assert!(e.run_pass().unwrap().errors.is_empty());
            assert_eq!(events(&w), first);
        }
        polylogue_testkit::delete_index(&w.root);
        let (mut e, _lock) = engine(&w, chunk);
        assert!(e.run_pass().unwrap().errors.is_empty());
        assert_eq!(events(&w), first);
    }
}

#[test]
fn malformed_observer_suffix_never_commits_a_valid_prefix() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("synthetic.atof.jsonl");
    fs::write(&path,b"{\"atof_version\":\"0.1\",\"kind\":\"mark\",\"uuid\":\"synthetic\",\"name\":\"neutral\"}\n{").unwrap();
    let (mut e, _lock) = engine(&w, 7);
    let _ = e.import(&[path]);
    assert!(events(&w).is_empty());
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM session"), 0);
}
