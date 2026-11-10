//! A conversation a newer complete export no longer holds is gone upstream
//! and keeps its rows (docs/archive.md).

use super::*;
use polylogue_testkit::ClaudeHome;
use serde_json::json;

/// A ChatGPT conversation with one prompt at `time` seconds.
fn conversation(id: &str, time: u64) -> Value {
    let node = format!("{id}-prompt");
    json!({"id": id, "create_time": time, "current_node": node, "mapping": {
        node.clone(): {"id": node, "parent": null, "children": [], "message": {
            "id": node, "author": {"role": "user"}, "create_time": time,
            "content": {"content_type": "text", "parts": [format!("prompt in {id}")]}}}}})
}

fn ids(home: &ClaudeHome, text: &str) -> Vec<String> {
    let pool = home.reads();
    let reader = pool.reader().unwrap();
    let fields = vec!["id".into()];
    let mut request = ask(text);
    request.fields = &fields;
    let answer = run(&reader, request).unwrap();
    let mut ids: Vec<String> = answer["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row["id"].as_str().unwrap().to_owned())
        .collect();
    ids.sort();
    ids
}

#[test]
fn upstream_gone_kept() {
    // The older export holds three conversations; the newer one, whose
    // latest message is later, no longer holds `deleted` or `recent`, but
    // `recent` has activity after it, so only `deleted` is gone.
    let older = json!([
        conversation("deleted", 1_000),
        conversation("kept", 1_000),
        conversation("recent", 5_000),
    ]);
    let newer = json!([conversation("kept", 2_000)]);
    for newer_first in [false, true] {
        let home = ClaudeHome::new();
        let mut paths = Vec::new();
        for (name, export) in [("older", &older), ("newer", &newer)] {
            let dir = home.dir.path().join(name);
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join("conversations.json");
            std::fs::write(&path, serde_json::to_vec(export).unwrap()).unwrap();
            paths.push(path);
        }
        if newer_first {
            paths.reverse();
        }
        let (mut engine, _lock) = home.engine(64 << 20);
        for path in &paths {
            let report = engine.import(std::slice::from_ref(path)).unwrap();
            assert!(report.refused.is_empty(), "{report:?}");
            assert!(report.pass.errors.is_empty(), "{report:?}");
        }
        drop(engine);
        let order = format!("newer first: {newer_first}");
        assert_eq!(
            ids(&home, "sessions where source:gone"),
            ["chatgpt:deleted"],
            "{order}"
        );
        assert_eq!(
            ids(&home, "sessions where source:present"),
            ["chatgpt:kept", "chatgpt:recent"],
            "{order}"
        );
        // A gone conversation keeps its rows.
        assert_eq!(
            ids(&home, "messages"),
            [
                "chatgpt:deleted-prompt",
                "chatgpt:kept-prompt",
                "chatgpt:recent-prompt"
            ],
            "{order}"
        );
    }
}
