//! A conversation a newer complete export no longer holds is gone upstream
//! and keeps its rows (docs/archive.md).

use super::*;
use polylogue_testkit::Home;
use serde_json::json;

/// A ChatGPT conversation with one prompt at `time` seconds.
fn conversation(id: &str, time: u64) -> Value {
    let node = format!("{id}-prompt");
    json!({"id": id, "create_time": time, "current_node": node, "mapping": {
        node.clone(): {"id": node, "parent": null, "children": [], "message": {
            "id": node, "author": {"role": "user"}, "create_time": time,
            "content": {"content_type": "text", "parts": [format!("prompt in {id}")]}}}}})
}

fn ids(home: &Home, text: &str) -> Vec<String> {
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
        let home = Home::new();
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
        polylogue_testkit::install_empty_index(&mut engine);
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

/// A prompt whose chunks are the user turns `texts`.
fn prompt(texts: &[&str]) -> Vec<u8> {
    let chunks: Vec<Value> = texts
        .iter()
        .map(|text| json!({"role": "user", "text": text}))
        .collect();
    serde_json::to_vec(&json!({"runSettings": {}, "chunkedPrompt": {"chunks": chunks}})).unwrap()
}

#[test]
fn a_prompt_deleted_in_drive_is_gone_and_keeps_its_revisions() {
    use polylogue_sources::ai_studio::drive::{Drive, fake::FakeDrive};
    let drive = FakeDrive::start();
    drive.create("1AbC", "Electrons", "2026-05-17T12:00:00Z");
    drive.save("1AbC", "r1", "2026-05-17T12:10:00Z", &prompt(&["first"]));
    drive.save(
        "1AbC",
        "r2",
        "2026-05-17T12:20:00Z",
        &prompt(&["first", "second"]),
    );
    drive.create("2Def", "Muons", "2026-05-17T13:00:00Z");
    drive.save("2Def", "m1", "2026-05-17T13:10:00Z", &prompt(&["muon"]));
    let home = Home::new();
    let (mut engine, _lock) = home.engine(64 << 20);
    let token_path = home.dir.path().join("drive-token.json");
    std::fs::write(&token_path, r#"{"client_id":"c","refresh_token":"r"}"#).unwrap();
    let collector = Drive {
        endpoints: drive.endpoints(),
        token_path,
        spool: engine.config.spool_dir("ai-studio"),
    };
    // Each sync publishes the next revision of each prompt; the archive
    // stores it before the collector moves on.
    let mut sync = || {
        let report = collector.sync().unwrap();
        let pass = engine.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{pass:?}");
        collector.advance(&|_, _, _| Ok(true)).unwrap();
        report
    };
    while sync().revisions_written > 0 {}
    assert_eq!(
        ids(&home, "sessions where source:gone"),
        Vec::<String>::new()
    );

    drive.delete("1AbC");
    let report = sync();
    assert_eq!((report.prompts, report.removed), (1, 1));
    assert_eq!(ids(&home, "sessions where source:gone"), ["ai-studio:1AbC"]);
    assert_eq!(
        ids(&home, "sessions where source:present"),
        ["ai-studio:2Def"]
    );
    // Its session still reads as its newest revision holds it.
    let texts: Vec<String> = home
        .reads()
        .reader()
        .unwrap()
        .messages(
            &polylogue_model::SessionId::parse("ai-studio:1AbC").unwrap(),
            false,
        )
        .unwrap()
        .iter()
        .map(|m| m.text())
        .collect();
    assert_eq!(texts, ["first", "second"]);
}
