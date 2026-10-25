use super::*;
use crate::import::tests::engine;
use polylogue_testkit::ClaudeHome;
use serde_json::json;
use std::fs;

fn conversation(id: &str, time: u64, messages: &[(&str, &str)]) -> serde_json::Value {
    let mapping: serde_json::Map<String,serde_json::Value> = messages.iter().enumerate().map(|(i,(id,text))| {
        (id.to_string(),json!({"id":id,"parent":if i==0 {None}else{Some(messages[i-1].0)},"children":[],
            "message":{"id":id,"author":{"role":"user"},"content":{"content_type":"text","parts":[text]},"create_time":i+1}}))
    }).collect();
    json!({"id":id,"create_time":time,"mapping":mapping})
}

fn texts(w: &ClaudeHome, id: &str) -> Vec<String> {
    w.reads()
        .reader()
        .unwrap()
        .messages(&polylogue_model::SessionId::parse(id).unwrap(), false)
        .unwrap()
        .iter()
        .map(|m| m.text())
        .collect()
}

#[test]
fn array_order_and_parent_ownership_survive_newest_first_storage_and_rebuild() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    let parent = conversation("parent", 1, &[("z-first", "First"), ("a-second", "Second")]);
    let child = conversation(
        "child",
        2,
        &[
            ("z-first", "First"),
            ("a-second", "Second"),
            ("third", "Third"),
        ],
    );
    let bytes = serde_json::to_vec_pretty(&vec![child, parent]).unwrap();
    fs::write(&path, &bytes).unwrap();
    {
        let (mut e, _lock) = engine(&w, 16);
        let report = e.import(std::slice::from_ref(&path)).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        assert_eq!(texts(&w, "chatgpt:parent"), ["First", "Second"]);
        assert_eq!(texts(&w, "chatgpt:child"), ["Third"]);
        assert_eq!(
            w.count("SELECT COUNT(*) AS n FROM link WHERE kind='continuation' AND how='fork'"),
            1
        );
        let reads = w.reads();
        let r = reads.reader().unwrap();
        for row in r
            .query("SELECT source_start,source_end FROM message_source", &[])
            .unwrap()
        {
            let range = row["source_start"].as_u64().unwrap() as usize
                ..row["source_end"].as_u64().unwrap() as usize;
            let raw: serde_json::Value = serde_json::from_slice(&bytes[range]).unwrap();
            assert!(raw.get("mapping").is_some());
        }
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
        assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 7);
    }
    fs::remove_file(&path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 16);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(texts(&w, "chatgpt:parent"), ["First", "Second"]);
    assert_eq!(texts(&w, "chatgpt:child"), ["Third"]);
    assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 7);
}

#[test]
fn malformed_array_suffix_commits_one_refusal_and_no_prefix_conversations() {
    for suffix in [",]", "] false", ""] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("conversations.json");
        let bytes = format!(
            "[{}{}",
            conversation("never-committed", 1, &[("u1", "prefix")]),
            suffix
        );
        fs::write(&path, bytes).unwrap();
        let (mut e, _lock) = engine(&w, 16);
        let report = e.import(&[path]).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 0);
        assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 1);
        assert_eq!(w.count("SELECT SUM(unreadable) AS n FROM parse_state"), 1);
    }
}

#[test]
fn claude_array_keeps_message_order_separate_from_one_conversation_span() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    let messages = json!([
        {"uuid":"z","sender":"human","text":"First","created_at":"2026-01-01T00:00:00Z"},
        {"uuid":"a","sender":"human","text":"Second","created_at":"2026-01-01T00:01:00Z"}]);
    let bytes = format!(" [ {{\"uuid\":\"c1\",\"chat_messages\":{messages}}} ] ");
    fs::write(&path, &bytes).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let report = e.import(&[path]).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(texts(&w, "claude-ai:c1"), ["First", "Second"]);
    let reads = w.reads();
    let r = reads.reader().unwrap();
    let locations = r
        .query(
            "SELECT source_start,source_end,seq AS source_order FROM message ORDER BY seq",
            &[],
        )
        .unwrap();
    assert_eq!(locations.len(), 2);
    assert_eq!(locations[0]["source_start"], locations[1]["source_start"]);
    assert_eq!(locations[0]["source_end"], locations[1]["source_end"]);
    assert_eq!(locations[0]["source_order"], 0);
    assert_eq!(locations[1]["source_order"], 1);
    let from = locations[0]["source_start"].as_u64().unwrap() as usize;
    let to = locations[0]["source_end"].as_u64().unwrap() as usize;
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&bytes[from..to]).unwrap()["uuid"],
        "c1"
    );
}

#[cfg(feature = "faults")]
#[test]
fn interrupted_array_delivery_replays_stored_bytes_and_resets_record_counts() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    let bytes = serde_json::to_vec(&vec![
        conversation("new", 2, &[("n", "new")]),
        conversation("old", 1, &[("o", "old")]),
    ])
    .unwrap();
    fs::write(&path, bytes).unwrap();
    {
        let (mut e, _lock) = engine(&w, 8);
        polylogue_store::fault::arm(
            "parse",
            &path.to_string_lossy(),
            polylogue_store::fault::Action::Fail,
            1,
            1,
        );
        let report = e.import(std::slice::from_ref(&path)).unwrap();
        assert_eq!(report.pass.errors.len(), 1);
        assert_eq!(w.count("SELECT parsed_upto AS n FROM parse_state"), 0);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 1);
        assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 2);
    }
    fs::remove_file(&path).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let retry = e.run_pass().unwrap();
    assert!(retry.errors.is_empty(), "{:?}", retry.errors);
    assert_eq!(texts(&w, "chatgpt:old"), ["old"]);
    assert_eq!(texts(&w, "chatgpt:new"), ["new"]);
    assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 4);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
}

#[test]
fn array_delivery_waits_for_the_previous_conversations_commit() {
    struct Probe(std::sync::mpsc::Sender<()>);
    impl ExportParser for Probe {
        fn order_key(&self, _: &serde_json::value::RawValue) -> Vec<u8> {
            Vec::new()
        }
        fn item(
            &mut self,
            _: &[u8],
            _: &mut polylogue_sources::document::Sink<'_>,
            _: &mut dyn ExportState,
        ) -> Result<(), String> {
            self.0.send(()).unwrap();
            Ok(())
        }
    }
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    fs::write(&path, "[{},{}]").unwrap();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel();
    let (go_tx, go_rx) = std::sync::mpsc::channel();
    let (item_tx, item_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let w = &w;
        let path = path.clone();
        scope.spawn(move || {
            let (mut e, _lock) = engine(w, 8);
            let captured =
                Captured::stream(&e.archive, &mut fs::File::open(&path).unwrap()).unwrap();
            let origin = polylogue_sources::origin::by_name("chatgpt").unwrap();
            let mut report = ImportReport::default();
            let mut plan = e
                .import_plan(&path, origin, captured, None, &mut report)
                .unwrap()
                .unwrap();
            ready_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            e.execute_export(&mut plan, &mut Probe(item_tx), &mut Pass::default())
                .unwrap();
        });
        ready_rx.recv().unwrap();
        let db = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        go_tx.send(()).unwrap();
        item_rx.recv().unwrap();
        assert!(matches!(
            item_rx.recv_timeout(std::time::Duration::from_millis(100)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
        db.execute_batch("COMMIT").unwrap();
        item_rx.recv().unwrap();
    });
}

#[test]
fn repeated_conversation_markers_follow_logical_order_and_repeated_facts_keep_the_latest_order() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    let marker = |time, leaf| {
        let mut value = conversation("same", time, &[("a", "A"), ("b", "B")]);
        value["mapping"]["b"]["parent"] = serde_json::Value::Null;
        value["current_node"] = json!(leaf);
        if time != 1 {
            // Later markers still name known IDs, but contain no usable message bodies.
            for node in value["mapping"].as_object_mut().unwrap().values_mut() {
                node["message"]["content"] = serde_json::Value::Null;
            }
        }
        value
    };
    // Physical order ends in B, while chronological order is A -> B -> A.
    fs::write(
        &path,
        serde_json::to_vec(&vec![marker(1, "a"), marker(3, "a"), marker(2, "b")]).unwrap(),
    )
    .unwrap();
    {
        let (mut e, _lock) = engine(&w, 8);
        assert!(
            e.import(std::slice::from_ref(&path))
                .unwrap()
                .pass
                .errors
                .is_empty()
        );
        assert_eq!(texts(&w, "chatgpt:same"), ["A", "B"]);
        assert_eq!(
            w.count("SELECT COUNT(*) AS n FROM message WHERE id='chatgpt:a' AND on_active_path=1"),
            1
        );
        assert_eq!(
            w.count("SELECT COUNT(*) AS n FROM message WHERE id='chatgpt:b' AND on_active_path=0"),
            1
        );
        let reads = w.reads();
        let r = reads.reader().unwrap();
        let orders = r.query("SELECT m.id,e.seq AS source_order FROM event e JOIN message m ON m.pk=e.after_message_pk WHERE e.kind='active_leaf' ORDER BY m.id", &[]).unwrap();
        assert_eq!(orders.len(), 2);
        assert_eq!(orders[0]["id"], "chatgpt:a");
        assert_eq!(orders[0]["source_order"], 3);
        assert_eq!(orders[1]["source_order"], 2);
    }
    fs::remove_file(path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 8);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(texts(&w, "chatgpt:same"), ["A", "B"]);
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM message WHERE id='chatgpt:a' AND on_active_path=1"),
        1
    );
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM message WHERE id='chatgpt:b' AND on_active_path=0"),
        1
    );
}

#[test]
fn pending_ordinary_live_input_commits_before_bulk_array_delivery() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    fs::write(
        &path,
        serde_json::to_vec(&vec![conversation("bulk", 1, &[("b", "bulk")])]).unwrap(),
    )
    .unwrap();
    let live_path = w.dir.path().join("live.jsonl");
    fs::write(
        &live_path,
        polylogue_testkit::claude_user("l", "live", "live"),
    )
    .unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let mut report = ImportReport::default();
    let mut plans = Vec::new();
    for (path, origin, priority) in [
        (&path, "chatgpt", Priority::Bulk),
        (&live_path, "claude-code", Priority::Live),
    ] {
        let captured = Captured::stream(&e.archive, fs::File::open(path).unwrap()).unwrap();
        let mut plan = e
            .import_plan(
                path,
                polylogue_sources::origin::by_name(origin).unwrap(),
                captured,
                None,
                &mut report,
            )
            .unwrap()
            .unwrap();
        plan.priority = priority;
        plans.push(plan);
    }
    let live_id = plans[1].input.id.clone();
    let db = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
    db.execute_batch(
        "CREATE TABLE delivery_order(seq INTEGER PRIMARY KEY,id TEXT);
        CREATE TRIGGER record_delivery AFTER INSERT ON parse_state BEGIN
          INSERT INTO delivery_order(id) SELECT id FROM input_ref WHERE pk=NEW.input_pk;
        END;",
    )
    .unwrap();
    let mut pass = Pass::default();
    e.execute(plans, &mut pass).unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let rows = e
        .reads
        .reader()
        .unwrap()
        .query("SELECT id FROM delivery_order ORDER BY seq", &[])
        .unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["id"], live_id);
}

#[test]
fn a_live_append_during_bulk_array_processing_is_served_by_the_next_pass() {
    struct Paused {
        parser: Box<dyn ExportParser>,
        entered: std::sync::mpsc::Sender<()>,
        resume: std::sync::mpsc::Receiver<()>,
    }
    impl ExportParser for Paused {
        fn order_key(&self, raw: &serde_json::value::RawValue) -> Vec<u8> {
            self.parser.order_key(raw)
        }
        fn item(
            &mut self,
            value: &[u8],
            sink: &mut polylogue_sources::document::Sink<'_>,
            owners: &mut dyn ExportState,
        ) -> Result<(), String> {
            if self.entered.send(()).is_ok() {
                self.resume.recv().unwrap();
            }
            self.parser.item(value, sink, owners)
        }
    }
    let w = ClaudeHome::new();
    w.append(
        "live.jsonl",
        &polylogue_testkit::claude_user("u1", "live", "before"),
    );
    let bulk = w.dir.path().join("conversations.json");
    fs::write(
        &bulk,
        serde_json::to_vec(&vec![conversation("bulk", 1, &[("b", "bulk")])]).unwrap(),
    )
    .unwrap();
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (resume_tx, resume_rx) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let w = &w;
        scope.spawn(move || {
            let (mut e, _lock) = engine(w, 8);
            e.run_pass().unwrap();
            let captured = Captured::stream(&e.archive, fs::File::open(&bulk).unwrap()).unwrap();
            let origin = polylogue_sources::origin::by_name("chatgpt").unwrap();
            let mut plan = e
                .import_plan(&bulk, origin, captured, None, &mut ImportReport::default())
                .unwrap()
                .unwrap();
            let Some(polylogue_sources::Parse::Export(parser)) = (origin.open)(&bulk) else {
                panic!("array");
            };
            e.execute_export(
                &mut plan,
                &mut Paused {
                    parser,
                    entered: entered_tx,
                    resume: resume_rx,
                },
                &mut Pass::default(),
            )
            .unwrap();
            assert_eq!(w.texts("live"), ["before"]);
            let pass = e.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
            assert_eq!(w.texts("live"), ["before", "during bulk"]);
        });
        entered_rx.recv().unwrap();
        w.append(
            "live.jsonl",
            &polylogue_testkit::claude_user("u2", "live", "during bulk"),
        );
        assert_eq!(w.texts("live"), ["before"]);
        resume_tx.send(()).unwrap();
    });
}

mod ai_studio;
mod gemini_cli;
mod openai_chat;

mod polylogue_export;

mod openrouter;

mod input_events;
