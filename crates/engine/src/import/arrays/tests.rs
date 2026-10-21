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
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 7);
    }
    fs::remove_file(&path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 16);
    assert!(e.converge().unwrap().errors.is_empty());
    assert_eq!(texts(&w, "chatgpt:parent"), ["First", "Second"]);
    assert_eq!(texts(&w, "chatgpt:child"), ["Third"]);
    assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 7);
}

#[test]
fn malformed_array_suffix_publishes_one_refusal_and_no_prefix_conversations() {
    for suffix in [",]", "] false", ""] {
        let w = ClaudeHome::new();
        let path = w.dir.path().join("conversations.json");
        let bytes = format!(
            "[{}{}",
            conversation("never-published", 1, &[("u1", "prefix")]),
            suffix
        );
        fs::write(&path, bytes).unwrap();
        let (mut e, _lock) = engine(&w, 16);
        let report = e.import(&[path]).unwrap();
        assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 0);
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 1);
        assert_eq!(w.count("SELECT SUM(undecodable) AS n FROM parse_state"), 1);
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
            "SELECT source_start,source_end,source_order FROM message_source ORDER BY source_order",
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
fn interrupted_array_delivery_replays_retained_bytes_and_resets_accounting() {
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
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 2);
    }
    fs::remove_file(&path).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let retry = e.converge().unwrap();
    assert!(retry.errors.is_empty(), "{:?}", retry.errors);
    assert_eq!(texts(&w, "chatgpt:old"), ["old"]);
    assert_eq!(texts(&w, "chatgpt:new"), ["new"]);
    assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 4);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
}

#[test]
fn array_delivery_waits_for_the_previous_conversations_commit() {
    struct Probe(std::sync::mpsc::Sender<()>);
    impl ArrayParser for Probe {
        fn order_key(&self, _: &serde_json::value::RawValue) -> Option<f64> {
            None
        }
        fn item(
            &mut self,
            _: &polylogue_model::Json,
            _: &mut polylogue_sources::document::Sink<'_>,
            _: &mut dyn Owners,
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
            let captured = Captured::read(&e.blobs, &mut fs::File::open(&path).unwrap()).unwrap();
            let origin = polylogue_sources::origin::by_name("chatgpt").unwrap();
            let mut report = ImportReport::default();
            let mut plan = e
                .import_plan(&path, origin, captured, None, &mut report)
                .unwrap()
                .unwrap();
            ready_tx.send(()).unwrap();
            go_rx.recv().unwrap();
            e.execute_array(&mut plan, &mut Probe(item_tx), &mut Pass::default())
                .unwrap();
        });
        ready_rx.recv().unwrap();
        let db = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        go_tx.send(()).unwrap();
        item_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert!(matches!(
            item_rx.recv_timeout(std::time::Duration::from_millis(100)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
        db.execute_batch("COMMIT").unwrap();
        item_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
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
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM current_message WHERE id='chatgpt:a' AND on_active_path=1"),1);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM current_message WHERE id='chatgpt:b' AND on_active_path=0"),1);
        let reads = w.reads();
        let r = reads.reader().unwrap();
        let orders = r.query("SELECT m.id,es.source_order FROM event_source es JOIN event e ON e.id=es.event_id JOIN message m ON m.pk=e.after_message_pk WHERE e.kind='active_leaf' ORDER BY m.id", &[]).unwrap();
        assert_eq!(orders.len(), 2);
        assert_eq!(orders[0]["id"], "chatgpt:a");
        assert_eq!(orders[0]["source_order"], 3);
        assert_eq!(orders[1]["source_order"], 2);
    }
    fs::remove_file(path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 8);
    assert!(e.converge().unwrap().errors.is_empty());
    assert_eq!(texts(&w, "chatgpt:same"), ["A", "B"]);
    assert_eq!(
        w.count(
            "SELECT COUNT(*) AS n FROM current_message WHERE id='chatgpt:a' AND on_active_path=1"
        ),
        1
    );
    assert_eq!(
        w.count(
            "SELECT COUNT(*) AS n FROM current_message WHERE id='chatgpt:b' AND on_active_path=0"
        ),
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
        let captured = Captured::read(&e.blobs, fs::File::open(path).unwrap()).unwrap();
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
fn retained_ai_studio_chunks_survive_source_deletion_and_index_rebuild() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("Electrons.json");
    // Settings follow chunks on the wire. Parsing cannot assume header order.
    let bytes = serde_json::to_vec(&json!({
        "chunkedPrompt":{"chunks":[
            {"role":"user","text":"Explain electrons","createTime":"2026-05-17T12:27:24Z",
             "branchParent":{"promptId":"Parent"},"driveDocument":{"id":"file-1"}},
            {"role":"model","parts":[{"text":"Considering charge.","thought":true,"thoughtSignature":"sig"}],"tokenCount":20},
            {"role":"model","id":"answer","text":"A lepton.","tokenCount":3,"finishReason":"STOP",
             "inlineImage":{"mimeType":"image/png","data":"aGVsbG8="}}
        ],"pendingInputs":[{"role":"user","text":"unsent"}]},
        "runSettings":{"model":"models/gemini-example","temperature":0.5},
        "systemInstruction":{"text":"Answer briefly."},"citations":[{"title":"reference"}]
    })).unwrap();
    fs::write(&path, &bytes).unwrap();
    {
        let (mut e, _lock) = engine(&w, 8);
        let captured = Captured::read(&e.blobs, fs::File::open(&path).unwrap()).unwrap();
        let origin = polylogue_sources::origin::by_name("ai-studio").unwrap();
        // Stop after durable acquisition. The first parse must use retained bytes.
        e.import_plan(&path, origin, captured, None, &mut ImportReport::default())
            .unwrap()
            .unwrap();
    }
    fs::remove_file(&path).unwrap();
    let mut previous = None;
    for rebuild in [false, true] {
        if rebuild {
            polylogue_testkit::delete_index(&w.root);
        }
        let (mut e, _lock) = engine(&w, 8);
        let pass = e.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        let r = e.reads.reader().unwrap();
        let sid = polylogue_model::SessionId::parse("ai-studio:Electrons").unwrap();
        let messages = r.messages(&sid, false).unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].text(), "Answer briefly.");
        assert_eq!(messages[1].id.to_string(), "ai-studio:Electrons#0");
        assert_eq!(messages[3].id.to_string(), "ai-studio:Electrons#answer");
        assert_eq!(messages[3].parent.as_ref(), Some(&messages[2].id));
        assert_eq!(messages[3].model.as_deref(), Some("gemini-example"));
        assert_eq!(messages[3].usage.output, Some(3));
        assert_eq!(messages[3].stop_reason.as_deref(), Some("STOP"));
        assert!(
            matches!(&messages[2].blocks[0],polylogue_model::Block::Thinking{signature:Some(s),..} if s=="sig")
        );
        assert!(
            matches!(&messages[3].blocks[1],polylogue_model::Block::Image(a) if a.blob.is_some())
        );
        assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 4);
        assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 1);
        let source_rows = r
            .query(
                "SELECT source_start,source_end FROM message_source WHERE source_start>0",
                &[],
            )
            .unwrap();
        assert_eq!(source_rows.len(), 3);
        for row in source_rows {
            let start = row["source_start"].as_u64().unwrap() as usize;
            let end = row["source_end"].as_u64().unwrap() as usize;
            let chunk: serde_json::Value = serde_json::from_slice(&bytes[start..end]).unwrap();
            assert!(chunk.get("role").is_some());
        }
        assert_eq!(
            r.session(&sid)
                .unwrap()
                .unwrap()
                .started
                .unwrap()
                .to_string(),
            "2026-05-17T12:27:24Z"
        );
        let settings = r
            .events(&sid)
            .unwrap()
            .into_iter()
            .find(|e| e.kind == "run_settings")
            .unwrap();
        assert_eq!(
            settings.data,
            json!({"model":"models/gemini-example","temperature":0.5})
        );
        let shown = serde_json::to_value(&messages).unwrap();
        if let Some(prior) = previous {
            assert_eq!(shown, prior);
        }
        previous = Some(shown);
    }
}

#[test]
fn watched_ai_studio_prompt_revisions_are_retained_before_chunk_parsing() {
    let w = ClaudeHome::new();
    let dir = w.config(8).spool_dir("ai-studio");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("drive-id.json");
    let prompt = |text: &str| {
        serde_json::to_vec(&json!({"chunkedPrompt":{"chunks":[{"role":"user","text":text}]},"runSettings":{"model":"example"}})).unwrap()
    };
    let (mut e, _lock) = engine(&w, 8);
    for text in ["first revision", "edited second revision"] {
        fs::write(&path, prompt(text)).unwrap();
        let pass = e.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        assert_eq!(texts(&w, "ai-studio:drive-id"), [text]);
        assert!(pass.bytes_stored > 0);
    }
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 2);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.extent"), 2);
    assert_eq!(e.converge().unwrap().bytes_read, 0);
    drop(e);
    drop(_lock);
    fs::remove_file(&path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 8);
    assert!(e.converge().unwrap().errors.is_empty());
    assert_eq!(texts(&w, "ai-studio:drive-id"), ["edited second revision"]);
}

#[test]
fn an_appended_ai_studio_suffix_reuses_the_retained_prefix_without_duplicate_bytes() {
    let w = ClaudeHome::new();
    let dir = w.config(8).spool_dir("ai-studio");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("unfinished.json");
    let full = br#"{"chunkedPrompt":{"chunks":[{"role":"user","text":"now complete"}]}}"#;
    fs::write(&path, &full[..full.len() - 1]).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(texts(&w, "ai-studio:unfinished").is_empty());
    use std::io::Write;
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"}")
        .unwrap();
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.bytes_stored, 1);
    assert_eq!(texts(&w, "ai-studio:unfinished"), ["now complete"]);
    assert_eq!(
        w.count("SELECT SUM(length) AS n FROM a.extent"),
        full.len() as i64
    );
    assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 2);
}

#[test]
fn a_live_append_during_bulk_array_processing_is_served_by_the_next_pass() {
    struct Paused {
        parser: Box<dyn ArrayParser>,
        entered: std::sync::mpsc::Sender<()>,
        resume: std::sync::mpsc::Receiver<()>,
    }
    impl ArrayParser for Paused {
        fn order_key(&self, raw: &serde_json::value::RawValue) -> Option<f64> {
            self.parser.order_key(raw)
        }
        fn item(
            &mut self,
            value: &polylogue_model::Json,
            sink: &mut polylogue_sources::document::Sink<'_>,
            owners: &mut dyn Owners,
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
            e.converge().unwrap();
            let captured = Captured::read(&e.blobs, fs::File::open(&bulk).unwrap()).unwrap();
            let origin = polylogue_sources::origin::by_name("chatgpt").unwrap();
            let mut plan = e
                .import_plan(&bulk, origin, captured, None, &mut ImportReport::default())
                .unwrap()
                .unwrap();
            let Some(polylogue_sources::Parse::Array(parser)) = (origin.open)(&bulk) else {
                panic!("array");
            };
            e.execute_array(
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
            let pass = e.converge().unwrap();
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
