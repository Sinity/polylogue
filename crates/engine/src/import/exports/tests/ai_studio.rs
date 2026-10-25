use super::*;

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
        let captured = Captured::stream(
            &e.archive,
            &e.config.blob_roots,
            fs::File::open(&path).unwrap(),
        )
        .unwrap();
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
        let bytes = prompt(text);
        fs::write(&path, &bytes).unwrap();
        let pass = e.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        assert_eq!(texts(&w, "ai-studio:drive-id"), [text]);
        assert!(pass.bytes_stored > 0);
        let settings = e
            .reads
            .reader()
            .unwrap()
            .events(&polylogue_model::SessionId::parse("ai-studio:drive-id").unwrap())
            .unwrap();
        assert_eq!(settings[0].range, Some((0, bytes.len() as u64)));
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
    let path = dir.join("incomplete.json");
    let full = br#"{"chunkedPrompt":{"chunks":[{"role":"user","text":"now complete"}]}}"#;
    fs::write(&path, &full[..full.len() - 1]).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(texts(&w, "ai-studio:incomplete").is_empty());
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
    assert_eq!(texts(&w, "ai-studio:incomplete"), ["now complete"]);
    assert_eq!(
        w.count("SELECT SUM(length) AS n FROM a.extent"),
        full.len() as i64
    );
    assert_eq!(w.count("SELECT SUM(framed) AS n FROM parse_state"), 2);
}
