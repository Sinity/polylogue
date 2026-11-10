use super::*;

#[test]
fn stored_ai_studio_chunks_survive_source_deletion_and_index_rebuild() {
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
        let captured = Captured::stream(&e.archive, fs::File::open(&path).unwrap()).unwrap();
        let origin = polylogue_sources::origin::by_name("ai-studio").unwrap();
        // Stop after durable import. The first parse must use stored bytes.
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
        let pass = e.run_pass().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        let r = e.reads.reader().unwrap();
        // An imported prompt carries no Drive file ID; it is named by its
        // first chunk (BLAKE3 of the tag, a zero byte and the chunk's
        // canonical JSON), and an ID-less chunk by its content.
        let sid =
            polylogue_model::SessionId::parse("ai-studio:content-712969f6f44e0e19c4f48ce5a97fad28")
                .unwrap();
        let messages = r.messages(&sid, false).unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[0].text(), "Answer briefly.");
        assert_eq!(
            messages[1].id.to_string(),
            "ai-studio:content-712969f6f44e0e19c4f48ce5a97fad28~user-38fe0b0b74f66011-0"
        );
        assert_eq!(
            messages[3].id.to_string(),
            "ai-studio:40:content-712969f6f44e0e19c4f48ce5a97fad28#id:6:answer"
        );
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
        assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 4);
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
fn watched_ai_studio_prompt_revisions_are_stored_before_chunk_parsing() {
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
        let pass = e.run_pass().unwrap();
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
    assert_eq!(e.run_pass().unwrap().bytes_read, 0);
    drop(e);
    drop(_lock);
    fs::remove_file(&path).unwrap();
    polylogue_testkit::delete_index(&w.root);
    let (mut e, _lock) = engine(&w, 8);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(texts(&w, "ai-studio:drive-id"), ["edited second revision"]);
}

#[test]
fn an_appended_ai_studio_suffix_reuses_the_stored_prefix_without_duplicate_bytes() {
    let w = ClaudeHome::new();
    let dir = w.config(8).spool_dir("ai-studio");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("incomplete.json");
    let full = br#"{"chunkedPrompt":{"chunks":[{"role":"user","text":"now complete"}]}}"#;
    fs::write(&path, &full[..full.len() - 1]).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert!(texts(&w, "ai-studio:incomplete").is_empty());
    let (input, old_chain, _) = native_source(&e, &path);
    assert_eq!(
        e.inputs[&path].facts,
        Some(FileFacts::of(&fs::metadata(&path).unwrap()))
    );
    assert_eq!(e.inputs[&path].chain, Some(old_chain));
    assert_eq!(e.inputs[&path].parsed_upto, 0);
    use std::io::Write;
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"}")
        .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.bytes_stored, 1);
    assert!(pass.bytes_read >= full.len() as u64, "{pass:?}");
    let (_, chain, length) = native_source(&e, &path);
    assert_eq!(chain, old_chain);
    assert_eq!(retained(&e, &input, chain, length), full);
    assert_eq!(texts(&w, "ai-studio:incomplete"), ["now complete"]);
    assert_eq!(
        w.count("SELECT SUM(length) AS n FROM a.extent"),
        full.len() as i64
    );
    assert_eq!(w.count("SELECT SUM(records) AS n FROM parse_state"), 2);
}

fn native_source(engine: &Engine, path: &Path) -> (String, i64, u64) {
    let known = engine
        .reads
        .reader()
        .unwrap()
        .known_watched_input_at("test-machine", path.as_os_str().as_encoded_bytes())
        .unwrap()
        .unwrap();
    (known.input_id, known.chain.unwrap(), known.stored)
}

fn retained(engine: &Engine, input: &str, chain: i64, length: u64) -> Vec<u8> {
    engine
        .reads
        .reader()
        .unwrap()
        .stored_range(input, chain, 0, length, &engine.blobs)
        .unwrap()
}

#[test]
fn proven_export_append_reads_the_prefix_but_stores_only_the_delta() {
    use std::io::Write;
    let w = ClaudeHome::new();
    let dir = w.config(8).spool_dir("ai-studio");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("append-proof.json");
    let text = "retained prefix ".repeat(1200);
    let full =
        serde_json::to_vec(&json!({"chunkedPrompt":{"chunks":[{"role":"user","text":text}]}}))
            .unwrap();
    assert!(full.len() > 2 * read::TAIL);
    fs::write(&path, &full[..full.len() - 1]).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let first = e.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    let (input, old_chain, _) = native_source(&e, &path);
    fs::OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(b"}")
        .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_stored, 1);
    // One archived full scan plus the actual provider prefix and delta. The
    // cached tail alone is too short to satisfy this independent IO bound.
    assert!(pass.bytes_read >= 2 * full.len() as u64, "{pass:?}");
    let (_, chain, length) = native_source(&e, &path);
    assert_eq!(chain, old_chain);
    assert_eq!(retained(&e, &input, chain, length), full);
    assert_eq!(texts(&w, "ai-studio:append-proof"), [text]);
    assert_eq!(
        w.count("SELECT sum(length) AS n FROM a.extent"),
        full.len() as i64
    );
    assert_eq!(w.count("SELECT count(*) AS n FROM a.extent"), 2);
}

#[test]
fn early_rewrite_with_an_unchanged_tail_keeps_both_export_chains() {
    use std::io::Write;
    let w = ClaudeHome::new();
    let dir = w.config(8).spool_dir("ai-studio");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("prefix.json");
    let original = serde_json::to_vec(&json!({"chunkedPrompt":{"chunks":[
        {"role":"user","text":format!("old {}", "x".repeat(8192))}]}}))
    .unwrap();
    let mut current = original.clone();
    let at = current
        .windows(4)
        .position(|bytes| bytes == b"old ")
        .unwrap();
    current[at..at + 3].copy_from_slice(b"new");
    let incomplete = &original[..original.len() - 1];
    assert_eq!(
        &current[current.len() - 1 - read::TAIL..current.len() - 1],
        &incomplete[incomplete.len() - read::TAIL..]
    );
    fs::write(&path, incomplete).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let first = e.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    let (input, old_chain, old_length) = native_source(&e, &path);
    fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .write_all(&current)
        .unwrap();
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert_eq!(pass.bytes_stored, current.len() as u64);
    let (new_input, chain, length) = native_source(&e, &path);
    assert_eq!(input, new_input);
    assert_ne!(chain, old_chain);
    assert_eq!(retained(&e, &input, old_chain, old_length), incomplete);
    assert_eq!(retained(&e, &input, chain, length), current);
    assert_eq!(
        texts(&w, "ai-studio:prefix"),
        [format!("new {}", "x".repeat(8192))]
    );
    assert_eq!(
        w.count("SELECT sum(length) AS n FROM a.extent"),
        (incomplete.len() + current.len()) as i64
    );
}

#[test]
fn missing_retained_export_prefix_captures_a_full_rewrite_without_repair() {
    let w = ClaudeHome::new();
    let dir = w.config(8).spool_dir("ai-studio");
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("missing.json");
    let full = br#"{"chunkedPrompt":{"chunks":[{"role":"user","text":"complete"}]}}"#;
    let incomplete = &full[..full.len() - 1];
    fs::write(&path, incomplete).unwrap();
    let (mut e, _lock) = engine(&w, 8);
    let first = e.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{first:?}");
    let (input, old_chain, old_length) = native_source(&e, &path);
    let stored = e
        .reads
        .reader()
        .unwrap()
        .stored_chain(&input, old_chain, old_length)
        .unwrap();
    let hash = stored.single_file_blob().unwrap();
    let extent = e.blobs.extent(&hashed_blobs::Hash(hash.0)).unwrap();
    let segment = e.blobs.root().join(&extent.segment);
    let held = segment.with_extension("held-prefix");
    fs::write(&path, full).unwrap();
    let facts = FileFacts::of(&fs::metadata(&path).unwrap());
    let origin = polylogue_sources::origin::by_name("ai-studio").unwrap();
    let context = e
        .sampling_context(&path, origin.name, &origin.reading_version())
        .unwrap();
    let mut plan = Plan::stored(
        path.clone(),
        origin,
        InputKey {
            id: input.clone(),
            format: origin.name.into(),
            route: origin.route,
        },
        Change::Appended { from: old_length },
        Some(facts),
        full.len() as u64,
        0,
        pool::Bytes::File,
        8,
        true,
    );
    plan.stored_upto = old_length;
    plan.basis = context.basis;
    plan.observation = Some(e.observation(&path, Some(facts)));
    fs::rename(&segment, &held).unwrap();
    let mut pass = Pass::default();
    let result = e.store_plan(&mut plan, &mut pass);
    // Restore the synthetic segment's name before decoding either retained body.
    fs::rename(&held, &segment).unwrap();
    result.unwrap();
    assert_eq!(pass.bytes_stored, full.len() as u64);
    let (_, chain, length) = native_source(&e, &path);
    assert_ne!(chain, old_chain);
    assert_eq!(retained(&e, &input, old_chain, old_length), incomplete);
    assert_eq!(retained(&e, &input, chain, length), full);
    let Some(polylogue_sources::Parse::Export(mut parser)) = (origin.open)(&path) else {
        panic!("export parser")
    };
    e.execute_export(&mut plan, &mut *parser, &mut pass)
        .unwrap();
    assert_eq!(texts(&w, "ai-studio:missing"), ["complete"]);
    assert_eq!(
        w.count("SELECT sum(length) AS n FROM a.extent"),
        (incomplete.len() + full.len()) as i64
    );
}

#[test]
fn replacement_during_export_prefix_comparison_refuses_the_generation() {
    let scratch = tempfile::tempdir().unwrap();
    let path = scratch.path().join("provider");
    let replacement = scratch.path().join("replacement");
    let original = vec![b'x'; 128 << 10];
    fs::write(&path, &original).unwrap();
    fs::write(&replacement, &original).unwrap();
    let mut file = fs::File::open(&path).unwrap();
    let expected = FileFacts::of(&file.metadata().unwrap());
    let mut moved = false;
    let result = prefix_matches(
        &mut file,
        &path,
        expected,
        original.len() as u64,
        std::io::Cursor::new(&original),
        |_| {
            if !moved {
                fs::rename(&replacement, &path).unwrap();
                moved = true;
            }
        },
    );
    assert!(moved);
    assert!(
        result.is_err(),
        "the old descriptor cannot certify its replacement path"
    );
}

/// A prompt read from Drive and its Takeout copy are one session, named by
/// the Drive file ID, whichever arrives first and after a rebuild: the Drive
/// reading records the content key the copy is named by.
#[test]
fn drive_and_takeout_copy_give_one_session() {
    let bytes = serde_json::to_vec(&json!({
        "chunkedPrompt":{"chunks":[
            {"role":"user","text":"About rivers","createTime":"2026-03-01T10:00:00Z"},
            {"role":"model","text":"Rivers flow to the sea.","finishReason":"STOP"}
        ]},
        "runSettings":{"model":"models/gemini-example"}
    }))
    .unwrap();
    let mut shown = Vec::new();
    for drive_first in [true, false] {
        let w = ClaudeHome::new();
        let spool = w.config(8).spool_dir("ai-studio");
        fs::create_dir_all(&spool).unwrap();
        let takeout = w.dir.path().join("Takeout");
        let copy = takeout.join("Drive/Google AI Studio/Rivers");
        fs::create_dir_all(copy.parent().unwrap()).unwrap();
        fs::write(&copy, &bytes).unwrap();
        let (mut e, lock) = engine(&w, 8);
        let drive = |e: &mut Engine| {
            fs::write(spool.join("drive-id.json"), &bytes).unwrap();
            let pass = e.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        };
        if drive_first {
            drive(&mut e);
        }
        let report = e.import(std::slice::from_ref(&takeout)).unwrap();
        assert!(report.refused.is_empty(), "{report:?}");
        assert!(report.pass.errors.is_empty(), "{report:?}");
        if !drive_first {
            drive(&mut e);
        }
        drop(e);
        drop(lock);
        for rebuild in [false, true] {
            if rebuild {
                polylogue_testkit::delete_index(&w.root);
                let (mut e, lock) = engine(&w, 8);
                let pass = e.run_pass().unwrap();
                assert!(pass.errors.is_empty(), "{:?}", pass.errors);
                drop(e);
                drop(lock);
            }
            assert_eq!(
                w.count("SELECT COUNT(*) AS n FROM session WHERE origin='ai-studio'"),
                1,
                "drive first {drive_first}, rebuild {rebuild}"
            );
            let messages = w
                .reads()
                .reader()
                .unwrap()
                .messages(
                    &polylogue_model::SessionId::parse("ai-studio:drive-id").unwrap(),
                    false,
                )
                .unwrap();
            assert_eq!(
                messages.iter().map(|m| m.text()).collect::<Vec<_>>(),
                ["About rivers", "Rivers flow to the sea."]
            );
            assert_eq!(
                w.count("SELECT COUNT(DISTINCT input_pk) AS n FROM message_source"),
                2,
                "both copies are readings of the one session"
            );
            shown.push(serde_json::to_value(&messages).unwrap());
        }
    }
    assert!(shown.windows(2).all(|pair| pair[0] == pair[1]));
}
