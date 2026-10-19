use super::*;
use polylogue_model::{Message, SessionId};
use std::fs::{self, OpenOptions};
use std::io::Write;

struct World {
    dir: tempfile::TempDir,
    home: PathBuf,
    root: PathBuf,
}

impl World {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let home = dir.path().join("home");
        fs::create_dir_all(home.join(".claude/projects/-proj")).unwrap();
        let root = dir.path().join("archive");
        Self { dir, home, root }
    }

    fn engine(&self, chunk: u64) -> (Engine, polylogue_store::ArchiveLock) {
        let (archive, lock) = Archive::open_or_create(&self.root).unwrap();
        let config = Config {
            home: self.home.clone(),
            tool_dirs: Default::default(),
            workers: 3,
            bytes_in_flight: 1 << 20,
            machine: "test-machine".into(),
            host: "test".into(),
            chunk,
        };
        (Engine::new(config, archive).unwrap(), lock)
    }

    fn file(&self, name: &str) -> PathBuf {
        self.home.join(".claude/projects/-proj").join(name)
    }

    fn append(&self, name: &str, text: &str) {
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.file(name))
            .unwrap();
        f.write_all(text.as_bytes()).unwrap();
    }

    fn texts(&self, session: &str) -> Vec<String> {
        let pool = ReadPool::open(&Archive::at(&self.root)).unwrap();
        let id = SessionId::parse(&format!("claude-code:{session}")).unwrap();
        pool.reader()
            .unwrap()
            .messages(&id, false)
            .unwrap()
            .iter()
            .map(Message::text)
            .collect()
    }

    fn count(&self, sql: &str) -> i64 {
        let pool = ReadPool::open(&Archive::at(&self.root)).unwrap();
        pool.reader().unwrap().query(sql, &[]).unwrap()[0]["n"]
            .as_i64()
            .unwrap()
    }
}

fn user(uuid: &str, session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "user", "uuid": uuid, "sessionId": session,
            "timestamp": "2026-03-01T10:00:00Z",
            "message": {"role": "user", "content": text}
        })
    )
}

fn assistant(uuid: &str, session: &str, text: &str) -> String {
    format!(
        "{}\n",
        serde_json::json!({
            "type": "assistant", "uuid": uuid, "sessionId": session,
            "timestamp": "2026-03-01T10:00:01Z",
            "message": {"id": format!("msg_{uuid}"), "role": "assistant", "stop_reason": "end_turn",
                        "content": [{"type": "text", "text": text}]}
        })
    )
}

#[test]
fn a_first_build_stores_and_parses_and_a_second_pass_reads_nothing() {
    let w = World::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "hello") + &assistant("a1", "s1", "hi")),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(pass.inputs_changed, 1);
    assert_eq!(w.texts("s1"), ["hello", "hi"]);
    let again = e.converge().unwrap();
    assert_eq!(again.inputs_changed, 0);
    assert_eq!(again.bytes_read, 0);
}

#[test]
fn appends_cost_only_their_bytes_and_a_partial_line_waits() {
    let w = World::new();
    let first = user("u1", "s1", "one");
    w.append("s1.jsonl", &first);
    let (mut e, _lock) = w.engine(64 << 20);
    e.converge().unwrap();
    let second = user("u2", "s1", "two");
    let (head, rest) = second.split_at(10);
    w.append("s1.jsonl", head);
    let pass = e.converge().unwrap();
    assert_eq!(pass.bytes_read, 10, "only the appended bytes are read");
    assert_eq!(w.texts("s1"), ["one"]);
    w.append("s1.jsonl", rest);
    let pass = e.converge().unwrap();
    assert_eq!(
        pass.bytes_read,
        second.len() as u64,
        "the waiting line and its rest"
    );
    assert_eq!(w.texts("s1"), ["one", "two"]);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 3);
    assert_eq!(
        w.count("SELECT SUM(length) AS n FROM a.extent"),
        (first.len() + second.len()) as i64
    );
}

#[test]
fn after_a_restart_state_is_rebuilt_only_when_the_input_changes() {
    let w = World::new();
    w.append("s1.jsonl", &user("u1", "s1", "one"));
    w.append("s2.jsonl", &user("v1", "s2", "other"));
    {
        let (mut e, _lock) = w.engine(64 << 20);
        e.converge().unwrap();
    }
    let (mut e, _lock) = w.engine(64 << 20);
    let quiet = e.converge().unwrap();
    assert_eq!(quiet.bytes_read, 0, "nothing is re-parsed at start-up");
    let line = user("u2", "s1", "two");
    w.append("s1.jsonl", &line);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert!(
        pass.bytes_read > line.len() as u64,
        "s1 was parsed again from the start"
    );
    assert_eq!(w.texts("s1"), ["one", "two"]);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM message"), 3);
}

#[test]
fn a_rewritten_file_starts_a_new_chain_and_its_content_wins() {
    let w = World::new();
    w.append("s1.jsonl", &user("u1", "s1", "draft"));
    let (mut e, _lock) = w.engine(64 << 20);
    e.converge().unwrap();
    fs::write(w.file("s1.jsonl"), user("u1", "s1", "final text, longer")).unwrap();
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(w.texts("s1"), ["final text, longer"]);
    assert_eq!(
        w.count("SELECT COUNT(DISTINCT chain) AS n FROM a.revision"),
        2
    );
}

#[test]
fn large_files_go_in_chunks_and_a_record_longer_than_a_chunk_still_parses() {
    let w = World::new();
    let mut text = String::new();
    for i in 0..40 {
        text += &user(&format!("u{i}"), "s1", &format!("message {i}"));
    }
    text += &user("big", "s1", &"x".repeat(2000));
    w.append("s1.jsonl", &text);
    let (mut e, _lock) = w.engine(300);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let got = w.texts("s1");
    assert_eq!(got.len(), 41);
    assert_eq!(got[40].len(), 2000);
    assert_eq!(w.count("SELECT COUNT(*) AS n FROM a.revision"), 1);
    assert_eq!(
        w.count("SELECT SUM(length) AS n FROM a.extent"),
        text.len() as i64
    );
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM parse_state WHERE status = 'done'"),
        1
    );
}

#[test]
fn a_vanished_file_is_observed_gone_and_its_messages_stay() {
    let w = World::new();
    w.append("s1.jsonl", &user("u1", "s1", "kept"));
    let (mut e, _lock) = w.engine(64 << 20);
    e.converge().unwrap();
    fs::remove_file(w.file("s1.jsonl")).unwrap();
    e.converge().unwrap();
    assert_eq!(w.texts("s1"), ["kept"]);
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.observation WHERE class = 'deleted'"),
        1
    );
}

#[test]
fn importing_the_same_bytes_twice_stores_them_once() {
    let w = World::new();
    let outside = w.dir.path().join("export/s9.jsonl");
    fs::create_dir_all(outside.parent().unwrap()).unwrap();
    fs::write(&outside, user("x1", "s9", "imported")).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let first = e.import(std::slice::from_ref(&outside)).unwrap();
    assert_eq!(first.imported, std::slice::from_ref(&outside));
    assert_eq!(w.texts("s9"), ["imported"]);
    let second = e.import(std::slice::from_ref(&outside)).unwrap();
    assert_eq!(second.already_stored, [outside]);
}

fn messages_of(root: &Path, session: &str) -> Vec<Message> {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let id = SessionId::parse(session).unwrap();
    pool.reader().unwrap().messages(&id, false).unwrap()
}

fn chatgpt_conversation(id: &str, text: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id, "conversation_id": id, "title": "T", "create_time": 1700000000.0,
        "current_node": "n1",
        "mapping": {
            "n1": {"id": "n1", "parent": null, "children": [], "message": {
                "id": format!("{id}-m1"), "author": {"role": "user"}, "create_time": 1700000000.0,
                "content": {"content_type": "multimodal_text", "parts": [
                    {"content_type": "image_asset_pointer", "asset_pointer": "file-service://file-Img"},
                    text
                ]},
                "status": "finished_successfully", "recipient": "all", "metadata": {}
            }}
        }
    })
}

#[test]
fn a_zip_export_imports_its_conversations_and_keeps_its_other_files() {
    use std::io::Write as _;
    let w = World::new();
    let zip_path = w.dir.path().join("chatgpt-data.zip");
    let mut zip = zip::ZipWriter::new(fs::File::create(&zip_path).unwrap());
    let opts = zip::write::SimpleFileOptions::default();
    zip.start_file("conversations-000.json", opts).unwrap();
    let convs = serde_json::json!([chatgpt_conversation("c1", "look at this")]);
    zip.write_all(&serde_json::to_vec(&convs).unwrap()).unwrap();
    zip.start_file("file-Img-photo.png", opts).unwrap();
    zip.write_all(b"\x89PNG not really").unwrap();
    zip.finish().unwrap();

    let (mut e, _lock) = w.engine(64 << 20);
    let report = e.import(std::slice::from_ref(&zip_path)).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(report.imported.len(), 2, "{:?}", report.imported);
    let msgs = messages_of(&w.root, "chatgpt:c1");
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].text(), "look at this");
    assert_eq!(
        w.count("SELECT COUNT(*) AS n FROM a.input WHERE format = 'export-file'"),
        1,
        "the image is kept as bytes"
    );
    let again = e.import(std::slice::from_ref(&zip_path)).unwrap();
    assert_eq!(again.already_stored.len(), 2);
    assert!(again.imported.is_empty());
}

#[test]
fn a_document_export_is_parsed_whole_however_small_the_chunk() {
    let w = World::new();
    let path = w.dir.path().join("conversations.json");
    let convs = serde_json::json!([
        chatgpt_conversation("c1", "one"),
        chatgpt_conversation("c2", "two")
    ]);
    fs::write(&path, serde_json::to_vec_pretty(&convs).unwrap()).unwrap();
    // A chunk far smaller than the document: a line-framed origin would
    // read it in steps; a document is read whole in one.
    let (mut e, _lock) = w.engine(64);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    assert_eq!(messages_of(&w.root, "chatgpt:c1")[0].text(), "one");
    assert_eq!(messages_of(&w.root, "chatgpt:c2")[0].text(), "two");
}

fn capture_file(texts: &[&str]) -> Vec<u8> {
    let turns: Vec<serde_json::Value> = texts
        .iter()
        .enumerate()
        .map(|(i, t)| serde_json::json!({"provider_turn_id": format!("t{i}"), "role": "user", "text": t, "ordinal": i}))
        .collect();
    serde_json::to_vec_pretty(&serde_json::json!({
        "polylogue_capture_kind": "browser_llm_session",
        "schema_version": 1,
        "source": "browser-extension",
        "provenance": {"captured_at": "2026-02-01T00:00:00Z"},
        "provider_meta": {},
        "raw_provider_payload": {},
        "session": {"provider": "grok", "provider_session_id": "g1", "turns": turns}
    }))
    .unwrap()
}

#[test]
fn a_capture_rewritten_in_place_is_parsed_again_whole() {
    let w = World::new();
    let dir = w.home.join(".local/state/polylogue/browser-capture/grok");
    fs::create_dir_all(&dir).unwrap();
    let file = dir.join("g1.json");
    fs::write(&file, capture_file(&["one"])).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    let texts = |root: &Path| -> Vec<String> {
        messages_of(root, "grok:g1")
            .iter()
            .map(Message::text)
            .collect()
    };
    assert_eq!(texts(&w.root), ["one"]);
    // The extension rewrites the whole file when it captures again.
    fs::write(&file, capture_file(&["one", "two"])).unwrap();
    let pass = e.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(texts(&w.root), ["one", "two"]);
}

#[test]
fn an_imported_file_keeps_the_origin_whose_layout_it_sits_in() {
    // Claude Code accepts any JSON-lines file on import; a Grok CLI history
    // copied with its directories still reads as Grok CLI.
    let head = br#"{"type":"system","content":"You are a CLI."}"#;
    let grok = Path::new("/backup/.grok/sessions/%2Fhome/01a0/chat_history.jsonl");
    assert_eq!(recognize_import(grok, head).unwrap().name, "grok-cli");
    let loose = Path::new("/backup/s1.jsonl");
    assert_eq!(recognize_import(loose, head).unwrap().name, "claude-code");
}

fn excise(e: &mut Engine, target: &str, dry_run: bool) -> Excised {
    e.excise(ExciseRequest {
        id: format!("req-{target}-{dry_run}"),
        scope: ExciseScope::parse(target).unwrap(),
        reason: Some("test".into()),
        dry_run,
    })
    .unwrap()
}

/// Whether any stored input byte range holds `needle`.
fn stored(root: &Path, needle: &str) -> bool {
    !polylogue_store::excise::stored_blobs_containing(&Archive::at(root), needle)
        .unwrap()
        .is_empty()
}

#[test]
fn an_excised_session_is_gone_from_every_file_and_never_comes_back() {
    let w = World::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "the launch code is zebra42") + &assistant("a1", "s1", "noted")),
    );
    w.append(
        "s2.jsonl",
        &(user("u2", "s2", "unrelated") + &assistant("a2", "s2", "fine")),
    );
    let (mut e, lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(stored(&w.root, "zebra42"));

    let preview = excise(&mut e, "session:claude-code:s1", true);
    assert!(preview.dry_run);
    assert_eq!(preview.messages, 2);
    assert_eq!(w.texts("s1"), ["the launch code is zebra42", "noted"]);

    let done = excise(&mut e, "session:claude-code:s1", false);
    assert_eq!(done.sessions, ["claude-code:s1"]);
    assert_eq!(done.inputs_removed.len(), 1);
    assert!(done.blobs_deleted >= 1, "{done:?}");
    assert!(w.texts("s1").is_empty());
    assert_eq!(w.texts("s2"), ["unrelated", "fine"]);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM session WHERE id = 'claude-code:s1'"),
        0
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'zebra42'"),
        0
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM removed_session WHERE session_id = 'claude-code:s1'"),
        1
    );
    assert!(!stored(&w.root, "zebra42"));

    // The file keeps growing; nothing of it is collected again.
    w.append("s1.jsonl", &user("u3", "s1", "zebra42 again"));
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(w.texts("s1").is_empty());
    // Nor after a restart.
    drop(e);
    drop(lock);
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    assert!(w.texts("s1").is_empty());
    assert!(!stored(&w.root, "zebra42"));
    // Nor by import of the same session from another file.
    let copy = w.dir.path().join("copy.jsonl");
    fs::write(&copy, user("u1", "s1", "the launch code is zebra42")).unwrap();
    e.import(std::slice::from_ref(&copy)).unwrap();
    assert!(w.texts("s1").is_empty());
}

#[test]
fn an_excised_text_is_overwritten_in_stored_bytes_and_the_index() {
    let w = World::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "token sk-abc123 here") + &assistant("a1", "s1", "ok")),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    let done = excise(&mut e, "text:sk-abc123", false);
    assert_eq!(done.inputs_overwritten.len(), 1, "{done:?}");
    assert_eq!(
        done.on_disk.len(),
        1,
        "the live file still holds it: {done:?}"
    );
    assert!(!stored(&w.root, "sk-abc123"));
    assert_eq!(
        w.count("SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'abc123'"),
        0
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'token'"),
        1
    );
    e.converge().unwrap();
    let texts = w.texts("s1");
    assert_eq!(texts.len(), 2, "{texts:?}");
    assert!(!texts[0].contains("sk-abc123"));
    assert!(texts[0].starts_with("token ") && texts[0].ends_with(" here"));
    let pool = ReadPool::open(&Archive::at(&w.root)).unwrap();
    let rows = pool
        .reader()
        .unwrap()
        .query("SELECT scope AS s FROM a.excision", &[])
        .unwrap();
    assert!(!rows[0]["s"].as_str().unwrap().contains("sk-abc123"));
}

#[test]
fn excising_a_session_from_a_shared_input_overwrites_only_its_records() {
    let w = World::new();
    // One file holding records of two sessions (a resumed session's file
    // names the session each record belongs to).
    w.append(
        "mix.jsonl",
        &(user("u1", "keep", "keep this")
            + &user("u2", "drop", "drop pelican77")
            + &user("u3", "keep", "and this")),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    assert_eq!(w.count("SELECT count(*) AS n FROM session"), 2);
    let target = w.count("SELECT count(*) AS n FROM session WHERE id = 'claude-code:drop'");
    assert_eq!(target, 1);
    let done = excise(&mut e, "session:claude-code:drop", false);
    assert!(done.inputs_removed.is_empty(), "{done:?}");
    assert_eq!(done.inputs_overwritten.len(), 1, "{done:?}");
    assert!(!stored(&w.root, "pelican77"));
    assert!(stored(&w.root, "keep this") && stored(&w.root, "and this"));
    assert_eq!(w.texts("keep").len(), 2);
}

fn export_of(root: &Path, session: &str) -> polylogue_model::export::SessionExport {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    pool.reader()
        .unwrap()
        .session_export(&SessionId::parse(session).unwrap())
        .unwrap()
        .unwrap()
}

#[test]
fn an_export_imported_into_a_fresh_archive_gives_the_same_sessions_and_ids() {
    let w = World::new();
    w.append(
        "s1.jsonl",
        &(user("u1", "s1", "hello") + &assistant("a1", "s1", "hi") + &user("u2", "s1", "more")),
    );
    let (mut e, _lock) = w.engine(64 << 20);
    assert!(e.converge().unwrap().errors.is_empty());
    let original = export_of(&w.root, "claude-code:s1");
    let doc = polylogue_model::export::Document::new(vec![original.clone()]);
    let file = w.dir.path().join("export.json");
    fs::write(&file, doc.to_json().unwrap()).unwrap();

    let fresh = World::new();
    let (mut e2, _lock2) = fresh.engine(64 << 20);
    let report = e2.import(std::slice::from_ref(&file)).unwrap();
    assert_eq!(report.imported, std::slice::from_ref(&file), "{report:?}");
    let back = export_of(&fresh.root, "claude-code:s1");
    assert_eq!(back.session.id, original.session.id);
    assert_eq!(back.session.title, original.session.title);
    assert_eq!(back.messages, original.messages);
    assert_eq!(back.events, original.events);
    assert_eq!(back.links, original.links);
}
