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
