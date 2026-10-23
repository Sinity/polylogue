use super::*;
use polylogue_model::{Author, MessageId, NewNote, NoteKind, Target};
use polylogue_store::{UserRequest, UserWrite};
use serde_json::json;

const THREAD: &str = "019a0000-0000-7000-8000-000000000099";

fn line(ordinal: u64, kind: &str, payload: serde_json::Value) -> String {
    format!(
        "{}\n",
        json!({"timestamp":"2026-10-23T10:00:03Z", "ordinal":ordinal,"type":kind,"payload":payload})
    )
}

fn header() -> String {
    line(
        1,
        "session_meta",
        json!({"id":THREAD,"source":"cli","cli_version":"0.151.0"}),
    )
}

fn message(text: &str) -> String {
    line(
        3,
        "response_item",
        json!({"id":"current-m1","type":"message","role":"assistant","content":[{"type":"output_text","text":text}]}),
    )
}

fn path(w: &ClaudeHome) -> std::path::PathBuf {
    let dir = w.home.join(".codex/sessions/2026/10/23");
    fs::create_dir_all(&dir).unwrap();
    dir.join(format!("rollout-synthetic-{THREAD}.jsonl"))
}

fn messages(w: &ClaudeHome) -> Vec<Message> {
    w.reads()
        .reader()
        .unwrap()
        .messages(
            &SessionId::parse(&format!("codex:{THREAD}")).unwrap(),
            false,
        )
        .unwrap()
}

fn versions(w: &ClaudeHome, id: &str) -> Vec<polylogue_store::read::MessageVersion> {
    w.reads()
        .reader()
        .unwrap()
        .message_versions(&MessageId::parse(id).unwrap())
        .unwrap()
}

#[test]
fn retained_codex_rewrite_items_versions_and_note_targets_survive_index_rebuild() {
    for remove_provider in [false, true] {
        let w = ClaudeHome::new();
        let file = path(&w);
        let historic = line(
            2,
            "response_item",
            json!({"id":"historic-r1","type":"reasoning","summary":[{"type":"summary_text","text":"synthetic retained reasoning"}]}),
        );
        fs::write(&file, header() + &historic + &message("current-v1")).unwrap();
        let before;
        let note;
        {
            let (mut e, _lock) = w.engine(64 << 20);
            assert!(e.converge().unwrap().errors.is_empty());
            e.writer()
                .unwrap()
                .user(UserRequest::now(
                    "about-old-reasoning".into(),
                    UserWrite::AddNote {
                        note: NewNote {
                            kind: NoteKind::Note,
                            key: None,
                            value: Some("Keep this exact reference".into()),
                            author: Author::Operator,
                            targets: vec![Target::Message(
                                MessageId::parse("codex:historic-r1").unwrap(),
                            )],
                            relations: vec![],
                            revises: None,
                            valid_until: None,
                        },
                    },
                ))
                .unwrap();
            fs::write(&file, header() + &message("current-v2")).unwrap();
            assert!(e.converge().unwrap().errors.is_empty());
            assert_eq!(messages(&w).len(), 2);
            before = (
                versions(&w, "codex:historic-r1"),
                versions(&w, "codex:current-m1"),
            );
            assert_eq!(before.0.len(), 1);
            assert_eq!(before.1.len(), 2);
            note = w
                .reads()
                .reader()
                .unwrap()
                .query("SELECT * FROM a.note_target ORDER BY ref", &[])
                .unwrap();
        }
        if remove_provider {
            fs::remove_file(&file).unwrap();
        }
        delete_index(&w.root);
        {
            let (mut rebuilt, _lock) = w.engine(64 << 20);
            let pass = rebuilt.converge().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
            assert_eq!(messages(&w).len(), 2, "provider removed={remove_provider}");
            assert_eq!(versions(&w, "codex:historic-r1"), before.0);
            assert_eq!(versions(&w, "codex:current-m1"), before.1);
            let reads = w.reads();
            let r = reads.reader().unwrap();
            assert_eq!(
                r.note("about-old-reasoning").unwrap().unwrap().targets,
                [Target::Message(
                    MessageId::parse("codex:historic-r1").unwrap()
                )]
            );
            assert_eq!(
                r.query("SELECT * FROM a.note_target ORDER BY ref", &[])
                    .unwrap(),
                note
            );
            assert_eq!(rebuilt.converge().unwrap().bytes_read, 0);
        }
        let (mut restarted, _lock) = w.engine(64 << 20);
        assert_eq!(restarted.converge().unwrap().bytes_read, 0);
        assert_eq!(messages(&w).len(), 2);
    }
}

#[test]
fn retained_append_boundaries_rebuild_open_item_versions_without_full_prefix_rereads() {
    let w = ClaudeHome::new();
    let file = path(&w);
    let mut total = 0;
    let before;
    let hashes;
    {
        let (mut e, _lock) = w.engine(64 << 20);
        fs::write(&file, header()).unwrap();
        total += header().len();
        e.converge().unwrap();
        for ordinal in 2..26 {
            let record = line(
                ordinal,
                "response_item",
                json!({"type":"function_call","name":"exec_command","arguments":"{\"cmd\":\"true\"}","call_id":format!("call-{ordinal}")}),
            );
            use std::io::Write;
            fs::OpenOptions::new()
                .append(true)
                .open(&file)
                .unwrap()
                .write_all(record.as_bytes())
                .unwrap();
            total += record.len();
            let pass = e.converge().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        }
        before = messages(&w)
            .iter()
            .map(|m| (m.id.clone(), versions(&w, &m.id.to_string())))
            .collect::<Vec<_>>();
        assert!(!before.is_empty());
        hashes = w
            .reads()
            .reader()
            .unwrap()
            .query(
                "SELECT lower(hex(hash)) AS hash FROM message_version ORDER BY hash",
                &[],
            )
            .unwrap();
        assert!(
            hashes.len() > before.len(),
            "appends retain changed open-item bodies"
        );
    }
    fs::remove_file(file).unwrap();
    delete_index(&w.root);
    let (mut rebuilt, _lock) = w.engine(64 << 20);
    let pass = rebuilt.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(
        pass.bytes_read, total as u64,
        "each appended range is parsed once"
    );
    for (id, expected) in before {
        assert_eq!(versions(&w, &id.to_string()), expected, "{id}");
    }
    assert_eq!(
        w.reads()
            .reader()
            .unwrap()
            .query(
                "SELECT lower(hex(hash)) AS hash FROM message_version ORDER BY hash",
                &[]
            )
            .unwrap(),
        hashes
    );
    assert_eq!(rebuilt.converge().unwrap().bytes_read, 0);
}

#[test]
fn historical_replay_keeps_exact_and_whole_input_excisions_after_source_removal() {
    for whole in [false, true] {
        let w = ClaudeHome::new();
        w.append(
            "history.jsonl",
            &(user("old", "history", "removed historical content")
                + &user("keep", "history", "surviving sibling")),
        );
        {
            let (mut e, _lock) = w.engine(64 << 20);
            e.converge().unwrap();
            fs::write(
                w.file("history.jsonl"),
                user("new", "history", "current sibling"),
            )
            .unwrap();
            e.converge().unwrap();
            let target = if whole {
                let input = w
                    .reads()
                    .reader()
                    .unwrap()
                    .known_inputs(&e.config.machine)
                    .unwrap()
                    .into_iter()
                    .find(|i| Path::new(&i.path) == w.file("history.jsonl"))
                    .unwrap()
                    .input_id;
                format!("input:{input}")
            } else {
                "text:removed historical content".into()
            };
            excise(&mut e, &target, false);
        }
        fs::remove_file(w.file("history.jsonl")).unwrap();
        delete_index(&w.root);
        let (mut e, _lock) = w.engine(64 << 20);
        let pass = e.converge().unwrap();
        assert!(pass.errors.is_empty(), "{:?}", pass.errors);
        if whole {
            assert!(w.texts("history").is_empty());
        } else {
            let texts = w.texts("history");
            assert!(texts.iter().any(|t| t == "surviving sibling"));
            assert!(texts.iter().any(|t| t == "current sibling"));
            assert!(
                texts
                    .iter()
                    .all(|t| !t.contains("removed historical content"))
            );
        }
    }
}
