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

fn add_note(e: &Engine, id: &str, target: &str) {
    e.writer()
        .unwrap()
        .user(UserRequest::now(
            id.into(),
            UserWrite::AddNote {
                note: NewNote {
                    kind: NoteKind::Note,
                    key: None,
                    value: Some("Keep this exact reference".into()),
                    author: Author::Operator,
                    targets: vec![Target::Message(MessageId::parse(target).unwrap())],
                    relations: vec![],
                    revises: None,
                    valid_until: None,
                },
            },
        ))
        .unwrap();
}

// Install the pre-canonicalization row keys while keeping the actual retained
// bodies and source associations. Reopening must replace this whole generation.
fn legacy_hash_generation(
    index: &mut rusqlite::Connection,
    messages: impl Iterator<Item = Message>,
) -> Vec<(String, String)> {
    let transaction = index.transaction().unwrap();
    let mut hashes = Vec::new();
    for message in messages {
        let legacy = blake3::hash(&serde_json::to_vec(&message).unwrap());
        let mut canonical = serde_json::to_value(&message).unwrap();
        canonical.sort_all_objects();
        let canonical = blake3::hash(&serde_json::to_vec(&canonical).unwrap());
        assert_ne!(legacy, canonical);
        assert_eq!(
            transaction
                .execute(
                    "UPDATE message_version SET hash=?1 WHERE hash=?2",
                    rusqlite::params![
                        legacy.as_bytes().as_slice(),
                        canonical.as_bytes().as_slice()
                    ],
                )
                .unwrap(),
            1
        );
        for table in ["message", "message_source"] {
            transaction
                .execute(
                    &format!("UPDATE {table} SET version_hash=?1 WHERE version_hash=?2"),
                    rusqlite::params![
                        legacy.as_bytes().as_slice(),
                        canonical.as_bytes().as_slice()
                    ],
                )
                .unwrap();
        }
        hashes.push((legacy.to_hex().to_string(), canonical.to_hex().to_string()));
    }
    transaction.pragma_update(None, "user_version", 29).unwrap();
    transaction.commit().unwrap();
    hashes
}

#[test]
fn retained_codex_rewrite_items_versions_and_note_targets_survive_index_rebuild() {
    for (remove_provider, replace_generation, truncate) in [
        (false, false, false),
        (true, false, false),
        (true, true, false),
        (true, true, true),
    ] {
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
            add_note(&e, "about-old-reasoning", "codex:historic-r1");
            add_note(&e, "cites-old-answer", "codex:current-m1");
            let basis = versions(&w, "codex:current-m1").remove(0);
            // A fixture with explicitly supplied citation fields, distinct from
            // the ordinary write above, whose author-read basis stays unknown.
            let hash = blake3::hash(basis.message.text().as_bytes());
            let parser = w
                .reads()
                .reader()
                .unwrap()
                .query(
                    "SELECT parser_version AS version FROM message_source LIMIT 1",
                    &[],
                )
                .unwrap()[0]["version"]
                .as_i64()
                .unwrap();
            let canonical = rusqlite::Connection::open(w.root.join("archive.db")).unwrap();
            canonical.execute("UPDATE note_target SET seen_hash=?1,anchor_input=?2,anchor_chain=?3,anchor_start=?4,anchor_end=?5,anchor_parser=?6 WHERE note_id='cites-old-answer'", rusqlite::params![hash.as_bytes().as_slice(),basis.input,basis.chain as i64,basis.start as i64,basis.end as i64,parser]).unwrap();
            drop(canonical);
            fs::write(&file, header() + &message("current-v2")).unwrap();
            assert!(e.converge().unwrap().errors.is_empty());
            assert_eq!(messages(&w).len(), 2);
            if truncate {
                fs::write(&file, "").unwrap();
                assert!(e.converge().unwrap().errors.is_empty());
                assert_eq!(messages(&w).len(), 2);
            }
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
        let generation = w
            .reads()
            .reader()
            .unwrap()
            .query("SELECT value FROM meta WHERE key='generation'", &[])
            .unwrap();
        let legacy_hashes = if replace_generation {
            let mut index = rusqlite::Connection::open(w.root.join("index.db")).unwrap();
            legacy_hash_generation(
                &mut index,
                before
                    .0
                    .iter()
                    .chain(&before.1)
                    .map(|version| version.message.clone()),
            )
        } else {
            delete_index(&w.root);
            Vec::new()
        };
        {
            let (mut rebuilt, _lock) = w.engine(64 << 20);
            let pass = rebuilt.converge().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
            assert_ne!(
                w.reads()
                    .reader()
                    .unwrap()
                    .query("SELECT value FROM meta WHERE key='generation'", &[])
                    .unwrap(),
                generation
            );
            assert_eq!(
                messages(&w).len(),
                2,
                "provider removed={remove_provider}, generation replaced={replace_generation}"
            );
            assert_eq!(versions(&w, "codex:historic-r1"), before.0);
            assert_eq!(versions(&w, "codex:current-m1"), before.1);
            let reads = w.reads();
            let r = reads.reader().unwrap();
            if replace_generation {
                let actual = r
                    .query(
                        "SELECT lower(hex(hash)) AS hash FROM message_version ORDER BY hash",
                        &[],
                    )
                    .unwrap()
                    .into_iter()
                    .map(|row| row["hash"].as_str().unwrap().to_owned())
                    .collect::<Vec<_>>();
                let mut expected = legacy_hashes
                    .iter()
                    .map(|(_, canonical)| canonical.clone())
                    .collect::<Vec<_>>();
                expected.sort();
                assert_eq!(
                    actual, expected,
                    "every retained version gets its canonical key in the replacement generation"
                );
                assert!(
                    legacy_hashes
                        .iter()
                        .all(|(legacy, _)| !actual.contains(legacy)),
                    "old/new key schemes never mix in place"
                );
                let source_keys = r
                    .query(
                        "SELECT lower(hex(version_hash)) AS hash FROM message_source",
                        &[],
                    )
                    .unwrap();
                assert!(source_keys.iter().all(|row| {
                    actual
                        .iter()
                        .any(|hash| row["hash"].as_str() == Some(hash.as_str()))
                }));
            }
            assert_eq!(
                r.note("about-old-reasoning").unwrap().unwrap().targets,
                [Target::Message(
                    MessageId::parse("codex:historic-r1").unwrap()
                )]
            );
            assert!(
                r.query(
                    "SELECT seen_hash FROM a.note_target WHERE note_id='about-old-reasoning'",
                    &[]
                )
                .unwrap()[0]["seen_hash"]
                    .is_null()
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
fn retained_append_chains_rebuild_the_observed_versions_without_full_prefix_rereads() {
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
        assert_eq!(
            hashes.len(),
            before.len(),
            "same-chain open-item updates replace their earlier projection"
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
