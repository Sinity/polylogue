use super::*;

#[test]
fn excising_a_session_from_a_shared_export_blanks_its_texts_and_notes() {
    use polylogue_model::{Author, NewNote, NoteKind, Target};
    use polylogue_store::{UserRequest, UserWrite};
    for reverse in [false, true] {
        let w = ClaudeHome::new();
        w.append(
            "s1.jsonl",
            &(user("u1", "s1", "keep this message please")
                + &assistant("a1", "s1", "kept reply text here")),
        );
        w.append(
            "s2.jsonl",
            &(user("u2", "s2", "the secret is walrus-9931 indeed")
                + &assistant("a2", "s2", "acknowledged the walrus thing")),
        );
        let (mut e, _lock) = w.engine(64 << 20);
        assert!(e.converge().unwrap().errors.is_empty());
        let mut sessions = vec![w.export("claude-code:s1"), w.export("claude-code:s2")];
        if reverse {
            sessions.reverse();
        }
        let doc = polylogue_model::export::Document::new(sessions);
        let file = w.dir.path().join("both.json");
        fs::write(&file, doc.to_json().unwrap()).unwrap();

        // One export file holds both sessions.
        let fresh = ClaudeHome::new();
        let (mut e2, _lock2) = fresh.engine(64 << 20);
        e2.import(std::slice::from_ref(&file)).unwrap();
        assert!(stored(&fresh.root, "walrus-9931"));
        let s2 = SessionId::parse("claude-code:s2").unwrap();
        let note = |id: &str, targets: Vec<Target>| {
            UserRequest::now(
                id.into(),
                UserWrite::AddNote {
                    note: NewNote {
                        kind: NoteKind::Note,
                        key: None,
                        value: Some("about the walrus".into()),
                        author: Author::Operator,
                        targets,
                        relations: vec![],
                        revises: None,
                        valid_until: None,
                    },
                },
            )
        };
        let writer = e2.writer().unwrap();
        writer
            .user(note("n-only", vec![Target::Session(s2.clone())]))
            .unwrap();
        writer
            .user(note(
                "n-both",
                vec![
                    Target::Session(s2.clone()),
                    Target::Session(SessionId::parse("claude-code:s1").unwrap()),
                ],
            ))
            .unwrap();
        drop(writer);

        let before_revision = revision(&fresh);
        assert_bad_ranges_refuse(&mut e2, &fresh, &doc.to_json().unwrap());
        assert_eq!(revision(&fresh), before_revision);
        let dry = excise(&mut e2, "session:claude-code:s2", true);
        assert!(dry.dry_run);
        assert_eq!(revision(&fresh), before_revision);
        assert!(stored(&fresh.root, "walrus-9931"));
        assert_eq!(fresh.count("SELECT count(*) AS n FROM a.note"), 2);
        assert_eq!(
            fresh.texts("s2"),
            [
                "the secret is walrus-9931 indeed",
                "acknowledged the walrus thing"
            ]
        );
        let done = excise(&mut e2, "session:claude-code:s2", false);
        assert_ne!(revision(&fresh), before_revision);
        assert!(done.inputs_removed.is_empty(), "{done:?}");
        assert!(done.inputs_kept.is_empty(), "{done:?}");
        assert_eq!(done.texts_left, 0, "{done:?}");
        assert_eq!(done.notes_removed, 1, "{done:?}");
        assert!(!stored(&fresh.root, "walrus-9931"));
        assert!(!stored(&fresh.root, "acknowledged the walrus thing"));
        assert!(stored(&fresh.root, "keep this message please"));
        assert_eq!(
            fresh.count("SELECT count(*) AS n FROM a.note WHERE id = 'n-only'"),
            0
        );
        assert_eq!(
            fresh.count("SELECT count(*) AS n FROM a.note_target WHERE note_id = 'n-both'"),
            1
        );
        // Rebuild from rewritten stored bytes after deleting the original export.
        drop(e2);
        drop(_lock2);
        fs::remove_file(&file).unwrap();
        delete_index(&fresh.root);
        let (mut e3, _lock3) = fresh.engine(64 << 20);
        assert!(e3.converge().unwrap().errors.is_empty());
        assert_eq!(
            fresh.texts("s1"),
            ["keep this message please", "kept reply text here"]
        );
        assert!(fresh.texts("s2").is_empty());
    }
}

fn assert_bad_ranges_refuse(e: &mut Engine, w: &ClaudeHome, raw: &str) {
    let index = polylogue_store::archive::open_write(
        &Archive::at(&w.root).index_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    let saved:Vec<(i64,i64,i64)>=index.prepare("SELECT message_pk,source_start,source_end FROM message_source WHERE message_pk IN (SELECT m.pk FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id='claude-code:s2')").unwrap().query_map([],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap().collect::<Result<_,_>>().unwrap();
    let other:(i64,i64)=index.query_row("SELECT source_start,source_end FROM message_source WHERE message_pk IN (SELECT m.pk FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id='claude-code:s1') LIMIT 1",[],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
    let inside = raw.find("walrus-9931").unwrap() as i64;
    for (label, range) in [
        ("partial scalar", (inside, inside + 11)),
        ("surviving record", other),
    ] {
        index.execute("UPDATE message_source SET source_start=?1,source_end=?2 WHERE message_pk IN (SELECT m.pk FROM message m JOIN session s ON s.pk=m.session_pk WHERE s.id='claude-code:s2')",(range.0,range.1)).unwrap();
        for dry_run in [true, false] {
            let error = e
                .excise(ExciseRequest {
                    id: format!("refuse-{label}-{dry_run}"),
                    scope: ExciseScope::Session(SessionId::parse("claude-code:s2").unwrap()),
                    reason: None,
                    dry_run,
                })
                .unwrap_err();
            assert!(
                error.to_string().contains(if label == "partial scalar" {
                    "scalar"
                } else {
                    "surviving session record"
                }),
                "{error}"
            );
            assert_eq!(
                w.texts("s1"),
                ["keep this message please", "kept reply text here"]
            );
            assert_eq!(
                w.texts("s2"),
                [
                    "the secret is walrus-9931 indeed",
                    "acknowledged the walrus thing"
                ]
            );
            assert!(stored(&w.root, "walrus-9931"));
            assert_eq!(w.count("SELECT count(*) AS n FROM a.note"), 2);
            assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
        }
    }
    for (message, start, end) in saved {
        index
            .execute(
                "UPDATE message_source SET source_start=?2,source_end=?3 WHERE message_pk=?1",
                (message, start, end),
            )
            .unwrap();
    }
}

fn revision(w: &ClaudeHome) -> String {
    w.reads()
        .reader()
        .unwrap()
        .query(
            "SELECT count(*) || ':' || COALESCE(max(commit_seq), 0) AS value FROM a.excision",
            &[],
        )
        .unwrap()[0]["value"]
        .as_str()
        .unwrap()
        .to_owned()
}

#[test]
fn excising_one_conversation_of_a_shared_export_leaves_every_other_byte_and_row() {
    let w = ClaudeHome::new();
    let path = w.dir.path().join("conversations.json");
    let raw = serde_json::to_vec(&serde_json::json!([
        chatgpt_conversation("c1", "first kept conversation"),
        chatgpt_conversation("c2", "excised heron-5521 conversation"),
        chatgpt_conversation("c3", "third kept conversation"),
    ]))
    .unwrap();
    fs::write(&path, &raw).unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    let report = e.import(std::slice::from_ref(&path)).unwrap();
    assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
    let input = blake3::hash(&raw).to_hex().to_string();
    let kept = ["chatgpt:c1", "chatgpt:c3"];
    let rows = || {
        let pool = w.reads();
        let reader = pool.reader().unwrap();
        [
            "SELECT * FROM session WHERE id <> 'chatgpt:c2' ORDER BY id",
            "SELECT m.* FROM message m JOIN session s ON s.pk = m.session_pk
             WHERE s.id <> 'chatgpt:c2' ORDER BY m.pk",
            "SELECT b.* FROM block b JOIN session s ON s.pk = b.session_pk
             WHERE s.id <> 'chatgpt:c2' ORDER BY 1, 2",
            "SELECT x.* FROM message_source x JOIN message m ON m.pk = x.message_pk
             JOIN session s ON s.pk = m.session_pk WHERE s.id <> 'chatgpt:c2' ORDER BY 1, 2",
        ]
        .map(|sql| reader.query(sql, &[]).unwrap())
    };
    let exports = || kept.map(|id| w.export(id));
    let before = (rows(), exports());
    assert_eq!(before.0[0].len(), 2);

    let done = excise(&mut e, "session:chatgpt:c2", false);
    assert_eq!(done.sessions, ["chatgpt:c2"]);
    assert!(done.inputs_removed.is_empty(), "{done:?}");
    assert!(done.inputs_kept.is_empty(), "{done:?}");
    assert!(done.inputs_overwritten.contains_key(&input), "{done:?}");
    assert_eq!((rows(), exports()), before);

    // Only the excised conversation's bytes changed, to spaces, in place.
    let store = hashed_blobs::Store::open(w.root.clone(), vec![], 0o600).unwrap();
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let progress = reader.input_progress(&input).unwrap().unwrap();
    let after = reader
        .stored_range(&input, progress.chain, 0, progress.length, &store)
        .unwrap();
    assert_eq!(after.len(), raw.len());
    let first = raw.iter().zip(&after).position(|(a, b)| a != b).unwrap();
    let last = raw.iter().zip(&after).rposition(|(a, b)| a != b).unwrap();
    let removed = String::from_utf8_lossy(&raw[first..=last]);
    assert!(removed.contains("heron-5521"), "{removed}");
    for other in ["\"c1\"", "\"c3\"", "first kept", "third kept"] {
        assert!(!removed.contains(other), "{removed}");
    }
    assert!(after[first..=last].iter().all(u8::is_ascii_whitespace));
    assert!(!stored(&w.root, "heron-5521"));
}
