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
        // Rebuild from rewritten retained bytes after deleting the original export.
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
    let saved:Vec<(i64,i64,i64)>=index.prepare("SELECT message_pk,source_start,source_end FROM message_source WHERE session_pk=(SELECT pk FROM session WHERE id='claude-code:s2')").unwrap().query_map([],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?))).unwrap().collect::<Result<_,_>>().unwrap();
    let other:(i64,i64)=index.query_row("SELECT source_start,source_end FROM message_source WHERE session_pk=(SELECT pk FROM session WHERE id='claude-code:s1') LIMIT 1",[],|r|Ok((r.get(0)?,r.get(1)?))).unwrap();
    let inside = raw.find("walrus-9931").unwrap() as i64;
    for (label, range) in [
        ("partial scalar", (inside, inside + 11)),
        ("surviving record", other),
    ] {
        index.execute("UPDATE message_source SET source_start=?1,source_end=?2 WHERE session_pk=(SELECT pk FROM session WHERE id='claude-code:s2')",(range.0,range.1)).unwrap();
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
            "SELECT value FROM a.meta WHERE key='excision_revision'",
            &[],
        )
        .unwrap()[0]["value"]
        .as_str()
        .unwrap()
        .to_owned()
}
