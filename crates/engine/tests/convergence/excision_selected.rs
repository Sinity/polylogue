use super::*;
use polylogue_store::{Delivery, UserRequest, UserWrite};

#[test]
fn whole_input_removal_refuses_new_messages_of_its_session_after_restart() {
    for restart in [false, true] {
        let w = ClaudeHome::new();
        w.append("original.jsonl", &user("old", "removed", "original body"));
        let (mut engine, mut lock) = w.engine(64 << 20);
        assert!(engine.converge().unwrap().errors.is_empty());
        let input = w
            .reads()
            .reader()
            .unwrap()
            .known_inputs(&engine.config.machine)
            .unwrap()
            .into_iter()
            .find(|input| Path::new(&input.path) == w.file("original.jsonl"))
            .unwrap()
            .input_id;
        for dry in [true, false] {
            let done = excise(&mut engine, &format!("input:{input}"), dry);
            assert_eq!(done.sessions, ["claude-code:removed"]);
            assert_eq!(w.texts("removed").len(), usize::from(dry));
        }
        if restart {
            drop(engine);
            drop(lock);
            (engine, lock) = w.engine(64 << 20);
        }
        w.append(
            "other.jsonl",
            &user("new", "removed", "new body from another input"),
        );
        w.append(
            "unrelated.jsonl",
            &user("independent", "unrelated", "stored independent body"),
        );
        assert!(engine.converge().unwrap().errors.is_empty());
        assert!(w.texts("removed").is_empty(), "restart={restart}");
        assert_eq!(w.texts("unrelated"), ["stored independent body"]);
        drop(engine);
        drop(lock);
    }
}

#[test]
fn large_shared_blob_remains_readable_after_two_separate_session_excisions() {
    let w = ClaudeHome::new();
    let mut expected = Vec::new();
    for n in 0..160 {
        for session in ["first", "second", "kept"] {
            let text = format!(
                "{session} unique content {n} {}",
                "synthetic filler ".repeat(8)
            );
            w.append(
                "large.jsonl",
                &user(&format!("{session}-{n}"), session, &text),
            );
            if session == "kept" {
                expected.push(text);
            }
        }
    }
    let (mut engine, lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    assert_eq!(w.count("SELECT count(*) AS n FROM a.extent"), 1);
    assert!(w.count("SELECT size AS n FROM a.blob") >= polylogue_store::blobs::INLINE_LIMIT as i64);
    for session in ["first", "second"] {
        for dry in [true, false] {
            let done = excise(&mut engine, &format!("session:claude-code:{session}"), dry);
            assert_eq!(done.messages, 160);
            assert_eq!(w.texts("kept"), expected);
            assert_eq!(w.texts(session).len(), if dry { 160 } else { 0 });
        }
        assert!(!stored(&w.root, &format!("{session} unique content")));
    }
    drop(engine);
    drop(lock);
    fs::remove_file(w.file("large.jsonl")).unwrap();
    delete_index(&w.root);
    let (mut restored, _lock) = w.engine(64 << 20);
    assert!(restored.converge().unwrap().errors.is_empty());
    assert_eq!(w.texts("kept"), expected);
    assert!(w.texts("first").is_empty());
    assert!(w.texts("second").is_empty());
}

#[test]
fn session_excision_checks_all_append_pieces_before_rewriting_and_rebuilding() {
    let w = ClaudeHome::new();
    let (mut engine, lock) = w.engine(64 << 20);
    let mut expected = Vec::new();
    for n in 0..48 {
        let keep = format!("stored public message number {n}");
        w.append(
            "shared.jsonl",
            &(user(
                &format!("remove-{n}"),
                "remove",
                &format!("removed private message {n}"),
            ) + &user(&format!("keep-{n}"), "keep", &keep)),
        );
        assert!(engine.converge().unwrap().errors.is_empty());
        expected.push(keep);
    }
    assert!(w.count("SELECT count(*) AS n FROM a.extent") >= 48);
    let before = w.count("SELECT count(*) AS n FROM a.blob");
    for dry_run in [true, false] {
        let done = excise(&mut engine, "session:claude-code:remove", dry_run);
        assert_eq!(done.messages, 48);
        assert_eq!(done.inputs_overwritten.values().sum::<u64>(), 48);
        assert_eq!(w.texts("keep"), expected);
        assert_eq!(w.texts("remove").len(), if dry_run { 48 } else { 0 });
        assert_eq!(w.count("SELECT count(*) AS n FROM a.blob"), before);
    }
    assert!(!stored(&w.root, "removed private message"));
    drop(engine);
    drop(lock);
    fs::remove_file(w.file("shared.jsonl")).unwrap();
    delete_index(&w.root);
    let (mut restored, _lock) = w.engine(64 << 20);
    assert!(restored.converge().unwrap().errors.is_empty());
    assert_eq!(w.texts("keep"), expected);
    assert!(w.texts("remove").is_empty());
}

#[test]
fn input_excision_owns_text_shared_only_by_its_removed_sessions() {
    for (survivor, title_only) in [(false, false), (true, false), (false, true), (true, true)] {
        let shared = "shared synthetic evidence held by removed sessions";
        let message = |id| {
            if title_only {
                format!("separate synthetic message {id}")
            } else {
                shared.to_owned()
            }
        };
        let exporter = ClaudeHome::new();
        for (n, id) in [(1, "first"), (2, "second"), (3, "third")] {
            exporter.append(
                &format!("{id}.jsonl"),
                &user(&format!("u{n}"), id, &message(id)),
            );
        }
        let (mut source, _source_lock) = exporter.engine(64 << 20);
        assert!(source.converge().unwrap().errors.is_empty());
        let export = |id| {
            let mut doc = exporter.export(&format!("claude-code:{id}"));
            if title_only {
                doc.session
                    .offer_title(shared, polylogue_model::TitleSource::Custom);
            }
            doc
        };
        let selected = exporter.dir.path().join("selected.json");
        fs::write(
            &selected,
            polylogue_model::export::Document::new(vec![export("first"), export("second")])
                .to_json()
                .unwrap(),
        )
        .unwrap();
        let w = ClaudeHome::new();
        let (mut engine, _lock) = w.engine(64 << 20);
        engine.import(std::slice::from_ref(&selected)).unwrap();
        if survivor {
            let kept = exporter.dir.path().join("kept.json");
            fs::write(
                &kept,
                polylogue_model::export::Document::new(vec![export("third")])
                    .to_json()
                    .unwrap(),
            )
            .unwrap();
            engine.import(&[kept]).unwrap();
        }
        let input = w
            .reads()
            .reader()
            .unwrap()
            .known_inputs(&engine.config.machine)
            .unwrap()
            .into_iter()
            .find(|input| Path::new(&input.path) == selected)
            .unwrap()
            .input_id;
        engine
            .writer()
            .unwrap()
            .user(UserRequest::now(
                "shared-summary".into(),
                UserWrite::Deliver(Delivery {
                    recipient_kind: "operator".into(),
                    recipient_id: None,
                    sender: "cli".into(),
                    event: "recall".into(),
                    cwd: None,
                    text: shared.into(),
                    refs: vec!["session:claude-code:first".into()],
                    left_out: vec![],
                }),
            ))
            .unwrap();
        for dry_run in [true, false] {
            let done = engine
                .excise(ExciseRequest {
                    id: "remove-selected".into(),
                    scope: ExciseScope::Input(input.clone()),
                    reason: None,
                    dry_run,
                })
                .unwrap();
            assert_eq!(done.sessions.len(), 2);
            assert_eq!(done.messages, 2);
            assert_eq!(
                done.deliveries_redacted,
                u64::from(!survivor),
                "survivor={survivor}, title={title_only}, preview={dry_run}"
            );
            let expected = if dry_run || survivor {
                shared.into()
            } else {
                " ".repeat(shared.len())
            };
            let archive_conn = rusqlite::Connection::open(Archive::at(&w.root).archive_db()).unwrap();
            let hash: Vec<u8> = archive_conn
                .query_row(
                    "SELECT content_hash FROM delivery WHERE id='shared-summary'",
                    [],
                    |r| r.get(0),
                )
                .unwrap();
            assert_eq!(hash, blake3::hash(expected.as_bytes()).as_bytes());
        }
        assert!(w.texts("first").is_empty());
        assert!(w.texts("second").is_empty());
        if survivor {
            assert_eq!(w.texts("third"), [message("third")]);
        }
    }
}
