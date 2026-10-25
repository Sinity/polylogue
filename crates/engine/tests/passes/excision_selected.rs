use super::*;
use polylogue_store::{Delivery, UserRequest, UserWrite};

#[test]
fn whole_input_removal_refuses_new_messages_of_its_session_after_restart() {
    for restart in [false, true] {
        let w = ClaudeHome::new();
        w.append("original.jsonl", &user("old", "removed", "original body"));
        let (mut engine, mut lock) = w.engine(64 << 20);
        assert!(engine.run_pass().unwrap().errors.is_empty());
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
        assert!(engine.run_pass().unwrap().errors.is_empty());
        assert!(w.texts("removed").is_empty(), "restart={restart}");
        assert_eq!(w.texts("unrelated"), ["stored independent body"]);
        drop(engine);
        drop(lock);
    }
}

#[test]
fn a_session_named_by_a_delivered_brief_refuses_and_nothing_changes() {
    let w = ClaudeHome::new();
    w.append(
        "delivered.jsonl",
        &user("u1", "delivered", "delivered text"),
    );
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    engine
        .writer()
        .unwrap()
        .user(UserRequest::now(
            "brief".into(),
            UserWrite::Deliver(Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "cli".into(),
                event: "recall".into(),
                cwd: None,
                text: "delivered text".into(),
                refs: vec!["session:claude-code:delivered".into()],
                left_out: vec![],
            }),
        ))
        .unwrap();
    let refused = engine
        .excise(ExciseRequest {
            id: "remove-delivered".into(),
            target: ExciseTarget::parse("session:claude-code:delivered").unwrap(),
            reason: None,
            dry_run: false,
        })
        .unwrap_err();
    assert!(refused.to_string().contains("delivery:brief"), "{refused}");
    assert_eq!(w.texts("delivered"), ["delivered text"]);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.excision"), 0);
}

/// Whether any database file or write-ahead log of the archive holds `text`.
fn in_database_files(root: &Path, text: &str) -> bool {
    std::fs::read_dir(root).unwrap().any(|entry| {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        (name.starts_with("archive.db") || name.starts_with("index.db"))
            && std::fs::read(&path)
                .unwrap()
                .windows(text.len())
                .any(|window| window == text.as_bytes())
    })
}

#[test]
fn an_excised_session_leaves_no_readable_copy_in_database_files() {
    let w = ClaudeHome::new();
    let secret = "zebrafinchsecret7731";
    w.append(
        "removed.jsonl",
        &user("u1", "removed", &format!("the value {secret} is private")),
    );
    w.append("other.jsonl", &user("o1", "other", "another kept message"));
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    assert!(
        in_database_files(&w.root, secret),
        "the text must be readable in a database file before the excision"
    );
    excise(&mut engine, "session:claude-code:removed", false);
    assert!(!in_database_files(&w.root, secret));
    assert_eq!(w.texts("other"), ["another kept message"]);
}
