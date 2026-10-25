use super::*;

#[test]
fn an_older_inputs_parse_prefix_does_not_commit_a_late_append_early() {
    let w = ClaudeHome::new();
    let first: String = (0..10)
        .map(|n| user(&format!("u{n}"), "late", "original message"))
        .collect();
    w.append("late.jsonl", &first);
    fs::File::open(w.file("late.jsonl"))
        .unwrap()
        .set_modified(std::time::SystemTime::UNIX_EPOCH)
        .unwrap();
    let (mut e, _lock) = w.engine(64 << 20);
    e.config.chunk = 256;
    assert!(e.run_pass().unwrap().errors.is_empty());
    let later = user("u-later", "late", "late appended message");
    w.append("late.jsonl", &later);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 2);
    assert_eq!(w.count("SELECT count(*) AS n FROM a.observation"), 2);
    // Length is the captured prefix; size is the observed file EOF.
    let captured = w.count("SELECT length AS n FROM a.observation WHERE class='appended'");
    assert!(captured > first.len() as i64 && captured <= (first.len() + later.len()) as i64);
    assert_eq!(
        w.count("SELECT size AS n FROM a.observation WHERE class='appended'"),
        (first.len() + later.len()) as i64
    );
    assert_eq!(
        w.count("SELECT length AS n FROM a.revision WHERE number=2"),
        (first.len() + later.len()) as i64
    );
    assert_eq!(w.texts("late").len(), 11);
}

#[test]
fn empty_live_inputs_keep_observations_and_later_appends_without_empty_extents() {
    let w = ClaudeHome::new();
    w.append("empty.jsonl", "");
    {
        let (mut e, _lock) = w.engine(64 << 20);
        let first = e.run_pass().unwrap();
        assert!(first.errors.is_empty(), "{:?}", first.errors);
        assert_eq!(first.bytes_read, 0);
        assert_eq!(
            w.count("SELECT count(*) AS n FROM a.observation WHERE class='first' AND length=0"),
            1
        );
        assert_eq!(
            w.count("SELECT count(*) AS n FROM a.revision WHERE length=0"),
            1
        );
        assert_eq!(w.count("SELECT count(*) AS n FROM a.extent"), 0);
        assert_eq!(w.count("SELECT count(*) AS n FROM session"), 0);
        assert_eq!(e.run_pass().unwrap().inputs_changed, 0);
        assert_eq!(w.count("SELECT count(*) AS n FROM a.observation"), 1);
    }
    let (mut e, lock) = w.engine(64 << 20);
    assert_eq!(e.run_pass().unwrap().inputs_changed, 0);
    let first = user("u1", "empty", "first actual message");
    w.append("empty.jsonl", &first);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(w.texts("empty"), ["first actual message"]);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.revision WHERE chain=1"),
        2
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.extent WHERE length=0"),
        0
    );
    fs::write(w.file("empty.jsonl"), "").unwrap();
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.observation WHERE class='rewritten' AND length=0"),
        1
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.revision WHERE chain=3 AND length=0"),
        1
    );
    drop(e);
    drop(lock);
    let (mut e, _lock) = w.engine(64 << 20);
    assert_eq!(e.run_pass().unwrap().inputs_changed, 0);
    let later = user("u2", "empty", "message after truncation");
    w.append("empty.jsonl", &later);
    assert!(e.run_pass().unwrap().errors.is_empty());
    assert_eq!(w.count("SELECT count(*) AS n FROM a.revision"), 4);
    assert_eq!(
        w.count("SELECT count(*) AS n FROM a.extent WHERE length=0"),
        0
    );
    let pool = w.reads();
    let reader = pool.reader().unwrap();
    let inputs = reader.known_inputs(&e.config.machine).unwrap();
    assert_eq!(inputs.len(), 1);
    let input = &inputs[0];
    let store = hashed_blobs::Store::open(w.root.clone(), vec![], 0o600).unwrap();
    assert_eq!(
        reader
            .stored_chain(&input.input_id, 1, first.len() as u64)
            .unwrap()
            .read_range(&store, 0, first.len() as u64)
            .unwrap(),
        first.as_bytes()
    );
    assert_eq!(
        reader
            .stored_chain(&input.input_id, 3, later.len() as u64)
            .unwrap()
            .read_range(&store, 0, later.len() as u64)
            .unwrap(),
        later.as_bytes()
    );
}
