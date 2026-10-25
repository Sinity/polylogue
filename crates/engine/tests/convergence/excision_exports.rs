use super::*;

#[test]
fn one_conversation_of_a_multi_conversation_export_refuses_and_nothing_changes() {
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
    let state = || {
        let pool = w.reads();
        let reader = pool.reader().unwrap();
        let rows = [
            "SELECT * FROM session ORDER BY id",
            "SELECT * FROM message ORDER BY pk",
            "SELECT * FROM a.extent",
            "SELECT * FROM a.excision",
        ]
        .map(|sql| reader.query(sql, &[]).unwrap());
        let store = hashed_blobs::Store::open(w.root.clone(), vec![], 0o600).unwrap();
        let progress = reader.input_progress(&input).unwrap().unwrap();
        let bytes = reader
            .stored_range(&input, progress.chain, 0, progress.length, &store)
            .unwrap();
        (rows, bytes)
    };
    let before = state();
    assert_eq!(before.0[0].len(), 3);
    assert_eq!(before.1, raw);
    for dry_run in [true, false] {
        let refused = e
            .excise(ExciseRequest {
                id: format!("one-of-three-{dry_run}"),
                scope: ExciseScope::parse("session:chatgpt:c2").unwrap(),
                reason: None,
                dry_run,
            })
            .unwrap_err();
        assert!(
            refused
                .to_string()
                .contains(&format!("excision refused for {input}")),
            "{refused}"
        );
        assert_eq!(state(), before);
    }
    // The whole export can go, with every conversation in it.
    let done = excise(&mut e, &format!("input:{input}"), false);
    assert_eq!(done.sessions, ["chatgpt:c1", "chatgpt:c2", "chatgpt:c3"]);
    assert!(!stored(&w.root, "heron-5521"));
}
