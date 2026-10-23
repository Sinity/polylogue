use polylogue_engine::{ExciseRequest, ExciseScope};
use polylogue_testkit::{ClaudeHome, claude_user as user, delete_index};

#[test]
fn single_input_excision_clears_session_records_without_touching_another_input() {
    let w = ClaudeHome::new();
    w.append(
        "removed.jsonl",
        &user("gone", "removed", "Removed first prompt"),
    );
    w.append("kept.jsonl", &user("kept", "kept", "Retained first prompt"));
    let (mut engine, lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    assert_eq!(w.count("SELECT count(*) AS n FROM session_record"), 2);
    let input = w
        .reads()
        .reader()
        .unwrap()
        .known_inputs(&engine.config.machine)
        .unwrap()
        .into_iter()
        .find(|i| std::path::Path::new(&i.path) == w.file("removed.jsonl"))
        .unwrap()
        .input_id;
    let records = || {
        let rows = w.reads().reader().unwrap().query(
            "SELECT s.id, r.facts, r.chain, r.parser_version FROM session_record r JOIN session s ON s.pk=r.session_pk ORDER BY s.id", &[]).unwrap();
        assert_eq!(
            w.count("SELECT count(*) AS n FROM session_record"),
            rows.len() as i64,
            "orphan session_record rows must not hide behind the join"
        );
        rows
    };
    let before = records();
    assert_eq!(before.len(), 2);
    let kept = before
        .iter()
        .find(|r| r["id"] == "claude-code:kept")
        .unwrap()
        .clone();
    for dry_run in [true, false] {
        engine
            .excise(ExciseRequest {
                id: format!("single-input-{dry_run}"),
                scope: ExciseScope::parse(&format!("input:{input}")).unwrap(),
                reason: Some("synthetic law test".into()),
                dry_run,
            })
            .unwrap();
        if dry_run {
            assert_eq!(records(), before);
        } else {
            assert_eq!(records(), vec![kept.clone()]);
        }
    }
    assert!(engine.converge().unwrap().errors.is_empty());
    assert_eq!(records(), vec![kept.clone()]);
    assert!(w.texts("removed").is_empty());
    assert_eq!(w.texts("kept"), ["Retained first prompt"]);
    drop(engine);
    drop(lock);
    delete_index(&w.root);
    let (mut engine, _lock) = w.engine(64 << 20);
    assert!(engine.converge().unwrap().errors.is_empty());
    assert_eq!(records(), vec![kept]);
    assert!(w.texts("removed").is_empty());
    assert_eq!(w.texts("kept"), ["Retained first prompt"]);
}
