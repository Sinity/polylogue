use super::*;

#[test]
fn changed_reading_replaces_its_chains_and_the_next_pass_is_quiet() {
    let w = ClaudeHome::new();
    w.append(
        "main.jsonl",
        &(user("one", "main", "first") + &assistant("two", "main", "second")),
    );
    let (mut engine, lock) = w.engine(64);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    let before = w.export("claude-code:main").messages;
    drop(engine);
    let archive = Archive::at(&w.root);
    let (chains, bytes) = polylogue_store::fixture::older_reading(&archive, "claude-code").unwrap();
    assert_eq!(chains, 1);
    assert!(bytes > 0);
    let config = w.config(64);
    let mut engine = Engine::new(config, archive).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{pass:?}");
    assert!(pass.bytes_read > 0);
    assert_eq!(w.export("claude-code:main").messages, before);
    let quiet = engine.run_pass().unwrap();
    assert!(quiet.errors.is_empty());
    assert_eq!(quiet.bytes_read, 0);
    drop(engine);
    drop(lock);
}
