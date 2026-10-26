//! Live inputs during a long first build: a hook event or capture that
//! arrives while the backlog is being stored goes ahead of it.

use super::*;

#[test]
fn a_spool_file_arriving_mid_pass_is_stored_before_the_backlog_ends() {
    let w = ClaudeHome::new();
    const BACKLOG: i64 = 1500;
    let mut lines = String::new();
    for i in 0..10 {
        lines.push_str(&user(&format!("u{i}"), "SESSION", &"backlog text ".repeat(40)));
        lines.push_str(&assistant(&format!("a{i}"), "SESSION", &"answer text ".repeat(40)));
    }
    for n in 0..BACKLOG {
        w.append(&format!("b{n}.jsonl"), &lines.replace("SESSION", &format!("b{n}")));
    }
    let mut config = w.config(64 << 20);
    config.workers = 1;
    config.live_every = std::time::Duration::ZERO;
    let spool = config.spool_dir(polylogue_sources::hook_spool::ORIGIN);
    let (archive, lock) = Archive::open_or_create(&w.root).unwrap();
    let mut engine = Engine::new(config, archive).unwrap();
    let pass = std::thread::spawn(move || {
        let pass = engine.run_pass();
        drop(lock);
        pass
    });
    let sessions = |w: &ClaudeHome| w.count("SELECT count(*) AS n FROM session WHERE id LIKE 'claude-code:b%'");
    while sessions(&w) == 0 {
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    fs::create_dir_all(&spool).unwrap();
    fs::write(
        spool.join("4242.ndjson"),
        r#"{"at_ns":"1790856000000000000","event":"UserPromptSubmit","harness_pid":4242,"provider":"claude-code","session_id":"live1","payload":{"session_id":"live1","cwd":"/x"}}
"#,
    )
    .unwrap();
    let live = |w: &ClaudeHome| w.count("SELECT count(*) AS n FROM session WHERE id = 'claude-code:live1'");
    while live(&w) == 0 {
        assert!(!pass.is_finished(), "the pass ended before the hook event was stored");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let behind = sessions(&w);
    assert!(
        behind < BACKLOG,
        "the hook event waited for the whole backlog ({behind} of {BACKLOG} sessions stored)"
    );
    let done = pass.join().unwrap().unwrap();
    assert!(done.errors.is_empty(), "{:?}", done.errors);
    assert_eq!(sessions(&w), BACKLOG);
}
