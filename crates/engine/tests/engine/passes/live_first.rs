//! Live inputs during a long first build: a hook event or capture that
//! arrives while the backlog is being stored goes ahead of it.

use super::*;

/// The pass is held before parsing its 21st backlog file while the hook event
/// arrives, so the event arrives mid-pass on any runner. On release the pass
/// sees it at its next live check and puts it ahead of the waiting backlog:
/// it is stored after at most the 16 files already in the pool and one
/// commit group (about 100 of these files under the 1 MiB in-flight limit),
/// well before the 200th backlog session.
#[cfg(feature = "faults")]
#[test]
fn a_spool_file_arriving_mid_pass_is_stored_before_the_backlog_ends() {
    use polylogue_store::fault::{self, Action};
    let w = ClaudeHome::new();
    const BACKLOG: i64 = 200;
    const HELD_AT: u32 = 20;
    let mut lines = String::new();
    for i in 0..10 {
        lines.push_str(&user(
            &format!("u{i}"),
            "SESSION",
            &"backlog text ".repeat(40),
        ));
        lines.push_str(&assistant(
            &format!("a{i}"),
            "SESSION",
            &"answer text ".repeat(40),
        ));
    }
    for n in 0..BACKLOG {
        w.append(
            &format!("b{n}.jsonl"),
            &lines.replace("SESSION", &format!("b{n}")),
        );
    }
    let mut config = w.config(64 << 20);
    config.workers = 1;
    config.live_every = std::time::Duration::ZERO;
    let spool = config.spool_dir(polylogue_sources::hook_spool::ORIGIN);
    // Backlog files end in `.jsonl`; the hook spool file does not.
    fault::arm("parse", ".jsonl", Action::Hold, HELD_AT, 1);
    let (archive, lock) = Archive::open_or_create(&w.root).unwrap();
    let mut engine = Engine::new(config, archive, polylogue_testkit::stopped_attempt()).unwrap();
    let pass = std::thread::spawn(move || {
        let pass = engine.run_pass();
        drop(lock);
        pass
    });
    let sessions = |w: &ClaudeHome| {
        w.count("SELECT count(*) AS n FROM session WHERE id LIKE 'claude-code:b%'")
    };
    while !fault::held() {
        assert!(
            !pass.is_finished(),
            "the pass ended without reaching backlog file {HELD_AT}"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(sessions(&w) <= i64::from(HELD_AT));
    fs::create_dir_all(&spool).unwrap();
    fs::write(
        spool.join("4242.ndjson"),
        r#"{"at_ns":"1790856000000000000","event":"UserPromptSubmit","harness_pid":4242,"provider":"claude-code","session_id":"live1","payload":{"session_id":"live1","cwd":"/x"}}
"#,
    )
    .unwrap();
    fault::release();
    let live = |w: &ClaudeHome| {
        w.count("SELECT count(*) AS n FROM session WHERE id = 'claude-code:live1'")
    };
    while live(&w) == 0 {
        assert!(
            !pass.is_finished(),
            "the pass ended before the hook event was stored"
        );
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
