use super::*;
use polylogue_store::fault::{self, Action};

/// A lone large input is read one range at a time. Each range goes to the
/// worker that ran the one before, so the pass holds one worker's memory,
/// not a range's worth with every worker in turn
/// (docs/architecture.md, pipeline).
#[test]
fn the_ranges_of_a_lone_input_run_on_one_worker() {
    let w = ClaudeHome::new();
    let filler = "quokka ".repeat(100);
    let mut lines = String::new();
    for i in 0..2_000 {
        lines += &user(&format!("u{i}"), "long", &format!("{filler}{i}"));
        lines += &assistant(&format!("a{i}"), "long", &format!("{filler}{i}"));
    }
    w.append("long.jsonl", &lines);
    const CHUNK: u64 = 64 << 10;
    let ranges = lines.len() as u64 / CHUNK;
    assert!(ranges >= 20, "{ranges} ranges");

    let (archive, _lock) = Archive::open_or_create(&w.root).unwrap();
    let mut config = w.config(CHUNK);
    config.workers = 8;
    let mut engine = Engine::new(config, archive).unwrap();
    // Upkeep reads the file: a first build reads each chain on a build
    // worker of its own.
    polylogue_testkit::install_empty_index(&mut engine);

    fault::clear();
    fault::arm("parse", "long.jsonl", Action::Record, 0, u32::MAX);
    let pass = engine.run_pass().unwrap();
    let parsed = fault::recorded("parse");
    fault::clear();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(w.texts("long").len(), 4_000);

    assert!(
        parsed.len() as u64 >= ranges,
        "{} ranges parsed",
        parsed.len()
    );
    let worker = &parsed[0].1;
    assert!(worker.starts_with("polylogue-parse-"), "{worker}");
    for (range, (_, thread)) in parsed.iter().enumerate() {
        assert_eq!(thread, worker, "range {range} ran on another worker");
    }
}
