use super::*;

/// An input holds its parser state from its first range to its last, so a
/// pass starts a new input only when that input's first range can be queued
/// at once, and otherwise finishes the inputs it started. With room for one
/// range in flight, large inputs are read one after another rather than one
/// range of each in turn (docs/architecture.md, pipeline).
#[test]
fn a_pass_finishes_started_inputs_before_starting_more() {
    let w = ClaudeHome::new();
    let filler = "quokka ".repeat(100);
    // Three sessions of different sizes, so the pass orders them largest
    // first; each spans several ranges.
    let sessions = [("large", 1_200), ("middle", 1_000), ("small", 800)];
    for (session, pairs) in sessions {
        let mut lines = String::new();
        for i in 0..pairs {
            lines += &user(&format!("{session}-u{i}"), session, &format!("{filler}{i}"));
            lines += &assistant(&format!("{session}-a{i}"), session, &format!("{filler}{i}"));
        }
        w.append(&format!("{session}.jsonl"), &lines);
    }
    const CHUNK: u64 = 256 << 10;
    let (archive, _lock) = Archive::open_or_create(&w.root).unwrap();
    let mut config = w.config(CHUNK);
    config.workers = 4;
    config.bytes_in_flight = CHUNK;
    let mut engine = Engine::new(config, archive, polylogue_testkit::stopped_attempt()).unwrap();
    let pass = engine.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);

    // Row keys follow the order ranges were read in.
    let reads = w.reads();
    let rows = reads
        .reader()
        .unwrap()
        .query(
            "SELECT s.id AS session, min(m.pk) AS first, max(m.pk) AS last, count(*) AS n \
             FROM message m JOIN session s ON s.pk = m.session_pk GROUP BY s.id ORDER BY first",
            &[],
        )
        .unwrap();
    let read: Vec<(String, u64, u64, u64)> = rows
        .iter()
        .map(|row| {
            (
                row["session"].as_str().unwrap().to_owned(),
                row["first"].as_u64().unwrap(),
                row["last"].as_u64().unwrap(),
                row["n"].as_u64().unwrap(),
            )
        })
        .collect();
    let order: Vec<_> = read.iter().map(|(s, ..)| s.as_str()).collect();
    assert_eq!(
        order,
        [
            "claude-code:large",
            "claude-code:middle",
            "claude-code:small"
        ]
    );
    let counts: Vec<_> = read.iter().map(|(.., n)| *n).collect();
    assert_eq!(counts, [2_400, 2_000, 1_600]);
    for pair in read.windows(2) {
        assert!(
            pair[0].2 < pair[1].1,
            "{} was read between ranges of {}: {read:?}",
            pair[1].0,
            pair[0].0
        );
    }
}
