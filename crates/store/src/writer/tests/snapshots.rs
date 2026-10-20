use super::*;

fn snapshot(input: &str, at: u64, complete: bool, texts: &[(&str, &str)]) -> Batch {
    let rows = texts
        .iter()
        .map(|(id, text)| MessageRow {
            offset: at,
            end: at + 100,
            message: text_message(id, "s", Role::User, text, None),
        })
        .collect();
    let mut b = batch(input, 0, at, rows, &["s"]);
    b.parse[0].snapshot = Some(polylogue_model::Snapshot {
        start: at,
        end: at + 100,
        covered: vec![sid("s")],
        complete,
    });
    b.parse[0].at_end = true;
    b
}
fn read(w: &World) -> Vec<String> {
    let pool = ReadPool::open(&w.archive).unwrap();
    pool.reader()
        .unwrap()
        .messages(&sid("s"), false)
        .unwrap()
        .iter()
        .flat_map(|m| {
            m.blocks
                .iter()
                .filter_map(|b| b.text().map(ToOwned::to_owned))
        })
        .collect()
}

#[test]
fn partial_snapshots_keep_absent_members_and_complete_snapshots_retire_them() {
    let w = world();
    w.writer
        .send(snapshot("a", 0, true, &[("a", "old A"), ("b", "old B")]))
        .unwrap();
    w.writer
        .send(snapshot("a", 100, false, &[("b", "partial B")]))
        .unwrap();
    w.writer.flush().unwrap();
    assert_eq!(read(&w), ["old A", "partial B"]);
    w.writer
        .send(snapshot("a", 200, true, &[("b", "complete B")]))
        .unwrap();
    w.writer.derive().unwrap();
    assert_eq!(read(&w), ["complete B"]);
    let pool = ReadPool::open(&w.archive).unwrap();
    let reader = pool.reader().unwrap();
    let history = reader
        .message_versions(&MessageId::parse("claude-code:a").unwrap())
        .unwrap();
    assert_eq!(history.len(), 1);
    assert!(!history[0].current);
    assert_eq!(reader.status().unwrap().messages, 1);
}

#[test]
fn reversed_snapshot_replay_and_other_source_promotion_match_forward_arrival() {
    for reverse in [false, true] {
        let w = world();
        let mut snapshots = vec![
            snapshot("a", 0, true, &[("a", "old A"), ("b", "old B")]),
            snapshot("a", 100, false, &[("b", "partial B")]),
            snapshot("a", 200, true, &[("b", "complete B")]),
        ];
        // Retained bytes already exist during rebuild; bind every replay to
        // one physical chain without manufacturing new archive revisions.
        let mut seed = batch("a", 0, 0, vec![], &[]);
        seed.length = 300;
        seed.parse.clear();
        w.writer.send(seed).unwrap();
        w.writer.flush().unwrap();
        if reverse {
            snapshots.reverse();
        }
        for mut s in snapshots {
            s.change = Change::Reparse { chain: 1 };
            s.length = 300;
            w.writer.send(s).unwrap();
        }
        w.writer
            .send(batch(
                "b",
                1,
                0,
                vec![MessageRow {
                    offset: 0,
                    end: 100,
                    message: text_message("a", "s", Role::User, "alternate current A", None),
                }],
                &["s"],
            ))
            .unwrap();
        w.writer.derive().unwrap();
        let mut result = read(&w);
        result.sort();
        assert_eq!(result, ["alternate current A", "complete B"]);
    }
}
