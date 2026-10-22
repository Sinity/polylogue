use super::*;

#[test]
fn a_failed_middle_batch_blocks_its_suffix_until_a_committed_progress_retry() {
    let w = world();
    let make = |input, start, id, text| {
        batch(
            input,
            0,
            start,
            vec![row(start, text_message(id, input, Role::User, text, None))],
            &[input],
        )
    };
    w.writer.send(make("input", 0, "first", "first")).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut bad = make("input", 100, "middle", "middle");
    bad.parse[0].accounting.framed = 0;
    w.writer.send(bad).unwrap();
    w.writer.send(make("input", 200, "last", "last")).unwrap();
    w.writer
        .send(make("unrelated", 0, "other", "other"))
        .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(!report.errors.is_empty());
    assert_eq!(
        report.failed_inputs,
        std::collections::BTreeSet::from(["input".into()])
    );
    assert_eq!(texts(&w, "input"), ["first"]);
    assert_eq!(texts(&w, "unrelated"), ["other"]);
    let reader = ReadPool::open(&w.archive).unwrap();
    let r = reader.reader().unwrap();
    let checkpoint = r.known_input_at("m", "/p/input").unwrap().unwrap();
    assert_eq!(checkpoint.parsed_upto, 100);
    drop(r);
    drop(reader);
    w.writer
        .send(make("input", 100, "middle", "middle"))
        .unwrap();
    w.writer.send(make("input", 200, "last", "last")).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert!(report.failed_inputs.is_empty());
    assert_eq!(texts(&w, "input"), ["first", "middle", "last"]);
}
