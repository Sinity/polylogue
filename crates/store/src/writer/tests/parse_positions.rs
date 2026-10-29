use super::*;

#[test]
fn a_failed_middle_batch_blocks_its_suffix_until_a_committed_progress_retry() {
    let w = test_archive();
    let make = |input, start, id, text| {
        batch(
            input,
            0,
            start,
            vec![row(start, text_message(id, input, Role::User, text, None))],
            &[input],
        )
    };
    let mut raw = make("input", 0, "first", "first");
    raw.parse.clear();
    raw.length = 300;
    w.store_reading(raw, Some(&[b'r'; 300])).unwrap();
    let reading = |start, id, text| {
        let mut b = make("input", start, id, text);
        b.change = Change::Reparse { chain: 1 };
        b.observation = None;
        b.length = 300;
        b
    };
    w.store_reading(reading(0, "first", "first"), None).unwrap();
    assert!(w.writer.flush().unwrap().errors.is_empty());
    let mut bad = reading(100, "middle", "middle");
    bad.parse[0].counts.records = 0;
    // Worker validation can refuse a batch before it reaches the writer.
    // The committed parse position remains the retry authority in either case.
    let refused_before_queue = match w.store_reading(bad, None) {
        Ok(_) => false,
        Err(StoreError::Counts(_)) => true,
        Err(error) => panic!("unexpected malformed-count refusal: {error}"),
    };
    match w.store_reading(reading(200, "last", "last"), None) {
        Ok(_) | Err(StoreError::Invalid(_)) => {}
        Err(error) => panic!("unexpected suffix refusal: {error}"),
    }
    w.store_reading(make("unrelated", 0, "other", "other"), Some(&SOURCE_RECORD))
        .unwrap();
    let report = w.writer.flush().unwrap();
    assert!(refused_before_queue || !report.errors.is_empty());
    assert_eq!(texts(&w, "input"), ["first"]);
    assert_eq!(texts(&w, "unrelated"), ["other"]);
    let reader = ReadPool::open(&w.archive).unwrap();
    let r = reader.reader().unwrap();
    let checkpoint = r.known_input_at("m", "/p/input").unwrap().unwrap();
    assert_eq!(checkpoint.parsed_upto, 100);
    drop(r);
    drop(reader);
    w.store_reading(reading(100, "middle", "middle"), None)
        .unwrap();
    w.store_reading(reading(200, "last", "last"), None).unwrap();
    let report = w.writer.flush().unwrap();
    assert!(report.errors.is_empty(), "{report:?}");
    assert_eq!(texts(&w, "input"), ["first", "middle", "last"]);
}
