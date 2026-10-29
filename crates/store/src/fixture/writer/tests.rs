use super::*;

#[test]
fn fixture_keeps_each_inputs_stored_bytes_and_committed_parse_position() {
    let mut fixture = WriterArchive::new();
    fixture.write("alpha", prose("alpha", 1, 1_000, "first"));
    fixture.write("beta", prose("beta", 1, 2_000, "other input"));
    fixture.write("alpha", prose("alpha", 2, 3_000, "appended"));
    let pool = ReadPool::open(&fixture.archive).unwrap();
    let reader = pool.reader().unwrap();
    let blobs = reader.blob_reader().unwrap();
    for (input, expected) in [("input-alpha", &b"r\nr\n"[..]), ("input-beta", &b"r\n"[..])] {
        let position = reader.input_progress(input).unwrap().unwrap();
        assert_eq!(position.parsed_upto, expected.len() as u64);
        assert_eq!(position.length, expected.len() as u64);
        let chain = reader
            .stored_chain(input, position.chain, position.length)
            .unwrap();
        assert_eq!(
            chain.read_range(&blobs, 0, position.length).unwrap(),
            expected
        );
    }
    assert_eq!(
        reader
            .messages(&sid("alpha"), false)
            .unwrap()
            .iter()
            .map(Message::text)
            .collect::<Vec<_>>(),
        ["first", "appended"]
    );
    assert_eq!(
        reader.messages(&sid("beta"), false).unwrap()[0].text(),
        "other input"
    );
}
