use super::*;

#[test]
fn counts_beyond_the_sql_integer_range_are_unknown_not_wrapped() {
    let archive = test_archive();
    let mut message = text_message("m", "s", Role::Assistant, "hello", Some("end_turn"));
    message.usage.input = Some(u64::MAX);
    message.usage.output = Some(7);
    message.usage.cache_write = Some(u64::MAX);
    message.usage.cache_write_1h = Some(1);
    archive
        .store_reading(
            batch("usage", Route::Live, 0, vec![row(0, message)], &["s"]),
            Some(&SOURCE_RECORD),
        )
        .unwrap();
    let pool = ReadPool::open(&archive.archive).unwrap();
    let reader = pool.reader().unwrap();
    let rows = reader
        .query(
            "SELECT input_tokens AS input, output_tokens AS output,
                    cache_write_tokens AS written, cache_write_5m_tokens AS five_minutes
             FROM message",
            &[],
        )
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["input"], Json::Null);
    assert_eq!(rows[0]["output"], 7);
    assert_eq!(rows[0]["written"], Json::Null);
    assert_eq!(rows[0]["five_minutes"], Json::Null);
}
