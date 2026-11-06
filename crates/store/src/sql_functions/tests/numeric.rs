use super::*;

#[test]
fn interval_union_keeps_unknown_distinct_from_zero_and_merges_overlaps() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let union: f64 = conn.query_row("WITH intervals(a,b) AS (VALUES (1.,4.),(3.,6.),(5.,5.),(NULL,100.),(9.5,11.)) SELECT pl_covered(a,b) FROM intervals", [], |row|row.get(0)).unwrap();
    assert_eq!(union, 6.5);
    let unknown: Option<f64> = conn
        .query_row("SELECT pl_covered(NULL,3)", [], |row| row.get(0))
        .unwrap();
    assert_eq!(unknown, None);
    let zero: f64 = conn
        .query_row("SELECT pl_covered(3,2)", [], |row| row.get(0))
        .unwrap();
    assert_eq!(zero, 0.);
}

#[test]
fn correlation_and_histogram_keep_their_typed_product_values() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let correlation: Option<f64> = conn.query_row("WITH pairs(x,y) AS (VALUES (1,4),(3,6),(5,5),(NULL,100)) SELECT pl_correlation(x,y) FROM pairs",[],|row|row.get(0)).unwrap();
    assert!((correlation.unwrap() - 0.5).abs() < 1e-12);
    let constant: Option<f64> = conn
        .query_row(
            "WITH pairs(x,y) AS (VALUES (1,2),(1,4)) SELECT pl_correlation(x,y) FROM pairs",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(constant, None);
    let histogram: String = conn.query_row("WITH items(value) AS (VALUES ('1'),('two'),('1'),(NULL)) SELECT pl_histogram(value,'text') FROM items",[],|row|row.get(0)).unwrap();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&histogram).unwrap(),
        serde_json::json!([{"value":"1","count":2},{"value":"two","count":1}])
    );
    assert!(
        conn.query_row("SELECT pl_finite(1e308 * 1e308)", [], |row| row
            .get::<_, f64>(0))
            .is_err()
    );
}

#[test]
fn text_length_counts_unicode_and_embedded_nuls() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let length: i64 = conn
        .query_row("SELECT pl_text_length(?1)", ["a\0é😀"], |row| row.get(0))
        .unwrap();
    assert_eq!(length, 4);
}

#[test]
fn finite_extreme_pairs_do_not_overflow_dimensionless_correlation() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let correlation: f64 = conn.query_row("WITH pairs(x,y) AS (VALUES (-1e308,1e308),(0.,0.),(1e308,-1e308)) SELECT pl_correlation(x,y) FROM pairs",[],|row|row.get(0)).unwrap();
    assert!((correlation + 1.).abs() < 1e-12);
}

#[test]
fn finite_interval_union_does_not_overflow_an_intermediate_sum() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let covered: f64 = conn.query_row("WITH intervals(a,b) AS (VALUES (-1e308,0.),(9e307,1e308)) SELECT pl_covered(a,b) FROM intervals", [], |row| row.get(0)).unwrap();
    assert!((covered / 1e308 - 1.1).abs() < 1e-12);
}

#[test]
fn absolute_values_preserve_unknowns_and_handle_the_minimum_integer() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let values: (Option<f64>, i64, f64, f64) = conn
        .query_row(
            "SELECT pl_abs(NULL), pl_abs(-7), pl_abs(-1.25), pl_abs(-9223372036854775808)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(values, (None, 7, 1.25, 9223372036854775808.0));
}

#[test]
fn plain_text_cast_distinguishes_stored_json_strings_from_encoded_objects() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    for value in ["42", "true", "{\"key\":1}", "\"quoted\"", "a\0é😀"] {
        let actual: String = conn
            .query_row("SELECT pl_query_text(?1, 0)", [value], |row| row.get(0))
            .unwrap();
        assert_eq!(actual, value);
        let encoded = serde_json::to_string(value).unwrap();
        let actual: String = conn
            .query_row("SELECT pl_query_text(?1, 1)", [encoded], |row| row.get(0))
            .unwrap();
        assert_eq!(actual, value);
    }
    for json in ["42", "true", "[]", "{\"key\":1}", "not valid JSON"] {
        assert!(
            conn.query_row("SELECT pl_query_text(?1, 1)", [json], |row| row
                .get::<_, String>(0))
                .is_err(),
            "{json}"
        );
    }
    for json in [0, 1] {
        let unknown: Option<String> = conn
            .query_row("SELECT pl_query_text(NULL, ?1)", [json], |row| row.get(0))
            .unwrap();
        assert_eq!(unknown, None);
    }
    assert!(
        conn.query_row("SELECT pl_query_text(42, 0)", [], |row| row
            .get::<_, String>(0))
            .is_err()
    );
}

#[test]
fn typed_natural_word_and_path_matches_preserve_their_boundaries() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let word = |text: &str, needle: &str| {
        conn.query_row("SELECT pl_query_word(?1,?2)", [text, needle], |row| {
            row.get::<_, bool>(0)
        })
        .unwrap()
    };
    assert!(word("École_read", "école"));
    assert!(word("École_read", "READ"));
    assert!(!word("École_read", "ecole"));
    assert!(!word("École_read", "écol"));
    let under = |path: &str, root: &str| {
        conn.query_row("SELECT pl_query_under(?1,?2)", [path, root], |row| {
            row.get::<_, bool>(0)
        })
        .unwrap()
    };
    assert!(under("/work", "/work/"));
    assert!(under("/work", "/"));
    assert!(!under("relative", "/"));
    assert!(!under("", "/"));
    assert!(under("/work/src", "/work"));
    assert!(!under("/work2/src", "/work"));
    assert!(!under("/Work/src", "/work"));
    let absent: Option<bool> = conn
        .query_row("SELECT pl_query_under(NULL,'/work')", [], |row| row.get(0))
        .unwrap();
    assert_eq!(absent, None);
}

#[test]
fn kept_time_and_native_identity_require_actual_captured_scalar_evidence() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let timestamp: i64 = conn
        .query_row(
            "SELECT pl_kept_time('1970-01-01T00:00:01.125Z')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(timestamp, 1125);
    let missing: Option<i64> = conn
        .query_row("SELECT pl_kept_time(NULL)", [], |row| row.get(0))
        .unwrap();
    assert_eq!(missing, None);
    assert!(
        conn.query_row(
            "SELECT pl_kept_time('1970-01-01T00:00:01.000000001Z')",
            [],
            |row| row.get::<_, i64>(0)
        )
        .is_err()
    );
    let opaque: String = conn
        .query_row(r#"SELECT pl_kept_identity('["001"]',0,1)"#, [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(opaque, "001");
    for (index, count) in [(-1, 1), (0, -1), (1, 1)] {
        assert!(
            conn.query_row(
                "SELECT pl_kept_identity('[1]',?1,?2)",
                [index, count],
                |row| row.get::<_, i64>(0)
            )
            .is_err()
        );
    }
    for value in ["null", "[]", "[null]", "[{}]", "[9223372036854775808]"] {
        assert!(
            conn.query_row("SELECT pl_kept_identity(?1,0,1)", [value], |row| row
                .get::<_, String>(0))
                .is_err(),
            "{value}"
        );
    }
}
