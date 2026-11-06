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
