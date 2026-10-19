use super::*;

#[test]
fn functions_read_compressed_text_and_logical_days() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let packed = zstd::bulk::compress("Łódź café".as_bytes(), 3).unwrap();
    let (text, key, folded): (String, String, String) = conn
        .query_row(
            "SELECT pl_text(?1), pl_key(pl_text(?1)), pl_fold('Łódź')",
            [&packed],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(text, "Łódź café");
    assert_eq!(key, "lodz cafe");
    assert_eq!(folded, "Lódź");
    // 01:30 local in Warsaw on Oct 6 belongs to logical Oct 5.
    let ms = "2026-10-05T23:30:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    let (day, week, month): (String, String, String) = conn
        .query_row(
            "SELECT pl_day(?1, 'Europe/Warsaw', 240), pl_week(?1, 'Europe/Warsaw', 240),
                    pl_month(?1, 'Europe/Warsaw', 240)",
            [ms],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        (day.as_str(), week.as_str(), month.as_str()),
        ("2026-10-05", "2026-W41", "2026-10")
    );
    let (p90, p50, empty): (f64, f64, Option<f64>) = conn
        .query_row(
            "SELECT (SELECT pl_percentile(column1, 90) FROM (VALUES (1),(2),(3),(4),(5),(6),(7),(8),(9),(10),(NULL))),
                    (SELECT pl_percentile(column1, 50) FROM (VALUES (3),(1),(2))),
                    (SELECT pl_percentile(column1, 50) FROM (VALUES (NULL)))",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    assert_eq!((p90, p50, empty), (9.0, 2.0, None));
    let none: Option<String> = conn
        .query_row("SELECT pl_day(NULL, 'UTC', 0)", [], |r| r.get(0))
        .unwrap();
    assert_eq!(none, None);
    assert!(
        conn.query_row("SELECT pl_day(0, 'Nowhere/Else', 0)", [], |r| r
            .get::<_, String>(0))
            .is_err()
    );
}
