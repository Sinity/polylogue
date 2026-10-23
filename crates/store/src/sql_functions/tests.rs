use super::*;

fn random_value(conn: &Connection, key: &str, seed: &str) -> i64 {
    conn.query_row("SELECT pl_random(?1, ?2)", [key, seed], |r| r.get(0))
        .unwrap()
}

#[test]
fn seeded_random_values_repeat_across_calls_and_connections() {
    let first = Connection::open_in_memory().unwrap();
    let second = Connection::open_in_memory().unwrap();
    register(&first).unwrap();
    register(&second).unwrap();
    for key in ["", "session:abc", "é😀", "9007199254740993"] {
        for seed in ["0", "7", "18446744073709551615"] {
            let value = random_value(&first, key, seed);
            assert_eq!(value, random_value(&first, key, seed));
            assert_eq!(value, random_value(&second, key, seed));
        }
    }
}

#[test]
fn seeded_random_keeps_opaque_key_spellings_and_large_integer_ids_distinct() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    let keys = [
        "1",
        "01",
        "1.0",
        "1e0",
        "+1",
        "9007199254740992",
        "9007199254740993",
        "9223372036854775807",
        "9223372036854775808",
        "18446744073709551615",
        "18446744073709551616",
        "session:1",
        "session:1\0suffix",
    ];
    let values: std::collections::BTreeSet<_> = keys
        .iter()
        .map(|key| random_value(&conn, key, "7"))
        .collect();
    assert_eq!(values.len(), keys.len());
}

#[test]
fn seeded_random_numeric_row_keys_use_the_same_decimal_text_identity() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    for key in [i64::MIN, -1, 0, 1, 9007199254740993, i64::MAX] {
        let numeric: i64 = conn
            .query_row(
                "SELECT pl_random(CAST(?1 AS TEXT), '18446744073709551615')",
                [key],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            numeric,
            random_value(&conn, &key.to_string(), "18446744073709551615")
        );
    }
}

#[test]
fn seeded_random_order_depends_on_seed_and_keys_instead_of_insertion_order() {
    let keys: Vec<_> = (0..32).map(|n| format!("session:{n:02}")).collect();
    let make_conn = |reverse: bool| {
        let conn = Connection::open_in_memory().unwrap();
        register(&conn).unwrap();
        conn.execute_batch("CREATE TABLE item(key TEXT NOT NULL)")
            .unwrap();
        for n in 0..keys.len() {
            let n = if reverse { keys.len() - 1 - n } else { n };
            conn.execute("INSERT INTO item(key) VALUES (?1)", [&keys[n]])
                .unwrap();
        }
        conn
    };
    let order = |conn: &Connection, seed: &str| {
        conn.prepare("SELECT key FROM item ORDER BY pl_random(key, ?1), key")
            .unwrap()
            .query_map([seed], |r| r.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    };
    let forward = make_conn(false);
    let reverse = make_conn(true);
    let first = order(&forward, "7");
    let changed = order(&forward, "8");
    assert_eq!(first, order(&forward, "7"));
    assert_eq!(first, order(&reverse, "7"));
    assert_eq!(changed, order(&reverse, "8"));
    assert_ne!(first, changed);
    assert_ne!(first, keys);
    assert_ne!(changed, keys);
}

#[test]
fn seeded_random_refuses_seeds_outside_unsigned_decimal_range() {
    let conn = Connection::open_in_memory().unwrap();
    register(&conn).unwrap();
    for seed in ["", "-1", "1.5", "18446744073709551616", "seed"] {
        assert!(
            conn.query_row("SELECT pl_random('session:1', ?1)", [seed], |r| {
                r.get::<_, i64>(0)
            })
            .is_err(),
            "seed {seed:?}"
        );
    }
}

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
