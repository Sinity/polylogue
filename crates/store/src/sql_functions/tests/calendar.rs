use super::super::*;
use query_language::time::DaySetting;
fn clock(zone: &str, start: &str) -> Clock {
    Clock::new(
        "2026-03-08T18:00:00Z".parse().unwrap(),
        DaySetting::new(
            jiff::tz::TimeZone::get(zone).unwrap(),
            start.parse().unwrap(),
        )
        .unwrap(),
    )
}
fn connection() -> (Connection, CalendarContext) {
    let conn = Connection::open_in_memory().unwrap();
    let context = CalendarContext::new();
    register_calendar(&conn, &context).unwrap();
    (conn, context)
}
fn labels(conn: &Connection, at: &str) -> Vec<String> {
    conn.query_row(
        "SELECT pl_day(?1),pl_week(?1),pl_month(?1)",
        [at.parse::<jiff::Timestamp>().unwrap().as_millisecond()],
        |row| Ok(vec![row.get(0)?, row.get(1)?, row.get(2)?]),
    )
    .unwrap()
}
#[test]
fn cached_and_nested_sql_uses_the_exact_connection_calendar() {
    let (conn, context) = connection();
    let (other, other_context) = connection();
    let los_angeles = clock("America/Los_Angeles", "06:00:01");
    let warsaw = clock("Europe/Warsaw", "04:00:00");
    let at = "2026-03-08T13:00:00Z";
    context.with_clock(&los_angeles, || {
        assert_eq!(labels(&conn, at), ["2026-03-07", "2026-W10", "2026-03"]);
        other_context.with_clock(&warsaw, || {
            assert_eq!(labels(&other, at), ["2026-03-08", "2026-W10", "2026-03"]);
            assert_eq!(labels(&conn, at)[0], "2026-03-07");
        });
        let mut statement = conn.prepare_cached("SELECT pl_day(?1)").unwrap();
        let ms = at.parse::<jiff::Timestamp>().unwrap().as_millisecond();
        assert_eq!(
            statement
                .query_row([ms], |row| row.get::<_, String>(0))
                .unwrap(),
            "2026-03-07"
        );
        context.with_clock(&warsaw, || {
            assert_eq!(
                statement
                    .query_row([ms], |row| row.get::<_, String>(0))
                    .unwrap(),
                "2026-03-08"
            )
        });
        assert_eq!(
            statement
                .query_row([ms], |row| row.get::<_, String>(0))
                .unwrap(),
            "2026-03-07"
        );
    });
    assert!(
        conn.query_row("SELECT pl_day(0)", [], |row| row.get::<_, String>(0))
            .is_err()
    );
    assert!(
        other
            .query_row("SELECT pl_day(0)", [], |row| row.get::<_, String>(0))
            .is_err()
    );
}
#[test]
fn nested_unwind_restores_the_previous_day_without_zone_lookup() {
    let (conn, context) = connection();
    context.with_clock(&clock("UTC", "00:00:01"), || {
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            context.with_clock(&clock("UTC", "12:00:01"), || panic!("synthetic unwind"))
        }));
        assert!(caught.is_err());
        assert_eq!(labels(&conn, "2026-10-05T06:00:00Z")[0], "2026-10-05");
    });
}
