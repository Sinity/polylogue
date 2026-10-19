use super::LogicalDay;
use jiff::civil::date;

fn warsaw() -> LogicalDay {
    LogicalDay::named("Europe/Warsaw", LogicalDay::DEFAULT_START_MINUTES).unwrap()
}

fn ms(text: &str) -> i64 {
    text.parse::<jiff::Timestamp>().unwrap().as_millisecond()
}

#[test]
fn work_after_midnight_belongs_to_the_day_before() {
    let d = warsaw();
    // 01:30 local on Oct 6 (CEST, +02:00) is logical Oct 5. Clocks change
    // at 02:00, before the 04:00 start, so the short and long logical
    // days are the Saturdays before the change.
    assert_eq!(d.date_of(ms("2026-10-05T23:30:00Z")), date(2026, 10, 5));
    assert_eq!(d.date_of(ms("2026-10-06T02:00:00Z")), date(2026, 10, 6));
    assert_eq!(d.day_label(ms("2026-10-06T01:59:59Z")), "2026-10-05");
}

#[test]
fn dst_days_are_23_and_25_hours() {
    let d = warsaw();
    let (s, e) = d.day(date(2026, 3, 28));
    assert_eq!(e - s, 23 * 3_600_000);
    let (s, e) = d.day(date(2026, 10, 24));
    assert_eq!(e - s, 25 * 3_600_000);
    let (s, e) = d.day(date(2026, 10, 5));
    assert_eq!(e - s, 24 * 3_600_000);
    assert_eq!(s, ms("2026-10-05T02:00:00Z"));
}

#[test]
fn weeks_start_on_monday_and_months_on_the_first() {
    let d = warsaw();
    let (s, e) = d.week(date(2026, 10, 8));
    assert_eq!(s, d.day(date(2026, 10, 5)).0);
    assert_eq!(e, d.day(date(2026, 10, 12)).0);
    assert_eq!(d.week_label(ms("2026-10-08T12:00:00Z")), "2026-W41");
    let (s, _) = d.month(date(2026, 10, 18));
    assert_eq!(s, d.day(date(2026, 10, 1)).0);
    assert_eq!(d.month_label(ms("2026-10-31T23:00:00Z")), "2026-10");
}
