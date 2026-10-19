use super::*;

fn clock() -> Clock {
    // 2026-10-07 12:00 in Warsaw (CEST, +02:00).
    let now = "2026-10-07T10:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    Clock::new(
        now,
        LogicalDay::named("Europe/Warsaw", LogicalDay::DEFAULT_START_MINUTES).unwrap(),
    )
}

fn ms(text: &str) -> i64 {
    text.parse::<jiff::Timestamp>().unwrap().as_millisecond()
}

#[test]
fn relative_times_count_exact_hours_back() {
    let c = clock();
    assert_eq!(
        resolve("7d", &c),
        Some(Time::Instant {
            ms: c.now - 168 * HOUR
        })
    );
    assert_eq!(
        resolve("3h", &c),
        Some(Time::Instant {
            ms: c.now - 3 * HOUR
        })
    );
}

#[test]
fn dates_are_logical_days() {
    let c = clock();
    assert_eq!(
        resolve("2026-10-05", &c),
        Some(Time::Span {
            start: ms("2026-10-05T02:00:00Z"),
            end: ms("2026-10-06T02:00:00Z")
        })
    );
    assert_eq!(resolve("today", &c), resolve("2026-10-07", &c));
    assert_eq!(resolve("yesterday", &c), resolve("2026-10-06", &c));
    assert_eq!(
        resolve("this week", &c).map(Time::start),
        resolve("2026-10-05", &c).map(Time::start)
    );
    assert_eq!(
        resolve("last week", &c).map(Time::start),
        resolve("2026-09-28", &c).map(Time::start)
    );
    assert_eq!(
        resolve("2026-10", &c).map(Time::start),
        resolve("2026-10-01", &c).map(Time::start)
    );
}

#[test]
fn times_of_day_and_datetimes() {
    let c = clock();
    assert_eq!(
        resolve("09:00", &c),
        Some(Time::Instant {
            ms: ms("2026-10-07T07:00:00Z")
        })
    );
    assert_eq!(
        resolve("2026-10-03T14:30", &c),
        Some(Time::Instant {
            ms: ms("2026-10-03T12:30:00Z")
        })
    );
    assert_eq!(
        resolve("2026-10-03T14:30:00Z", &c),
        Some(Time::Instant {
            ms: ms("2026-10-03T14:30:00Z")
        })
    );
    assert_eq!(resolve("soon", &c), None);
}

#[test]
fn durations_and_money() {
    assert_eq!(duration_ms("5m"), Some(300_000));
    assert_eq!(duration_ms("1.5h"), Some(5_400_000));
    assert_eq!(duration_ms("250ms"), Some(250));
    assert_eq!(duration_ms("2w"), Some(1_209_600_000));
    assert_eq!(duration_ms("5x"), None);
    assert_eq!(money_nusd("2.90"), Some(2_900_000_000));
    assert_eq!(money_nusd("$1usd"), Some(1_000_000_000));
}
