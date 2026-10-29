use super::*;
fn fixture() -> Clock {
    clock(
        1_791_367_200_000,
        LogicalDay::named("Europe/Warsaw", 240).unwrap(),
    )
    .unwrap()
}
fn ms(text: &str) -> i64 {
    text.parse::<jiff::Timestamp>().unwrap().as_millisecond()
}
#[test]
fn source_time_expressions_use_one_shared_clock() {
    let clock = fixture();
    assert_eq!(clock.now.to_string(), "2026-10-07T10:00:00Z");
    for (text, expected) in [
        ("7d", "2026-09-30T10:00:00Z"),
        ("3h", "2026-10-07T07:00:00Z"),
        ("09:00", "2026-10-07T07:00:00Z"),
        ("2026-10-03T14:30", "2026-10-03T12:30:00Z"),
    ] {
        assert_eq!(
            resolve(text, &clock).unwrap().value.start_ms(),
            ms(expected),
            "{text}"
        );
    }
    let day = resolve("2026-10-05", &clock).unwrap().value;
    assert_eq!(day.start_ms(), ms("2026-10-05T02:00:00Z"));
    assert_eq!(day.end_ms().unwrap(), ms("2026-10-06T02:00:00Z"));
    assert_eq!(
        resolve("today", &clock).unwrap().value,
        resolve("2026-10-07", &clock).unwrap().value
    );
    assert_eq!(
        resolve("this week", &clock).unwrap().value.start_ms(),
        day.start_ms()
    );
    assert!(resolve("soon", &clock).is_err());
}
#[test]
fn native_zone_and_valid_boundaries_are_required_at_the_adapter() {
    assert!(
        clock(
            0,
            LogicalDay {
                zone: jiff::tz::TimeZone::fixed(jiff::tz::Offset::from_seconds(3_600).unwrap()),
                start_minutes: 240
            }
        )
        .is_err()
    );
    for start_minutes in [1500, 15360, u16::MAX] {
        assert!(
            clock(
                0,
                LogicalDay {
                    zone: jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
                    start_minutes
                }
            )
            .is_err()
        );
    }
}
#[test]
fn exact_shared_durations_and_product_money_keep_storage_units() {
    for (text, expected) in [
        ("5m", 300_000),
        ("1.5h", 5_400_000),
        ("250ms", 250),
        ("2w", 1_209_600_000),
    ] {
        assert_eq!(duration_ms(text), Some(expected));
    }
    assert_eq!(duration_ms("5x"), None);
    assert_eq!(duration_ms("0.0000001ms"), None);
    assert_eq!(money_nusd("2.90"), Some(2_900_000_000));
    assert_eq!(money_nusd("$1usd"), Some(1_000_000_000));
}
