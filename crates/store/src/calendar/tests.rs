use super::*;
fn day(zone: &str, start: &str) -> DaySetting {
    DaySetting::new(
        jiff::tz::TimeZone::get(zone).unwrap(),
        start.parse().unwrap(),
    )
    .unwrap()
}
#[test]
fn shared_boundaries_preserve_dst_fold_and_fractional_day_start() {
    let at = |text: &str| text.parse::<Timestamp>().unwrap().as_millisecond();
    let los_angeles = day("America/Los_Angeles", "06:00:01.000000001");
    // The day starts at 14:00:01.000000001Z, so its first whole millisecond is .001.
    assert_eq!(
        day_label(&los_angeles, at("2026-03-07T14:00:01.000Z")).unwrap(),
        "2026-03-06"
    );
    assert_eq!(
        day_label(&los_angeles, at("2026-03-07T14:00:01.001Z")).unwrap(),
        "2026-03-07"
    );
    // The 23-hour day after the spring change ends at 13:00:01.000000001Z.
    assert_eq!(
        day_label(&los_angeles, at("2026-03-08T13:00:01.000Z")).unwrap(),
        "2026-03-07"
    );
    assert_eq!(
        day_label(&los_angeles, at("2026-03-08T13:00:01.001Z")).unwrap(),
        "2026-03-08"
    );
    // A start inside the autumn fold is its first occurrence.
    let fold = day("Europe/Warsaw", "02:30:01");
    assert_eq!(
        day_label(&fold, at("2026-10-25T00:30:00.999Z")).unwrap(),
        "2026-10-24"
    );
    assert_eq!(
        day_label(&fold, at("2026-10-25T00:30:01Z")).unwrap(),
        "2026-10-25"
    );
}

#[test]
fn logical_week_and_month_labels_use_the_same_shared_day() {
    let warsaw = day("Europe/Warsaw", "04:00:00");
    let instant = "2026-10-05T23:30:00Z"
        .parse::<Timestamp>()
        .unwrap()
        .as_millisecond();
    assert_eq!(day_label(&warsaw, instant).unwrap(), "2026-10-05");
    assert_eq!(week_label(&warsaw, instant).unwrap(), "2026-W41");
    assert_eq!(month_label(&warsaw, instant).unwrap(), "2026-10");
}

#[test]
fn calendar_sql_daily_series_and_pooled_reads_agree_at_fractional_dst_boundaries() {
    use crate::fixture::{Fixture, MessageFacts, SessionFacts};
    let temp = tempfile::tempdir().unwrap();
    let mut fixture = Fixture::new(temp.path()).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "test:calendar",
            ..Default::default()
        })
        .unwrap();
    for (id, at) in [
        ("before", "2026-03-07T14:00:01.000Z"),
        ("first", "2026-03-07T14:00:01.001Z"),
        ("last", "2026-03-08T13:00:01.000Z"),
        ("after", "2026-03-08T13:00:01.001Z"),
    ] {
        fixture
            .message(
                session,
                MessageFacts {
                    id,
                    role: "user",
                    written_by: "human",
                    prompt_kind: Some("prose"),
                    at: Some(at.parse::<Timestamp>().unwrap().as_millisecond()),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let clock = query_language::time::Clock::new(
        "2026-03-09T00:00:00Z".parse().unwrap(),
        day("America/Los_Angeles", "06:00:01.000000001"),
    );
    let pool = fixture.reads().unwrap();
    {
        let reader = pool.reader().unwrap();
        let series = crate::measure::daily::daily(
            &reader,
            &crate::measure::Selection::default(),
            "prompts",
            &clock.day,
        )
        .unwrap();
        assert_eq!(
            series.days,
            vec![
                ("2026-03-06".into(), 1),
                ("2026-03-07".into(), 2),
                ("2026-03-08".into(), 1)
            ]
        );
        let rhythm = crate::measure::daily::rhythm(
            &reader,
            &crate::measure::Selection::default(),
            &clock.day,
        )
        .unwrap();
        assert_eq!(rhythm.prompts, 4);
        // Friday's logical day, Saturday's clock hour.
        let mut grid = [[0u64; 24]; 7];
        grid[4][6] = 1;
        grid[5][6] = 2;
        grid[6][6] = 1;
        assert_eq!(rhythm.grid, grid);
        reader.with_clock(&clock, || {
            assert_eq!(
                reader
                    .query(
                        "SELECT pl_day(?1) AS day",
                        &[crate::SqlValue::Int(
                            "2026-03-08T13:00:01.000Z"
                                .parse::<Timestamp>()
                                .unwrap()
                                .as_millisecond()
                        )]
                    )
                    .unwrap()[0]["day"],
                "2026-03-07"
            )
        });
    }
    let reader = pool.reader().unwrap();
    let utc = query_language::time::Clock::new(clock.now, day("UTC", "00:00:00"));
    reader.with_clock(&utc, || {
        assert_eq!(
            reader
                .query(
                    "SELECT pl_day(?1) AS day",
                    &[crate::SqlValue::Int(
                        "2026-03-08T13:00:01.000Z"
                            .parse::<Timestamp>()
                            .unwrap()
                            .as_millisecond()
                    )]
                )
                .unwrap()[0]["day"],
            "2026-03-08"
        )
    });
}

#[test]
fn a_skipped_start_uses_the_first_valid_instant_and_a_skipped_date_is_empty() {
    let gap = day("Europe/Warsaw", "02:30:01");
    assert_eq!(
        day_label(
            &gap,
            "2026-03-29T00:59:59.999Z"
                .parse::<Timestamp>()
                .unwrap()
                .as_millisecond()
        )
        .unwrap(),
        "2026-03-28"
    );
    assert_eq!(
        day_label(
            &gap,
            "2026-03-29T01:00:00Z"
                .parse::<Timestamp>()
                .unwrap()
                .as_millisecond()
        )
        .unwrap(),
        "2026-03-29"
    );
    let apia = day("Pacific/Apia", "04:00:00");
    assert_eq!(
        day_label(
            &apia,
            "2011-12-30T12:00:00Z"
                .parse::<Timestamp>()
                .unwrap()
                .as_millisecond()
        )
        .unwrap(),
        "2011-12-29"
    );
}

#[test]
fn calendar_labels_keep_iso_years_before_the_common_era() {
    let day = day("UTC", "00:00:00");
    let at = "-000400-02-29T12:00:00Z"
        .parse::<Timestamp>()
        .unwrap()
        .as_millisecond();
    assert_eq!(day_label(&day, at).unwrap(), "-0400-02-29");
    assert_eq!(month_label(&day, at).unwrap(), "-0400-02");
    // Gregorian weekdays repeat every 400 years: this is the Tuesday of ISO week 9.
    assert_eq!(week_label(&day, at).unwrap(), "-0400-W09");
}

#[test]
fn month_labels_follow_the_logical_day_across_a_clock_change() {
    let warsaw = day("Europe/Warsaw", "04:00:00");
    let at = |text: &str| text.parse::<Timestamp>().unwrap().as_millisecond();
    assert_eq!(
        month_label(&warsaw, at("2026-10-01T01:59:59.999Z")).unwrap(),
        "2026-09"
    );
    assert_eq!(
        month_label(&warsaw, at("2026-10-01T02:00:00Z")).unwrap(),
        "2026-10"
    );
    // 02:30Z on 1 November is 03:30 local, before the 04:00 day start.
    assert_eq!(
        month_label(&warsaw, at("2026-11-01T02:30:00Z")).unwrap(),
        "2026-10"
    );
    assert_eq!(
        month_label(&warsaw, at("2026-11-01T03:00:00Z")).unwrap(),
        "2026-11"
    );
}
