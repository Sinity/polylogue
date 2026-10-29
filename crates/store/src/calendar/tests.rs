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
    let los_angeles = day("America/Los_Angeles", "06:00:01.000000001");
    let (lower, upper) = range(&los_angeles, "2026-03-07".parse().unwrap()).unwrap();
    assert_eq!(
        lower,
        "2026-03-07T14:00:01.001Z"
            .parse::<Timestamp>()
            .unwrap()
            .as_millisecond()
    );
    assert_eq!(
        upper,
        "2026-03-08T13:00:01.001Z"
            .parse::<Timestamp>()
            .unwrap()
            .as_millisecond()
    );
    assert_eq!(day_label(&los_angeles, lower - 1).unwrap(), "2026-03-06");
    assert_eq!(day_label(&los_angeles, lower).unwrap(), "2026-03-07");
    let fold = day("Europe/Warsaw", "02:30:01");
    assert_eq!(
        start(&fold, "2026-10-25".parse().unwrap()).unwrap(),
        "2026-10-25T00:30:01Z"
            .parse::<Timestamp>()
            .unwrap()
            .as_millisecond()
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
    let bounds = week(&warsaw, "2026-10-08".parse().unwrap()).unwrap();
    assert_eq!(
        bounds,
        (
            "2026-10-05T02:00:00Z"
                .parse::<Timestamp>()
                .unwrap()
                .as_millisecond(),
            "2026-10-12T02:00:00Z"
                .parse::<Timestamp>()
                .unwrap()
                .as_millisecond()
        )
    );
    assert_eq!(
        range(&warsaw, "2026-03-28".parse().unwrap()).unwrap().1
            - range(&warsaw, "2026-03-28".parse().unwrap()).unwrap().0,
        23 * 3_600_000
    );
    assert_eq!(
        range(&warsaw, "2026-10-24".parse().unwrap()).unwrap().1
            - range(&warsaw, "2026-10-24".parse().unwrap()).unwrap().0,
        25 * 3_600_000
    );
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
                ("2026-03-06".into(), 1.0),
                ("2026-03-07".into(), 2.0),
                ("2026-03-08".into(), 1.0)
            ]
        );
        let rhythm = crate::measure::daily::rhythm(
            &reader,
            &crate::measure::Selection::default(),
            &clock.day,
        )
        .unwrap();
        assert_eq!(rhythm.prompts, 4);
        assert_eq!(rhythm.grid[4][6], 1); // Friday's logical day, Saturday's clock hour.
        assert_eq!(rhythm.grid[5][6], 2);
        assert_eq!(rhythm.grid[6][6], 1);
        let calendar =
            crate::timeline::calendar(&reader, &clock, "2026-03-07".parse().unwrap()).unwrap();
        assert_eq!(
            calendar
                .iter()
                .map(|row| (
                    row["date"].as_str().unwrap(),
                    row["records"].as_u64().unwrap()
                ))
                .collect::<Vec<_>>(),
            [("2026-03-06", 1), ("2026-03-07", 2), ("2026-03-08", 1)]
        );
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
    assert!(reader.query("SELECT pl_day(0) AS day", &[]).is_err());
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
        start(&gap, "2026-03-29".parse().unwrap()).unwrap(),
        "2026-03-29T01:00:00Z"
            .parse::<Timestamp>()
            .unwrap()
            .as_millisecond()
    );
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
    let skipped = range(&apia, "2011-12-30".parse().unwrap()).unwrap();
    assert_eq!(skipped.0, skipped.1);
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
