use super::*;
use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};

fn setup(root: &std::path::Path) -> Fixture {
    let mut f = Fixture::new(root).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:old",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    for (id, at, model) in [
        ("codex:a", Some("2024-03-30T02:59:59Z"), Some("test")),
        ("codex:b", Some("2024-03-30T03:00:00Z"), Some("test")),
        ("codex:c", Some("2024-03-31T01:59:59Z"), Some("test")),
        ("codex:d", Some("2024-03-31T02:00:00Z"), Some("test")),
        ("codex:unknown", None, None),
        ("codex:unknown-time-model", None, Some("test")),
    ] {
        f.message(
            s,
            MessageFacts {
                id,
                at: at.map(|s| s.parse::<jiff::Timestamp>().unwrap().as_millisecond()),
                model,
                role: "assistant",
                written_by: "model",
                cost_nusd: Some(5_000_000),
                ..Default::default()
            },
        )
        .unwrap();
    }
    f
}
fn options<'a>(date: Option<&'a str>, cursor: Option<&'a str>) -> Cost<'a> {
    Cost {
        embeddings: false,
        near: None,
        month: None,
        query: "sessions",
        date,
        model: None,
        undated: false,
        unknown_model: false,
        cursor,
        page_size: Some(1),
    }
}
#[test]
fn period_totals_models_days_and_paged_responses_are_one_population() {
    let temp = tempfile::tempdir().unwrap();
    let f = setup(temp.path());
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let result = cost(
        &r,
        options(Some("2024-03-30"), None),
        &fixture_clock("04:00:00", 0),
    )
    .unwrap();
    assert_eq!(result["usage"]["data"]["known_response_messages"], 2);
    assert_eq!(result["usage"]["data"]["by_model"][0]["messages"], 2);
    assert_eq!(result["days"].as_array().unwrap().len(), 1);
    assert_eq!(result["days"][0]["records"], 2);
    assert_eq!(
        result["usage"]["data"]["api_cost"]["known_usd"],
        result["days"][0]["known_usd"]
    );
    assert_eq!(result["responses"]["rows"][0]["id"], "codex:b");
    let next = result["responses"]["next"].as_str().unwrap();
    let second = cost(
        &r,
        options(Some("2024-03-30"), Some(next)),
        &fixture_clock("04:00:00", 10_000_000),
    )
    .unwrap();
    assert_eq!(second["responses"]["rows"][0]["id"], "codex:c");
    assert!(second["responses"]["next"].is_null());
    assert!(
        cost(
            &r,
            options(Some("2024-03-31"), Some(next)),
            &fixture_clock("04:00:00", 0)
        )
        .is_err()
    );
    let mut undated = options(None, None);
    undated.undated = true;
    undated.unknown_model = true;
    let result = cost(&r, undated, &fixture_clock("04:00:00", 0)).unwrap();
    assert_eq!(result["usage"]["data"]["known_response_messages"], 2);
    assert_eq!(result["responses"]["rows"][0]["id"], "codex:unknown");
}

#[test]
fn owner_day_binds_timeline_continuation_and_leap_days_remain_real() {
    let temp = tempfile::tempdir().unwrap();
    let f = setup(temp.path());
    let pool = f.reads().unwrap();
    let cursor = {
        let reader = pool.reader().unwrap();
        let result = timeline(
            &reader,
            Timeline {
                date: Some("2024-03-30"),
                across_years: false,
                undated: false,
                cursor: None,
                page_size: Some(1),
            },
            &fixture_clock("04:00:00", 0),
        )
        .unwrap();
        result["activity"]["next"].as_str().unwrap().to_owned()
    };
    let reader = pool.reader().unwrap();
    assert!(
        timeline(
            &reader,
            Timeline {
                date: Some("2024-03-30"),
                across_years: false,
                undated: false,
                cursor: Some(&cursor),
                page_size: Some(1)
            },
            &fixture_clock("00:00:00", 0)
        )
        .is_err()
    );
    let leap = timeline(
        &reader,
        Timeline {
            date: Some("2024-02-29"),
            across_years: true,
            undated: false,
            cursor: None,
            page_size: Some(1),
        },
        &fixture_clock("04:00:00", 0),
    )
    .unwrap();
    assert_eq!(leap["calendar"].as_array().unwrap().len(), 29);
    assert!(
        timeline(
            &reader,
            Timeline {
                date: Some("2023-02-29"),
                across_years: true,
                undated: false,
                cursor: None,
                page_size: Some(1)
            },
            &fixture_clock("04:00:00", 0)
        )
        .is_err()
    );
}

#[test]
fn invalid_period_is_refused_before_semantic_selection_runs() {
    let temp = tempfile::tempdir().unwrap();
    let f = setup(temp.path());
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let near = |_: &str| -> Result<std::sync::Arc<polylogue_store::vectors::Similarity>, String> {
        panic!("invalid period must not request semantic work")
    };
    let mut request = options(Some("not-a-date"), None);
    request.query = "sessions similar:synthetic";
    request.embeddings = true;
    request.near = Some(&near);
    assert_eq!(
        cost(&reader, request, &fixture_clock("04:00:00", 0))
            .unwrap_err()
            .kind,
        "bad_request"
    );
}

#[test]
fn all_time_model_drilldown_reaches_every_dated_and_undated_response() {
    let temp = tempfile::tempdir().unwrap();
    let f = setup(temp.path());
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut next = None;
    let mut ids = Vec::new();
    let mut count = None;
    loop {
        let mut request = options(None, next.as_deref());
        request.model = Some("test");
        let result = cost(&reader, request, &fixture_clock("04:00:00", 0)).unwrap();
        assert_eq!(result["unplaced"]["count"], 2);
        count.get_or_insert(
            result["usage"]["data"]["by_model"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["model"] == "test")
                .unwrap()["messages"]
                .as_u64()
                .unwrap(),
        );
        ids.extend(
            result["responses"]["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| row["id"].as_str().unwrap().to_owned()),
        );
        next = result["responses"]["next"].as_str().map(str::to_owned);
        if next.is_none() {
            break;
        }
    }
    assert_eq!(ids.len() as u64, count.unwrap());
    assert_eq!(ids.last().unwrap(), "codex:unknown-time-model");
    ids.sort();
    ids.dedup();
    assert_eq!(ids.len() as u64, count.unwrap());
}

#[test]
fn unrepresentable_price_totals_leave_response_records_readable() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(temp.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:large-cost",
            kind: "main",
            ..Default::default()
        })
        .unwrap();
    let at = "2024-03-30T12:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    for id in ["codex:large-a", "codex:large-b"] {
        f.message(
            s,
            MessageFacts {
                id,
                role: "assistant",
                written_by: "model",
                at: Some(at),
                cost_nusd: Some(9_000_000_000_000_000_000),
                ..Default::default()
            },
        )
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let mut request = options(Some("2024-03-30"), None);
    request.page_size = Some(50);
    let result = cost(&reader, request, &fixture_clock("04:00:00", 0)).unwrap();
    assert_eq!(result["responses"]["rows"].as_array().unwrap().len(), 2);
    assert_eq!(result["usage"]["error"], "store");
    assert!(result["days"].is_null());
    assert_eq!(result["days_error"]["error"], "store");
}

fn fixture_clock(start: &str, now: i64) -> Clock {
    Clock::new(
        jiff::Timestamp::from_millisecond(now).unwrap(),
        query_language::time::DaySetting::new(
            jiff::tz::TimeZone::get("Europe/Warsaw").unwrap(),
            start.parse().unwrap(),
        )
        .unwrap(),
    )
}
