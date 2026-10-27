use super::*;
use polylogue_store::fixture::{Fixture, MessageFacts, SessionFacts};

fn setup(root: &std::path::Path) -> Fixture {
    let mut f = Fixture::new(root).unwrap();
    f.setting("day.zone", "Europe/Warsaw").unwrap();
    f.setting("day.logical_start", "04:00").unwrap();
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
    let result = cost(&r, options(Some("2024-03-30"), None), 0).unwrap();
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
    let second = cost(&r, options(Some("2024-03-30"), Some(next)), 0).unwrap();
    assert_eq!(second["responses"]["rows"][0]["id"], "codex:c");
    assert!(second["responses"]["next"].is_null());
    assert!(cost(&r, options(Some("2024-03-31"), Some(next)), 0).is_err());
    let mut undated = options(None, None);
    undated.undated = true;
    undated.unknown_model = true;
    let result = cost(&r, undated, 0).unwrap();
    assert_eq!(result["usage"]["data"]["known_response_messages"], 1);
    assert_eq!(result["responses"]["rows"][0]["id"], "codex:unknown");
}

#[test]
fn logical_day_settings_bind_timeline_continuation_and_leap_days_remain_real() {
    let temp = tempfile::tempdir().unwrap();
    let mut f = setup(temp.path());
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
            0,
        )
        .unwrap();
        result["activity"]["next"].as_str().unwrap().to_owned()
    };
    f.setting("day.logical_start", "00:00").unwrap();
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
            0
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
        0,
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
            0
        )
        .is_err()
    );
}
