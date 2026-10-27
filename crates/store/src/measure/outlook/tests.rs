use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use serde_json::json;
const START: i64 = 1_777_593_600_000; // 2026-05-01 UTC

fn setting(f: &mut Fixture, quota: Option<f64>, unit: Option<&str>, anchor: Option<i8>) {
    f.setting("cost.plans",&json!({"sample":{"origin":"codex","cycle_anchor_day":anchor,"quota_unit":unit,"quota":quota,"price_usd":20.0,"overage_rate_usd_per_unit":0.1}}).to_string()).unwrap();
}
fn observed(f: &mut Fixture, s: i64, day: i64, amount: f64) {
    f.message(
        s,
        MessageFacts {
            id: &format!("m{day}"),
            role: "assistant",
            written_by: "model",
            at: Some(START + day * DAY),
            cost_nusd: Some((amount * 1e9) as i64),
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn projections_preserve_history_zero_elapsed_and_recent_spike() {
    assert_eq!(project(&[], 0.0, 30.0, "linear").unwrap().projected, 0.0);
    assert_eq!(project(&[1.0], 0.0, 30.0, "linear").unwrap().projected, 1.0);
    assert_eq!(
        project(&[100.0; 10], 10.0, 30.0, "linear")
            .unwrap()
            .projected,
        3000.0
    );
    assert_eq!(
        project(&[100.0; 10], 10.0, 30.0, "trailing-7d-mean")
            .unwrap()
            .projected,
        3000.0
    );
    let mut spike = [0.0; 10];
    spike[9] = 700.0;
    assert_eq!(
        project(&spike, 10.0, 30.0, "trailing-7d-mean")
            .unwrap()
            .projected,
        2700.0
    );
    assert_eq!(
        project(&spike, 10.0, 30.0, "linear").unwrap().projected,
        2100.0
    );
}

#[test]
fn monthly_window_uses_utc_and_excludes_cycle_end() {
    let w = window(START + 10 * DAY, 1).unwrap();
    assert_eq!(w.start, START);
    assert_eq!(w.end, START + 31 * DAY);
    assert_eq!(w.elapsed_days, 10.0);
    assert_eq!(window(w.end, 1).unwrap().start, w.end);
    let w = window(START, 31).unwrap();
    assert!(w.start <= START && w.end > START);
}

#[test]
fn missing_plan_anchor_quota_and_unobserved_unit_stay_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let s = f
        .session(SessionFacts {
            id: "codex:cost",
            ..Default::default()
        })
        .unwrap();
    observed(&mut f, s, 0, 40_000.0);
    {
        let p = f.reads().unwrap();
        let r = p.reader().unwrap();
        assert_eq!(
            outlook(&r, &Selection::default(), START + DAY, "sample", "x")
                .unwrap()
                .reason,
            Some("plans_not_configured")
        );
    }
    setting(&mut f, None, None, None);
    {
        let p = f.reads().unwrap();
        let r = p.reader().unwrap();
        assert_eq!(
            outlook(&r, &Selection::default(), START + DAY, "sample", "x")
                .unwrap()
                .reason,
            Some("no_cycle_anchor")
        );
        assert!(outlook(&r, &Selection::default(), START + DAY, "unknown", "x").is_err());
    }
    setting(&mut f, None, None, Some(1));
    {
        let p = f.reads().unwrap();
        let r = p.reader().unwrap();
        let x = outlook(&r, &Selection::default(), START + DAY, "sample", "x").unwrap();
        assert_eq!(x.quota_reason, Some("no_quota_configured"));
        assert_eq!(x.units["api_usd"].used, 40_000.0);
    }
    setting(&mut f, Some(30_000.0), Some("credits"), Some(1));
    let p = f.reads().unwrap();
    let r = p.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + DAY, "sample", "x").unwrap();
    let q = x.quota_pressure.unwrap();
    assert_eq!(q.reason, Some("quota_unit_unobserved"));
    assert_eq!(q.used, None);
    assert_eq!(q.breach_day, None);
    assert_eq!(q.projected_overage_cost_usd, None);
}

#[test]
fn missing_day_lowers_coverage_and_breach_uses_own_unit() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting(&mut f, Some(30.0), Some("api_usd"), Some(1));
    let s = f
        .session(SessionFacts {
            id: "codex:cost",
            ..Default::default()
        })
        .unwrap();
    for day in 0..10 {
        if day != 5 {
            observed(&mut f, s, day, 4.0);
        }
    }
    observed(&mut f, s, 31, 100_000.0);
    let p = f.reads().unwrap();
    let r = p.reader().unwrap();
    let x = outlook(
        &r,
        &Selection::default(),
        START + 10 * DAY,
        "sample:trailing-7d-mean",
        "x",
    )
    .unwrap();
    assert_eq!(x.incomplete_days, vec![START + 5 * DAY]);
    assert_eq!(x.coverage_ratio, Some(0.9));
    assert_eq!(x.units["api_usd"].used, 36.0);
    let q = x.quota_pressure.unwrap();
    assert_eq!(q.breach_day, Some(START + 8 * DAY));
    assert_eq!(q.actual_overage, Some(6.0));
    assert!(q.projected_overage_cost_usd.unwrap() > 0.0);
    assert!(x.incomplete_usage); // Tool-reported money was not reported.
}

#[test]
fn zero_usage_is_unknown_until_observed_zero_and_projection_can_breach() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting(&mut f, Some(20.0), Some("api_usd"), Some(1));
    let s = f
        .session(SessionFacts {
            id: "codex:cost",
            ..Default::default()
        })
        .unwrap();
    {
        let p = f.reads().unwrap();
        let r = p.reader().unwrap();
        let x = outlook(&r, &Selection::default(), START + 10 * DAY, "sample", "x").unwrap();
        assert!(x.units.is_empty());
        assert_eq!(x.quota_pressure.unwrap().used, None);
        assert_eq!(x.coverage_ratio, Some(0.0));
    }
    observed(&mut f, s, 0, 0.0);
    {
        let p = f.reads().unwrap();
        let r = p.reader().unwrap();
        let x = outlook(&r, &Selection::default(), START + DAY, "sample", "x").unwrap();
        assert_eq!(x.units["api_usd"].used, 0.0);
    }
    for day in 1..10 {
        observed(&mut f, s, day, 1.0);
    }
    let p = f.reads().unwrap();
    let r = p.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + 10 * DAY, "sample", "x").unwrap();
    let q = x.quota_pressure.unwrap();
    assert!(q.projected.unwrap() > 20.0);
    assert!(q.breach_day.is_some());
    assert_eq!(q.actual_overage, Some(0.0));
}

#[test]
fn units_remain_separate_and_user_only_days_do_not_prove_usage_coverage() {
    let mut fixture = crate::fixture::writer::WriterArchive::new();
    let mut message = crate::fixture::writer::message(
        "a",
        "units",
        polylogue_model::Role::Assistant,
        START,
        vec![polylogue_model::Block::Text {
            text: "Reply".into(),
        }],
    );
    message.model = Some("gpt-6.1-sol".into());
    message.usage = polylogue_model::Usage {
        input: Some(100),
        output: Some(20),
        cached_input: Some(0),
        cache_write: Some(0),
        reasoning: Some(0),
        ..Default::default()
    };
    message.reported_cost_usd = Some(serde_json::Number::from(7));
    fixture.write(
        "units",
        vec![
            message,
            crate::fixture::writer::message(
                "u",
                "units",
                polylogue_model::Role::User,
                START + DAY,
                vec![polylogue_model::Block::Text {
                    text: "Prompt only".into(),
                }],
            ),
        ],
    );
    fixture.writer.derive().unwrap();
    let pool = crate::ReadPool::open(&fixture.archive).unwrap();
    let r = pool.reader().unwrap();
    let report = super::super::usage(&r, &Selection::default()).unwrap();
    let mut units = std::collections::BTreeMap::new();
    units.insert("reported_usd", report.reported_cost.known_usd.unwrap());
    units.insert("api_usd", report.api_cost.known_usd.unwrap());
    units.insert("input_tokens", report.tokens.input.tokens as f64);
    units.insert("output_tokens", report.tokens.output.tokens as f64);
    assert_eq!(units["reported_usd"], 7.0);
    assert_eq!(units["input_tokens"], 100.0);
    assert_eq!(units["output_tokens"], 20.0);
    assert!(units["api_usd"] > 0.0 && units["api_usd"] < 7.0);
    drop(r);
    drop(pool);
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting(&mut f, None, None, Some(1));
    let s = f
        .session(SessionFacts {
            id: "codex:user",
            ..Default::default()
        })
        .unwrap();
    f.message(
        s,
        MessageFacts {
            id: "u",
            role: "user",
            written_by: "human",
            at: Some(START),
            ..Default::default()
        },
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + DAY, "sample", "x").unwrap();
    assert_eq!(x.coverage_ratio, Some(0.0));
}

#[test]
fn zero_quota_has_no_breach_and_removed_method_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting(&mut f, Some(0.0), Some("api_usd"), Some(1));
    let pk = f
        .session(SessionFacts {
            id: "codex:zero",
            ..Default::default()
        })
        .unwrap();
    observed(&mut f, pk, 0, 1.0);
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let x = outlook(&reader, &Selection::default(), START + DAY, "sample", "x").unwrap();
    assert_eq!(x.quota_pressure.unwrap().breach_day, None);
    assert!(project(&[1.0], 1.0, 31.0, "eom-naive").is_err());
}

#[test]
fn grouped_units_preserve_zero_and_untimed_count_uses_response_rows() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting(&mut f, None, None, Some(1));
    let pk = f
        .session(SessionFacts {
            id: "codex:grouped",
            ..Default::default()
        })
        .unwrap();
    let timed = f
        .message(
            pk,
            MessageFacts {
                id: "timed",
                role: "assistant",
                written_by: "model",
                at: Some(START),
                cost_nusd: Some(7_000_000_000),
                ..Default::default()
            },
        )
        .unwrap();
    f.usage(
        timed,
        &polylogue_model::Usage {
            input: Some(100),
            cached_input: Some(0),
            cache_write: Some(0),
            output: Some(20),
            reasoning: Some(0),
            ..Default::default()
        },
        None,
    )
    .unwrap();
    for (id, author) in [
        ("unknown-boundaries", "model"),
        ("untimed-response", "model"),
        ("harness", "harness"),
    ] {
        f.message(
            pk,
            MessageFacts {
                id,
                role: "assistant",
                written_by: author,
                ..Default::default()
            },
        )
        .unwrap();
    }
    let conn = crate::archive::open_write(&f.archive.index_db(), 64).unwrap();
    conn.execute(
        "UPDATE message SET attrs_json=?1 WHERE id='unknown-boundaries'",
        [r#"{"production":{"response_boundary_unknown":true}}"#],
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let result = outlook(&reader, &Selection::default(), START + DAY, "sample", "x").unwrap();
    assert_eq!(result.untimed_responses, 1);
    assert_eq!(result.units["api_usd"].used, 7.0);
    assert_eq!(result.units["input_tokens"].used, 100.0);
    assert_eq!(result.units["cached_input_tokens"].used, 0.0);
    assert_eq!(result.units["output_tokens"].used, 20.0);
}
