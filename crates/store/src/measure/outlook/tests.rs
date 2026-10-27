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
        project(&spike, 10.0, 30.0, "eom-naive").unwrap().projected,
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
