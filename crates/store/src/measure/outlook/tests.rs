use super::*;
use crate::fixture::{Fixture, MessageFacts, SessionFacts};
use serde_json::json;
const START: i64 = 1_777_593_600_000;

fn plan(name: &str, budget: Option<Budget>) -> Plan {
    Plan {
        name: name.into(),
        origin: "codex".into(),
        cycle_anchor_day: Some(1),
        price_nusd: Some(31_000_000_000),
        budget,
    }
}
fn setting_at(f: &mut Fixture, at: i64, plan: &Plan) {
    f.setting("plan.example", &serde_json::to_string(plan).unwrap())
        .unwrap();
    let conn = crate::archive::open_write(&f.archive.archive_db(), 64).unwrap();
    conn.execute("UPDATE setting SET set_at=?1 WHERE id=(SELECT id FROM setting WHERE key='plan.example' ORDER BY commit_seq DESC LIMIT 1)",[at]).unwrap();
}
fn observed(f: &mut Fixture, s: i64, id: &str, at: Option<i64>, cost: Option<i64>) {
    f.message(
        s,
        MessageFacts {
            id,
            role: "assistant",
            written_by: "model",
            at,
            cost_nusd: cost,
            ..Default::default()
        },
    )
    .unwrap();
}
fn projection(values: &[i64], method: &str) -> Projection {
    let mut a = Amounts::default();
    for (i, n) in values.iter().enumerate() {
        a.add(i as i64 * DAY, Some(*n), 1, 0).unwrap();
    }
    project(&a, 0, 10 * DAY, 30 * DAY, method).unwrap()
}

#[test]
fn archive_plan_history_ignores_host_rows_and_future_terms() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting_at(&mut f, START, &plan("first", None));
    setting_at(&mut f, START + 5 * DAY, &plan("host-only", None));
    let conn = crate::archive::open_write(&f.archive.archive_db(), 64).unwrap();
    conn.execute(
        "UPDATE setting SET machine='other-host',value='not a plan' WHERE key='plan.example' AND set_at=?1",
        [START + 5 * DAY],
    )
    .unwrap();
    setting_at(&mut f, START + 10 * DAY, &plan("second", None));
    setting_at(&mut f, START + 20 * DAY, &plan("future", None));
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let answer = outlook(&reader, &Selection::default(), START + 15 * DAY, "example").unwrap();
    assert_eq!(
        answer
            .segments
            .iter()
            .map(|s| (
                s.plan.name.as_str(),
                s.setting_at,
                s.from,
                s.until,
                s.active
            ))
            .collect::<Vec<_>>(),
        vec![
            ("first", START, START, START + 10 * DAY, false),
            (
                "second",
                START + 10 * DAY,
                START + 10 * DAY,
                START + 15 * DAY,
                true
            ),
        ]
    );
    assert_eq!(
        answer.window.as_ref().map(|w| (w.start, w.end)),
        Some((START, START + 31 * DAY))
    );
}

#[test]
fn projections_preserve_history_zero_elapsed_and_recent_spike() {
    assert_eq!(
        project(&Amounts::default(), 0, 0, 30 * DAY, "linear")
            .unwrap()
            .projected,
        None
    );
    let mut zero = Amounts::default();
    zero.add(0, Some(0), 1, 0).unwrap();
    assert_eq!(
        project(&zero, 0, 0, 30 * DAY, "linear").unwrap().projected,
        Some(0)
    );
    assert_eq!(projection(&[100; 10], "linear").projected, Some(3000));
    assert_eq!(
        projection(&[100; 10], "trailing-7d-mean").projected,
        Some(3000)
    );
    let mut spike = [0; 10];
    spike[9] = 700;
    assert_eq!(projection(&spike, "trailing-7d-mean").projected, Some(2700));
    assert_eq!(projection(&spike, "linear").projected, Some(2100));
    let mut a = Amounts::default();
    for (i, n) in spike.iter().enumerate() {
        a.add(i as i64 * DAY, Some(*n), 1, 0).unwrap();
    }
    assert_eq!(
        project_with_tail(&a, 0, 10 * DAY, 30 * DAY, "trailing-7d-mean", 10)
            .unwrap()
            .projected,
        Some(2100)
    );
    assert!(validate_method("eom-naive").is_err());
}

#[test]
fn monthly_window_clamps_anchors_and_excludes_cycle_end() {
    let w = window(START + 10 * DAY, 1).unwrap();
    assert_eq!((w.start, w.end), (START, START + 31 * DAY));
    assert_eq!(window(w.end, 1).unwrap().start, w.end);
    let feb = "2026-02-28T12:00:00Z"
        .parse::<jiff::Timestamp>()
        .unwrap()
        .as_millisecond();
    let w = window(feb, 31).unwrap();
    assert_eq!(
        jiff::Timestamp::from_millisecond(w.start)
            .unwrap()
            .to_string(),
        "2026-02-28T00:00:00Z"
    );
    assert_eq!(
        jiff::Timestamp::from_millisecond(w.end)
            .unwrap()
            .to_string(),
        "2026-03-31T00:00:00Z"
    );
}

#[test]
fn missing_configuration_anchor_and_unobserved_budget_stay_explicit() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        assert_eq!(
            outlook(&r, &Selection::default(), START + DAY, "")
                .unwrap()
                .reason,
            Some("provider_required")
        );
        assert_eq!(
            outlook(&r, &Selection::default(), START + DAY, "example")
                .unwrap()
                .reason,
            Some("plan_not_configured")
        );
    }
    let mut p = plan("pro", None);
    p.cycle_anchor_day = None;
    setting_at(&mut f, START, &p);
    {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        assert_eq!(
            outlook(&r, &Selection::default(), START + DAY, "example")
                .unwrap()
                .reason,
            Some("no_cycle_anchor")
        );
    }
    p.cycle_anchor_day = Some(1);
    p.budget = Some(Budget {
        population: Population::Responses,
        unit: "input_tokens".into(),
        amount: 10,
        overage_rate_nusd_per_unit: None,
    });
    setting_at(&mut f, START, &p);
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + DAY, "example").unwrap();
    let pressure = x.segments[0].budget_pressure.as_ref().unwrap();
    assert_eq!(pressure.used, None);
    assert_eq!(pressure.breach_day, None);
    assert_eq!(pressure.reason, Some("budget_unit_unobserved"));
}

#[test]
fn grouped_billing_keeps_nano_usd_unknowns_and_future_rows_apart() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting_at(&mut f, START, &plan("pro", None));
    let s = f
        .session(SessionFacts {
            id: "codex:billing",
            ..Default::default()
        })
        .unwrap();
    observed(&mut f, s, "a", Some(START), Some(9_007_199_254_740_993));
    observed(&mut f, s, "b", Some(START + 1), Some(1));
    observed(&mut f, s, "unknown", Some(START + 2), None);
    observed(&mut f, s, "future", Some(START + 2 * DAY), Some(99));
    observed(&mut f, s, "untimed", None, Some(99));
    let conn = crate::archive::open_write(&f.archive.index_db(), 64).unwrap();
    conn.execute("UPDATE message SET provider_cost_nusd=0 WHERE id='a'", [])
        .unwrap();
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + DAY, "example").unwrap();
    let seg = &x.segments[0];
    assert_eq!(seg.responses["api_nusd"].used, Some(9_007_199_254_740_994));
    assert_eq!(seg.responses["api_nusd"].unknown_items, 1);
    assert!(seg.responses["api_nusd"].lower_bound);
    assert_eq!(seg.responses["reported_nusd"].used, Some(0));
    assert_eq!(seg.responses["reported_nusd"].unknown_items, 2);
    assert_eq!(x.untimed_responses, 1);
    assert!(seg.recorded_reports.is_empty());
}

#[test]
fn missing_days_lower_coverage_and_zero_budget_never_divides() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    let budget = Budget {
        population: Population::Responses,
        unit: "api_nusd".into(),
        amount: 30,
        overage_rate_nusd_per_unit: Some(2),
    };
    setting_at(&mut f, START, &plan("pro", Some(budget)));
    let s = f
        .session(SessionFacts {
            id: "codex:days",
            ..Default::default()
        })
        .unwrap();
    for day in 0..10 {
        if day != 5 {
            observed(
                &mut f,
                s,
                &format!("m{day}"),
                Some(START + day * DAY),
                Some(4),
            );
        }
    }
    {
        let pool = f.reads().unwrap();
        let r = pool.reader().unwrap();
        let x = outlook(
            &r,
            &Selection::default(),
            START + 10 * DAY,
            "example:trailing-7d-mean",
        )
        .unwrap();
        let seg = &x.segments[0];
        assert_eq!(seg.responses["api_nusd"].used, Some(36));
        assert_eq!(
            seg.responses["api_nusd"].missing_days,
            vec![START + 5 * DAY]
        );
        assert_eq!(seg.responses["api_nusd"].coverage_ratio, Some(0.9));
        let p = seg.budget_pressure.as_ref().unwrap();
        assert_eq!(p.label, "your budget");
        assert_eq!(p.breach_day, Some(START + 8 * DAY));
        assert_eq!(p.actual_overage, Some(6));
        assert!(p.projected_overage_cost_nusd.unwrap() > 0);
    }
    let mut p = plan("pro", None);
    p.budget = Some(Budget {
        population: Population::Responses,
        unit: "api_nusd".into(),
        amount: 0,
        overage_rate_nusd_per_unit: None,
    });
    setting_at(&mut f, START, &p);
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + 10 * DAY, "example").unwrap();
    let p = x.segments[0].budget_pressure.as_ref().unwrap();
    assert_eq!(p.breach_day, None);
    assert_eq!(p.used_ratio, None);
}

#[test]
fn a_plan_change_splits_the_cycle_and_prices_only_its_own_span() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting_at(&mut f, START, &plan("first", None));
    setting_at(&mut f, START + 10 * DAY, &plan("second", None));
    let s = f
        .session(SessionFacts {
            id: "codex:plans",
            ..Default::default()
        })
        .unwrap();
    for (id, day) in [
        ("early", 2),
        ("boundary", 10),
        ("late", 12),
        ("after-now", 16),
    ] {
        observed(&mut f, s, id, Some(START + day * DAY), Some(1));
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + 15 * DAY, "example").unwrap();
    assert_eq!(x.segments.len(), 2);
    assert_eq!(x.segments[0].responses["api_nusd"].used, Some(1));
    assert_eq!(x.segments[1].responses["api_nusd"].used, Some(2));
    assert_eq!(x.segments[0].responses["api_nusd"].projected, Some(1));
    assert_eq!(
        x.segments
            .iter()
            .map(|s| s.subscription_price_share_nusd.unwrap())
            .sum::<i64>(),
        31_000_000_000
    );
    let earlier = outlook(&r, &Selection::default(), START + 5 * DAY, "example").unwrap();
    assert_eq!(earlier.segments.len(), 1);
    assert_eq!(earlier.segments[0].plan.name, "first");
}

#[test]
fn usage_before_a_plan_and_other_origins_never_enter_its_projection() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting_at(&mut f, START + DAY, &plan("pro", None));
    let s = f
        .session(SessionFacts {
            id: "codex:selection",
            ..Default::default()
        })
        .unwrap();
    let other = f
        .session(SessionFacts {
            id: "claude-code:other",
            ..Default::default()
        })
        .unwrap();
    observed(&mut f, s, "before", Some(START), Some(100));
    observed(&mut f, s, "own", Some(START + DAY), Some(1));
    observed(&mut f, other, "other", Some(START + DAY), Some(100));
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + 2 * DAY, "example").unwrap();
    assert_eq!(x.segments[0].responses["api_nusd"].used, Some(1));
    assert_eq!(x.billing_before_setting, "unknown");
    let x = outlook(
        &r,
        &Selection {
            origins: Some(vec!["claude-code".into()]),
            ..Default::default()
        },
        START + 2 * DAY,
        "example",
    )
    .unwrap();
    assert!(x.segments[0].responses.is_empty());
}

#[test]
fn recorded_reports_remain_a_separate_population_with_confirmation_counts() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting_at(&mut f, START, &plan("pro", None));
    let s = f
        .session(SessionFacts {
            id: "codex:reports",
            ..Default::default()
        })
        .unwrap();
    observed(&mut f, s, "response", Some(START), Some(1));
    for (i, confirmed, at) in [
        (0, true, Some(START)),
        (1, false, Some(START)),
        (2, true, None),
        (3, true, Some(START + 2 * DAY)),
    ] {
        f.event(s, "usage_report", at).unwrap();
        let conn = crate::archive::open_write(&f.archive.index_db(), 64).unwrap();
        let value = json!({"report_id":format!("r{i}"),"usage":{"input":100,"cached_input":0,"cache_write":0,"output":1,"reasoning":0},"model":"gpt-6.1-sol","response":null,"measured_completion":confirmed,"measured_at":at.and_then(polylogue_model::Timestamp::from_unix_ms),"input_including_cache":null,"output_including_reasoning":null});
        let packed = crate::encoding::pack_json(&value).unwrap();
        conn.execute(
            "UPDATE event SET data=?1 WHERE pk=(SELECT max(pk) FROM event)",
            [packed],
        )
        .unwrap();
    }
    let pool = f.reads().unwrap();
    let r = pool.reader().unwrap();
    let x = outlook(&r, &Selection::default(), START + DAY, "example").unwrap();
    assert_eq!(x.untimed_reports, 1);
    assert_eq!(x.segments[0].unconfirmed_reports, 1);
    assert_eq!(x.segments[0].responses["api_nusd"].used, Some(1));
    assert_eq!(
        x.segments[0].recorded_reports["input_tokens"].used,
        Some(100)
    );
    assert!(x.segments[0].recorded_reports["input_tokens"].lower_bound);
    assert_eq!(
        x.segments[0].recorded_reports["input_tokens"].known_items,
        1
    );
}

#[test]
fn plan_validation_refuses_ambiguous_and_mismatched_units() {
    let mut p = plan("max", None);
    assert!(validate_plan(&p).is_err());
    p.name = "max-20x".into();
    p.price_nusd = Some(-1);
    assert!(validate_plan(&p).is_err());
    p.price_nusd = None;
    p.budget = Some(Budget {
        population: Population::Responses,
        unit: "invented credits".into(),
        amount: 10,
        overage_rate_nusd_per_unit: None,
    });
    assert!(validate_plan(&p).is_err());
    assert!(
        serde_json::from_value::<Plan>(json!({"name":"pro","origin":"codex","price_usd":20}))
            .is_err()
    );
}

#[test]
fn known_usage_is_a_lower_bound_when_response_boundaries_are_unknown() {
    let dir = tempfile::tempdir().unwrap();
    let mut f = Fixture::new(dir.path()).unwrap();
    setting_at(&mut f, START, &plan("pro", None));
    let s = f
        .session(SessionFacts {
            id: "codex:boundaries",
            ..Default::default()
        })
        .unwrap();
    observed(&mut f, s, "known", Some(START), Some(1));
    observed(&mut f, s, "unknown-boundary", Some(START + 1), Some(100));
    observed(&mut f, s, "untimed-boundary", None, Some(100));
    let conn = crate::archive::open_write(&f.archive.index_db(), 64).unwrap();
    conn.execute(
        "UPDATE message SET attrs_json=?1 WHERE id IN ('unknown-boundary','untimed-boundary')",
        [r#"{"production":{"response_boundary_unknown":true}}"#],
    )
    .unwrap();
    let pool = f.reads().unwrap();
    let reader = pool.reader().unwrap();
    let outlook = outlook(&reader, &Selection::default(), START + DAY, "example").unwrap();
    let segment = &outlook.segments[0];
    assert_eq!(segment.unknown_boundary_items, 1);
    assert_eq!(outlook.untimed_boundary_items, 1);
    assert_eq!(outlook.untimed_responses, 0);
    assert_eq!(segment.responses["api_nusd"].used, Some(1));
    assert_eq!(segment.responses["api_nusd"].unknown_items, 0);
    assert!(segment.responses["api_nusd"].lower_bound);
}
