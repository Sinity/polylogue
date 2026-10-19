use super::*;

fn usage(input: u64, cached: Option<u64>, write: Option<u64>, output: u64) -> Usage {
    Usage {
        input: Some(input),
        cached_input: cached,
        cache_write: write,
        output: Some(output),
        ..Usage::default()
    }
}

#[test]
fn names_resolve_through_prefixes_dates_and_tags() {
    let c = Catalog::builtin();
    assert!(c.price("claude-opus-4-1").is_some());
    assert_eq!(
        c.price("anthropic/claude-opus-4-1-20250805"),
        c.price("claude-opus-4-1")
    );
    assert_eq!(
        c.price("claude-sonnet-4-5[1m]"),
        c.price("claude-sonnet-4-5")
    );
    assert!(c.price("gpt-5").is_some());
    assert_eq!(strip_date("gpt-5-2025-08-07"), "gpt-5");
    assert_eq!(strip_date("o3"), "o3");
}

#[test]
fn cost_sums_each_lane_at_its_rate() {
    let c = Catalog::builtin();
    // claude-opus-4-1: 15 in, 1.5 cache read, 18.75 cache write, 75 out.
    let u = usage(1_000_000, Some(1_000_000), Some(1_000_000), 1_000_000);
    let usd = 15.0 + 1.5 + 18.75 + 75.0;
    assert_eq!(
        c.cost_nusd(Some("claude-opus-4-1"), &u),
        Some((usd * 1e9) as i64)
    );
}

#[test]
fn unknown_is_never_zero() {
    let c = Catalog::builtin();
    let u = usage(10, None, None, 10);
    assert_eq!(c.cost_nusd(Some("no-such-model"), &u), None);
    assert_eq!(c.cost_nusd(None, &u), None);
    assert_eq!(
        c.cost_nusd(Some("claude-opus-4-1"), &Usage::default()),
        None
    );
    // A response that really used no tokens costs exactly zero.
    assert_eq!(
        c.cost_nusd(Some("claude-opus-4-1"), &usage(0, None, None, 0)),
        Some(0)
    );
}

#[test]
fn reasoning_is_billed_at_the_output_rate() {
    let c = Catalog::builtin();
    let mut u = usage(0, None, None, 0);
    u.reasoning = Some(1_000_000);
    let out = c.price("gpt-5").unwrap().output;
    assert_eq!(c.cost_nusd(Some("gpt-5"), &u), Some((out * 1e9) as i64));
}

#[test]
fn the_models_in_use_are_priced() {
    let c = Catalog::builtin();
    for m in [
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-fable-5-1",
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "gpt-6.1-sol",
        "gpt-6-luna",
    ] {
        assert!(c.price(m).is_some(), "{m} is unpriced");
    }
    // An alias names no version, so no price.
    assert!(c.price("opus").is_none());
}

#[test]
fn one_hour_cache_writes_cost_more_than_short_ones() {
    let c = Catalog::builtin();
    // claude-opus-5-5: 4 in, 5 short write, 8 one-hour write, 20 out.
    let mut u = usage(0, None, Some(3_000_000), 0);
    u.cache_write_1h = Some(1_000_000);
    let usd = 2.0 * 5.0 + 8.0;
    assert_eq!(
        c.cost_nusd(Some("claude-opus-5-5"), &u),
        Some((usd * 1e9) as i64)
    );
}

#[test]
fn long_prompts_switch_to_the_long_context_rates() {
    let c = Catalog::builtin();
    let short = usage(272_000, None, None, 1_000_000);
    let long = usage(272_001, None, None, 1_000_000);
    let at = |u: &Usage| c.cost_nusd(Some("gpt-6.1-sol"), u).unwrap() as f64 / 1e9;
    assert!((at(&short) - (0.272 * 2.0 + 10.0)).abs() < 1e-6);
    assert!((at(&long) - (0.272001 * 4.0 + 15.0)).abs() < 1e-6);
    // Cached input counts toward the prompt size.
    let mut cached = usage(1_000, Some(300_000), None, 0);
    cached.output = Some(0);
    let usd = 0.001 * 4.0 + 0.3 * 0.20;
    assert!((at(&cached) - usd).abs() < 1e-9);
}

#[test]
fn tiers_multiply_and_an_unknown_tier_is_unknown() {
    let c = Catalog::builtin();
    let base = usage(1_000_000, None, None, 0);
    let cost = |tier: Option<&str>, speed: Option<&str>, model: &str| {
        let mut u = base.clone();
        u.service_tier = tier.map(str::to_owned);
        u.speed = speed.map(str::to_owned);
        c.cost_nusd(Some(model), &u)
    };
    let std = cost(None, None, "gpt-6.1-sol").unwrap();
    assert_eq!(cost(Some("default"), None, "gpt-6.1-sol"), Some(std));
    assert_eq!(cost(Some("flex"), None, "gpt-6.1-sol"), Some(std / 2));
    assert_eq!(cost(Some("priority"), None, "gpt-6.1-sol"), Some(std * 2));
    assert_eq!(cost(Some("ultrafast"), None, "gpt-6.1-sol"), Some(std * 6));
    assert_eq!(cost(Some("ultrafast"), None, "gpt-6-luna"), None);
    let opus = cost(Some("standard"), Some("standard"), "claude-opus-5-5").unwrap();
    assert_eq!(
        cost(Some("standard"), Some("fast"), "claude-opus-5-5"),
        Some(opus * 2)
    );
    assert_eq!(cost(None, Some("fast"), "claude-sonnet-5-5"), None);
}
