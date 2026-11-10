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
    assert_eq!(
        c.cost_nusd(Some("claude-opus-4-1"), &u),
        Some(110_250_000_000)
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
    // gpt-5 output is $10 per million tokens.
    assert_eq!(c.cost_nusd(Some("gpt-5"), &u), Some(10_000_000_000));
}

#[test]
fn claude_thinking_is_priced_once() {
    // A response of 1M output tokens of which 400k are thinking: the parser
    // stores 600k output and 400k reasoning, billed as 1M at the output rate.
    let c = Catalog::builtin();
    let mut u = usage(0, None, None, 600_000);
    u.reasoning = Some(400_000);
    // claude-opus-4-1 output is $75 per million tokens.
    assert_eq!(
        c.cost_nusd(Some("claude-opus-4-1"), &u),
        Some(75_000_000_000)
    );
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
    // 2M short writes at $5 and 1M one-hour writes at $8.
    assert_eq!(
        c.cost_nusd(Some("claude-opus-5-5"), &u),
        Some(18_000_000_000)
    );
}

#[test]
fn long_prompts_switch_to_the_long_context_rates() {
    let c = Catalog::builtin();
    let short = usage(272_000, None, None, 1_000_000);
    let long = usage(272_001, None, None, 1_000_000);
    let at = |u: &Usage| c.cost_nusd(Some("gpt-6.1-sol"), u);
    // 272k at $2 plus 1M out at $10; then 272 001 at $4 plus 1M out at $15.
    assert_eq!(at(&short), Some(10_544_000_000));
    assert_eq!(at(&long), Some(16_088_004_000));
    // Cached input counts toward the prompt size.
    let mut cached = usage(1_000, Some(300_000), None, 0);
    cached.output = Some(0);
    assert_eq!(at(&cached), Some(64_000_000));
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

fn catalog(models: &str) -> Catalog {
    serde_json::from_str(&format!(r#"{{"version":"test","models":{models}}}"#)).unwrap()
}

#[test]
fn money_round_trips_exactly() {
    // $0.01875 and $1.23457 per million tokens are not whole nano-USD per
    // token; pico-USD holds them exactly, and the cost rounds once.
    let c = catalog(r#"{"m":{"input":0.01875,"output":1.23457,"tiers":{"half":0.5}}}"#);
    let price = c.price("m").unwrap();
    assert_eq!(price.input.pico_usd_per_token(), 18_750);
    assert_eq!(price.output.pico_usd_per_token(), 1_234_570);
    // 1 000 003 input tokens are 18 750 056.25 nano-USD and 7 output tokens
    // 8 641.99; their sum, 18 758 698.24, rounds to 18 758 698.
    let mut u = usage(1_000_003, None, None, 7);
    assert_eq!(c.cost_nusd(Some("m"), &u), Some(18_758_698));
    u.service_tier = Some("half".into());
    // Half of 18 758 698 240 pico-USD is 9 379 349.12 nano-USD.
    assert_eq!(c.cost_nusd(Some("m"), &u), Some(9_379_349));
    // The rates serialize back to the spelling they were read from.
    let text = serde_json::to_string(price).unwrap();
    assert_eq!(
        text,
        r#"{"input":0.01875,"cached_input":null,"cache_write":null,"cache_write_1h":null,"output":1.23457,"long":null,"tiers":{"half":0.5}}"#
    );
}

#[test]
fn rates_finer_than_pico_usd_or_negative_are_refused() {
    for models in [
        r#"{"m":{"input":0.0000001,"output":1}}"#,
        r#"{"m":{"input":-1,"output":1}}"#,
        r#"{"m":{"input":1,"output":1,"tiers":{"x":0.0000001}}}"#,
    ] {
        let text = format!(r#"{{"version":"test","models":{models}}}"#);
        assert!(serde_json::from_str::<Catalog>(&text).is_err(), "{models}");
    }
}

#[test]
fn more_one_hour_writes_than_writes_is_unknown() {
    let c = Catalog::builtin();
    let mut u = usage(0, None, Some(1), 0);
    u.cache_write_1h = Some(2);
    assert_eq!(c.cost_nusd(Some("claude-opus-5-5"), &u), None);
}
