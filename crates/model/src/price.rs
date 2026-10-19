//! One price catalog for API-equivalent cost. A model the catalog lacks is
//! unpriced, and an unpriced or unmeasured message has no cost: unknown,
//! never zero. Subscription credit is a different quantity and is never
//! derived from these prices.

use crate::Usage;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// List prices in USD per million tokens.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Price {
    pub input: f64,
    #[serde(default)]
    pub cached_input: Option<f64>,
    #[serde(default)]
    pub cache_write: Option<f64>,
    /// Cache writes with a one-hour lifetime, where the vendor prices them
    /// apart from the default writes.
    #[serde(default)]
    pub cache_write_1h: Option<f64>,
    pub output: f64,
    /// Rates for a response whose prompt is longer than `above` tokens.
    #[serde(default)]
    pub long: Option<LongContext>,
    /// Multipliers for service tiers other than the standard one, by the
    /// name the tool records (`fast`, `flex`, `priority`, ...). A tier the
    /// model has no multiplier for leaves the cost unknown.
    #[serde(default)]
    pub tiers: BTreeMap<String, f64>,
}

/// The long-context rates of a model whose price rises with prompt size.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LongContext {
    /// Prompt tokens (every input lane together) above which these apply.
    pub above: u64,
    pub input: f64,
    #[serde(default)]
    pub cached_input: Option<f64>,
    #[serde(default)]
    pub cache_write: Option<f64>,
    #[serde(default)]
    pub cache_write_1h: Option<f64>,
    pub output: f64,
}

/// Tier names that mean the standard rates.
const STANDARD_TIERS: [&str; 4] = ["standard", "default", "auto", "on_demand"];

impl Price {
    /// The multiplier for a recorded service tier, or `None` when the tier
    /// is not standard and the model has no rate for it.
    fn tier_multiplier(&self, tier: Option<&str>) -> Option<f64> {
        match tier {
            None => Some(1.0),
            Some(t) if STANDARD_TIERS.contains(&t) => Some(1.0),
            Some(t) => self.tiers.get(t).copied(),
        }
    }

    /// The multiplier for a response's service tier and speed together.
    fn multiplier(&self, usage: &Usage) -> Option<f64> {
        Some(
            self.tier_multiplier(usage.service_tier.as_deref())?
                * self.tier_multiplier(usage.speed.as_deref())?,
        )
    }
}

#[derive(Debug, Deserialize)]
pub struct Catalog {
    /// Changes whenever a price changes; stored costs computed under another
    /// version are recomputed.
    pub version: String,
    models: BTreeMap<String, Price>,
}

impl Catalog {
    /// The catalog compiled into this build.
    pub fn builtin() -> &'static Catalog {
        static CATALOG: OnceLock<Catalog> = OnceLock::new();
        CATALOG.get_or_init(|| {
            serde_json::from_str(include_str!("prices.json")).expect("prices.json is valid")
        })
    }

    /// The price for a model as a tool names it. Tools add vendor prefixes,
    /// release dates and context-size tags to the catalog's names; those are
    /// removed in turn until a name matches.
    pub fn price(&self, model: &str) -> Option<&Price> {
        let mut name = model.trim();
        if let Some(p) = self.models.get(name) {
            return Some(p);
        }
        if let Some((_, rest)) = name.rsplit_once('/') {
            name = rest;
        }
        let name = name.split('[').next().unwrap_or(name).trim();
        if let Some(p) = self.models.get(name) {
            return Some(p);
        }
        self.models.get(strip_date(name))
    }

    /// API-equivalent cost of one response in nano-USD, or `None` when the
    /// model is unpriced or a lane that carries tokens has no price or was
    /// not reported.
    pub fn cost_nusd(&self, model: Option<&str>, usage: &Usage) -> Option<i64> {
        let price = self.price(model?)?;
        let multiplier = price.multiplier(usage)?;
        // Input and output are reported for every response; without them
        // the response's size is unknown.
        let (input, output) = (usage.input?, usage.output?);
        let long = price
            .long
            .as_ref()
            .filter(|l| usage.total_input().is_some_and(|t| t > l.above));
        let (in_rate, cached_rate, write_rate, write_1h_rate, out_rate) = match long {
            Some(l) => (
                l.input,
                l.cached_input,
                l.cache_write,
                l.cache_write_1h,
                l.output,
            ),
            None => (
                price.input,
                price.cached_input,
                price.cache_write,
                price.cache_write_1h,
                price.output,
            ),
        };
        // One-hour writes are part of `cache_write`; the rest are written
        // with the default lifetime.
        let write_1h = usage.cache_write_1h.filter(|_| usage.cache_write.is_some());
        let write_short = usage
            .cache_write
            .map(|w| w.saturating_sub(write_1h.unwrap_or(0)));
        let mut usd_e6 = input as f64 * in_rate + output as f64 * out_rate;
        // Reasoning is billed at the output rate; sources that count it
        // inside output store it in its own lane only.
        usd_e6 += usage.reasoning.unwrap_or(0) as f64 * out_rate;
        for (tokens, rate) in [
            (usage.cached_input, cached_rate),
            (write_short, write_rate),
            (write_1h, write_1h_rate),
        ] {
            match (tokens, rate) {
                (None | Some(0), _) => {}
                (Some(t), Some(r)) => usd_e6 += t as f64 * r,
                (Some(_), None) => return None,
            }
        }
        // USD per million tokens times tokens is micro-USD per token-million:
        // 1e-6 USD, which is 1e3 nano-USD.
        Some((usd_e6 * multiplier * 1e3).round() as i64)
    }
}

/// `claude-opus-4-1-20250805` → `claude-opus-4-1`, `gpt-5-2025-08-07` →
/// `gpt-5`.
fn strip_date(name: &str) -> &str {
    let b = name.as_bytes();
    let digits = |s: &[u8]| s.iter().all(u8::is_ascii_digit);
    if b.len() > 9 && b[b.len() - 9] == b'-' && digits(&b[b.len() - 8..]) {
        return &name[..name.len() - 9];
    }
    if b.len() > 11
        && b[b.len() - 11] == b'-'
        && digits(&b[b.len() - 10..b.len() - 6])
        && b[b.len() - 6] == b'-'
        && b[b.len() - 3] == b'-'
    {
        return &name[..name.len() - 11];
    }
    name
}

#[cfg(test)]
mod tests {
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
}
