//! One price catalog for API-equivalent cost. A model the catalog lacks is
//! unpriced, and an unpriced or unmeasured message has no cost: unknown,
//! never zero. Subscription credit is a different quantity and is never
//! derived from these prices.

use crate::Usage;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// List prices in USD per million tokens.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
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
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LongContext {
    /// Prompt tokens (every kind of input together) above which these apply.
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

#[derive(Debug, Deserialize, Serialize)]
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
    /// model is unpriced or a kind of token it carries has no price or was
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
        // inside output store it in its own count only.
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
mod tests;
