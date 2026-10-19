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
    pub output: f64,
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
        // Input and output are reported for every response; without them
        // the response's size is unknown.
        let (input, output) = (usage.input?, usage.output?);
        let mut usd_e6 = input as f64 * price.input + output as f64 * price.output;
        // Reasoning is billed at the output rate; sources that count it
        // inside output store it in its own lane only.
        usd_e6 += usage.reasoning.unwrap_or(0) as f64 * price.output;
        for (tokens, rate) in [
            (usage.cached_input, price.cached_input),
            (usage.cache_write, price.cache_write),
        ] {
            match (tokens, rate) {
                (None | Some(0), _) => {}
                (Some(t), Some(r)) => usd_e6 += t as f64 * r,
                (Some(_), None) => return None,
            }
        }
        // USD per million tokens times tokens is micro-USD per token-million:
        // 1e-6 USD, which is 1e3 nano-USD.
        Some((usd_e6 * 1e3).round() as i64)
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
}
