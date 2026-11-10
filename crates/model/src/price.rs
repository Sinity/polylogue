//! One price catalog for API-equivalent cost. A model the catalog lacks is
//! unpriced, and an unpriced or unmeasured message has no cost: unknown,
//! never zero. Subscription credit is a different quantity and is never
//! derived from these prices.

use crate::Usage;
use query_language::number::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::OnceLock;

/// A list price, written in the catalog as USD per million tokens and held
/// exactly as integer pico-USD per token (`0.01875` USD/M is 18 750 pico-USD
/// per token). A rate that pico-USD cannot hold exactly is refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(try_from = "serde_json::Number", into = "serde_json::Number")]
pub struct Rate {
    pico_usd_per_token: i128,
}

impl Rate {
    pub const ZERO: Rate = Rate {
        pico_usd_per_token: 0,
    };

    pub fn pico_usd_per_token(self) -> i128 {
        self.pico_usd_per_token
    }
}

impl TryFrom<serde_json::Number> for Rate {
    type Error = String;
    fn try_from(number: serde_json::Number) -> Result<Self, String> {
        let text = number.to_string();
        Decimal::parse(&text)
            .filter(|rate| !rate.is_negative())
            .and_then(|rate| rate.exact_scaled_i128(6))
            .map(|pico_usd_per_token| Self { pico_usd_per_token })
            .ok_or_else(|| {
                format!("price {text} is not a non-negative USD rate with at most six decimals")
            })
    }
}

impl From<Rate> for serde_json::Number {
    fn from(rate: Rate) -> Self {
        // Pico-USD per token is micro-USD per million tokens.
        let micro = rate.pico_usd_per_token;
        let fraction = format!("{:06}", micro % 1_000_000);
        let fraction = fraction.trim_end_matches('0');
        format!(
            "{}{}{fraction}",
            micro / 1_000_000,
            if fraction.is_empty() { "" } else { "." }
        )
        .parse()
        .expect("decimal rate text is a JSON number")
    }
}

impl schemars::JsonSchema for Rate {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Rate".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"type":"number","minimum":0,"description":"USD per million tokens"})
    }
}

/// A service-tier multiplier (`2`, `0.5`), held exactly in millionths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(try_from = "serde_json::Number", into = "serde_json::Number")]
pub struct Multiplier {
    millionths: i128,
}

const ONE: Multiplier = Multiplier {
    millionths: 1_000_000,
};

impl TryFrom<serde_json::Number> for Multiplier {
    type Error = String;
    fn try_from(number: serde_json::Number) -> Result<Self, String> {
        let text = number.to_string();
        Decimal::parse(&text)
            .filter(|value| !value.is_negative())
            .and_then(|value| value.exact_scaled_i128(6))
            .map(|millionths| Self { millionths })
            .ok_or_else(|| {
                format!("multiplier {text} is not a non-negative number with at most six decimals")
            })
    }
}

impl From<Multiplier> for serde_json::Number {
    fn from(multiplier: Multiplier) -> Self {
        Rate {
            pico_usd_per_token: multiplier.millionths,
        }
        .into()
    }
}

impl schemars::JsonSchema for Multiplier {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        "Multiplier".into()
    }
    fn json_schema(_: &mut schemars::SchemaGenerator) -> schemars::Schema {
        schemars::json_schema!({"type":"number","minimum":0})
    }
}

/// List prices in USD per million tokens.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Price {
    pub input: Rate,
    #[serde(default)]
    pub cached_input: Option<Rate>,
    #[serde(default)]
    pub cache_write: Option<Rate>,
    /// Cache writes with a one-hour lifetime, where the vendor prices them
    /// apart from the default writes.
    #[serde(default)]
    pub cache_write_1h: Option<Rate>,
    pub output: Rate,
    /// Rates for a response whose prompt is longer than `above` tokens.
    #[serde(default)]
    pub long: Option<LongContext>,
    /// Multipliers for service tiers other than the standard one, by the
    /// name the tool records (`fast`, `flex`, `priority`, ...). A tier the
    /// model has no multiplier for leaves the cost unknown.
    #[serde(default)]
    pub tiers: BTreeMap<String, Multiplier>,
}

/// The long-context rates of a model whose price rises with prompt size.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LongContext {
    /// Prompt tokens (every kind of input together) above which these apply.
    pub above: u64,
    pub input: Rate,
    #[serde(default)]
    pub cached_input: Option<Rate>,
    #[serde(default)]
    pub cache_write: Option<Rate>,
    #[serde(default)]
    pub cache_write_1h: Option<Rate>,
    pub output: Rate,
}

/// Tier names that mean the standard rates.
const STANDARD_TIERS: [&str; 4] = ["standard", "default", "auto", "on_demand"];

impl Price {
    /// The multiplier for a recorded service tier, or `None` when the tier
    /// is not standard and the model has no rate for it.
    fn tier_multiplier(&self, tier: Option<&str>) -> Option<Multiplier> {
        match tier {
            None => Some(ONE),
            Some(t) if STANDARD_TIERS.contains(&t) => Some(ONE),
            Some(t) => self.tiers.get(t).copied(),
        }
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
        let tier = price.tier_multiplier(usage.service_tier.as_deref())?;
        let speed = price.tier_multiplier(usage.speed.as_deref())?;
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
        // with the default lifetime. More one-hour writes than writes is an
        // inconsistent record, priced as unknown.
        let write_1h = usage.cache_write_1h.filter(|_| usage.cache_write.is_some());
        let write_short = match usage.cache_write {
            None => None,
            Some(w) => Some(w.checked_sub(write_1h.unwrap_or(0))?),
        };
        let lane =
            |tokens: u64, rate: Rate| i128::from(tokens).checked_mul(rate.pico_usd_per_token);
        // Reasoning is billed at the output rate and is never part of
        // `output` (see `Usage::output`).
        let mut pico = lane(input, in_rate)?
            .checked_add(lane(output, out_rate)?)?
            .checked_add(lane(usage.reasoning.unwrap_or(0), out_rate)?)?;
        for (tokens, rate) in [
            (usage.cached_input, cached_rate),
            (write_short, write_rate),
            (write_1h, write_1h_rate),
        ] {
            match (tokens, rate) {
                (None | Some(0), _) => {}
                (Some(t), Some(r)) => pico = pico.checked_add(lane(t, r)?)?,
                (Some(_), None) => return None,
            }
        }
        // Pico-USD times two millionth multipliers, rounded once to nano-USD.
        let scaled = pico
            .checked_mul(tier.millionths)?
            .checked_mul(speed.millionths)?;
        let nusd = crate::money::rounded_quotient(scaled, 1_000 * 1_000_000 * 1_000_000)?;
        i64::try_from(nusd).ok()
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
