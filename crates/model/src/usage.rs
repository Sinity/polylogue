use serde::{Deserialize, Serialize};

/// Token counts for one model response or recorded usage observation, by kind: uncached input, cached
/// input, output and reasoning, plus cache writes. A count the source did
/// not report is `None`, never 0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct Usage {
    /// Input tokens not served from a cache.
    pub input: Option<u64>,
    /// Input tokens read from a cache.
    pub cached_input: Option<u64>,
    /// Input tokens written to a cache, billed apart from plain input.
    pub cache_write: Option<u64>,
    pub output: Option<u64>,
    /// Reasoning tokens, when the source reports them apart from output.
    pub reasoning: Option<u64>,
    /// The part of `cache_write` written with a one-hour lifetime, which is
    /// billed at a higher rate than the default five-minute writes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write_1h: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    /// A generation mode priced differently from the standard one, as the
    /// tool records it (`fast`, `standard`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speed: Option<String>,
}

impl Usage {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Sum of the input counts, or `None` when none was reported.
    pub fn total_input(&self) -> Option<u64> {
        let inputs = [self.input, self.cached_input, self.cache_write];
        inputs
            .iter()
            .flatten()
            .copied()
            .reduce(|a, b| a.saturating_add(b))
    }
}

#[cfg(test)]
mod tests;
