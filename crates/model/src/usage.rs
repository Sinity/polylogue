use serde::{Deserialize, Serialize};

/// Token counts for one model response, in four lanes. A lane the source
/// did not report is `None`, never 0.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
}

impl Usage {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Sum of the input lanes, or `None` when no input lane was reported.
    pub fn total_input(&self) -> Option<u64> {
        let lanes = [self.input, self.cached_input, self.cache_write];
        lanes
            .iter()
            .flatten()
            .copied()
            .reduce(|a, b| a.saturating_add(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_lanes_stay_unknown() {
        let u = Usage::default();
        assert!(u.is_empty());
        assert_eq!(u.total_input(), None);
        let u = Usage {
            input: Some(3),
            cached_input: Some(10),
            ..Usage::default()
        };
        assert_eq!(u.total_input(), Some(13));
    }
}
