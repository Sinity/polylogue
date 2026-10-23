//! Recorded usage is independent of output-item membership. The producer
//! persists items before completion and can retry without a durable boundary.
use super::*;
use polylogue_model::{BlobHash, UsageObservation};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
struct Counters {
    input: u64,
    cached: u64,
    output: u64,
    reasoning: u64,
    total: u64,
}

impl Counters {
    fn read(value: &Json) -> Option<Self> {
        Some(Self {
            input: value.get("input_tokens")?.as_u64()?,
            cached: value.get("cached_input_tokens")?.as_u64()?,
            output: value.get("output_tokens")?.as_u64()?,
            reasoning: value.get("reasoning_output_tokens")?.as_u64()?,
            total: value.get("total_tokens")?.as_u64()?,
        })
    }

    fn ordinary(self) -> bool {
        self.cached <= self.input
            && self.reasoning <= self.output
            && self.input.checked_add(self.output) == Some(self.total)
            && self.total > 0
    }

    fn follows(self, previous: Self, last: Self) -> bool {
        previous.input.checked_add(last.input) == Some(self.input)
            && previous.cached.checked_add(last.cached) == Some(self.cached)
            && previous.output.checked_add(last.output) == Some(self.output)
            && previous.reasoning.checked_add(last.reasoning) == Some(self.reasoning)
            && previous.total.checked_add(last.total) == Some(self.total)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(super) struct Observer {
    observed: Option<Counters>,
    report: Option<String>,
    recorded: Option<String>,
}

pub(super) fn identity(value: &impl Serialize) -> String {
    BlobHash::of(&serde_json::to_vec(value).expect("serializable reading")).to_string()
}

impl Observer {
    pub(super) fn start(&mut self, offset: u64, copied: bool) {
        if offset == 0 && !copied {
            self.observed = Some(Counters::default());
        }
    }

    pub(super) fn read(
        &mut self,
        info: &Json,
        observation: String,
        model: Option<String>,
        tier: Option<String>,
    ) -> Option<UsageObservation> {
        let total = info.get("total_token_usage").and_then(Counters::read);
        let last_value = info.get("last_token_usage");
        let last = last_value.and_then(Counters::read);
        let previous = self.observed;
        // An orphan still advances the observed baseline. Missing/malformed
        // state invalidates it instead of retaining a stale associated one.
        self.observed = total;
        let report = identity(info);
        if self.report.replace(report.clone()).as_ref() == Some(&report) {
            return None;
        }
        if total.is_some() && total == previous && last.is_some_and(Counters::ordinary) {
            return None;
        }
        let raw = last_value.filter(|value| value.is_object())?;
        // These helpers put context size only in total, with ordinary
        // counters zero. They are not measured usage or completion reports.
        if last.is_some_and(|last| last.input == 0 && last.output == 0 && last.total > 0) {
            return None;
        }
        if self.recorded.as_ref() == Some(&observation) {
            return None;
        }
        let n = |key: &str| raw.get(key).and_then(Json::as_u64);
        let input = n("input_tokens");
        let output = n("output_tokens");
        let cached =
            n("cached_input_tokens").filter(|cached| input.is_none_or(|input| *cached <= input));
        let reasoning = n("reasoning_output_tokens")
            .filter(|reasoning| output.is_none_or(|output| *reasoning <= output));
        let usage = Usage {
            input: input
                .zip(cached)
                .and_then(|(input, cached)| input.checked_sub(cached)),
            cached_input: cached,
            cache_write: n("cache_write_input_tokens"),
            output: output
                .zip(reasoning)
                .and_then(|(output, reasoning)| output.checked_sub(reasoning)),
            reasoning,
            service_tier: tier,
            ..Usage::default()
        };
        if [
            usage.input,
            usage.cached_input,
            usage.cache_write,
            usage.output,
            usage.reasoning,
            input,
            output,
        ]
        .iter()
        .all(Option::is_none)
        {
            return None;
        }
        let completion = match (total, previous, last) {
            (Some(total), Some(previous), Some(last)) => {
                last.ordinary() && total.follows(previous, last)
            }
            _ => false,
        };
        self.recorded = Some(observation.clone());
        Some(UsageObservation {
            observation,
            usage,
            model,
            response: None,
            completion_observed: completion.then_some(true),
            observed_at: None,
            input_including_cache: input,
            output_including_reasoning: output,
        })
    }
}
