//! A page owns consecutive display fragments, independently of semantic blocks.
use super::{MAX_BUDGET, MINIMUM, Position, Window, refusal, request::Request, unavailable};
use serde_json::{Value, json};

struct Fragment {
    identity: Value,
    window: Window,
    block: u64,
}
impl Fragment {
    fn units(&self) -> usize {
        // An empty or fully suppressed block still consumes its boundary.
        self.window.ends.len().max(1)
    }
}

pub(super) struct Assembler<'a, 'r> {
    envelope: Value,
    request: &'r Request<'a>,
    basis: [u8; 32],
    count: u64,
    start: Position,
    fragments: Vec<Fragment>,
    units: usize,
}
impl<'a, 'r> Assembler<'a, 'r> {
    pub fn new(
        envelope: Value,
        request: &'r Request<'a>,
        basis: [u8; 32],
        count: u64,
        start: Position,
    ) -> Self {
        Self {
            envelope,
            request,
            basis,
            count,
            start,
            fragments: Vec::new(),
            units: 0,
        }
    }

    /// Keep one admitted page and at most one candidate window. A null-next
    /// envelope is the lower bound, since a complete final page can be smaller
    /// than a nonfinal prefix. Never stop solely on that larger prefix cost.
    pub fn push(&mut self, identity: Value, window: Window, block: u64) -> bool {
        let complete = !window.full;
        let fragment = Fragment {
            identity,
            window,
            block,
        };
        self.units += fragment.units();
        self.fragments.push(fragment);
        complete && self.request.cost(&self.candidate(self.units, true)) <= self.request.budget
    }

    fn candidate(&self, mut units: usize, provisional_final: bool) -> Value {
        let mut shown = Vec::new();
        let mut position = self.start.clone();
        position.budget = self.request.budget;
        for fragment in &self.fragments {
            if units == 0 {
                break;
            }
            let take = units.min(fragment.units());
            let w = &fragment.window;
            let (bytes, end) = if w.ends.is_empty() {
                (0, w.start)
            } else {
                w.ends[take - 1]
            };
            let mut value = fragment.identity.clone();
            value["display_start"] = json!(w.start);
            value["display_end"] = json!(end);
            if w.present {
                value["text"] = json!(&w.text[..bytes]);
            }
            shown.push(value);
            position.block = fragment.block;
            position.scalar = end;
            if !w.full && take == fragment.units() {
                position.block += 1;
                position.scalar = 0;
            }
            units -= take;
        }
        let next = if provisional_final || position.block >= self.count {
            Value::Null
        } else {
            self.request.follow_up(&self.basis, &position)
        };
        let mut page = self.envelope.clone();
        page["primary_content_complete"] = json!(next.is_null());
        page["next"] = next;
        page["shown"] = json!(shown);
        page
    }

    pub fn finish(self) -> Result<Value, Value> {
        let budget = self.request.budget;
        let all = self.candidate(self.units, false);
        // Check the whole candidate first: final next:null breaks monotonicity
        // at the last atom. All proper prefixes have ordinary continuation.
        if budget >= MINIMUM && self.request.cost(&all) <= budget {
            return Ok(all);
        }
        let first = self.candidate(self.units.min(1), false);
        if budget < MINIMUM || self.request.cost(&first) > budget {
            let sufficient = self.request.cost(&first).max(MINIMUM) + 32;
            if sufficient > MAX_BUDGET {
                return Err(unavailable("required_metadata_exceeds_maximum_budget"));
            }
            let mut error = refusal("insufficient_budget");
            error["error"]["requested_budget"] = json!(budget);
            error["error"]["minimum_budget"] = json!(MINIMUM);
            error["error"]["sufficient_budget"] = json!(sufficient);
            error["error"]["retry"] =
                json!("same reference and continuation with sufficient_budget");
            return Err(error);
        }
        let (mut low, mut high) = (1, self.units.saturating_sub(1).max(1));
        while low < high {
            let middle = (low + high).div_ceil(2);
            if self.request.cost(&self.candidate(middle, false)) <= budget {
                low = middle;
            } else {
                high = middle - 1;
            }
        }
        Ok(self.candidate(low, false))
    }
}

#[cfg(test)]
mod tests;
