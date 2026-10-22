use super::super::{Plan, Range, Ranges};
use serde_json::value::RawValue;

#[derive(Debug, PartialEq, Eq)]
pub(in crate::excise) struct Erasure {
    pub ranges: Vec<Range>,
    pub documents: usize,
    pub erased_documents: usize,
}

pub(in crate::excise) fn plan(bytes: &[u8], requested: &[Range]) -> Result<Erasure, String> {
    let mut ranges = Vec::new();
    for &(start, end) in requested {
        let raw = bytes
            .get(start..end)
            .ok_or("source range is outside its blob")?;
        let trimmed = raw.trim_ascii();
        if trimmed.is_empty() {
            return Err("source range contains no JSON record".into());
        }
        let leading = raw.len() - raw.trim_ascii_start().len();
        ranges.push((start + leading, start + leading + trimmed.len()));
    }
    ranges.sort_unstable();
    let mut merged: Vec<Range> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = merged.last_mut().filter(|last| last.1 >= start) {
            last.1 = last.1.max(end);
        } else {
            merged.push((start, end));
        }
    }
    let mut state = Memory {
        bytes,
        seen: vec![(false, false); merged.len()],
        requested: merged,
        output: Vec::new(),
    };
    let mut plan = Plan {
        bytes,
        base: 0,
        ranges: &mut state,
    };
    let mut documents = 0;
    let mut erased_documents = 0;
    for raw in serde_json::Deserializer::from_slice(bytes).into_iter::<&RawValue>() {
        let raw =
            raw.map_err(|e| format!("source piece is not a complete JSON record stream: {e}"))?;
        documents += 1;
        let range = plan.range(raw)?;
        if plan.ranges.covered(range)? {
            plan.ranges.mark(range);
            plan.ranges.push(range)?;
            erased_documents += 1;
        } else if plan.ranges.touched(range)? {
            plan.descend(raw)?;
        }
    }
    if state.seen.iter().any(|&(start, end)| !start || !end) {
        return Err("source citation does not follow complete JSON value boundaries".into());
    }
    Ok(Erasure {
        ranges: state.output,
        documents,
        erased_documents,
    })
}

struct Memory<'a> {
    bytes: &'a [u8],
    requested: Vec<Range>,
    seen: Vec<(bool, bool)>,
    output: Vec<Range>,
}
impl Memory<'_> {
    fn first(&self, range: Range) -> usize {
        self.requested.partition_point(|r| r.1 <= range.0)
    }
}
impl Ranges for Memory<'_> {
    fn touched(&mut self, range: Range) -> Result<bool, String> {
        Ok(self
            .requested
            .get(self.first(range))
            .is_some_and(|r| r.0 < range.1))
    }
    fn covered(&mut self, range: Range) -> Result<bool, String> {
        Ok(self
            .requested
            .get(self.first(range))
            .is_some_and(|r| r.0 <= range.0 && r.1 >= range.1))
    }
    fn mark(&mut self, range: Range) {
        let index = self.first(range);
        self.seen[index].0 |= self.requested[index].0 == range.0;
        self.seen[index].1 |= self.requested[index].1 == range.1;
    }
    fn push(&mut self, range: Range) -> Result<(), String> {
        if let Some(last) = self.output.last_mut()
            && last.1 <= range.0
            && self.bytes[last.1..range.0]
                .iter()
                .all(u8::is_ascii_whitespace)
        {
            last.1 = range.1;
        } else {
            self.output.push(range);
        }
        Ok(())
    }
}
