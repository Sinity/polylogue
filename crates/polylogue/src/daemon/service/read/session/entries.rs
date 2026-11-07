//! A composed reading over cached physical-session prefix sums. Only the
//! selected page's positions are collected; composition costs one span each.
use super::{Composed, Runs};
use crate::daemon::service::Daemon;
use polylogue_model::read::{EntryKind, FoldProfile, RunningSums, View};
use std::{ops::Range, sync::Arc};

struct Run {
    part: usize,
    sums: Arc<RunningSums>,
    first: usize,
    end: usize,
    shown_before: usize,
    tokens_before: u64,
}

pub(super) struct ShownEntries {
    runs: Vec<Run>,
    len: usize,
    tokens: u64,
}
impl ShownEntries {
    pub(super) fn new(d: &Daemon, composed: &Composed, runs: &Runs, profile: FoldProfile) -> Self {
        let mut out = Self {
            runs: Vec::new(),
            len: 0,
            tokens: 0,
        };
        for &(part, start, end) in &runs.spans {
            let sums = d.warm.items.visible(&composed.indexes[part], profile);
            let first = sums.shown.partition_point(|shown| shown.entry < start);
            let mut through = sums.shown.partition_point(|shown| shown.entry < end);
            // A clipped span can end after a header but before any shown
            // block. Whole-session preparation saw that block outside the
            // span; this span must apply the header rule to its own range.
            if !matches!(profile.view, View::Transcript | View::Context | View::Raw)
                && through > first
                && composed.entries(part)[sums.shown[through - 1].entry].kind == EntryKind::Header
            {
                through -= 1;
            }
            out.push(part, sums, first..through);
        }
        out
    }
    fn push(&mut self, part: usize, sums: Arc<RunningSums>, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        let tokens = sums.sums[range.end].saturating_sub(sums.sums[range.start]);
        let len = range.len();
        self.runs.push(Run {
            part,
            sums,
            first: range.start,
            end: range.end,
            shown_before: self.len,
            tokens_before: self.tokens,
        });
        self.len += len;
        self.tokens = self.tokens.saturating_add(tokens);
    }
    pub(super) fn len(&self) -> usize {
        self.len
    }
    pub(super) fn get(&self, at: usize) -> Option<(usize, usize)> {
        if at >= self.len {
            return None;
        }
        let run = &self.runs[self
            .runs
            .partition_point(|run| run.shown_before + run.end - run.first <= at)];
        Some((
            run.part,
            run.sums.shown[run.first + at - run.shown_before].entry,
        ))
    }
    pub(super) fn range(&self, range: Range<usize>) -> Vec<(usize, usize)> {
        range
            .map(|at| self.get(at).expect("a selected page is inside its reading"))
            .collect()
    }
    pub(super) fn partition_point(&self, predicate: impl Fn((usize, usize)) -> bool) -> usize {
        let (mut low, mut high) = (0, self.len);
        while low < high {
            let mid = low + (high - low) / 2;
            if predicate(self.get(mid).expect("inside reading")) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low
    }
    fn prefix(&self, at: usize) -> u64 {
        if at == self.len {
            return self.tokens;
        }
        let run = &self.runs[self
            .runs
            .partition_point(|run| run.shown_before + run.end - run.first <= at)];
        run.tokens_before.saturating_add(
            run.sums.sums[run.first + at - run.shown_before]
                .saturating_sub(run.sums.sums[run.first]),
        )
    }
    pub(super) fn page_after(&self, start: usize, tokens: u64, items: usize) -> usize {
        if start >= self.len {
            return start;
        }
        let limit = self.prefix(start).saturating_add(tokens);
        let (mut low, mut high) = (start + 1, self.len + 1);
        while low < high {
            let mid = low + (high - low) / 2;
            if self.prefix(mid) <= limit {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        (low - 1)
            .max(start + 1)
            .min(start.saturating_add(items.max(1)))
    }
    pub(super) fn page_before(&self, end: usize, tokens: u64, items: usize) -> usize {
        let end = end.min(self.len);
        if end == 0 {
            return 0;
        }
        let floor = self.prefix(end).saturating_sub(tokens);
        let (mut low, mut high) = (0, end);
        while low < high {
            let mid = low + (high - low) / 2;
            if self.prefix(mid) < floor {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        low.min(end - 1).max(end.saturating_sub(items.max(1)))
    }
}

#[cfg(test)]
mod tests;
