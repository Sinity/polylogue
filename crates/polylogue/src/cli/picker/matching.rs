use super::*;
use fuzzy_matcher::{FuzzyMatcher, skim::SkimMatcherV2};
use std::{
    cmp::{Ordering, Reverse},
    collections::BinaryHeap,
};

#[derive(
    Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, serde::Serialize, serde::Deserialize,
)]
struct Rank {
    score: i64,
    position: Reverse<u64>,
}

struct Ranked {
    rank: Rank,
    row: Candidate,
}
impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }
}
impl Eq for Ranked {}
impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

/// Rows in one page of fuzzy candidates.
pub const CANDIDATE_PAGE: usize = 64;

/// Ranks catalog rows by title, first prompt and repository, retaining one
/// transfer page. The iterator has a stable order for the catalog snapshot.
/// Callers stream catalog rows here; no archive read or daemon call is needed.
pub fn fuzzy_page(
    rows: impl IntoIterator<Item = Candidate>,
    request: &PageRequest,
    page_size: usize,
) -> Result<Page> {
    let mut matched = FuzzyRows::new(request, page_size)?;
    matched.extend(rows);
    matched.finish()
}

/// Builds one ranked page while source pages arrive in their stable order.
/// Only the requested page and its next candidate are retained.
pub struct FuzzyRows {
    text: String,
    after: Option<Rank>,
    matcher: SkimMatcherV2,
    best: BinaryHeap<Reverse<Ranked>>,
    count: u64,
    position: u64,
    page_size: usize,
}
impl FuzzyRows {
    pub fn new(request: &PageRequest, page_size: usize) -> Result<Self> {
        if page_size == 0 {
            return Err(Failure::bad_request(
                "a candidate page size must be positive",
            ));
        }
        let text = request
            .fuzzy
            .as_deref()
            .ok_or_else(|| Failure::bad_request("query mode needs query rows"))?;
        let after: Option<Rank> = request
            .cursor
            .as_deref()
            .map(serde_json::from_str)
            .transpose()
            .map_err(Failure::bad_request)?;
        let text = text.to_lowercase();
        let matcher = SkimMatcherV2::default();
        Ok(Self {
            text,
            after,
            matcher,
            best: BinaryHeap::new(),
            count: 0,
            position: 0,
            page_size,
        })
    }
    pub fn extend(&mut self, rows: impl IntoIterator<Item = Candidate>) {
        for row in rows {
            let position = self.position;
            self.position += 1;
            let score = if self.text.is_empty() {
                Some(0)
            } else {
                [&row.title, &row.first_prompt, &row.repository]
                    .into_iter()
                    .filter_map(|field| self.matcher.fuzzy_match(&field.to_lowercase(), &self.text))
                    .max()
            };
            let Some(score) = score else { continue };
            self.count += 1;
            let rank = Rank {
                score,
                position: Reverse(position),
            };
            if self.after.is_some_and(|after| rank >= after) {
                continue;
            }
            if self.best.len() <= self.page_size
                || self.best.peek().is_some_and(|worst| rank > worst.0.rank)
            {
                self.best.push(Reverse(Ranked { rank, row }));
                if self.best.len() > self.page_size.saturating_add(1) {
                    self.best.pop();
                }
            }
        }
    }
    pub fn finish(self) -> Result<Page> {
        let mut best: Vec<_> = self.best.into_iter().map(|row| row.0).collect();
        best.sort_unstable_by_key(|row| Reverse(row.rank));
        let more = best.len() > self.page_size;
        best.truncate(self.page_size);
        let next = if more {
            best.last()
                .map(|last| serde_json::to_string(&last.rank))
                .transpose()
                .map_err(Failure::internal)?
        } else {
            None
        };
        Ok(Page {
            rows: best.into_iter().map(|ranked| ranked.row).collect(),
            count: Some(self.count),
            next,
        })
    }
}
