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

/// Ranks catalog rows by title, first prompt and repository, retaining one
/// transfer page. The iterator has a stable order for the catalog snapshot.
/// Callers stream catalog rows here; no archive read or daemon call is needed.
pub fn fuzzy_page(
    rows: impl IntoIterator<Item = Candidate>,
    request: &PageRequest,
    page_size: usize,
) -> Result<Page> {
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
    let matcher = SkimMatcherV2::default().ignore_case();
    let mut best: BinaryHeap<Reverse<Ranked>> = BinaryHeap::new();
    let mut count = 0;
    for (position, row) in rows.into_iter().enumerate() {
        let score = if text.is_empty() {
            Some(0)
        } else {
            [&row.title, &row.first_prompt, &row.repository]
                .into_iter()
                .filter_map(|field| matcher.fuzzy_match(&field.to_lowercase(), &text))
                .max()
        };
        let Some(score) = score else { continue };
        count += 1;
        let rank = Rank {
            score,
            position: Reverse(position as u64),
        };
        if after.is_some_and(|after| rank >= after) {
            continue;
        }
        if best.len() <= page_size || best.peek().is_some_and(|worst| rank > worst.0.rank) {
            best.push(Reverse(Ranked { rank, row }));
            if best.len() > page_size.saturating_add(1) {
                best.pop();
            }
        }
    }
    let mut best: Vec<_> = best.into_iter().map(|row| row.0).collect();
    best.sort_unstable_by_key(|row| Reverse(row.rank));
    let more = best.len() > page_size;
    best.truncate(page_size);
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
        count: Some(count),
        next,
    })
}
