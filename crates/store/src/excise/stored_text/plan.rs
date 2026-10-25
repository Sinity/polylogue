//! What one text excision found and will change, held in memory: the matches
//! in each chain, the snapshot frames whose digests change, and the blobs and
//! extents the rewrite replaces.

use super::{Chain, Extent, extent};
use crate::Result;
use crate::excise::text_matches::{HitSink, Scope};
use polylogue_model::BlobHash;
use polylogue_model::snapshot_log::Header;
use rusqlite::{Connection, params};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Default)]
pub(super) struct Plan {
    state: RefCell<State>,
}

#[derive(Default)]
struct State {
    next_scope: Scope,
    /// Accepted matches per chain, overlaps kept apart.
    matches: BTreeMap<Chain, BTreeSet<(u64, u64)>>,
    /// Hits inside a text whose acceptance is not yet decided.
    candidates: HashMap<Scope, BTreeSet<(u64, u64)>>,
    /// Matches merged into ordered, nonoverlapping ranges to blank.
    blank: BTreeMap<Chain, Vec<(u64, u64)>>,
    /// Replacement snapshot headers, in frame order.
    patches: BTreeMap<Chain, Vec<(u64, Vec<u8>)>>,
    framed: BTreeSet<Chain>,
    frames: BTreeMap<Chain, Vec<(u64, Header)>>,
    /// Where each decoded blob sits in the scratch file.
    places: HashMap<BlobHash, (u64, u64)>,
    affected: BTreeSet<BlobHash>,
    observations: Vec<(String, serde_json::Value)>,
    referring: Vec<Extent>,
}

impl Plan {
    pub(super) fn new() -> Result<Self> {
        Ok(Self::default())
    }

    #[cfg(test)]
    pub(super) fn matches(&self, input: &str, chain: i64, ranges: &[(u64, u64)]) -> Result<()> {
        self.state
            .borrow_mut()
            .matches
            .entry((input.to_owned(), chain))
            .or_default()
            .extend(ranges.iter().copied());
        Ok(())
    }

    pub(super) fn patch(&self, input: &str, chain: i64, start: u64, bytes: &[u8]) -> Result<()> {
        self.state
            .borrow_mut()
            .patches
            .entry((input.to_owned(), chain))
            .or_default()
            .push((start, bytes.to_vec()));
        Ok(())
    }

    pub(super) fn placement(&self, hash: &BlobHash) -> Result<Option<(u64, u64)>> {
        Ok(self.state.borrow().places.get(hash).copied())
    }

    pub(super) fn remember(&self, hash: &BlobHash, start: u64, length: u64) -> Result<()> {
        self.state
            .borrow_mut()
            .places
            .insert(*hash, (start, length));
        Ok(())
    }

    pub(super) fn inputs(&self) -> Result<BTreeSet<String>> {
        Ok(self
            .state
            .borrow()
            .matches
            .iter()
            .filter(|(_, ranges)| !ranges.is_empty())
            .map(|((input, _), _)| input.clone())
            .collect())
    }

    pub(super) fn counts(&self) -> Result<BTreeMap<String, u64>> {
        let mut counts = BTreeMap::new();
        for ((input, _), ranges) in &self.state.borrow().matches {
            if !ranges.is_empty() {
                *counts.entry(input.clone()).or_default() += ranges.len() as u64;
            }
        }
        Ok(counts)
    }

    /// Every changed range: a chain's matches, then its header patches.
    pub(super) fn changed(
        &self,
        mut visit: impl FnMut(&str, i64, u64, u64) -> Result<()>,
    ) -> Result<()> {
        let state = self.state.borrow();
        let chains: BTreeSet<&Chain> = state.matches.keys().chain(state.patches.keys()).collect();
        for key in chains {
            for &(start, end) in state.matches.get(key).into_iter().flatten() {
                visit(&key.0, key.1, start, end)?;
            }
            for (start, bytes) in state.patches.get(key).into_iter().flatten() {
                visit(&key.0, key.1, *start, start + bytes.len() as u64)?;
            }
        }
        Ok(())
    }

    pub(super) fn edits(&self, key: &Chain, from: u64) -> Result<Edits> {
        let state = self.state.borrow();
        let ranges = state.blank.get(key).cloned().unwrap_or_default();
        let next = ranges.partition_point(|&(_, end)| end <= from);
        Ok(Edits {
            ranges,
            next,
            at: from,
            patches: state.patches.get(key).cloned().unwrap_or_default(),
        })
    }

    pub(super) fn hits<'a>(&'a self, input: &'a str, chain: i64) -> Hits<'a> {
        Hits {
            plan: self,
            input,
            chain,
        }
    }

    /// Writes use ordered, nonoverlapping ranges; counts keep exact hits.
    pub(super) fn normalize(&self, input: &str, chain: i64) -> Result<()> {
        use polylogue_model::snapshot_log::{HEADER_LEN, MAGIC};
        let key = (input.to_owned(), chain);
        let mut state = self.state.borrow_mut();
        // A hit in the log's magic or a frame header is structure, not text.
        if state.framed.contains(&key) {
            let headers: Vec<u64> = state
                .frames
                .get(&key)
                .into_iter()
                .flatten()
                .map(|(at, _)| *at)
                .collect();
            if let Some(matches) = state.matches.get_mut(&key) {
                matches.retain(|&(start, end)| {
                    let header = headers.iter().take_while(|&&at| at < end).last().copied();
                    start >= MAGIC.len() as u64
                        && header.is_none_or(|at| at + HEADER_LEN as u64 <= start)
                });
            }
        }
        let mut merged: Vec<(u64, u64)> = Vec::new();
        for &(start, end) in state.matches.get(&key).into_iter().flatten() {
            match merged.last_mut() {
                Some((_, until)) if start <= *until => *until = (*until).max(end),
                _ => merged.push((start, end)),
            }
        }
        state.blank.insert(key, merged);
        Ok(())
    }

    pub(super) fn framed(&self, key: &Chain) -> Result<()> {
        self.state.borrow_mut().framed.insert(key.clone());
        Ok(())
    }

    pub(super) fn frame(&self, key: &Chain, at: u64, header: &Header) -> Result<()> {
        self.state
            .borrow_mut()
            .frames
            .entry(key.clone())
            .or_default()
            .push((at, header.clone()));
        Ok(())
    }

    pub(super) fn frames(&self, key: &Chain) -> Vec<(u64, Header)> {
        self.state
            .borrow()
            .frames
            .get(key)
            .cloned()
            .unwrap_or_default()
    }

    /// The blobs holding a changed range, and the spans each `excised`
    /// observation records.
    pub(super) fn stage_affected(&self, archive: &Connection) -> Result<()> {
        let mut affected = archive.prepare(
            "SELECT blob,offset,length,blob_offset FROM extent WHERE input=?1 AND chain=?2
             AND offset<?4 AND offset+length>?3",
        )?;
        let mut blobs = BTreeSet::new();
        let mut observations = Vec::new();
        self.changed(|input, chain, start, end| {
            let mut rows = affected.query(params![input, chain, start as i64, end as i64])?;
            while let Some(row) = rows.next()? {
                let hash: Vec<u8> = row.get(0)?;
                let offset = row.get::<_, i64>(1)? as u64;
                let length = row.get::<_, i64>(2)? as u64;
                let blob_offset = row.get::<_, i64>(3)? as u64;
                let from = start.max(offset) - offset + blob_offset;
                let to = end.min(offset + length) - offset + blob_offset;
                observations.push((
                    input.to_owned(),
                    serde_json::json!([crate::excise::hex(&hash), from, to - from]),
                ));
                blobs.insert(BlobHash(hash.try_into().map_err(|_| {
                    crate::StoreError::Invalid("blob hash is not 32 bytes".into())
                })?));
            }
            Ok(())
        })?;
        let mut state = self.state.borrow_mut();
        state.affected = blobs;
        state.observations = observations;
        Ok(())
    }

    pub(super) fn visit_blobs(&self, mut visit: impl FnMut(BlobHash) -> Result<()>) -> Result<()> {
        let blobs = self.state.borrow().affected.clone();
        for hash in blobs {
            visit(hash)?;
        }
        Ok(())
    }

    /// Takes one blob's referring extents before any of those rows change.
    pub(super) fn freeze_extents(&self, archive: &Connection, hash: BlobHash) -> Result<()> {
        let referring = archive
            .prepare("SELECT input,chain,offset,length,blob,blob_offset FROM extent WHERE blob=?1 ORDER BY input,chain,offset")?
            .query_map([&hash.0[..]], extent)?
            .collect::<rusqlite::Result<_>>()?;
        self.state.borrow_mut().referring = referring;
        Ok(())
    }

    pub(super) fn visit_extents(&self, mut visit: impl FnMut(Extent) -> Result<()>) -> Result<()> {
        let referring = self.state.borrow().referring.clone();
        for e in referring {
            visit(e)?;
        }
        Ok(())
    }

    pub(super) fn visit_observations(
        &self,
        mut visit: impl FnMut(&str, serde_json::Value) -> Result<()>,
    ) -> Result<()> {
        let mut by_input: BTreeMap<String, Vec<serde_json::Value>> = BTreeMap::new();
        for (input, span) in &self.state.borrow().observations {
            by_input
                .entry(input.clone())
                .or_default()
                .push(span.clone());
        }
        for (input, spans) in by_input {
            visit(&input, serde_json::Value::Array(spans))?;
        }
        Ok(())
    }
}

/// The scanner's hits for one chain.
pub(super) struct Hits<'a> {
    plan: &'a Plan,
    input: &'a str,
    chain: i64,
}

impl HitSink for Hits<'_> {
    fn scope(&mut self) -> Result<Scope> {
        let mut state = self.plan.state.borrow_mut();
        state.next_scope = state
            .next_scope
            .checked_add(1)
            .ok_or_else(|| crate::StoreError::Invalid("match scope overflow".into()))?;
        Ok(state.next_scope)
    }

    fn hit(&mut self, scope: Scope, start: u64, end: u64) -> Result<()> {
        self.plan
            .state
            .borrow_mut()
            .candidates
            .entry(scope)
            .or_default()
            .insert((start, end));
        Ok(())
    }

    fn accept(&mut self, scope: Scope, parent: Scope) -> Result<()> {
        let mut state = self.plan.state.borrow_mut();
        if let Some(hits) = state.candidates.remove(&scope) {
            state.candidates.entry(parent).or_default().extend(hits);
        }
        Ok(())
    }

    fn discard(&mut self, scope: Scope) -> Result<()> {
        self.plan.state.borrow_mut().candidates.remove(&scope);
        Ok(())
    }
}

impl Hits<'_> {
    pub(super) fn finish(self, scope: Scope) -> Result<()> {
        let mut state = self.plan.state.borrow_mut();
        let hits = state.candidates.remove(&scope).unwrap_or_default();
        state
            .matches
            .entry((self.input.to_owned(), self.chain))
            .or_default()
            .extend(hits);
        Ok(())
    }
}

/// A forward-only cursor that blanks the matched ranges and writes the new
/// snapshot headers into chunks as they pass.
pub(super) struct Edits {
    ranges: Vec<(u64, u64)>,
    next: usize,
    at: u64,
    patches: Vec<(u64, Vec<u8>)>,
}

impl Edits {
    pub(super) fn blank(&mut self, at: u64, bytes: &mut [u8], fill: u8) -> Result<()> {
        if at < self.at {
            return Err(crate::StoreError::Invalid(
                "text-excision chunks moved backwards".into(),
            ));
        }
        let end = at + bytes.len() as u64;
        while let Some(&(start, until)) = self.ranges.get(self.next) {
            if start >= end {
                break;
            }
            let from = start.max(at);
            let to = until.min(end);
            if from < to {
                for b in &mut bytes[(from - at) as usize..(to - at) as usize] {
                    if *b != b'\n' {
                        *b = fill;
                    }
                }
            }
            if until > end {
                break;
            }
            self.next += 1;
        }
        self.at = end;
        Ok(())
    }

    pub(super) fn rewrite(&mut self, at: u64, bytes: &mut [u8], fill: u8) -> Result<()> {
        self.blank(at, bytes, fill)?;
        let end = at + bytes.len() as u64;
        for (start, patch) in &self.patches {
            let patch_end = start + patch.len() as u64;
            let from = (*start).max(at);
            let to = patch_end.min(end);
            if from < to {
                bytes[(from - at) as usize..(to - at) as usize]
                    .copy_from_slice(&patch[(from - start) as usize..(to - start) as usize]);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
