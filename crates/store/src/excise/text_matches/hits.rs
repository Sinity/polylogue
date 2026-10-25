use super::{HitGroup, Error, TextMatches};
use crate::{Result, StoreError};
use std::collections::{BTreeMap, BTreeSet};

pub(in crate::excise) type Scope = u64;

/// Tentative hits become visible only after their enclosing text is accepted.
pub(in crate::excise) trait HitSink {
    fn scope(&mut self) -> Result<Scope>;
    fn hit(&mut self, scope: Scope, start: u64, end: u64) -> Result<()>;
    fn accept(&mut self, scope: Scope, parent: Scope) -> Result<()>;
    fn discard(&mut self, scope: Scope) -> Result<()>;
}

#[derive(Default)]
struct MemoryHits {
    next: Scope,
    scopes: BTreeMap<Scope, BTreeSet<(u64, u64)>>,
}

impl HitSink for MemoryHits {
    fn scope(&mut self) -> Result<Scope> {
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| StoreError::Invalid("match scope overflow".into()))?;
        Ok(self.next)
    }

    fn hit(&mut self, scope: Scope, start: u64, end: u64) -> Result<()> {
        self.scopes.entry(scope).or_default().insert((start, end));
        Ok(())
    }

    fn accept(&mut self, scope: Scope, parent: Scope) -> Result<()> {
        if let Some(hits) = self.scopes.remove(&scope) {
            self.scopes.entry(parent).or_default().extend(hits);
        }
        Ok(())
    }

    fn discard(&mut self, scope: Scope) -> Result<()> {
        self.scopes.remove(&scope);
        Ok(())
    }
}

/// Whole-value consumers already own their input and require a final hit list.
pub(in crate::excise) struct Collected {
    scanner: TextMatches,
    hits: MemoryHits,
}

impl Collected {
    pub(in crate::excise) fn new(needle: &str) -> Result<Self> {
        let mut hits = MemoryHits::default();
        let scanner = TextMatches::new(needle, &mut hits)?;
        Ok(Self { scanner, hits })
    }

    pub(in crate::excise) fn plain(needle: &str) -> Result<Self> {
        let mut collected = Self::new(needle)?;
        collected.scanner.json = Some(false);
        let raw = collected.scanner.raw_group.take().unwrap();
        raw.accept(&mut collected.scanner.output, &mut collected.hits)?;
        Ok(collected)
    }

    pub(in crate::excise) fn feed(&mut self, bytes: &[u8]) -> Result<()> {
        self.scanner
            .feed(bytes, &mut self.hits)
            .map_err(Error::into_store)
    }

    pub(in crate::excise) fn finish(mut self) -> Result<Vec<(u64, u64)>> {
        let HitGroup { scope, .. } = self
            .scanner
            .finish(&mut self.hits)
            .map_err(Error::into_store)?;
        Ok(self
            .hits
            .scopes
            .remove(&scope)
            .unwrap_or_default()
            .into_iter()
            .collect())
    }
}
