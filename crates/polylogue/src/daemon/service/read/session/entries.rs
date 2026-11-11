//! The entries a composed reading shows, read from the session item index
//! in windows next to a page. Where an entry sits among the shown entries
//! comes from the counts the writer stores per entry class, so a page in the
//! middle of a long session reads its own entries and the unit it starts
//! in, not the entries before it.
use super::{Composed, Runs};
use crate::daemon::service::Failure;
use polylogue_model::read::{EntryKind, FoldProfile, IndexEntry, ItemCounts, View};
use polylogue_store::Reader;

/// One shown entry: its part, its place in the part's session, and itself.
#[derive(Debug, Clone)]
pub(super) struct PageEntry {
    pub part: usize,
    pub seq: usize,
    pub entry: IndexEntry,
}
impl PageEntry {
    pub(super) fn place(&self) -> (usize, usize) {
        (self.part, self.seq)
    }
}

/// The entries of one page, and the shown entry after them.
pub(super) struct Page {
    pub entries: Vec<PageEntry>,
    pub next: Option<PageEntry>,
}

/// A run of one part's entries in the reading.
struct Span {
    part: usize,
    start: usize,
    end: usize,
    /// A header the span shows none of the blocks of: hidden in this span.
    dropped: Option<usize>,
    /// Shown entries before the span.
    before: u64,
}

pub(super) struct ShownEntries<'r, 'c> {
    reader: &'r Reader<'r>,
    composed: &'c Composed,
    profile: FoldProfile,
    spans: Vec<Span>,
    total: u64,
}

/// Rows read per window at first; later windows of one page read more.
const FIRST_WINDOW: usize = 16;

impl<'r, 'c> ShownEntries<'r, 'c> {
    pub(super) fn new(
        reader: &'r Reader<'r>,
        composed: &'c Composed,
        runs: &Runs,
        profile: FoldProfile,
    ) -> Result<Self, Failure> {
        let mut out = Self {
            reader,
            composed,
            profile,
            spans: Vec::new(),
            total: 0,
        };
        let headers_alone = matches!(profile.view, View::Transcript | View::Context | View::Raw);
        for &(part, start, end) in &runs.spans {
            if start >= end {
                continue;
            }
            let shown = composed
                .counts_at(reader, part, end)?
                .since(&composed.counts_at(reader, part, start)?)
                .shown(&profile);
            // A span cut inside a message can end after a header but before
            // any of its shown blocks; the header is then not shown here.
            let index = &composed.indexes[part];
            let mut dropped = None;
            if !headers_alone && end < index.entries as usize && shown > 0 {
                let last =
                    reader.item_window(index.pk, start as u64..end as u64, false, 1, |class| {
                        profile.shows(class)
                    })?;
                if let Some(last) = last.first()
                    && last.entry.kind == EntryKind::Header
                {
                    dropped = Some(last.seq as usize);
                }
            }
            let shown = shown - u64::from(dropped.is_some());
            out.spans.push(Span {
                part,
                start,
                end,
                dropped,
                before: out.total,
            });
            out.total += shown;
        }
        Ok(out)
    }

    /// How many entries the reading shows.
    pub(super) fn len(&self) -> u64 {
        self.total
    }

    /// The shown entries before a place.
    pub(super) fn position(&self, (part, seq): (usize, usize)) -> Result<u64, Failure> {
        let Some(at) = self.span_at(part, seq) else {
            return Ok(self.total);
        };
        let span = &self.spans[at];
        if (part, seq) <= (span.part, span.start) {
            return Ok(span.before);
        }
        let shown = self
            .composed
            .counts_at(self.reader, part, seq)?
            .since(&self.composed.counts_at(self.reader, part, span.start)?)
            .shown(&self.profile);
        let dropped = span.dropped.is_some_and(|dropped| dropped < seq);
        Ok(span.before + shown - u64::from(dropped))
    }

    /// The first span that holds a place or comes after it.
    fn span_at(&self, part: usize, seq: usize) -> Option<usize> {
        let at = self
            .spans
            .partition_point(|span| (span.part, span.end) <= (part, seq));
        (at < self.spans.len()).then_some(at)
    }

    /// Shown entries from a place on, in order, while `more` accepts them.
    fn walk(
        &self,
        from: (usize, usize),
        mut more: impl FnMut(&PageEntry) -> bool,
    ) -> Result<(), Failure> {
        let Some(first) = self.span_at(from.0, from.1) else {
            return Ok(());
        };
        let mut window = FIRST_WINDOW;
        for span in &self.spans[first..] {
            let index = &self.composed.indexes[span.part];
            let mut seq = if span.part == from.0 {
                from.1.max(span.start)
            } else {
                span.start
            };
            while seq < span.end {
                let rows = self.reader.item_window(
                    index.pk,
                    seq as u64..span.end as u64,
                    true,
                    window,
                    |class| self.profile.shows(class),
                )?;
                window = window.saturating_mul(2);
                let Some(last) = rows.last() else { break };
                seq = last.seq as usize + 1;
                for stored in rows {
                    if span.dropped == Some(stored.seq as usize) {
                        continue;
                    }
                    let entry = PageEntry {
                        part: span.part,
                        seq: stored.seq as usize,
                        entry: stored.entry,
                    };
                    if !more(&entry) {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    /// Shown entries before a place, nearest first, while `more` accepts them.
    fn walk_back(
        &self,
        before: (usize, usize),
        mut more: impl FnMut(&PageEntry) -> bool,
    ) -> Result<(), Failure> {
        let last = match self.span_at(before.0, before.1) {
            Some(at) => at + 1,
            None => self.spans.len(),
        };
        let mut window = FIRST_WINDOW;
        for span in self.spans[..last].iter().rev() {
            let index = &self.composed.indexes[span.part];
            let mut end = if span.part == before.0 {
                before.1.min(span.end)
            } else if (span.part, span.start) >= before {
                continue;
            } else {
                span.end
            };
            while end > span.start {
                let rows = self.reader.item_window(
                    index.pk,
                    span.start as u64..end as u64,
                    false,
                    window,
                    |class| self.profile.shows(class),
                )?;
                window = window.saturating_mul(2);
                let Some(last) = rows.last() else { break };
                end = last.seq as usize;
                for stored in rows {
                    if span.dropped == Some(stored.seq as usize) {
                        continue;
                    }
                    let entry = PageEntry {
                        part: span.part,
                        seq: stored.seq as usize,
                        entry: stored.entry,
                    };
                    if !more(&entry) {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    fn cost(&self, entry: &IndexEntry) -> u64 {
        self.profile.cost(entry).unwrap_or(0)
    }

    /// The page that starts at the first shown entry at or after `from`: as
    /// many whole entries as fit in `tokens` and `items`, at least one, and
    /// the shown entry after them.
    pub(super) fn page_after(
        &self,
        from: (usize, usize),
        tokens: u64,
        items: usize,
    ) -> Result<Page, Failure> {
        let items = items.max(1);
        let mut page = Page {
            entries: Vec::new(),
            next: None,
        };
        let mut used = 0u64;
        self.walk(from, |entry| {
            let cost = self.cost(&entry.entry);
            if page.entries.len() >= items
                || (!page.entries.is_empty() && used.saturating_add(cost) > tokens)
            {
                page.next = Some(entry.clone());
                return false;
            }
            used = used.saturating_add(cost);
            page.entries.push(entry.clone());
            true
        })?;
        Ok(page)
    }

    /// The first shown entry at or after a place.
    pub(super) fn first_at(&self, from: (usize, usize)) -> Result<Option<PageEntry>, Failure> {
        let mut found = None;
        self.walk(from, |entry| {
            found = Some(entry.clone());
            false
        })?;
        Ok(found)
    }

    /// The last shown entry before a place.
    pub(super) fn last_before(&self, before: (usize, usize)) -> Result<Option<PageEntry>, Failure> {
        let mut found = None;
        self.walk_back(before, |entry| {
            found = Some(entry.clone());
            false
        })?;
        Ok(found)
    }

    /// The page that ends just before a place: the most whole entries before
    /// it that fit in `tokens` and `items`, at least one, in reading order.
    pub(super) fn page_before(
        &self,
        before: (usize, usize),
        tokens: u64,
        items: usize,
    ) -> Result<Vec<PageEntry>, Failure> {
        let items = items.max(1);
        let mut out: Vec<PageEntry> = Vec::new();
        let mut used = 0u64;
        self.walk_back(before, |entry| {
            let cost = self.cost(&entry.entry);
            if out.len() >= items || (!out.is_empty() && used.saturating_add(cost) > tokens) {
                return false;
            }
            used = used.saturating_add(cost);
            out.push(entry.clone());
            true
        })?;
        out.reverse();
        Ok(out)
    }

    /// Where the reading starts and where it ends.
    pub(super) fn start(&self) -> (usize, usize) {
        self.spans
            .first()
            .map_or((0, 0), |span| (span.part, span.start))
    }
    pub(super) fn end(&self) -> (usize, usize) {
        self.spans
            .last()
            .map_or((0, 0), |span| (span.part, span.end))
    }
}

/// The counts of a part's session before `seq`.
pub(super) fn counts_before(
    reader: &Reader<'_>,
    index: &super::SessionIndex,
    seq: usize,
) -> Result<ItemCounts, Failure> {
    Ok(if seq == 0 {
        ItemCounts::default()
    } else if seq as u64 >= index.entries {
        index.counts.clone()
    } else {
        reader.item_counts_before(index.pk, seq as u64)?
    })
}
