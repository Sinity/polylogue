//! Presentation paging only. The store owns selection and the service seals
//! `ReadPosition` into a cursor bound to that selection and view.
use polylogue_model::read::*;
use std::num::{NonZeroU64, NonZeroUsize};

#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub items: NonZeroUsize,
    pub tokens: NonZeroU64,
    pub bytes: NonZeroUsize,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            items: NonZeroUsize::new(50).unwrap(),
            tokens: NonZeroU64::new(4000).unwrap(),
            bytes: NonZeroUsize::new(21_000).unwrap(),
        }
    }
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageError {
    MetadataTooLarge,
    InvalidPosition,
    AlreadyFull,
    Serialization(String),
}
impl std::fmt::Display for PageError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for PageError {}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Push {
    Accepted,
    Full,
}
/// Holds at most one transport page. A huge item is fragmented by Unicode
/// character positions, with exact JSON for a structured item's reconstruction.
/// The enclosing transport reserves its own envelope bytes before setting budget.
pub struct PageBuilder {
    header: ReadHeader,
    items: Vec<ReadItem>,
    budget: Budget,
    bytes: usize,
    tokens: u64,
    start: Option<ReadPosition>,
    next: Option<ReadPosition>,
    full: bool,
    boundaries: Vec<ReadPosition>,
}
impl PageBuilder {
    pub fn new(
        header: ReadHeader,
        budget: Budget,
        start: Option<ReadPosition>,
    ) -> Result<Self, PageError> {
        let bytes = crate::encode::json_bytes(&header)
            .map_err(|e| PageError::Serialization(e.to_string()))?
            .len()
            + 512;
        if bytes >= budget.bytes.get() {
            return Err(PageError::MetadataTooLarge);
        }
        Ok(Self {
            header,
            items: vec![],
            budget,
            bytes,
            tokens: 0,
            start,
            next: None,
            full: false,
            boundaries: vec![],
        })
    }
    /// `record` is the producer's stable transfer identity (a canonical record
    /// reference or prepared presentation-row identity). `ordinal` is separate
    /// from a paired item's displayed source reference.
    pub fn push_at(
        &mut self,
        record: &str,
        ordinal: u64,
        item: ReadItem,
    ) -> Result<Push, PageError> {
        if self.full {
            return Err(PageError::AlreadyFull);
        }
        let from = if let Some(start) = &self.start {
            if start.reference != record {
                return Err(PageError::InvalidPosition);
            }
            if ordinal < start.item {
                return Ok(Push::Accepted);
            }
            if ordinal > start.item && start.character != 0 {
                return Err(PageError::InvalidPosition);
            }
            if ordinal == start.item {
                start.character
            } else {
                0
            }
        } else {
            0
        };
        let encoded = crate::encode::json_bytes(&item)
            .map_err(|e| PageError::Serialization(e.to_string()))?;
        let cost = if matches!(&item.kind,ItemKind::Text {text} if text.is_empty()) {
            0
        } else {
            estimate_tokens(std::str::from_utf8(&encoded).expect("JSON is UTF-8")) as u64
        };
        if from == 0
            && self.items.len() < self.budget.items.get()
            && self.bytes + encoded.len() <= self.budget.bytes.get()
            && self.tokens.saturating_add(cost) <= self.budget.tokens.get()
        {
            self.bytes += encoded.len();
            self.tokens += cost;
            self.boundaries.push(ReadPosition {
                reference: record.into(),
                item: ordinal,
                character: from,
            });
            self.items.push(item);
            self.start = None;
            return Ok(Push::Accepted);
        }
        if !self.items.is_empty() {
            self.stop(record, ordinal, from);
            return Ok(Push::Full);
        }
        let (encoding, text) = match &item.kind {
            ItemKind::Text { text } => (FragmentEncoding::Text, text.clone()),
            _ => (
                FragmentEncoding::ReadItemJson,
                String::from_utf8(encoded).expect("JSON is UTF-8"),
            ),
        };
        let total = text.chars().count();
        let from = usize::try_from(from).map_err(|_| PageError::InvalidPosition)?;
        if from >= total {
            return Err(PageError::InvalidPosition);
        }
        // At most one record-sized buffer; binary search avoids quadratic work
        // for an enormous first item and accounts for JSON escaping overhead.
        let boundaries: Vec<usize> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect();
        let make = |until: usize| ReadItem {
            source: item.source.clone(),
            kind: ItemKind::Fragment {
                encoding,
                text: text[boundaries[from]..boundaries[until]].into(),
                from: from as u64,
                until: until as u64,
                total: total as u64,
            },
        };
        let payload_cost = |text: &str| -> Result<u64, PageError> {
            let encoded = crate::encode::json_bytes(&text)
                .map_err(|e| PageError::Serialization(e.to_string()))?;
            Ok(estimate_tokens(std::str::from_utf8(&encoded).expect("JSON is UTF-8")) as u64)
        };
        let minimum = payload_cost(&text[boundaries[from]..boundaries[from + 1]])?;
        let token_limit = self.budget.tokens.get().max(minimum);
        let fits = |candidate: &ReadItem| -> Result<bool, PageError> {
            let json = crate::encode::json_bytes(candidate)
                .map_err(|e| PageError::Serialization(e.to_string()))?;
            let ItemKind::Fragment { text, .. } = &candidate.kind else {
                unreachable!("fragment candidate")
            };
            Ok(self.bytes + json.len() <= self.budget.bytes.get()
                && payload_cost(text)? <= token_limit)
        };
        if !fits(&make(from + 1))? {
            return Err(PageError::MetadataTooLarge);
        }
        let (mut lo, mut hi) = (from + 1, total);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fits(&make(mid))? {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let fragment = make(lo);
        let json = crate::encode::json_bytes(&fragment)
            .map(|bytes| String::from_utf8(bytes).expect("JSON is UTF-8"))
            .map_err(|e| PageError::Serialization(e.to_string()))?;
        self.bytes += json.len();
        self.tokens += estimate_tokens(&json) as u64;
        self.boundaries.push(ReadPosition {
            reference: record.into(),
            item: ordinal,
            character: from as u64,
        });
        self.items.push(fragment);
        self.start = None;
        if lo < total {
            self.stop(record, ordinal, lo as u64);
            Ok(Push::Full)
        } else {
            Ok(Push::Accepted)
        }
    }
    /// Build follow-ups only for the final boundary. Large footer metadata can
    /// reduce this page; every retry receives the corresponding unread position.
    /// The callback seals cursors in the product; the renderer does no I/O.
    pub fn finish_with(
        mut self,
        mut footer: impl FnMut(Option<&ReadPosition>) -> Result<ReadFooter, PageError>,
    ) -> Result<ReadDocument, PageError> {
        loop {
            let mut footer = footer(self.next.as_ref())?;
            footer.position = self.next.clone();
            footer.shown = self.items.len() as u64;
            footer.estimator = Some(TOKEN_ESTIMATOR.into());
            let document = ReadDocument {
                header: self.header.clone(),
                items: self.items.clone(),
                footer,
            };
            let size = crate::encode::json_bytes(&document)
                .map_err(|e| PageError::Serialization(e.to_string()))?
                .len();
            if size <= self.budget.bytes.get() {
                return Ok(document);
            }
            if self.items.len() > 1 {
                self.items.pop();
                self.next = self.boundaries.pop();
                continue;
            }
            let Some(item) = self.items.first_mut() else {
                return Err(PageError::MetadataTooLarge);
            };
            let boundary = self
                .boundaries
                .first()
                .expect("accepted item has a boundary");
            let (encoding, text, from, total) = match &item.kind {
                ItemKind::Fragment {
                    encoding,
                    text,
                    from,
                    total,
                    ..
                } => (*encoding, text.clone(), *from, *total),
                ItemKind::Text { text } => (
                    FragmentEncoding::Text,
                    text.clone(),
                    0,
                    text.chars().count() as u64,
                ),
                _ => {
                    let text = String::from_utf8(
                        crate::encode::json_bytes(item)
                            .map_err(|e| PageError::Serialization(e.to_string()))?,
                    )
                    .expect("JSON is UTF-8");
                    let total = text.chars().count() as u64;
                    (FragmentEncoding::ReadItemJson, text, 0, total)
                }
            };
            let count = text.chars().count();
            if count <= 1 {
                return Err(PageError::MetadataTooLarge);
            }
            let keep = count.div_ceil(2);
            let until = from + keep as u64;
            let end = text
                .char_indices()
                .nth(keep)
                .map_or(text.len(), |(index, _)| index);
            item.kind = ItemKind::Fragment {
                encoding,
                text: text[..end].into(),
                from,
                until,
                total,
            };
            self.next = Some(ReadPosition {
                reference: boundary.reference.clone(),
                item: boundary.item,
                character: until,
            });
        }
    }
    fn stop(&mut self, record: &str, item: u64, character: u64) {
        self.next = Some(ReadPosition {
            reference: record.into(),
            item,
            character,
        });
        self.full = true;
    }
    pub fn position(&self) -> Option<&ReadPosition> {
        self.next.as_ref()
    }
    pub fn is_full(&self) -> bool {
        self.full
    }
    /// The service supplies ready calls/coverage and must reserve their encoded
    /// size in `Budget::bytes`. A limit violation is explicit, never truncation.
    pub fn finish(self, mut footer: ReadFooter) -> Result<ReadDocument, PageError> {
        footer.shown = self.items.len() as u64;
        footer.estimator = Some(TOKEN_ESTIMATOR.into());
        footer.position = self.next;
        let document = ReadDocument {
            header: self.header,
            items: self.items,
            footer,
        };
        let bytes = crate::encode::json_bytes(&document)
            .map_err(|e| PageError::Serialization(e.to_string()))?
            .len();
        if bytes > self.budget.bytes.get() {
            return Err(PageError::MetadataTooLarge);
        }
        Ok(document)
    }
}
