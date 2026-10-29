//! Layout paging under bounded prepared selection coordinates. Store owns
//! metadata seeks; the service authenticates and seals the captured read scope.
use polylogue_model::{BlobHash, read::*};
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
    ItemChanged,
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

/// At most one page and one encoded item are held. Every fragment uses the
/// complete original ReadItem JSON domain, including otherwise unbounded Source.
/// Its outer Source is the genuine bounded question supplied by the service.
pub struct PageBuilder {
    header: ReadHeader,
    framing: ReadFraming,
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
        framing: ReadFraming,
    ) -> Result<Self, PageError> {
        let bytes = crate::encode::json_bytes(&header)
            .map_err(serialization)?
            .len()
            + 512;
        if bytes >= budget.bytes.get() {
            return Err(PageError::MetadataTooLarge);
        }
        Ok(Self {
            header,
            framing,
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
    /// Coordinates come from the same prepared metadata view as resume seeks.
    /// No native identifier is copied into either the frame or its position.
    pub fn push_at(
        &mut self,
        selection: SelectionLocator,
        ordinal: u64,
        item: ReadItem,
    ) -> Result<Push, PageError> {
        if self.full {
            return Err(PageError::AlreadyFull);
        }
        let from = if let Some(start) = self.start {
            if start.selection != selection {
                return Err(PageError::InvalidPosition);
            }
            if ordinal < start.item {
                return Ok(Push::Accepted);
            }
            if ordinal > start.item {
                return Err(PageError::InvalidPosition);
            }
            start.character
        } else {
            0
        };
        let encoded = crate::encode::json_bytes(&item).map_err(serialization)?;
        let item_hash = BlobHash::of(&encoded);
        if self.start.is_some_and(|start| start.item_hash != item_hash) {
            return Err(PageError::ItemChanged);
        }
        let position = ReadPosition {
            selection,
            item: ordinal,
            character: from,
            item_hash,
        };
        let cost = if matches!(&item.kind,ItemKind::Text{text} if text.is_empty()) {
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
            self.boundaries.push(position);
            self.items.push(item);
            self.start = None;
            return Ok(Push::Accepted);
        }
        if !self.items.is_empty() {
            self.stop(position);
            return Ok(Push::Full);
        }
        let text = String::from_utf8(encoded).expect("JSON is UTF-8");
        let total = text.chars().count();
        let from = usize::try_from(from).map_err(|_| PageError::InvalidPosition)?;
        if from >= total {
            return Err(PageError::InvalidPosition);
        }
        let boundaries: Vec<_> = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()))
            .collect();
        let make = |until: usize| ReadItem {
            source: self.framing.source(),
            kind: ItemKind::Fragment {
                text: text[boundaries[from]..boundaries[until]].into(),
                position,
                until: until as u64,
                total: total as u64,
            },
        };
        let payload_cost = |text: &str| -> Result<u64, PageError> {
            let bytes = crate::encode::json_bytes(&text).map_err(serialization)?;
            Ok(estimate_tokens(std::str::from_utf8(&bytes).expect("JSON is UTF-8")) as u64)
        };
        let minimum = payload_cost(&text[boundaries[from]..boundaries[from + 1]])?;
        let token_limit = self.budget.tokens.get().max(minimum);
        let fits = |candidate: &ReadItem| -> Result<bool, PageError> {
            let bytes = crate::encode::json_bytes(candidate).map_err(serialization)?;
            let ItemKind::Fragment { text, .. } = &candidate.kind else {
                unreachable!("fragment candidate")
            };
            Ok(self.bytes + bytes.len() <= self.budget.bytes.get()
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
        let bytes = crate::encode::json_bytes(&fragment).map_err(serialization)?;
        self.bytes += bytes.len();
        self.tokens += estimate_tokens(std::str::from_utf8(&bytes).expect("JSON is UTF-8")) as u64;
        self.boundaries.push(position);
        self.items.push(fragment);
        self.start = None;
        if lo < total {
            self.stop(ReadPosition {
                character: lo as u64,
                ..position
            });
            Ok(Push::Full)
        } else {
            Ok(Push::Accepted)
        }
    }
    /// Follow-ups are regenerated only for the finalized unread coordinate.
    pub fn finish_with(
        mut self,
        mut footer: impl FnMut(Option<&ReadPosition>) -> Result<ReadFooter, PageError>,
    ) -> Result<ReadDocument, PageError> {
        if self.start.is_some() {
            return Err(PageError::InvalidPosition);
        }
        loop {
            let mut footer = footer(self.next.as_ref())?;
            footer.position = self.next;
            footer.shown = self.items.len() as u64;
            footer.estimator = Some(TOKEN_ESTIMATOR.into());
            let document = ReadDocument {
                header: self.header.clone(),
                items: self.items.clone(),
                footer,
            };
            if crate::encode::json_bytes(&document)
                .map_err(serialization)?
                .len()
                <= self.budget.bytes.get()
            {
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
            let boundary = *self
                .boundaries
                .first()
                .expect("accepted item has a boundary");
            let (text, position, total) = match &item.kind {
                ItemKind::Fragment {
                    text,
                    position,
                    total,
                    ..
                } => (text.clone(), *position, *total),
                _ => {
                    let text =
                        String::from_utf8(crate::encode::json_bytes(item).map_err(serialization)?)
                            .expect("JSON is UTF-8");
                    let total = text.chars().count() as u64;
                    (text, boundary, total)
                }
            };
            let count = text.chars().count();
            if count <= 1 {
                return Err(PageError::MetadataTooLarge);
            }
            let keep = count.div_ceil(2);
            let until = position.character + keep as u64;
            let end = text
                .char_indices()
                .nth(keep)
                .map_or(text.len(), |(index, _)| index);
            item.source = self.framing.source();
            item.kind = ItemKind::Fragment {
                text: text[..end].into(),
                position,
                until,
                total,
            };
            self.next = Some(ReadPosition {
                character: until,
                ..boundary
            });
        }
    }
    fn stop(&mut self, position: ReadPosition) {
        self.next = Some(position);
        self.full = true;
    }
    pub fn position(&self) -> Option<&ReadPosition> {
        self.next.as_ref()
    }
    pub fn is_full(&self) -> bool {
        self.full
    }
    /// Static footers must already have their encoded space reserved. Dynamic
    /// product follow-ups use finish_with so any trim regenerates the cursor.
    pub fn finish(self, mut footer: ReadFooter) -> Result<ReadDocument, PageError> {
        if self.start.is_some() {
            return Err(PageError::InvalidPosition);
        }
        footer.shown = self.items.len() as u64;
        footer.estimator = Some(TOKEN_ESTIMATOR.into());
        footer.position = self.next;
        let document = ReadDocument {
            header: self.header,
            items: self.items,
            footer,
        };
        if crate::encode::json_bytes(&document)
            .map_err(serialization)?
            .len()
            > self.budget.bytes.get()
        {
            return Err(PageError::MetadataTooLarge);
        }
        Ok(document)
    }
}
fn serialization(error: std::io::Error) -> PageError {
    PageError::Serialization(error.to_string())
}
