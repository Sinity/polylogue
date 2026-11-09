//! Layout paging under bounded prepared selection coordinates. Store owns
//! metadata seeks; the service authenticates and seals the captured read scope.
use polylogue_model::{BlobHash, read::*};
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
        match self {
            Self::MetadataTooLarge => f.write_str("the page's header and footer exceed its budget"),
            Self::InvalidPosition => f.write_str("the page position is not in this reading"),
            Self::ItemChanged => f.write_str("the item at the page position changed"),
            Self::AlreadyFull => f.write_str("the page is already full"),
            Self::Serialization(error) => write!(f, "a page item could not be encoded: {error}"),
        }
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
    budget: PageBudget,
    bytes: usize,
    tokens: u64,
    start: Option<ItemPlace>,
    next: Option<ItemPlace>,
    full: bool,
    boundaries: Vec<ItemPlace>,
    before: Option<ItemPlace>,
    previous_input: Option<(SelectionLocator, u64)>,
}
impl PageBuilder {
    pub fn new(
        header: ReadHeader,
        budget: PageBudget,
        start: Option<ItemPlace>,
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
            before: None,
            previous_input: None,
        })
    }
    /// Input starts with the exact anchor item, even when character zero
    /// excludes it, then proceeds through earlier items in metadata order.
    pub fn new_before(
        header: ReadHeader,
        budget: PageBudget,
        anchor: ItemPlace,
        framing: ReadFraming,
    ) -> Result<Self, PageError> {
        let mut page = Self::new(header, budget, Some(anchor), framing)?;
        page.before = Some(anchor);
        Ok(page)
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
        if self.before.is_some() {
            return self.push_before(selection, ordinal, item);
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
        let position = ItemPlace {
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
            self.stop(ItemPlace {
                character: lo as u64,
                ..position
            });
            Ok(Push::Full)
        } else {
            Ok(Push::Accepted)
        }
    }
    fn push_before(
        &mut self,
        selection: SelectionLocator,
        ordinal: u64,
        item: ReadItem,
    ) -> Result<Push, PageError> {
        if self
            .previous_input
            .is_some_and(|last| (selection, ordinal) >= last)
        {
            return Err(PageError::InvalidPosition);
        }
        if self
            .start
            .is_some_and(|anchor| anchor.selection != selection || anchor.item != ordinal)
        {
            return Err(PageError::InvalidPosition);
        }
        let encoded = crate::encode::json_bytes(&item).map_err(serialization)?;
        let item_hash = BlobHash::of(&encoded);
        if self
            .start
            .is_some_and(|anchor| anchor.item_hash != item_hash)
        {
            return Err(PageError::ItemChanged);
        }
        let total = std::str::from_utf8(&encoded)
            .expect("JSON is UTF-8")
            .chars()
            .count();
        let until = self.start.map_or(Ok(total), |anchor| {
            usize::try_from(anchor.character).map_err(|_| PageError::InvalidPosition)
        })?;
        if until > total {
            return Err(PageError::InvalidPosition);
        }
        self.start = None;
        self.previous_input = Some((selection, ordinal));
        if until == 0 {
            return Ok(Push::Accepted);
        }
        let position = ItemPlace {
            selection,
            item: ordinal,
            character: 0,
            item_hash,
        };
        let cost = if matches!(&item.kind,ItemKind::Text{text} if text.is_empty()) {
            0
        } else {
            estimate_tokens(std::str::from_utf8(&encoded).expect("JSON is UTF-8")) as u64
        };
        if until == total
            && self.items.len() < self.budget.items.get()
            && self.bytes + encoded.len() <= self.budget.bytes.get()
            && self.tokens.saturating_add(cost) <= self.budget.tokens.get()
        {
            self.bytes += encoded.len();
            self.tokens += cost;
            self.items.push(item);
            self.boundaries.push(position);
            return Ok(Push::Accepted);
        }
        if !self.items.is_empty() {
            self.stop(*self.first_position().expect("accepted item boundary"));
            return Ok(Push::Full);
        }
        let text = String::from_utf8(encoded).expect("JSON is UTF-8");
        let boundaries: Vec<_> = text
            .char_indices()
            .map(|(index, _)| index)
            .chain(std::iter::once(text.len()))
            .collect();
        let make = |keep: usize| ReadItem {
            source: self.framing.source(),
            kind: ItemKind::Fragment {
                text: text[boundaries[until - keep]..boundaries[until]].into(),
                position: ItemPlace {
                    character: (until - keep) as u64,
                    ..position
                },
                until: until as u64,
                total: total as u64,
            },
        };
        let payload_cost = |text: &str| -> Result<u64, PageError> {
            let bytes = crate::encode::json_bytes(&text).map_err(serialization)?;
            Ok(estimate_tokens(std::str::from_utf8(&bytes).expect("JSON is UTF-8")) as u64)
        };
        let minimum = payload_cost(&text[boundaries[until - 1]..boundaries[until]])?;
        let token_limit = self.budget.tokens.get().max(minimum);
        let fits = |candidate: &ReadItem| -> Result<bool, PageError> {
            let bytes = crate::encode::json_bytes(candidate).map_err(serialization)?;
            let ItemKind::Fragment { text, .. } = &candidate.kind else {
                unreachable!("fragment candidate")
            };
            Ok(self.bytes + bytes.len() <= self.budget.bytes.get()
                && payload_cost(text)? <= token_limit)
        };
        if !fits(&make(1))? {
            return Err(PageError::MetadataTooLarge);
        }
        let (mut lo, mut hi) = (1, until);
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
        let lower = ItemPlace {
            character: (until - lo) as u64,
            ..position
        };
        self.items.push(fragment);
        self.boundaries.push(lower);
        if lo < until {
            self.stop(lower);
            Ok(Push::Full)
        } else {
            Ok(Push::Accepted)
        }
    }
    fn ordered_items(&self) -> Vec<ReadItem> {
        let mut items = self.items.clone();
        if self.before.is_some() {
            items.reverse();
        }
        items
    }
    /// The first emitted encoded position; Before inputs are kept descending
    /// internally, then emitted in forward order.
    pub fn first_position(&self) -> Option<&ItemPlace> {
        if self.before.is_some() {
            self.boundaries.last()
        } else {
            self.boundaries.first()
        }
    }
    /// Fixed exclusive upper boundary for the next forward page.
    pub fn upper_position(&self) -> Option<&ItemPlace> {
        self.before.as_ref()
    }
    /// Follow-ups are regenerated only for the finalized unread coordinate.
    pub fn finish_with(
        mut self,
        mut footer: impl FnMut(Option<&ItemPlace>) -> Result<ReadFooter, PageError>,
    ) -> Result<ReadDocument, PageError> {
        if self.start.is_some() {
            return Err(PageError::InvalidPosition);
        }
        loop {
            let mut footer = footer(self.next.as_ref())?;
            footer.shown = self.items.len() as u64;
            footer.estimator = Some(TOKEN_ESTIMATOR.into());
            let document = ReadDocument {
                header: self.header.clone(),
                items: self.ordered_items(),
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
                let removed = self.boundaries.pop();
                self.next = if self.before.is_some() {
                    self.boundaries.last().copied()
                } else {
                    removed
                };
                self.full = true;
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
            if self.before.is_some() {
                let drop = count - keep;
                let begin = text
                    .char_indices()
                    .nth(drop)
                    .map_or(text.len(), |(index, _)| index);
                let until = position.character + count as u64;
                let lower = ItemPlace {
                    character: position.character + drop as u64,
                    ..position
                };
                item.source = self.framing.source();
                item.kind = ItemKind::Fragment {
                    text: text[begin..].into(),
                    position: lower,
                    until,
                    total,
                };
                self.boundaries[0] = lower;
                self.next = Some(lower);
                self.full = true;
                continue;
            }
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
            self.next = Some(ItemPlace {
                character: until,
                ..boundary
            });
        }
    }
    fn stop(&mut self, position: ItemPlace) {
        self.next = Some(position);
        self.full = true;
    }
    pub fn position(&self) -> Option<&ItemPlace> {
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
        let mut items = self.items;
        if self.before.is_some() {
            items.reverse();
        }
        let document = ReadDocument {
            header: self.header,
            items,
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
