//! Reading for models, consuming one message at a time. Whole original blocks
//! are masked before splitting; no unmasked text is carried in a position.

use crate::{View, rendered_blocks};
use polylogue_model::Message;
use polylogue_model::masking::MessageMasker;
use serde::{Deserialize, Serialize};
use std::ops::ControlFlow;

const CHARS_PER_TOKEN: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub text: String,
    pub next: Option<Position>,
    pub attachments: Vec<crate::AttachmentRef>,
    pub slices: Vec<BlockSlice>,
}

/// One masked portion of an original block. Offsets count Unicode characters
/// in the projected block text, before HTML escaping. The source block index
/// survives view filtering and continuation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockSlice {
    pub message: Option<MessageHeader>,
    pub block: usize,
    pub kind: String,
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<polylogue_model::CallOutcome>,
    pub text: String,
    pub from: usize,
    pub until: usize,
    pub more: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MessageHeader {
    pub id: polylogue_model::MessageId,
    pub session: polylogue_model::SessionId,
    pub role: polylogue_model::Role,
    pub written_by: polylogue_model::WrittenBy,
    pub at: Option<polylogue_model::Timestamp>,
    pub model: Option<String>,
}

/// A position in the rendered stream. The caller binds it to the read view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    pub message: usize,
    pub block: usize,
    pub character: usize,
}

/// Private source-stream state before one message. It can contain source
/// characters from an open PEM block; never serialize it into a read cursor.
#[derive(Clone)]
pub struct PageEnd {
    message: usize,
    masker: MessageMasker,
}

impl PageEnd {
    /// The state before `message`: the masker after the messages before it.
    pub fn new(message: usize, masker: MessageMasker) -> Self {
        Self { message, masker }
    }

    pub fn message(&self) -> usize {
        self.message
    }
}

pub struct Pager {
    view: View,
    budget: usize,
    at: Position,
    resumed: bool,
    message: usize,
    masker: MessageMasker,
    page_end: Option<PageEnd>,
    validated: bool,
    used: usize,
    text: String,
    slices: Vec<BlockSlice>,
    source_blocks: Vec<usize>,
    header: Option<MessageHeader>,
    first: Option<Position>,
    latest: bool,
    until: Option<Position>,
    next: Option<Position>,
    attachments: Vec<crate::AttachmentRef>,
    seen_attachments: std::collections::HashSet<crate::AttachmentRef>,
}

impl Pager {
    pub fn new(view: View, budget_tokens: usize, position: Option<Position>) -> Self {
        Self {
            view,
            budget: budget_tokens.max(1).saturating_mul(CHARS_PER_TOKEN),
            at: position.unwrap_or_default(),
            resumed: position.is_some(),
            message: 0,
            masker: MessageMasker::default(),
            page_end: None,
            validated: position.is_none(),
            used: 0,
            text: String::new(),
            slices: Vec::new(),
            source_blocks: Vec::new(),
            header: None,
            first: None,
            latest: false,
            until: None,
            next: None,
            attachments: Vec::new(),
            seen_attachments: std::collections::HashSet::new(),
        }
    }

    /// Restore private state only after the caller validates the content binding.
    pub fn resume(view: View, budget: usize, position: Position, page_end: PageEnd) -> Self {
        let mut pager = Self::new(view, budget, Some(position));
        pager.message = page_end.message;
        pager.masker = page_end.masker;
        pager
    }

    /// An anchor begins a fresh selection at this message. Unlike a sealed
    /// continuation, its first message may have no blocks in the chosen view.
    pub fn at_anchor(mut self) -> Self {
        self.resumed = false;
        self.validated = true;
        self
    }

    pub fn at_latest(mut self) -> Self {
        self.latest = true;
        self.at_anchor()
    }

    /// The first emitted character, after resolving an anchor or latest start.
    pub fn first_position(&self) -> Option<Position> {
        self.first
    }

    pub fn page_end(&self) -> Option<PageEnd> {
        self.next.and(self.page_end.clone())
    }

    /// Consume the message at the requested position, then following messages.
    /// Break as soon as a continuation is known, before decoding later bodies.
    pub fn push(&mut self, message: Message) -> Result<ControlFlow<()>, &'static str> {
        self.push_with_attachments(message, |_| Vec::new())
    }

    pub fn push_with_attachments(
        &mut self,
        mut message: Message,
        labels: impl FnOnce(&Message) -> Vec<crate::AttachmentLabel>,
    ) -> Result<ControlFlow<()>, &'static str> {
        self.page_end = Some(PageEnd {
            message: self.message,
            masker: self.masker.clone(),
        });
        // Replay prior source messages on resume. Hidden blocks establish the
        // same context ordinary and structured outputs use before projection.
        // Pages are text: values no text view shows stay unmasked.
        self.masker.message_text(&mut message);
        if self.message < self.at.message {
            return self.push_blocks(Vec::new(), header(&message), Vec::new());
        }
        let labels = labels(&message);
        let annotations = crate::attachments::apply(&mut message, self.view, labels);
        self.header = Some(MessageHeader {
            id: message.id.clone(),
            session: message.session.clone(),
            role: message.role,
            at: message.at,
            model: message.model.clone(),
            written_by: polylogue_model::WrittenBy::of(message.role, message.prompt_kind),
        });
        let blocks = rendered_blocks(&message, self.view);
        self.source_blocks = blocks.iter().map(|block| block.source_index).collect();
        self.push_blocks(blocks, header(&message), annotations)
    }

    /// Stop at an exclusive position, used for a page read backwards.
    pub fn until(mut self, end: Position) -> Self {
        self.until = Some(end);
        self
    }

    pub fn visible_attachments(&self) -> &[crate::AttachmentRef] {
        &self.attachments
    }

    fn push_blocks(
        &mut self,
        blocks: Vec<crate::RenderedBlock>,
        heading: String,
        annotations: Vec<crate::attachments::Annotation>,
    ) -> Result<ControlFlow<()>, &'static str> {
        let current = self.message;
        self.message += 1;
        if current < self.at.message {
            return Ok(ControlFlow::Continue(()));
        }

        if self.resumed && !self.validated && self.at.block >= blocks.len() {
            return Err("the read cursor block is outside the selected record");
        }
        if self.latest && !blocks.is_empty() {
            self.at.block = blocks.len() - 1;
        }
        while self.at.block < blocks.len() {
            let p = &blocks[self.at.block];
            let body = block_body(p, &heading, self.at.message, self.at.block);
            let prefix = body
                .chars()
                .count()
                .saturating_sub(p.text.chars().count() + 1);
            let whole_length = body.chars().count();
            let end = self
                .until
                .filter(|end| end.message == current && end.block == self.at.block);
            let length = end.map_or(whole_length, |end| end.character.min(whole_length));
            if end.is_some() && self.at.character >= length {
                self.next = self.until;
                return Ok(ControlFlow::Break(()));
            }
            if self.latest {
                self.latest = false;
                self.at.character = length.saturating_sub(self.budget);
            }
            if self.at.character >= length {
                return Err("the read cursor character is outside the block");
            }
            self.validated = true;
            let room = self.budget.saturating_sub(self.used);
            let remaining = length - self.at.character;
            let take = remaining.min(room);
            for annotation in annotations
                .iter()
                .filter(|annotation| annotation.block == self.at.block)
            {
                if prefix + annotation.from < self.at.character + take
                    && prefix + annotation.until > self.at.character
                    && self.seen_attachments.insert(annotation.reference.clone())
                {
                    self.attachments.push(annotation.reference.clone());
                }
            }
            if take > 0 {
                self.first.get_or_insert(self.at);
                let text_length = p.text.chars().count();
                let from = self.at.character.saturating_sub(prefix).min(text_length);
                let until = (self.at.character + take)
                    .saturating_sub(prefix)
                    .min(text_length);
                self.slices.push(BlockSlice {
                    message: self.header.clone(),
                    block: self
                        .source_blocks
                        .get(self.at.block)
                        .copied()
                        .unwrap_or(current),
                    kind: p.kind.into(),
                    label: p.label.clone(),
                    outcome: None,
                    text: p.text.chars().skip(from).take(until - from).collect(),
                    from,
                    until,
                    more: until < text_length,
                });
            }
            self.text
                .extend(body.chars().skip(self.at.character).take(take));
            self.used += take;
            if end.is_some() && take == remaining {
                self.next = self.until;
                return Ok(ControlFlow::Break(()));
            }
            if take < remaining {
                self.at.character += take;
                self.next = Some(self.at);
                return Ok(ControlFlow::Break(()));
            }
            self.at.block += 1;
            self.at.character = 0;
        }
        self.at.message += 1;
        self.at.block = 0;
        Ok(ControlFlow::Continue(()))
    }

    pub fn finish(self) -> Result<Page, &'static str> {
        if !self.validated {
            return Err("the read cursor record is outside the selected content");
        }
        Ok(Page {
            text: self.text,
            next: self.next,
            attachments: self.attachments,
            slices: self.slices,
        })
    }
}

fn block_body(block: &crate::RenderedBlock, heading: &str, message: usize, index: usize) -> String {
    let body = match (block.kind, &block.label) {
        ("tool_call", Some(label)) => format!("→ {label}: {}\n", block.text),
        ("tool_result", Some(label)) => format!("← {label}\n{}\n", block.text),
        _ => format!("{}\n", block.text),
    };
    if index == 0 {
        format!("{}{heading}{body}", if message > 0 { "\n" } else { "" })
    } else {
        body
    }
}

/// Find a suffix before an exclusive position using the same masked, annotated
/// block formatting as forward pages. The returned size counts characters.
pub fn previous_start(
    mut message: Message,
    view: View,
    index: usize,
    end: Option<Position>,
    characters: usize,
    mut masker: MessageMasker,
    labels: impl FnOnce(&Message) -> Vec<crate::AttachmentLabel>,
) -> (Option<Position>, usize) {
    masker.message_text(&mut message);
    let labels = labels(&message);
    crate::attachments::apply(&mut message, view, labels);
    let blocks = rendered_blocks(&message, view);
    let heading = header(&message);
    let mut used = 0;
    let mut start = None;
    for (block, item) in blocks.iter().enumerate().rev() {
        if end.is_some_and(|end| block > end.block) {
            continue;
        }
        let length = block_body(item, &heading, index, block).chars().count();
        let until = end
            .filter(|end| end.block == block)
            .map_or(length, |end| end.character.min(length));
        let take = until.min(characters - used);
        if take > 0 {
            start = Some(Position {
                message: index,
                block,
                character: until - take,
            });
            used += take;
        }
        if used == characters {
            break;
        }
    }
    (start, used)
}

/// Role-free content is projected one ordered block at a time, replaying masking on
/// continuation before projecting it. Positions use the same character splitter.
pub struct BlockPager {
    stream: Pager,
    heading: String,
    heading_emitted: bool,
}

impl BlockPager {
    pub fn new(view: View, budget: usize, position: Option<Position>, heading: String) -> Self {
        Self {
            stream: Pager::new(view, budget, position),
            heading,
            heading_emitted: false,
        }
    }
    pub fn push(
        &mut self,
        mut block: polylogue_model::Block,
    ) -> Result<ControlFlow<()>, &'static str> {
        self.stream.masker.block(&mut block);
        self.stream.source_blocks = vec![self.stream.message];
        let blocks = if self.stream.view.keeps_block(false, &block) {
            crate::block(&block).into_iter().collect()
        } else {
            Vec::new()
        };
        let heading = if !self.heading_emitted && !blocks.is_empty() {
            self.heading_emitted = true;
            self.heading.clone()
        } else {
            String::new()
        };
        self.stream.push_blocks(blocks, heading, Vec::new())
    }
    pub fn finish(self) -> Result<Page, &'static str> {
        self.stream.finish()
    }
}

fn header(m: &Message) -> String {
    let who = crate::who(m);
    match m.at {
        Some(t) => format!("[{who} {t}]\n"),
        None => format!("[{who}]\n"),
    }
}

#[cfg(test)]
mod tests;
