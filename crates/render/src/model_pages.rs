//! Reading for models, consuming one message at a time. Whole original blocks
//! are masked before splitting; no unmasked text is carried in a position.

use crate::{View, pieces};
use polylogue_model::Message;
use polylogue_model::masking::MessageMasker;
use std::ops::ControlFlow;

const CHARS_PER_TOKEN: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub text: String,
    pub next: Option<Position>,
    pub attachments: Vec<crate::AttachmentRef>,
}

/// A position in the rendered stream. The caller binds it to the read view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    pub message: usize,
    pub piece: usize,
    pub character: usize,
}

/// Private source-stream state before one message. It can contain source
/// characters from an open PEM block; never serialize it into a read cursor.
#[derive(Clone)]
pub struct Checkpoint {
    message: usize,
    masker: MessageMasker,
}

impl Checkpoint {
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
    checkpoint: Option<Checkpoint>,
    validated: bool,
    used: usize,
    text: String,
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
            checkpoint: None,
            validated: position.is_none(),
            used: 0,
            text: String::new(),
            next: None,
            attachments: Vec::new(),
            seen_attachments: std::collections::HashSet::new(),
        }
    }

    /// Restore private state only after the caller validates the content binding.
    pub fn resume(view: View, budget: usize, position: Position, checkpoint: Checkpoint) -> Self {
        let mut pager = Self::new(view, budget, Some(position));
        pager.message = checkpoint.message;
        pager.masker = checkpoint.masker;
        pager
    }

    pub fn checkpoint(&self) -> Option<Checkpoint> {
        self.next.and(self.checkpoint.clone())
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
        self.checkpoint = Some(Checkpoint {
            message: self.message,
            masker: self.masker.clone(),
        });
        // Replay prior source messages on resume. Hidden blocks establish the
        // same context ordinary and structured outputs use before projection.
        self.masker.message(&mut message);
        if self.message < self.at.message {
            return self.push_pieces(Vec::new(), header(&message), Vec::new());
        }
        let labels = labels(&message);
        let annotations = crate::attachments::apply(&mut message, self.view, labels);
        let ps = pieces(&message, self.view);
        self.push_pieces(ps, header(&message), annotations)
    }

    pub fn visible_attachments(&self) -> &[crate::AttachmentRef] {
        &self.attachments
    }

    fn push_pieces(
        &mut self,
        ps: Vec<crate::Piece>,
        heading: String,
        annotations: Vec<crate::attachments::Annotation>,
    ) -> Result<ControlFlow<()>, &'static str> {
        let current = self.message;
        self.message += 1;
        if current < self.at.message {
            return Ok(ControlFlow::Continue(()));
        }

        if self.resumed && !self.validated && self.at.piece >= ps.len() {
            return Err("the read cursor piece is outside the selected record");
        }
        while self.at.piece < ps.len() {
            let p = &ps[self.at.piece];
            let mut body = match (p.kind, &p.label) {
                ("tool_call", Some(l)) => format!("→ {l}: {}\n", p.text),
                ("tool_result", Some(l)) => format!("← {l}\n{}\n", p.text),
                _ => format!("{}\n", p.text),
            };
            if self.at.piece == 0 {
                let sep = if self.at.message > 0 { "\n" } else { "" };
                body = format!("{sep}{}{body}", heading);
            }
            let prefix = body
                .chars()
                .count()
                .saturating_sub(p.text.chars().count() + 1);
            let length = body.chars().count();
            if self.at.character >= length {
                return Err("the read cursor character is outside the piece");
            }
            self.validated = true;
            let room = self.budget.saturating_sub(self.used);
            let remaining = length - self.at.character;
            let take = remaining.min(room);
            for annotation in annotations
                .iter()
                .filter(|annotation| annotation.piece == self.at.piece)
            {
                if prefix + annotation.from < self.at.character + take
                    && prefix + annotation.until > self.at.character
                    && self.seen_attachments.insert(annotation.reference.clone())
                {
                    self.attachments.push(annotation.reference.clone());
                }
            }
            self.text
                .extend(body.chars().skip(self.at.character).take(take));
            self.used += take;
            if take < remaining {
                self.at.character += take;
                self.next = Some(self.at);
                return Ok(ControlFlow::Break(()));
            }
            self.at.piece += 1;
            self.at.character = 0;
        }
        self.at.message += 1;
        self.at.piece = 0;
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
        })
    }
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
        let ps = if self.stream.view.keeps_block(false, &block) {
            crate::piece(&block).into_iter().collect()
        } else {
            Vec::new()
        };
        let heading = if !self.heading_emitted && !ps.is_empty() {
            self.heading_emitted = true;
            self.heading.clone()
        } else {
            String::new()
        };
        self.stream.push_pieces(ps, heading, Vec::new())
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
