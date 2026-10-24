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
}

/// A position in the rendered stream. The caller binds it to the read view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Position {
    pub message: usize,
    pub piece: usize,
    pub character: usize,
}

pub struct Pager {
    view: View,
    budget: usize,
    at: Position,
    resumed: bool,
    message: usize,
    masker: MessageMasker,
    validated: bool,
    used: usize,
    text: String,
    next: Option<Position>,
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
            validated: position.is_none(),
            used: 0,
            text: String::new(),
            next: None,
        }
    }

    /// Consume the message at the requested position, then following messages.
    /// Break as soon as a continuation is known, before decoding later bodies.
    pub fn push(&mut self, mut message: Message) -> Result<ControlFlow<()>, &'static str> {
        // Replay prior source messages on resume. Hidden blocks establish the
        // same context ordinary and structured outputs use before projection.
        self.masker.message(&mut message);
        let ps = pieces(&message, self.view);
        self.push_pieces(ps, header(&message))
    }

    fn push_pieces(
        &mut self,
        ps: Vec<crate::Piece>,
        heading: String,
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
            let length = body.chars().count();
            if self.at.character >= length {
                return Err("the read cursor character is outside the piece");
            }
            self.validated = true;
            let room = self.budget.saturating_sub(self.used);
            let remaining = length - self.at.character;
            let take = remaining.min(room);
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
        self.stream.push_pieces(ps, heading)
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
