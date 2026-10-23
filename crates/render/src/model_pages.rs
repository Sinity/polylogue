//! Reading for models, consuming one message at a time. Whole original blocks
//! are masked before splitting; no unmasked text is carried in a position.

use crate::{View, unmasked_piece};
use polylogue_model::Message;
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
    masker: credential_masking::Masker,
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
            masker: credential_masking::Masker::new(),
            validated: position.is_none(),
            used: 0,
            text: String::new(),
            next: None,
        }
    }

    /// Consume the message at the requested position, then following messages.
    /// Break as soon as a continuation is known, before decoding later bodies.
    pub fn push(&mut self, message: &Message) -> Result<ControlFlow<()>, &'static str> {
        let mut ps = Vec::new();
        // Replay prior source blocks to reconstruct PEM state on every request.
        // Hidden blocks still establish the context of following visible text.
        for block in &message.blocks {
            let Some(mut piece) = unmasked_piece(block) else {
                continue;
            };
            piece.label = piece.label.map(|label| self.mask_fragment(&label));
            let text = self.mask_fragment(&piece.text);
            if self.view.keeps(message, block) {
                piece.text = text;
                ps.push(piece);
            }
        }
        let current = self.message;
        self.message += 1;
        if current < self.at.message {
            return Ok(ControlFlow::Continue(()));
        }

        if self.resumed && !self.validated && self.at.piece >= ps.len() {
            return Err("the read cursor piece is outside the message");
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
                body = format!("{sep}{}{body}", header(message));
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

    fn mask_fragment(&mut self, text: &str) -> String {
        let mut bytes = Vec::new();
        self.masker
            .push(text, &mut bytes)
            .expect("Vec writes cannot fail");
        self.masker
            .block_boundary(&mut bytes)
            .expect("Vec writes cannot fail");
        String::from_utf8(bytes).expect("masking preserves UTF-8")
    }

    pub fn finish(self) -> Result<Page, &'static str> {
        if !self.validated {
            return Err("the read cursor message is outside the session");
        }
        Ok(Page {
            text: self.text,
            next: self.next,
        })
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
