//! Semantic cards: a compact summary of a session built from outcomes
//! (tool results, edits, tests run), not from guessing at the text.

use polylogue_model::{Message, Session};

pub struct Card {
    pub title: String,
    pub lines: Vec<String>,
}

pub fn card(_session: &Session, _messages: &[Message]) -> Card {
    // Architecture 7: semantic cards from outcomes (old ap7).
    todo!()
}
