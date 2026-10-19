//! Reading for models: a session in pages sized to a token budget, with a
//! plain cursor to continue from. Size is a parameter, never a refusal: a
//! block longer than the budget is split across pages, and every page
//! carries at least some text. Credentials are masked per block before
//! anything is measured.

use crate::{View, pieces};
use polylogue_model::Message;

/// About four characters per token: the estimate pages are sized by.
const CHARS_PER_TOKEN: usize = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Page {
    pub text: String,
    /// Where the next page starts; `None` at the end.
    pub next: Option<String>,
}

/// A position: message, piece within it, character within the piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct At {
    message: usize,
    piece: usize,
    char: usize,
}

fn parse_cursor(c: &str) -> Option<At> {
    let mut parts = c.split('.');
    let message = parts.next()?.strip_prefix('m')?.parse().ok()?;
    let piece = parts.next()?.strip_prefix('b')?.parse().ok()?;
    let char = parts.next()?.strip_prefix('c')?.parse().ok()?;
    Some(At {
        message,
        piece,
        char,
    })
}

fn cursor(at: At) -> String {
    format!("m{}.b{}.c{}", at.message, at.piece, at.char)
}

fn header(m: &Message) -> String {
    let who = match (m.role, m.prompt_kind) {
        (polylogue_model::Role::Assistant, _) => "assistant",
        (polylogue_model::Role::System, _) => "system",
        (_, Some(polylogue_model::PromptKind::Harness)) => "harness",
        (_, Some(polylogue_model::PromptKind::Model)) => "model",
        _ => "user",
    };
    match m.at {
        Some(t) => format!("[{who} {t}]\n"),
        None => format!("[{who}]\n"),
    }
}

/// One page of the dialogue and actions of `messages`, at most
/// `budget_tokens` (at least one character), starting at `cursor`.
pub fn page(messages: &[Message], budget_tokens: usize, cursor_at: Option<&str>) -> Page {
    let budget = budget_tokens.max(1) * CHARS_PER_TOKEN;
    let mut at = cursor_at.and_then(parse_cursor).unwrap_or_default();
    let mut out = String::new();
    while at.message < messages.len() {
        let m = &messages[at.message];
        let ps = pieces(m, View::Full);
        while at.piece < ps.len() {
            let p = &ps[at.piece];
            let mut body = match (p.kind, &p.label) {
                ("tool_call", Some(l)) => format!("→ {l}: {}\n", p.text),
                ("tool_result", Some(l)) => format!("← {l}\n{}\n", p.text),
                (_, _) => format!("{}\n", p.text),
            };
            // The message's header leads its first piece, so it splits
            // across pages like any text and a cursor can point inside it.
            if at.piece == 0 {
                let sep = if at.message > 0 { "\n" } else { "" };
                body = format!("{sep}{}{body}", header(m));
            }
            let rest: String = body.chars().skip(at.char).collect();
            let room = budget.saturating_sub(out.chars().count());
            let len = rest.chars().count();
            if len <= room {
                out.push_str(&rest);
                at.piece += 1;
                at.char = 0;
                continue;
            }
            // Split the piece; a page always carries something.
            let take = if out.is_empty() { room.max(1) } else { room };
            if take == 0 {
                return Page {
                    text: out,
                    next: Some(cursor(at)),
                };
            }
            out.extend(rest.chars().take(take));
            at.char += take;
            return Page {
                text: out,
                next: Some(cursor(at)),
            };
        }
        at.message += 1;
        at.piece = 0;
        at.char = 0;
    }
    Page {
        text: out,
        next: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::conversation;

    fn all_pages(budget: usize) -> Vec<Page> {
        let ms = conversation();
        let mut pages = vec![];
        let mut c: Option<String> = None;
        loop {
            let p = page(&ms, budget, c.as_deref());
            c = p.next.clone();
            pages.push(p);
            if c.is_none() {
                break;
            }
            assert!(pages.len() < 10_000, "pages must make progress");
        }
        pages
    }

    #[test]
    fn pages_cover_the_whole_session_once_at_any_budget() {
        let whole = all_pages(1_000_000);
        assert_eq!(whole.len(), 1);
        let full = whole[0].text.clone();
        assert!(full.contains("→ Bash: cargo test"));
        assert!(full.contains("← error (exit 101)"));
        assert!(
            !full.contains("sk-ant-api03-AAAA"),
            "masked before measuring"
        );
        for budget in [1, 2, 5, 13, 40] {
            let joined: String = all_pages(budget).into_iter().map(|p| p.text).collect();
            assert_eq!(joined, full, "budget {budget}");
        }
    }

    #[test]
    fn a_page_stays_within_its_budget() {
        for p in all_pages(10) {
            assert!(p.text.chars().count() <= 40, "{:?}", p.text);
            assert!(!p.text.is_empty());
        }
    }
}
