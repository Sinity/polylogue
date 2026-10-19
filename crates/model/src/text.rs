//! Text as search sees it. The full-text index (SQLite FTS5, `unicode61`
//! with `remove_diacritics 2`) folds case and accents and splits on
//! anything that is not a letter or digit. It does not fold letters with a
//! stroke, so `fold` maps those first, both when text is indexed and when a
//! query is written; `Łódź` is then found by `lodz` (spec query.md).
//!
//! `tokens` and `key` reproduce the tokenizer's view in Rust, for marking
//! matches in snippets and for turning a typed term into an FTS5 phrase.

use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

/// Letters with a stroke mapped to their base letter, the one fold the
/// index tokenizer does not do itself. Everything else is left as written.
pub fn fold(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'ß' => out.push_str("ss"),
            'ẞ' => out.push_str("SS"),
            'ł' => out.push('l'),
            'Ł' => out.push('L'),
            'ø' => out.push('o'),
            'Ø' => out.push('O'),
            'đ' => out.push('d'),
            'Đ' => out.push('D'),
            'ħ' => out.push('h'),
            'Ħ' => out.push('H'),
            'ı' => out.push('i'),
            'ŧ' => out.push('t'),
            'Ŧ' => out.push('T'),
            'ƀ' => out.push('b'),
            'ɨ' => out.push('i'),
            'ʉ' => out.push('u'),
            _ => out.push(c),
        }
    }
    out
}

/// The comparison key of one token: stroke-folded, accents removed,
/// lower-cased, as the index compares it.
pub fn key(token: &str) -> String {
    fold(token)
        .nfd()
        .filter(|c| !is_combining_mark(*c))
        .flat_map(char::to_lowercase)
        .collect()
}

/// The tokens of `text` with their byte ranges: maximal runs of letters and
/// digits (combining marks stay inside the run they follow).
pub fn tokens(text: &str) -> Vec<(usize, usize)> {
    let mut out = vec![];
    let mut start = None;
    for (i, c) in text.char_indices() {
        let inside = c.is_alphanumeric() || (start.is_some() && is_combining_mark(c));
        match (inside, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, i));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
    out
}

/// One typed search term as an FTS5 expression: its tokens as a phrase
/// (`weekday_offset()` becomes `"weekday offset"`), with a trailing `*`
/// kept as a prefix search on the last token. `None` when the term holds
/// no letter or digit, so there is nothing the index could match.
pub fn fts_phrase(term: &str) -> Option<String> {
    let prefix = term.ends_with('*');
    let folded = fold(term.trim_end_matches('*'));
    let words: Vec<&str> = tokens(&folded)
        .into_iter()
        .map(|(s, e)| &folded[s..e])
        .collect();
    if words.is_empty() {
        return None;
    }
    let mut phrase = format!("\"{}\"", words.join(" ").replace('"', "\"\""));
    if prefix {
        phrase.push('*');
    }
    Some(phrase)
}

#[cfg(test)]
mod tests;
