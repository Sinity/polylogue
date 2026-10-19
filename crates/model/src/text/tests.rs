use super::*;

#[test]
fn strokes_fold_like_accents() {
    assert_eq!(key("Łódź"), "lodz");
    assert_eq!(key("łódź"), "lodz");
    assert_eq!(key("Straße"), "strasse");
    assert_eq!(fold("Ørsted"), "Orsted");
}

#[test]
fn tokens_split_on_anything_but_letters_and_digits() {
    let t = "weekday_offset() in 2026, café";
    let words: Vec<&str> = tokens(t).into_iter().map(|(s, e)| &t[s..e]).collect();
    assert_eq!(words, ["weekday", "offset", "in", "2026", "café"]);
}

#[test]
fn a_term_becomes_a_phrase_or_nothing() {
    assert_eq!(
        fts_phrase("weekday_offset()").as_deref(),
        Some("\"weekday offset\"")
    );
    assert_eq!(fts_phrase("refactor*").as_deref(), Some("\"refactor\"*"));
    assert_eq!(fts_phrase("Łódź").as_deref(), Some("\"Lódź\""));
    assert_eq!(fts_phrase("->"), None);
    assert_eq!(fts_phrase("say \"hi\"").as_deref(), Some("\"say hi\""));
}
