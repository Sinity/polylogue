use super::*;

#[test]
fn strokes_fold_like_accents() {
    assert_eq!(key("Łódź"), "lodz");
    assert_eq!(key("łódź"), "lodz");
    assert_eq!(key("Straße"), "strasse");
    assert_eq!(fold("Ørsted"), "Orsted");
    // A capital folds to the capital of the same base letter as its small
    // form, so the tokenizer's case fold brings both to one key.
    for (capital, small, base) in [
        ("Ƀ", "ƀ", "b"),
        ("Ɨ", "ɨ", "i"),
        ("Ʉ", "ʉ", "u"),
        ("Ƶ", "ƶ", "z"),
        ("Ɉ", "ɉ", "j"),
        ("Ɍ", "ɍ", "r"),
        ("Ǥ", "ǥ", "g"),
    ] {
        assert_eq!(key(capital), base);
        assert_eq!(key(small), base);
    }
}

#[test]
fn tokens_split_on_anything_but_letters_and_digits() {
    let t = "weekday_offset() in 2026, café";
    let words: Vec<&str> = tokens(t).into_iter().map(|(s, e)| &t[s..e]).collect();
    assert_eq!(words, ["weekday", "offset", "in", "2026", "café"]);
}

#[test]
fn private_use_characters_stay_inside_a_token() {
    let t = "build\u{E0B0}done next";
    let words: Vec<&str> = tokens(t).into_iter().map(|(s, e)| &t[s..e]).collect();
    assert_eq!(words, ["build\u{E0B0}done", "next"]);
    assert_eq!(
        fts_phrase("build\u{E0B0}done").as_deref(),
        Some("\"build\u{E0B0}done\"")
    );
    assert_eq!(key("ƚ"), "l");
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
