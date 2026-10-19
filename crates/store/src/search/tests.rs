use super::*;

#[test]
fn punctuation_splits_words_instead_of_failing() {
    assert_eq!(
        words("/realm/project/polylogue: cargo-nextest!"),
        ["realm", "project", "polylogue", "cargo", "nextest"]
    );
    assert_eq!(match_all("a.b").as_deref(), Some("\"a\" AND \"b\""));
    assert_eq!(match_all("--- !!"), None);
}

#[test]
fn words_fold_strokes_as_the_index_does() {
    assert_eq!(words("Łódź"), ["lódź"]);
}

#[test]
fn snippets_centre_on_the_first_match() {
    let text = format!(
        "{} the needle is here {}",
        "x ".repeat(200),
        "y ".repeat(200)
    );
    let s = snippet(&text, &["needle".into()], 60);
    assert!(s.contains("needle"));
    assert!(s.starts_with('…') && s.ends_with('…'));
}

#[test]
fn cuts_around_the_first_match_and_marks_every_match() {
    let text = "one two three four five six seven eight nine ten eleven twelve thirteen chrono fourteen fifteen sixteen seventeen eighteen nineteen twenty chrono";
    let (cut_text, marks) = cut(text, &["chrono".into()]);
    assert!(cut_text.starts_with('…'));
    assert!(cut_text.ends_with('…'));
    assert_eq!(marks.len(), 1);
    let (s, e) = marks[0];
    assert_eq!(&cut_text[s..e], "chrono");
}

#[test]
fn folding_and_prefixes_mark_like_the_index_matches() {
    let (t, marks) = cut("Wyjazd do Łodzi w piątek", &["lodzi".into()]);
    assert_eq!(&t[marks[0].0..marks[0].1], "Łodzi");
    let (t, marks) = cut("we refactored the parser", &["refactor*".into()]);
    assert_eq!(&t[marks[0].0..marks[0].1], "refactored");
    let (t, marks) = cut("call weekday_offset() here", &["weekday_offset()".into()]);
    assert_eq!(marks.len(), 2);
    assert_eq!(&t[marks[0].0..marks[0].1], "weekday");
}
