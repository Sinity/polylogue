use super::*;

#[test]
fn punctuation_splits_words_instead_of_failing() {
    assert_eq!(
        words("/workspace/project/polylogue: cargo-nextest!"),
        ["workspace", "project", "polylogue", "cargo", "nextest"]
    );
    assert!(words("--- !!").is_empty());
}

#[test]
fn words_fold_strokes_as_the_index_does() {
    assert_eq!(words("Łódź"), ["lódź"]);
}

/// The marked text of a snippet, with character offsets.
fn marked(text: &str, (start, end): (usize, usize)) -> String {
    text.chars().skip(start).take(end - start).collect()
}

#[test]
fn cuts_around_the_first_match_and_marks_every_match() {
    let text = "one two three four five six seven eight nine ten eleven twelve thirteen chrono fourteen fifteen sixteen seventeen eighteen nineteen twenty chrono";
    let (cut_text, marks) = cut(text, &["chrono"]);
    assert!(cut_text.starts_with('…'));
    assert!(cut_text.ends_with('…'));
    assert_eq!(marks.len(), 1);
    assert_eq!(marked(&cut_text, marks[0]), "chrono");
}

#[test]
fn folding_and_prefixes_mark_like_the_index_matches() {
    let (t, marks) = cut("Wyjazd do Łodzi w piątek", &["lodzi"]);
    assert_eq!(marked(&t, marks[0]), "Łodzi");
    let (t, marks) = cut("call weekday_offset() here", &["weekday_offset()"]);
    assert_eq!(marks.len(), 2);
    assert_eq!(marked(&t, marks[0]), "weekday");
}

#[test]
fn cut_marks_a_prefix_term_by_its_marker() {
    let (t, marks) = cut("we refactored the parser", &["refactor*"]);
    assert_eq!(marks.len(), 1);
    assert_eq!(marked(&t, marks[0]), "refactored");
    // Without the marker the word is whole and nothing matches.
    assert!(cut("we refactored the parser", &["refactor"]).1.is_empty());
}

mod source_text;
