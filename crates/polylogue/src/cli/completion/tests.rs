use super::*;

#[test]
fn full_query_context_preserves_suffix_and_cursor_inside_a_word() {
    let args = [
        "polylogue",
        "messages",
        "where",
        "model:alpha",
        "|",
        "limit",
        "5",
    ]
    .map(OsString::from);
    let (text, position, start, end) =
        context(&args, 3, 2, "model:alpha", Some("model:al")).unwrap();
    assert_eq!(text, "messages where model:alpha | limit 5");
    assert_eq!(&text[..position], "messages where model:al");
    assert_eq!(&text[start..end], "model:alpha");
    assert_eq!(&text[end..], " | limit 5");
}

#[test]
fn quoted_whole_query_uses_its_original_utf8_offsets() {
    let args = ["polylogue", "messages where body:żółw"].map(OsString::from);
    let (text, position, start, end) = context(
        &args,
        1,
        0,
        "messages where body:żółw",
        Some("messages where body:żó"),
    )
    .unwrap();
    assert_eq!(position, "messages where body:żó".len());
    assert!(text.is_char_boundary(position));
    assert_eq!(&text[start..end], "messages where body:żółw");
}
