use super::*;

#[test]
fn the_logical_session_default_names_every_lifting_field() {
    let why = DefaultId::LogicalSessions.describe();
    for field in DefaultId::LogicalSessions.lifted_by() {
        assert!(why.contains(&format!("{field}:")), "{field}: {why}");
    }
    assert!(why.ends_with("or a reference lifts it)"), "{why}");
}

#[test]
fn id_shaped_words_are_eight_or_more_hex_or_dash_characters() {
    assert!(id_shaped("0198ab12"));
    assert!(id_shaped("ab-cdefa"));
    assert!(id_shaped("deadbeef"));
    assert!(!id_shaped("--------"));
    assert!(!id_shaped("0198ab1"));
    assert!(!id_shaped("0198abxz"));
}
