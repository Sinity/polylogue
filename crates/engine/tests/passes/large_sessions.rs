use super::*;

/// More messages than SQLite takes parameters in one statement.
const PAST_THE_PARAMETER_LIMIT: usize = 32_767 + 233;

/// A session past SQLite's parameter limit exports every message, and
/// excising it removes every row, text and search entry it had; nothing
/// lists its messages as statement parameters.
#[test]
fn a_session_past_the_parameter_limit_exports_and_excises_whole() {
    let w = ClaudeHome::new();
    let mut lines = String::new();
    for i in 0..PAST_THE_PARAMETER_LIMIT {
        lines += &if i % 2 == 0 {
            user(&format!("u{i}"), "big", &format!("quokka prompt {i}"))
        } else {
            assistant(&format!("a{i}"), "big", &format!("quokka answer {i}"))
        };
    }
    w.append("big.jsonl", &lines);
    w.append("small.jsonl", &user("s1", "small", "kept"));
    let (mut e, _lock) = w.engine(64 << 20);
    let pass = e.run_pass().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);

    let export = w.export("claude-code:big");
    assert_eq!(export.messages.len(), PAST_THE_PARAMETER_LIMIT);

    let done = excise(&mut e, "session:claude-code:big", false);
    assert_eq!(done.sessions, ["claude-code:big"]);
    assert_eq!(done.messages, PAST_THE_PARAMETER_LIMIT as u64);
    assert!(w.texts("big").is_empty());
    assert_eq!(
        w.count(
            "SELECT count(*) AS n FROM message m JOIN session s ON s.pk = m.session_pk \
             WHERE s.id = 'claude-code:big'"
        ),
        0
    );
    assert_eq!(
        w.count("SELECT count(*) AS n FROM text_fts WHERE text_fts MATCH 'quokka'"),
        0
    );
    assert!(!stored(&w.root, "quokka"));
    assert_eq!(w.texts("small"), ["kept"]);
}
