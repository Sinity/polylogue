use super::*;

/// More messages than SQLite takes parameters in one statement.
const PAST_THE_PARAMETER_LIMIT: usize = 32_767 + 233;

/// A session past SQLite's parameter limit exports every message without
/// listing message IDs as parameters.
#[test]
fn a_session_past_the_parameter_limit_exports_whole() {
    let w = Home::new();
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
    let exported: Vec<String> = export
        .iter()
        .filter_map(|record| match record {
            polylogue_model::bundle::Record::Message(record) => Some(record.message.text()),
            _ => None,
        })
        .collect();
    let expected: Vec<String> = (0..PAST_THE_PARAMETER_LIMIT)
        .map(|i| {
            if i % 2 == 0 {
                format!("quokka prompt {i}")
            } else {
                format!("quokka answer {i}")
            }
        })
        .collect();
    assert_eq!(exported, expected);

    assert_eq!(w.texts("small"), ["kept"]);
}
