use super::*;

#[test]
fn the_manual_gives_working_example_queries() {
    let guide = include_str!("../../../../docs/agent-manual.md");
    let examples: Vec<_> = guide
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|text| text.starts_with("sessions where ") || text.starts_with("messages where "))
        .collect();
    assert!(!examples.is_empty());
    let options = polylogue_query::Options {
        embeddings: true,
        ..polylogue_query::Options::default()
    };
    for query in examples {
        let answer = polylogue_query::compile(query, &options);
        assert!(answer.is_ok(), "{query}: {answer:?}");
    }
    let manual = text();
    for tool in [
        "brief", "recall", "takeover", "query", "read", "note_add", "status",
    ] {
        assert!(
            crate::mcp::tools().iter().any(|entry| entry.name == tool),
            "no tool {tool}"
        );
        assert!(
            manual.contains(&format!("`{tool}`")),
            "{tool} is not described"
        );
    }
}
