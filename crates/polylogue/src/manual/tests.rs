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
        clock: polylogue_query::time::clock(
            1_790_000_000_000,
            polylogue_model::LogicalDay::named("UTC", 240).unwrap(),
        ).unwrap(),
        origins: Vec::new(),
        command_words: Vec::new(),
        page_size: None,
        random_seed: None,
        embeddings: true,
        self_session: None,
        saved: Default::default(),
        fields: Vec::new(),
        kept: Default::default(),
        chat_pages: Vec::new(),
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
            crate::mcp::tools()
                .iter()
                .any(|entry| entry["name"] == tool),
            "no tool {tool}"
        );
        assert!(
            manual.contains(&format!("`{tool}`")),
            "{tool} is not described"
        );
    }
}
