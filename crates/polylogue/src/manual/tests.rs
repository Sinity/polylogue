use super::*;

#[test]
fn the_manual_gives_working_example_queries() {
    let guide = text();
    let examples: Vec<_> = guide
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|text| text.starts_with("sessions where ") || text.starts_with("messages where "))
        .collect();
    assert!(!examples.is_empty());
    let options = polylogue_query::Options {
        clock: query_language::time::Clock::new(
            jiff::Timestamp::from_millisecond(1_790_000_000_000).unwrap(),
            query_language::time::DaySetting::new(
                jiff::tz::TimeZone::get("UTC").unwrap(),
                jiff::civil::Time::constant(4, 0, 0, 0),
            )
            .unwrap(),
        ),
        origins: Vec::new(),
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
        let answer = polylogue_store::query::compile(query, &options);
        assert!(answer.is_ok(), "{query}: {answer:?}");
    }
    let manual = text();
    assert!(manual.starts_with(card()));
    for tool in crate::commands::Tool::ALL {
        assert!(
            manual.contains(&format!("## `{}`", tool.name())),
            "{} has no section",
            tool.name()
        );
    }
    for command in crate::mcp::tool_commands() {
        let prefix = format!(
            "- `{}` (CLI `{}`)",
            command.name,
            command.cli_words().join(" ")
        );
        let line = manual
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("{} has no runnable CLI spelling", command.name));
        // Every argument is named, unlisted ones included.
        let (_, named) = line.rsplit_once("; arguments: ").unwrap();
        let named: Vec<&str> = named.split(", ").collect();
        for arg in command.args_for(crate::commands::Surface::Mcp) {
            if arg.name != "request_id" {
                assert!(named.contains(&arg.name), "{} {}", command.name, arg.name);
            }
        }
    }
}
