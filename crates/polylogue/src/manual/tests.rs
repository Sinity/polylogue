use super::*;

#[test]
fn the_manual_lists_each_tool_with_its_operations_and_arguments() {
    let manual = text();
    assert!(manual.starts_with(card()));
    // Each operation is listed under its own tool's heading, which is the
    // last occurrence of the heading (the guide comes first).
    let section = |tool: crate::commands::Tool| {
        let heading = format!("\n## `{}`\n", tool.name());
        let start = manual
            .rfind(&heading)
            .unwrap_or_else(|| panic!("{} has no section", tool.name()))
            + heading.len();
        let rest = &manual[start..];
        &rest[..rest.find("\n## `").unwrap_or(rest.len())]
    };
    for tool in crate::commands::Tool::ALL {
        let body = section(tool);
        for command in crate::mcp::operations(tool) {
            let prefix = format!(
                "- `{}` (CLI `{}`): ",
                command.name,
                command.cli_words().join(" ")
            );
            let line = body
                .lines()
                .find(|line| line.starts_with(&prefix))
                .unwrap_or_else(|| {
                    panic!(
                        "{} has no runnable CLI spelling under {}",
                        command.name,
                        tool.name()
                    )
                });
            // Every argument is named, unlisted ones included.
            let (_, named) = line.rsplit_once("; arguments: ").unwrap();
            let named: Vec<&str> = named.split(", ").collect();
            let input = &crate::commands::declarations()
                .descriptions()
                .iter()
                .find(|description| description.spec.name == command.name)
                .unwrap()
                .input_schema;
            for name in input["properties"]
                .as_object()
                .into_iter()
                .flatten()
                .map(|(name, _)| name)
            {
                if name != "request_id" {
                    assert!(named.contains(&name.as_str()), "{} {}", command.name, name);
                }
            }
        }
    }
}
