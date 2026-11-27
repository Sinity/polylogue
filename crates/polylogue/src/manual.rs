//! Agent guidance: the short card MCP sends as its instructions, and the
//! full manual (`polylogue help --all`): the card, the guide and the operation
//! reference.
use std::fmt::Write;

/// The card every MCP session receives; detail is in `help --all`.
pub fn card() -> &'static str {
    include_str!("../../../docs/agent-manual.md")
}

/// The card, the guide, and every tool operation with its CLI spelling,
/// help and arguments, unlisted ones included, and every verb the tool adds
/// with its signature.
pub fn text() -> String {
    let mut text = format!(
        "{}\n{}",
        card(),
        include_str!("../../../docs/agent-guide.md")
    );
    let offered = crate::daemon::verbs::offered_over_mcp();
    let verbs = crate::daemon::verbs::declarations();
    for tool in crate::commands::Tool::ALL {
        let _ = writeln!(text, "\n## `{}`\n", tool.name());
        for verb in verbs.verbs() {
            if offered.contains(&(verb.verb(), tool.access())) {
                let _ = writeln!(
                    text,
                    "- step `{}`: {}",
                    verb.signature(),
                    crate::commands::verb_sentence(verb)
                );
            }
        }
        for command in crate::mcp::operations(tool) {
            let arguments: Vec<&str> = crate::commands::declarations()
                .spec_named(command.name)
                .map(|spec| spec.arguments.as_slice())
                .unwrap_or_default()
                .iter()
                .filter(|arg| !arg.calling_process && arg.name != "request_id")
                .map(|arg| arg.name.as_str())
                .collect();
            let _ = writeln!(
                text,
                "- `{}` (CLI `{}`): {}; arguments: {}",
                command.name,
                command.cli_words().join(" "),
                command.help,
                if arguments.is_empty() {
                    "none".to_owned()
                } else {
                    arguments.join(", ")
                }
            );
        }
    }
    text
}

#[cfg(test)]
mod tests;
