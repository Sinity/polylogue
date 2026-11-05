//! Agent guidance: the short card MCP sends as its instructions, and the
//! full manual (`polylogue manual`): the card, the guide and the operation
//! reference.
use std::fmt::Write;

/// The card every MCP session receives; detail is in `explain`.
pub fn card() -> &'static str {
    include_str!("../../../docs/agent-manual.md")
}

/// The card, the guide, and every tool operation with its CLI spelling,
/// help and arguments, unlisted ones included.
pub fn text() -> String {
    let mut text = format!(
        "{}\n{}",
        card(),
        include_str!("../../../docs/agent-guide.md")
    );
    for tool in crate::commands::Tool::ALL {
        let _ = writeln!(text, "\n## `{}`\n", tool.name());
        for command in crate::mcp::operations(tool) {
            let arguments: Vec<&str> = command
                .args_for(crate::commands::Surface::Mcp)
                .map(|arg| arg.name)
                .filter(|name| *name != "request_id")
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
