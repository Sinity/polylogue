//! Standing guidance shared by MCP and the offline CLI help topic.
use std::fmt::Write;

pub fn text() -> String {
    let mut text = include_str!("../../../docs/agent-manual.md").to_owned();
    for command in crate::commands::COMMANDS.iter().filter(|c| c.mcp) {
        let _ = writeln!(
            text,
            "\n- `{}` (CLI `{}`): {}",
            command.mcp_name(),
            command.name,
            command.help
        );
    }
    text
}

#[cfg(test)]
mod tests;
