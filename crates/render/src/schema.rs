//! Offline schema reports contain structure and counts only.
use crate::Format;
use serde_json::Value;
use std::io::{self, Write};

pub fn render(data: &Value, format: Format, out: &mut dyn Write) -> io::Result<()> {
    if format == Format::Json || data.get("fields").is_some() {
        serde_json::to_writer_pretty(&mut *out, data)?;
        return writeln!(out);
    }
    writeln!(out, "Rule version: {}", data["rule_version"])?;
    for side in ["before", "after"] {
        let sample = &data[side];
        writeln!(
            out,
            "{side}: {} (sample: {}), {} records, {} unreadable inputs, {} unreadable records",
            sample["origin"].as_str().unwrap_or_default().escape_debug(),
            sample["sample"],
            sample["records"],
            sample["unreadable_inputs"],
            sample["unreadable_records"]
        )?;
        if let Some(selection) = sample["selection"].as_array() {
            for input in selection {
                writeln!(
                    out,
                    "  input {} revision {}",
                    input["input"].as_str().unwrap_or_default().escape_debug(),
                    input["revision"]
                )?;
            }
        }
    }
    writeln!(out, "Comparable samples: {}", data["comparable"])?;
    if let Some(reasons) = data["reasons"].as_array() {
        for reason in reasons {
            writeln!(
                out,
                "{}",
                reason.as_str().unwrap_or_default().escape_debug()
            )?;
        }
    }
    if let Some(changes) = data["changes"].as_array() {
        for change in changes {
            writeln!(
                out,
                "{} {}: types {} -> {}, required {} -> {}",
                change["kind"].as_str().unwrap_or_default().escape_debug(),
                change["path"]
                    .as_str()
                    .filter(|p| !p.is_empty())
                    .unwrap_or("/ (root)")
                    .escape_debug(),
                change["before"]["types"],
                change["after"]["types"],
                change["before"]["required"],
                change["after"]["required"]
            )?;
        }
        if changes.is_empty() {
            writeln!(out, "No schema differences")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
