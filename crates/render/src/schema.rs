//! Offline schema reports contain structure and counts only.
use crate::Format;
use serde_json::Value;
use std::io::{self, Write};

pub fn render(data: &Value, format: Format, out: &mut dyn Write) -> io::Result<()> {
    if format == Format::Json || data.get("fields").is_some() {
        write!(out, "{}", crate::terminal::json(data, true))?;
        return writeln!(out);
    }
    let mut text = Vec::new();
    render_text(data, &mut text)?;
    out.write_all(
        crate::terminal::escape(std::str::from_utf8(&text).expect("rendered strings are UTF-8"))
            .as_bytes(),
    )
}

fn render_text(data: &Value, out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "Rule version: {}", data["rule_version"])?;
    for side in ["before", "after"] {
        let population = &data[side];
        writeln!(
            out,
            "{side}: {} (sample: {}), {} records, {} unreadable inputs, {} unreadable records",
            population["origin"].as_str().unwrap_or_default(),
            population["sample"],
            population["records"],
            population["unreadable_inputs"],
            population["unreadable_records"]
        )?;
        if let Some(selection) = population["selection"].as_array() {
            for input in selection {
                writeln!(
                    out,
                    "  input {} revision {}",
                    input["input"].as_str().unwrap_or_default(),
                    input["revision"]
                )?;
            }
        }
    }
    writeln!(out, "Comparable populations: {}", data["comparable"])?;
    if let Some(reasons) = data["reasons"].as_array() {
        for reason in reasons {
            writeln!(out, "{}", reason.as_str().unwrap_or_default())?;
        }
    }
    if let Some(changes) = data["changes"].as_array() {
        for change in changes {
            writeln!(
                out,
                "{} {}: types {} -> {}, required {} -> {}",
                change["kind"].as_str().unwrap_or_default(),
                change["path"]
                    .as_str()
                    .filter(|p| !p.is_empty())
                    .unwrap_or("/ (root)"),
                change["before"]["types"],
                change["after"]["types"],
                change["before"]["required"],
                change["after"]["required"]
            )?;
        }
        if changes.is_empty() {
            writeln!(out, "No structural changes")?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
