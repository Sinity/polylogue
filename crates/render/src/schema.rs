//! Offline schema reports contain structure and counts only.
use crate::Format;
use serde_json::Value;
use std::io::{self, Write};

pub fn render(data: &Value, format: Format, out: &mut dyn Write) -> io::Result<()> {
    if matches!(format, Format::Json | Format::Jsonl) || data.get("fields").is_some() {
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

fn inline(text: &str) -> String {
    crate::terminal::escape(text)
        .replace('\n', "\\n")
        .replace('\t', "\\t")
}

fn render_text(data: &Value, out: &mut dyn Write) -> io::Result<()> {
    writeln!(out, "Rule version: {}", data["rule_version"])?;
    for side in ["before", "after"] {
        let sample = &data[side];
        writeln!(
            out,
            "{side}: {} (sample: {}), {} records, {} unreadable inputs, {} unreadable records",
            inline(sample["origin"].as_str().unwrap_or_default()),
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
                    inline(input["input"].as_str().unwrap_or_default()),
                    input["revision"]
                )?;
            }
        }
    }
    writeln!(out, "Comparable samples: {}", data["comparable"])?;
    if let Some(reasons) = data["reasons"].as_array() {
        for reason in reasons {
            writeln!(out, "{}", inline(reason.as_str().unwrap_or_default()))?;
        }
    }
    if let Some(changes) = data["changes"].as_array() {
        for change in changes {
            writeln!(
                out,
                "{} {}: types {} -> {}, required {} -> {}",
                inline(change["kind"].as_str().unwrap_or_default()),
                inline(
                    change["path"]
                        .as_str()
                        .filter(|p| !p.is_empty())
                        .unwrap_or("/ (root)")
                ),
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
