//! A query's answer in every format: rows as a table, groups and summaries
//! the same way, each text hit with its snippet (matches in bold on a
//! terminal, `«…»` in Markdown, `<mark>` in HTML), then the count, what the
//! next page needs, notes, and the `--why` breakdown of an empty result.
//! JSON is the answer as the daemon gave it.

use crate::Format;
use serde_json::Value;
use std::io::{self, Write};

pub fn render(data: &Value, format: Format, out: &mut dyn Write) -> io::Result<()> {
    if format == Format::Json {
        return writeln!(out, "{data}");
    }
    let columns: Vec<String> = data["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["name"].as_str().map(str::to_owned))
        .collect();
    let rows: Vec<&Value> = data["rows"].as_array().into_iter().flatten().collect();
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|r| columns.iter().map(|c| cell(&r[c.as_str()])).collect())
        .collect();
    match format {
        Format::Html => html(&columns, &cells, &rows, out)?,
        Format::Markdown => markdown(&columns, &cells, &rows, out)?,
        _ => terminal(&columns, &cells, &rows, out)?,
    }
    footer(data, format, out)
}

fn cell(v: &Value) -> String {
    match v {
        Value::Null => "–".into(),
        Value::String(s) => one_line(s),
        Value::Array(items) => format!("[{} rows]", items.len()),
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => {
                let s = format!("{f:.4}");
                s.trim_end_matches('0').trim_end_matches('.').to_owned()
            }
            _ => n.to_string(),
        },
        other => other.to_string(),
    }
}

fn one_line(s: &str) -> String {
    let flat: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() > 80 {
        format!("{}…", flat.chars().take(79).collect::<String>())
    } else {
        flat
    }
}

/// The snippet with its marks wrapped by `open`/`close`.
fn marked(snippet: &Value, open: &str, close: &str, escape: fn(&str) -> String) -> Option<String> {
    let text = snippet["text"].as_str()?;
    let mut marks: Vec<(usize, usize)> = snippet["marks"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|m| Some((m[0].as_u64()? as usize, m[1].as_u64()? as usize)))
        .filter(|(s, e)| {
            s <= e && *e <= text.len() && text.is_char_boundary(*s) && text.is_char_boundary(*e)
        })
        .collect();
    marks.sort_unstable();
    let mut out = String::new();
    let mut at = 0;
    for (s, e) in marks {
        if s < at {
            continue;
        }
        out.push_str(&escape(&text[at..s]));
        out.push_str(open);
        out.push_str(&escape(&text[s..e]));
        out.push_str(close);
        at = e;
    }
    out.push_str(&escape(&text[at..]));
    Some(one_line(&out))
}

fn plain(s: &str) -> String {
    s.to_owned()
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

fn terminal(
    columns: &[String],
    cells: &[Vec<String>],
    rows: &[&Value],
    out: &mut dyn Write,
) -> io::Result<()> {
    if cells.is_empty() {
        return Ok(());
    }
    let widths: Vec<usize> = columns
        .iter()
        .enumerate()
        .map(|(i, c)| {
            cells
                .iter()
                .map(|r| r[i].chars().count())
                .chain([c.chars().count()])
                .max()
                .unwrap_or(0)
                .min(60)
        })
        .collect();
    let line = |vals: &[String]| -> String {
        vals.iter()
            .zip(&widths)
            .map(|(v, w)| {
                let v: String = v.chars().take(*w).collect();
                format!("{v:<w$}")
            })
            .collect::<Vec<_>>()
            .join("  ")
            .trim_end()
            .to_owned()
    };
    writeln!(out, "\x1b[2m{}\x1b[0m", line(columns))?;
    for (r, row) in cells.iter().zip(rows) {
        writeln!(out, "{}", line(r))?;
        if let Some(s) = marked(&row["snippet"], "\x1b[1m", "\x1b[0m", plain) {
            writeln!(out, "    {s}")?;
        }
        attached(row, out, "    ")?;
    }
    Ok(())
}

fn attached(row: &Value, out: &mut dyn Write, indent: &str) -> io::Result<()> {
    let Some(obj) = row.as_object() else {
        return Ok(());
    };
    for (name, v) in obj {
        if let Value::Array(items) = v
            && items.iter().all(Value::is_object)
            && !items.is_empty()
        {
            writeln!(out, "{indent}{name}:")?;
            for item in items {
                let fields: Vec<String> = item
                    .as_object()
                    .into_iter()
                    .flatten()
                    .map(|(k, v)| format!("{k}={}", cell(v)))
                    .collect();
                writeln!(out, "{indent}  {}", fields.join("  "))?;
            }
        }
    }
    Ok(())
}

fn markdown(
    columns: &[String],
    cells: &[Vec<String>],
    rows: &[&Value],
    out: &mut dyn Write,
) -> io::Result<()> {
    if cells.is_empty() {
        return Ok(());
    }
    let esc = |s: &str| s.replace('|', "\\|");
    let with_snippet = rows.iter().any(|r| r["snippet"].is_object());
    let mut head: Vec<String> = columns.iter().map(|c| esc(c)).collect();
    if with_snippet {
        head.push("snippet".into());
    }
    writeln!(out, "| {} |", head.join(" | "))?;
    writeln!(out, "|{}|", vec!["---"; head.len()].join("|"))?;
    for (r, row) in cells.iter().zip(rows) {
        let mut vals: Vec<String> = r.iter().map(|v| esc(v)).collect();
        if with_snippet {
            vals.push(esc(
                &marked(&row["snippet"], "«", "»", plain).unwrap_or_default()
            ));
        }
        writeln!(out, "| {} |", vals.join(" | "))?;
    }
    Ok(())
}

fn html(
    columns: &[String],
    cells: &[Vec<String>],
    rows: &[&Value],
    out: &mut dyn Write,
) -> io::Result<()> {
    writeln!(out, "<table>")?;
    let with_snippet = rows.iter().any(|r| r["snippet"].is_object());
    write!(out, "<thead><tr>")?;
    for c in columns {
        write!(out, "<th>{}</th>", html_escape(c))?;
    }
    if with_snippet {
        write!(out, "<th>snippet</th>")?;
    }
    writeln!(out, "</tr></thead><tbody>")?;
    for (r, row) in cells.iter().zip(rows) {
        write!(out, "<tr>")?;
        for v in r {
            write!(out, "<td>{}</td>", html_escape(v))?;
        }
        if with_snippet {
            let s = marked(&row["snippet"], "<mark>", "</mark>", html_escape).unwrap_or_default();
            write!(out, "<td>{s}</td>")?;
        }
        writeln!(out, "</tr>")?;
    }
    writeln!(out, "</tbody></table>")
}

fn footer(data: &Value, format: Format, out: &mut dyn Write) -> io::Result<()> {
    let count = data["count"].as_u64().unwrap_or(0);
    let shown = data["rows"].as_array().map_or(0, Vec::len);
    let mut lines = vec![];
    match data["outcome"].as_str() {
        Some("empty") => lines.push("no rows".to_owned()),
        _ => lines.push(if shown as u64 == count {
            format!("{count} row{}", if count == 1 { "" } else { "s" })
        } else {
            format!("{shown} of {count} rows")
        }),
    }
    if let Some(next) = data["next"].as_str() {
        lines.push(format!("more: --cursor {next}"));
    }
    for n in data["degraded"]
        .as_array()
        .into_iter()
        .flatten()
        .chain(data["notes"].as_array().into_iter().flatten())
    {
        if let Some(n) = n.as_str() {
            lines.push(n.to_owned());
        }
    }
    for w in data["why"].as_array().into_iter().flatten() {
        lines.push(format!(
            "{} → {} of {}",
            w["clause"].as_str().unwrap_or(""),
            w["matches"],
            w["of"]
        ));
    }
    for l in lines {
        match format {
            Format::Html => writeln!(out, "<p>{}</p>", html_escape(&l))?,
            Format::Terminal => writeln!(out, "\x1b[2m{l}\x1b[0m")?,
            _ => writeln!(out, "{l}")?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
