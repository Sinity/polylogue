//! Data formats keep complete values, including nested columns and multiline text.
use crate::Format;
use serde_json::Value;
use std::io;

pub fn text(data: &Value, format: Format) -> io::Result<String> {
    if !matches!(format, Format::Csv | Format::Org | Format::Yaml) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "expected csv, org or yaml",
        ));
    }
    let mut data = data.clone();
    crate::mask_json(&mut data);
    if format == Format::Yaml {
        let mut out = String::from("---\n");
        yaml(&data, 0, &mut out);
        return Ok(out);
    }
    let rows = data["rows"]
        .as_array()
        .map_or_else(|| std::slice::from_ref(&data), Vec::as_slice);
    let mut columns: Vec<String> = data["columns"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c["name"].as_str().map(str::to_owned))
        .collect();
    for row in rows {
        for key in row.as_object().into_iter().flatten().map(|(key, _)| key) {
            if !columns.contains(key) {
                columns.push(key.clone());
            }
        }
    }
    let cell = |v: &Value| match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        _ => v.to_string(),
    };
    let mut out = String::new();
    let encode = |s: &str| match format {
        Format::Csv => {
            let safe = if s.starts_with(['=', '+', '-', '@']) {
                format!("'{s}")
            } else {
                s.to_owned()
            };
            format!("\"{}\"", safe.replace('"', "\"\""))
        }
        _ => s
            .replace('\\', "\\\\")
            .replace('|', "\\vert{}")
            .replace('\r', "\\r")
            .replace('\n', "\\n"),
    };
    let mut line = |cells: Vec<String>| {
        let cells: Vec<_> = cells.iter().map(|s| encode(s)).collect();
        if format == Format::Csv {
            out.push_str(&cells.join(","));
            out.push_str("\r\n");
        } else {
            out.push_str("| ");
            out.push_str(&cells.join(" | "));
            out.push_str(" |\n");
        }
    };
    line(columns.clone());
    for row in rows {
        line(columns.iter().map(|c| cell(&row[c])).collect());
    }
    if format == Format::Org {
        let mut metadata = data.clone();
        if let Some(obj) = metadata.as_object_mut() {
            obj.remove("rows");
            obj.remove("columns");
        }
        out.push_str("\n#+begin_example\n");
        out.push_str(&serde_json::to_string_pretty(&metadata).map_err(io::Error::other)?);
        out.push_str("\n#+end_example\n");
    }
    Ok(out)
}

fn yaml(value: &Value, indent: usize, out: &mut String) {
    let pad = " ".repeat(indent);
    match value {
        Value::Object(fields) if !fields.is_empty() => {
            for (key, value) in fields {
                out.push_str(&pad);
                out.push_str(&serde_json::to_string(key).expect("string"));
                out.push(':');
                child(value, indent, out);
            }
        }
        Value::Array(values) if !values.is_empty() => {
            for value in values {
                out.push_str(&pad);
                out.push('-');
                child(value, indent, out);
            }
        }
        _ => {
            out.push_str(&pad);
            out.push_str(&value.to_string());
            out.push('\n');
        }
    }
}

fn child(value: &Value, indent: usize, out: &mut String) {
    if value.as_object().is_some_and(|v| !v.is_empty())
        || value.as_array().is_some_and(|v| !v.is_empty())
    {
        out.push('\n');
        yaml(value, indent + 2, out);
    } else {
        out.push(' ');
        out.push_str(&value.to_string());
        out.push('\n');
    }
}
