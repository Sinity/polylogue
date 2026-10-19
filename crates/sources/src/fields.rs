//! Every wire field is read, or excluded with a declared reason. An origin
//! declares its fields per record kind; [`unaccounted`] lists the fields a
//! record carries that the declaration does not mention.

use polylogue_model::Json;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    /// The parser reads it into a row.
    Read,
    /// Kept whole in the stored record or an event's data, not interpreted
    /// yet, for the stated reason.
    Kept(&'static str),
    /// Not read, for the stated reason.
    Excluded(&'static str),
}

/// Paths are dotted object keys; `[]` stands for every element of an array
/// (`message.content[].text`). A declared path covers everything below it
/// unless deeper paths are declared too.
pub type Fields = &'static [(&'static str, Rule)];

pub fn unaccounted(value: &Json, fields: Fields) -> Vec<String> {
    let mut out = Vec::new();
    walk(value, "", fields, &mut out);
    out
}

fn has_deeper(path: &str, fields: Fields) -> bool {
    fields.iter().any(|(p, _)| {
        p.len() > path.len()
            && p.starts_with(path)
            && matches!(p.as_bytes()[path.len()], b'.' | b'[')
    })
}

fn walk(value: &Json, path: &str, fields: Fields, out: &mut Vec<String>) {
    match value {
        Json::Object(map) => {
            for (key, child) in map {
                let child_path = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{path}.{key}")
                };
                let declared = fields.iter().any(|(p, _)| *p == child_path);
                if declared {
                    if has_deeper(&child_path, fields) {
                        walk(child, &child_path, fields, out);
                    }
                } else if has_deeper(&child_path, fields) {
                    walk(child, &child_path, fields, out);
                } else {
                    out.push(child_path);
                }
            }
        }
        Json::Array(items) => {
            let element = format!("{path}[]");
            if has_deeper(&element, fields) {
                for item in items {
                    walk(item, &element, fields, out);
                }
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
