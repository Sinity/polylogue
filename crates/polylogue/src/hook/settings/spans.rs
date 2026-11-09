//! A JSON text read with the byte span of every value, so an edit replaces
//! only the spans it names and every other byte stays as the user wrote it.

use std::io;
use std::ops::Range;

/// One value of the text and where it lies.
pub(super) struct Spanned {
    pub start: usize,
    pub end: usize,
    pub kind: Kind,
}

pub(super) enum Kind {
    /// Members in written order: the decoded key, where the member starts
    /// (its key) and its value.
    Object(Vec<(String, usize, Spanned)>),
    Array(Vec<Spanned>),
    Scalar,
}

impl Spanned {
    pub fn member(&self, key: &str) -> Option<&Spanned> {
        match &self.kind {
            Kind::Object(members) => members
                .iter()
                .find(|(name, _, _)| name == key)
                .map(|(_, _, value)| value),
            _ => None,
        }
    }

    pub fn items(&self) -> &[Spanned] {
        match &self.kind {
            Kind::Array(items) => items,
            _ => &[],
        }
    }
}

fn malformed(at: usize) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("settings: malformed JSON at byte {at}"),
    )
}

/// Reads `text`, which a JSON parser has already accepted, into spans.
pub(super) fn read(text: &str) -> io::Result<Spanned> {
    let bytes = text.as_bytes();
    let mut at = skip_space(bytes, 0);
    let value = value(text, &mut at)?;
    if skip_space(bytes, at) != bytes.len() {
        return Err(malformed(at));
    }
    Ok(value)
}

fn skip_space(bytes: &[u8], mut at: usize) -> usize {
    while bytes.get(at).is_some_and(u8::is_ascii_whitespace) {
        at += 1;
    }
    at
}

fn value(text: &str, at: &mut usize) -> io::Result<Spanned> {
    let bytes = text.as_bytes();
    let start = *at;
    match bytes.get(start) {
        Some(b'{') => {
            *at += 1;
            let mut members = Vec::new();
            loop {
                *at = skip_space(bytes, *at);
                match bytes.get(*at) {
                    Some(b'}') => break,
                    Some(b',') if !members.is_empty() => {
                        *at = skip_space(bytes, *at + 1);
                    }
                    Some(_) if members.is_empty() => {}
                    _ => return Err(malformed(*at)),
                }
                let key_start = *at;
                string(bytes, at)?;
                let key: String = serde_json::from_str(&text[key_start..*at])
                    .map_err(|_| malformed(key_start))?;
                *at = skip_space(bytes, *at);
                if bytes.get(*at) != Some(&b':') {
                    return Err(malformed(*at));
                }
                *at = skip_space(bytes, *at + 1);
                let member = value(text, at)?;
                members.push((key, key_start, member));
            }
            *at += 1;
            Ok(Spanned {
                start,
                end: *at,
                kind: Kind::Object(members),
            })
        }
        Some(b'[') => {
            *at += 1;
            let mut items = Vec::new();
            loop {
                *at = skip_space(bytes, *at);
                match bytes.get(*at) {
                    Some(b']') => break,
                    Some(b',') if !items.is_empty() => {
                        *at = skip_space(bytes, *at + 1);
                    }
                    Some(_) if items.is_empty() => {}
                    _ => return Err(malformed(*at)),
                }
                items.push(value(text, at)?);
            }
            *at += 1;
            Ok(Spanned {
                start,
                end: *at,
                kind: Kind::Array(items),
            })
        }
        Some(b'"') => {
            string(bytes, at)?;
            Ok(Spanned {
                start,
                end: *at,
                kind: Kind::Scalar,
            })
        }
        Some(_) => {
            while bytes
                .get(*at)
                .is_some_and(|b| !b.is_ascii_whitespace() && !b",]}".contains(b))
            {
                *at += 1;
            }
            if *at == start {
                return Err(malformed(start));
            }
            Ok(Spanned {
                start,
                end: *at,
                kind: Kind::Scalar,
            })
        }
        None => Err(malformed(start)),
    }
}

fn string(bytes: &[u8], at: &mut usize) -> io::Result<()> {
    let start = *at;
    if bytes.get(start) != Some(&b'"') {
        return Err(malformed(start));
    }
    *at += 1;
    loop {
        match bytes.get(*at) {
            Some(b'"') => {
                *at += 1;
                return Ok(());
            }
            Some(b'\\') => *at += 2,
            Some(_) => *at += 1,
            None => return Err(malformed(start)),
        }
    }
}

/// A replacement of one byte range.
pub(super) struct Edit {
    pub range: Range<usize>,
    pub text: String,
}

/// Applies non-overlapping edits. Of two edits starting at one byte, the
/// removal applies before the insertion, so inserted text is never removed.
pub(super) fn apply(text: &str, mut edits: Vec<Edit>) -> String {
    edits.sort_by(|a, b| (b.range.start, b.range.end).cmp(&(a.range.start, a.range.end)));
    let mut out = text.to_owned();
    for edit in edits {
        out.replace_range(edit.range, &edit.text);
    }
    out
}

/// The edits that remove the marked items of a container (members of an
/// object, given as `start..end` from key to value end, or array items) and
/// append `appended`, already rendered for `layout`. The bytes of kept items
/// and of the separators between them do not change.
pub(super) fn splice(
    text: &str,
    container: &Spanned,
    items: &[Range<usize>],
    removed: &[bool],
    appended: impl Fn(&Layout) -> Vec<String>,
) -> Vec<Edit> {
    let open = container.start;
    let close = container.end - 1;
    let layout = Layout::of(text, open, items);
    let appended = appended(&layout);
    let kept: Vec<usize> = (0..items.len()).filter(|&i| !removed[i]).collect();
    let Some(&last_kept) = kept.last() else {
        if items.is_empty() && appended.is_empty() {
            return Vec::new();
        }
        let interior = if appended.is_empty() {
            String::new()
        } else {
            let separator = format!(",{}", layout.gap);
            format!(
                "{}{}{}",
                layout.gap,
                appended.join(&separator),
                layout.closing
            )
        };
        return vec![Edit {
            range: open + 1..close,
            text: interior,
        }];
    };
    let mut edits = Vec::new();
    for (index, item) in items.iter().enumerate() {
        if !removed[index] {
            continue;
        }
        let range = if index < last_kept {
            item.start..items[index + 1].start
        } else {
            items[index - 1].end..item.end
        };
        edits.push(Edit {
            range,
            text: String::new(),
        });
    }
    let at = items[last_kept].end;
    for item in appended {
        edits.push(Edit {
            range: at..at,
            text: format!(",{}{item}", layout.gap),
        });
    }
    edits
}

/// How a container lays out its items: the whitespace before each item,
/// the whitespace before the closing bracket and whether it spans lines.
pub(super) struct Layout {
    pub gap: String,
    pub closing: String,
    /// The indentation of the container's items.
    pub indent: String,
    pub multiline: bool,
}

impl Layout {
    fn of(text: &str, open: usize, items: &[Range<usize>]) -> Self {
        let close_gap = |end: usize| {
            text[end..]
                .find(|c: char| !c.is_ascii_whitespace())
                .map_or("", |n| &text[end..end + n])
        };
        let line_indent = {
            let line = text[..open].rfind('\n').map_or(0, |n| n + 1);
            text[line..open]
                .chars()
                .take_while(|c| *c == ' ' || *c == '\t')
                .collect::<String>()
        };
        let gap = match items {
            [] => None,
            [only] => Some(text[open + 1..only.start].to_owned()),
            [first, second, ..] => {
                let after = &text[first.end..second.start];
                Some(after.trim_start_matches(|c: char| c != ',')[1..].to_owned())
            }
        };
        match gap {
            Some(gap) => {
                let indent = gap.rsplit('\n').next().unwrap_or("").to_owned();
                Self {
                    multiline: gap.contains('\n'),
                    closing: close_gap(items[items.len() - 1].end).to_owned(),
                    gap,
                    indent,
                }
            }
            None => {
                let indent = format!("{line_indent}  ");
                Self {
                    gap: format!("\n{indent}"),
                    closing: format!("\n{line_indent}"),
                    indent,
                    multiline: true,
                }
            }
        }
    }

    /// `value` written at this layout's indentation.
    pub fn render(&self, value: &serde_json::Value) -> String {
        if self.multiline {
            serde_json::to_string_pretty(value)
                .expect("JSON value serializes")
                .replace('\n', &format!("\n{}", self.indent))
        } else {
            value.to_string()
        }
    }

    /// An object member `"key": value` at this layout.
    pub fn member(&self, key: &str, value: &serde_json::Value) -> String {
        let key = serde_json::Value::from(key).to_string();
        let separator = if self.multiline { ": " } else { ":" };
        format!("{key}{separator}{}", self.render(value))
    }
}
