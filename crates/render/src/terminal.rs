//! Terminal text uses visible hexadecimal escapes before optional ANSI styling.
use std::fmt::{self, Write as _};
use std::io::{self, IsTerminal, Write};

/// Preserve tabs and newlines; show C0, DEL, C1 and Unicode direction controls.
/// This is idempotent so already rendered terminal text can cross another text surface.
pub fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\n' | '\t' => out.push(c),
            '\u{00}'..='\u{1f}' | '\u{7f}'..='\u{9f}' => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            '\u{061c}'
            | '\u{200e}'
            | '\u{200f}'
            | '\u{202a}'..='\u{202e}'
            | '\u{2066}'..='\u{2069}' => {
                let _ = write!(out, "\\u{{{:04x}}}", c as u32);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Serialized JSON stays valid and round trips controls even when shown on a terminal.
pub fn json(value: &serde_json::Value, pretty: bool) -> String {
    let text = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .expect("JSON values serialize");
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if matches!(c, '\u{7f}'..='\u{9f}' | '\u{061c}' | '\u{200e}' | '\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}')
        {
            let _ = write!(out, "\\u{:04x}", c as u32);
        } else {
            out.push(c);
        }
    }
    out
}

/// File and pipe exports preserve their format; terminal display escapes text.
pub fn stdout_text(text: &str) -> String {
    if std::io::stdout().is_terminal() {
        escape(text)
    } else {
        text.to_owned()
    }
}

pub fn color_enabled() -> bool {
    color_for(
        std::io::stdout().is_terminal(),
        std::env::var_os("NO_COLOR").is_some(),
    )
}

fn color_for(terminal: bool, no_color: bool) -> bool {
    terminal && !no_color
}

/// The daemon produces plain terminal text; only the displaying client adds styling.
pub(crate) fn line(out: &mut dyn Write, args: fmt::Arguments<'_>) -> io::Result<()> {
    writeln!(out, "{}", escape(&args.to_string()))
}

/// Add a fixed heading style after escaping at the displaying client.
pub fn heading(text: &str) -> String {
    let text = escape(text);
    if color_enabled() {
        let (first, rest) = text.split_once('\n').unwrap_or((&text, ""));
        format!("\x1b[1m{first}\x1b[0m\n{rest}")
    } else {
        text
    }
}

#[cfg(test)]
mod tests;
