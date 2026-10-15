//! One binder for the registered Session subset, shared by every surface.
use polylogue_model::{
    SessionKind,
    query::{SessionQuery, TextTerm},
    types::{ErrorKind, Failure},
};
pub const SUPPORTED: &str = "sessions; conjunctive origin, kind, id and prose words/phrases/prefixes; sort by last_active desc; limit";
fn invalid(at: usize, text: impl std::fmt::Display) -> Failure {
    Failure::new(ErrorKind::InvalidQuery, format!("{text} at position {at}"))
}
fn unsupported(at: usize, text: &str) -> Failure {
    Failure::new(
        ErrorKind::NotYetAvailable,
        format!(
            "{text} is outside the registered Session query subset at position {at}; supported: {SUPPORTED}"
        ),
    )
}
struct Token {
    value: String,
    quoted: bool,
    literal: bool,
    at: usize,
}
fn tokens(text: &str) -> Result<Vec<Token>, Failure> {
    let mut chars = text.char_indices().peekable();
    let mut result = vec![];
    while let Some((at, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        let mut value = String::new();
        let mut quoted = false;
        let mut quote = false;
        if c == '"' {
            quoted = true;
            quote = true;
        } else {
            value.push(c);
        }
        while let Some(&(_, c)) = chars.peek() {
            if !quote && c.is_whitespace() {
                break;
            }
            chars.next();
            if c == '"' {
                quote = !quote;
                quoted = true;
            } else if c == '\\' {
                return Err(unsupported(at, "query escapes"));
            } else {
                value.push(c);
            }
        }
        if quote {
            return Err(invalid(at, "unclosed quote; add a closing double quote"));
        }
        result.push(Token {
            value,
            quoted,
            literal: c == '"',
            at,
        });
    }
    Ok(result)
}
pub fn bind(text: &str) -> Result<SessionQuery, Failure> {
    if text.len() > 16384 {
        return Err(invalid(0, "query exceeds 16384-byte admission"));
    }
    let tokens = tokens(text)?;
    let mut plan = SessionQuery::default();
    let mut i = 0;
    if tokens
        .first()
        .is_some_and(|t| !t.quoted && t.value == "sessions")
    {
        i += 1;
        if tokens.get(i).is_some_and(|t| t.value == "where") {
            i += 1;
        }
    }
    let mut condition = false;
    while i < tokens.len() {
        let t = &tokens[i];
        if !t.quoted && t.value == "|" {
            i += 1;
            let stage = tokens
                .get(i)
                .ok_or_else(|| invalid(t.at, "missing stage"))?;
            match stage.value.as_str() {
                "limit" => {
                    if plan.limit.is_some() {
                        return Err(unsupported(stage.at, "repeated limit"));
                    }
                    i += 1;
                    let n = tokens
                        .get(i)
                        .ok_or_else(|| invalid(stage.at, "limit needs a nonnegative integer"))?;
                    plan.limit = Some(
                        n.value
                            .parse()
                            .map_err(|_| invalid(n.at, "limit needs a nonnegative integer"))?,
                    );
                    i += 1;
                }
                "sort" => {
                    if plan.limit.is_some()
                        || plan.recent
                        || tokens.get(i + 1).is_none_or(|t| t.value != "by")
                        || tokens.get(i + 2).is_none_or(|t| t.value != "last_active")
                        || tokens.get(i + 3).is_none_or(|t| t.value != "desc")
                    {
                        return Err(unsupported(stage.at, "sort shape"));
                    }
                    plan.recent = true;
                    i += 4;
                }
                _ => return Err(unsupported(stage.at, &stage.value)),
            }
            if tokens.get(i).is_some_and(|t| t.value != "|") {
                return Err(unsupported(tokens[i].at, "condition after a stage"));
            }
            continue;
        }
        if !t.quoted && t.value == "AND" {
            if !condition
                || tokens
                    .get(i + 1)
                    .is_none_or(|t| t.value == "|" || t.value == "AND")
            {
                return Err(invalid(t.at, "AND needs conditions on both sides"));
            }
            condition = false;
            i += 1;
            continue;
        }
        if !t.quoted
            && (matches!(
                t.value.as_str(),
                "OR" | "NOT"
                    | "where"
                    | "messages"
                    | "blocks"
                    | "tool_calls"
                    | "mcp"
                    | "config"
                    | "import"
                    | "progress"
                    | "retry"
                    | "input"
                    | "explain"
                    | "query"
                    | "sql"
                    | "search"
                    | "export"
                    | "read"
                    | "list"
                    | "brief"
                    | "status"
                    | "daemon"
            ) || t.value.starts_with("--")
                || t.value.contains(['(', ')', '|', '<', '>', '=']))
        {
            return Err(unsupported(t.at, &t.value));
        }
        let (field, value) = if t.literal {
            ("text", t.value.as_str())
        } else if let Some((f, v)) = t.value.split_once(':') {
            if matches!(f, "claude-code" | "codex" | "chatgpt" | "claude-ai") {
                ("id", t.value.as_str())
            } else {
                (f, v)
            }
        } else {
            ("text", t.value.as_str())
        };
        if value.is_empty() {
            return Err(invalid(t.at, "empty condition"));
        }
        match field {
            "origin" => {
                polylogue_model::Origin::new(value)
                    .map_err(|_| invalid(t.at, "origin must be a lowercase origin identifier"))?;
                if plan.origin.replace(value.to_owned()).is_some() {
                    return Err(unsupported(t.at, "repeated origin"));
                }
            }
            "id" => {
                if plan.id.replace(value.to_owned()).is_some() {
                    return Err(unsupported(t.at, "repeated id"));
                }
            }
            "kind" => {
                if plan.kind.is_some() || plan.all_kinds {
                    return Err(unsupported(t.at, "repeated kind"));
                }
                match value {
                    "*" => plan.all_kinds = true,
                    "main" => plan.kind = Some(SessionKind::Main),
                    "chat" => plan.kind = Some(SessionKind::Chat),
                    "subagent" => plan.kind = Some(SessionKind::Subagent),
                    _ => return Err(invalid(t.at, "kind must be main, chat, subagent or *")),
                }
            }
            "text" => {
                if t.quoted {
                    if !value.chars().any(char::is_alphanumeric) || value.contains('*') {
                        return Err(unsupported(t.at, "empty phrase or phrase wildcard"));
                    }
                    plan.text.push(TextTerm::Phrase(value.to_owned()));
                } else {
                    let prefix = value.ends_with('*');
                    let word = if prefix {
                        &value[..value.len() - 1]
                    } else {
                        value
                    };
                    if word.is_empty() || !word.chars().all(char::is_alphanumeric) {
                        return Err(unsupported(t.at, "word shape"));
                    }
                    plan.text.push(if prefix {
                        TextTerm::Prefix(word.to_owned())
                    } else {
                        TextTerm::Word(word.to_owned())
                    });
                }
            }
            "reop" => return Err(invalid(t.at, "unknown field reop; did you mean repo?")),
            _ => return Err(unsupported(t.at, field)),
        }
        condition = true;
        i += 1;
    }
    Ok(plan)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn binds_only_registered_shapes() {
        let p=bind("sessions where origin:claude-code kind:* text:\"weekday offset\" refactor* | sort by last_active desc | limit 3").unwrap();
        assert!(p.all_kinds && p.recent);
        assert_eq!(p.limit, Some(3));
        assert_eq!(p.text.len(), 2);
        for q in [
            "near:chrono",
            "messages chrono",
            "chrono OR parser",
            "chrono | limit 1 | where x",
            "origin:",
            "\"unfinished",
            "chrono AND",
            "chrono | sort by cost desc",
            "chrono | limit 3 | sort by last_active desc",
            "--reveal",
            "reop:morrow",
        ] {
            assert!(bind(q).is_err(), "{q}");
        }
    }
}
