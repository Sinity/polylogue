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
impl Token {
    fn syntax(&self, value: &str) -> bool {
        !self.quoted && self.value == value
    }
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
    let explicit = tokens.first().is_some_and(|t| t.syntax("find"));
    let mut i = usize::from(explicit);
    if explicit && i == tokens.len() {
        return Err(invalid(0, "find needs query text"));
    }
    if !explicit && tokens.first().is_some_and(|t| t.syntax("sessions")) {
        i += 1;
        if tokens.get(i).is_some_and(|t| t.syntax("where")) {
            let at = tokens[i].at;
            i += 1;
            if tokens.get(i).is_none_or(|t| t.syntax("|")) {
                return Err(invalid(at, "where needs a condition"));
            }
        }
    }
    let mut condition = false;
    while i < tokens.len() {
        let t = &tokens[i];
        if t.syntax("|") {
            i += 1;
            let stage = tokens
                .get(i)
                .ok_or_else(|| invalid(t.at, "missing stage"))?;
            if stage.quoted {
                return Err(unsupported(stage.at, "quoted stage name"));
            }
            match stage.value.as_str() {
                "limit" => {
                    if plan.limit.is_some() {
                        return Err(unsupported(stage.at, "repeated limit"));
                    }
                    i += 1;
                    let n = tokens
                        .get(i)
                        .ok_or_else(|| invalid(stage.at, "limit needs a nonnegative integer"))?;
                    if n.quoted {
                        return Err(unsupported(n.at, "quoted limit operand"));
                    }
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
                        || tokens.get(i + 1).is_none_or(|t| !t.syntax("by"))
                        || tokens.get(i + 2).is_none_or(|t| !t.syntax("last_active"))
                        || tokens.get(i + 3).is_none_or(|t| !t.syntax("desc"))
                    {
                        return Err(unsupported(stage.at, "sort shape"));
                    }
                    plan.recent = true;
                    i += 4;
                }
                _ => return Err(unsupported(stage.at, &stage.value)),
            }
            if tokens.get(i).is_some_and(|t| !t.syntax("|")) {
                return Err(unsupported(tokens[i].at, "condition after a stage"));
            }
            continue;
        }
        if t.syntax("AND") {
            if !condition
                || tokens
                    .get(i + 1)
                    .is_none_or(|t| t.syntax("|") || t.syntax("AND"))
            {
                return Err(invalid(t.at, "AND needs conditions on both sides"));
            }
            condition = false;
            i += 1;
            continue;
        }
        if !t.quoted
            && (matches!(t.value.as_str(), "OR" | "NOT" | "where")
                || !explicit
                    && matches!(
                        t.value.as_str(),
                        "messages"
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
                    )
                || t.value.starts_with("--")
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
                    .map_err(|_| invalid(t.at, "origin must be a valid origin identifier"))?;
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
        assert_eq!(
            bind("find list").unwrap().text,
            vec![TextTerm::Word("list".into())]
        );
        assert_eq!(
            bind("find messages").unwrap().text,
            vec![TextTerm::Word("messages".into())]
        );
        for q in [
            "find",
            "find token OR other",
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

#[cfg(test)]
mod quoted_syntax_tests {
    use super::*;
    #[test]
    fn quoted_words_stay_data_at_leading_and_optional_where_positions() {
        for query in [
            "\"where\"",
            "sessions \"where\"",
            "sessions where \"where\"",
        ] {
            let plan = bind(query).unwrap();
            assert_eq!(plan.text, vec![TextTerm::Phrase("where".into())], "{query}");
        }
        assert_eq!(
            bind("\"sessions\"").unwrap().text,
            vec![TextTerm::Phrase("sessions".into())]
        );
        for query in ["where chrono", "sessions where", "sessions where | limit 0"] {
            assert!(bind(query).is_err(), "{query}");
        }
    }
    #[test]
    fn quoted_stage_names_and_parameters_cannot_acquire_syntax_authority() {
        for query in [
            "chrono | \"limit\" 0",
            "chrono | limit \"0\"",
            "chrono | \"sort\" by last_active desc",
            "chrono | sort \"by\" last_active desc",
            "chrono | sort by \"last_active\" desc",
            "chrono | sort by last_active \"desc\"",
            "chrono | limit 1 \"|\" limit 0",
        ] {
            assert!(bind(query).is_err(), "{query}");
        }
    }
    #[test]
    fn quoted_conjunction_is_data_and_quoted_pipe_is_an_unsupported_phrase() {
        assert_eq!(
            bind("\"AND\"").unwrap().text,
            vec![TextTerm::Phrase("AND".into())]
        );
        assert_eq!(
            bind("chrono AND \"AND\"").unwrap().text,
            vec![
                TextTerm::Word("chrono".into()),
                TextTerm::Phrase("AND".into())
            ]
        );
        for query in ["\"|\"", "chrono AND \"|\""] {
            let error = bind(query).unwrap_err();
            assert!(error.message.contains("empty phrase"), "{query}: {error:?}");
        }
    }
}

/// Bounded exact-Session operation reads use the ordinary query transport.
/// This binder never guesses a Session from operation reference punctuation.
pub fn bind_tools(text: &str) -> Result<Option<polylogue_model::query::ToolQuery>, Failure> {
    if text.len() > 16384 {
        return Err(invalid(0, "query exceeds 16384-byte admission"));
    }
    let tokens = tokens(text)?;
    let mut i = usize::from(tokens.first().is_some_and(|t| t.syntax("find")));
    let Some(table) = tokens.get(i) else {
        return Ok(None);
    };
    if !table.syntax("tool_items") && !table.syntax("tool_calls") {
        return Ok(None);
    }
    let calls_only = table.syntax("tool_calls");
    i += 1;
    if tokens.get(i).is_some_and(|t| t.syntax("where")) {
        i += 1;
    }
    let mut session = None;
    let mut text = None;
    let mut limit = 100;
    while let Some(t) = tokens.get(i) {
        if t.syntax("|") {
            if tokens.get(i + 1).is_none_or(|t| !t.syntax("limit")) || tokens.len() != i + 3 {
                return Err(unsupported(t.at, "operation query pipeline"));
            }
            limit = tokens[i + 2]
                .value
                .parse::<usize>()
                .ok()
                .filter(|n| *n > 0 && *n <= 4096)
                .ok_or_else(|| invalid(t.at, "operation limit must be 1..4096"))?;
            break;
        }
        if let Some(v) = t.value.strip_prefix("session:").filter(|v| !v.is_empty()) {
            if session.replace(v.to_owned()).is_some() {
                return Err(invalid(t.at, "repeated operation Session"));
            }
        } else if let Some(v) = t.value.strip_prefix("text:").filter(|v| !v.is_empty()) {
            if text.replace(v.to_owned()).is_some() {
                return Err(invalid(t.at, "repeated operation text"));
            }
        } else {
            return Err(unsupported(
                t.at,
                "operation filter; use exact session: and optional text:",
            ));
        }
        i += 1;
    }
    Ok(Some(polylogue_model::query::ToolQuery {
        calls_only,
        session: session.ok_or_else(|| invalid(0, "operation query requires exact session:"))?,
        text,
        limit,
    }))
}
