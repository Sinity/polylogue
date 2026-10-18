//! Parses query text into a `SessionQuery` or `ToolQuery`. The CLI and the
//! daemon both call these, so they agree on what a query means.
use polylogue_model::{
    Origin, SessionKind,
    query::{SessionQuery, TextTerm, ToolQuery},
    types::{ErrorKind, Failure},
};

#[cfg(test)]
mod tests;

const MAX_QUERY_BYTES: usize = 16384;
const SUPPORTED: &str = "sessions, origin:, kind:, id, words, \"phrases\", prefix*, AND, \
| sort by last_active desc, | limit N";
/// Command names that would be ambiguous as bare search words; `find` allows them.
const COMMAND_WORDS: &[&str] = &[
    "messages",
    "blocks",
    "tool_calls",
    "mcp",
    "config",
    "import",
    "progress",
    "retry",
    "input",
    "explain",
    "query",
    "sql",
    "search",
    "export",
    "read",
    "list",
    "brief",
    "status",
    "daemon",
];
/// Origins whose session IDs are written `origin:id`.
const ID_ORIGINS: &[&str] = &["claude-code", "codex", "chatgpt", "claude-ai"];

fn invalid(at: usize, text: impl std::fmt::Display) -> Failure {
    Failure::new(ErrorKind::InvalidQuery, format!("{text} at position {at}"))
}

fn unsupported(at: usize, text: &str) -> Failure {
    Failure::new(
        ErrorKind::NotYetAvailable,
        format!("{text} is not supported (position {at}); supported: {SUPPORTED}"),
    )
}

struct Token {
    value: String,
    /// Contains a quote anywhere; quoted tokens are never keywords.
    quoted: bool,
    /// Starts with a quote, so it is search text even if it contains `:`.
    starts_quoted: bool,
    at: usize,
}

impl Token {
    fn is_keyword(&self, keyword: &str) -> bool {
        !self.quoted && self.value == keyword
    }
}

fn tokens(text: &str) -> Result<Vec<Token>, Failure> {
    if text.len() > MAX_QUERY_BYTES {
        return Err(invalid(
            0,
            format!("query is longer than {MAX_QUERY_BYTES} bytes"),
        ));
    }
    let mut chars = text.char_indices().peekable();
    let mut result = vec![];
    while let Some((at, first)) = chars.next() {
        if first.is_whitespace() {
            continue;
        }
        let starts_quoted = first == '"';
        let mut value = String::new();
        if !starts_quoted {
            value.push(first);
        }
        let mut quoted = starts_quoted;
        let mut in_quote = starts_quoted;
        while let Some(&(_, c)) = chars.peek() {
            if !in_quote && c.is_whitespace() {
                break;
            }
            chars.next();
            match c {
                '"' => {
                    in_quote = !in_quote;
                    quoted = true;
                }
                '\\' => return Err(unsupported(at, "query escapes")),
                _ => value.push(c),
            }
        }
        if in_quote {
            return Err(invalid(at, "unclosed quote; add a closing double quote"));
        }
        result.push(Token {
            value,
            quoted,
            starts_quoted,
            at,
        });
    }
    Ok(result)
}

pub fn bind(text: &str) -> Result<SessionQuery, Failure> {
    let tokens = tokens(text)?;
    let mut plan = SessionQuery::default();
    let find = tokens.first().is_some_and(|t| t.is_keyword("find"));
    let mut i = usize::from(find);
    if find && i == tokens.len() {
        return Err(invalid(0, "find needs query text"));
    }
    if !find && tokens.first().is_some_and(|t| t.is_keyword("sessions")) {
        i += 1;
        if tokens.get(i).is_some_and(|t| t.is_keyword("where")) {
            let at = tokens[i].at;
            i += 1;
            if tokens.get(i).is_none_or(|t| t.is_keyword("|")) {
                return Err(invalid(at, "where needs a condition"));
            }
        }
    }
    let mut after_condition = false;
    while let Some(t) = tokens.get(i) {
        if t.is_keyword("|") {
            i = bind_stage(&tokens, i + 1, t.at, &mut plan)?;
            if let Some(next) = tokens.get(i).filter(|t| !t.is_keyword("|")) {
                return Err(unsupported(next.at, "condition after a stage"));
            }
        } else if t.is_keyword("AND") {
            if !after_condition
                || tokens
                    .get(i + 1)
                    .is_none_or(|t| t.is_keyword("|") || t.is_keyword("AND"))
            {
                return Err(invalid(t.at, "AND needs conditions on both sides"));
            }
            after_condition = false;
            i += 1;
        } else {
            bind_condition(t, find, &mut plan)?;
            after_condition = true;
            i += 1;
        }
    }
    Ok(plan)
}

/// Binds the stage starting at `tokens[i]`; returns the index after it.
fn bind_stage(
    tokens: &[Token],
    mut i: usize,
    pipe_at: usize,
    plan: &mut SessionQuery,
) -> Result<usize, Failure> {
    let stage = tokens
        .get(i)
        .ok_or_else(|| invalid(pipe_at, "missing stage"))?;
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
                return Err(unsupported(n.at, "quoted limit"));
            }
            let limit = n.value.parse();
            plan.limit =
                Some(limit.map_err(|_| invalid(n.at, "limit needs a nonnegative integer"))?);
            Ok(i + 1)
        }
        "sort" => {
            let shape_ok = ["by", "last_active", "desc"]
                .iter()
                .enumerate()
                .all(|(k, word)| tokens.get(i + 1 + k).is_some_and(|t| t.is_keyword(word)));
            // Sorting after limit would change which rows are kept.
            if plan.limit.is_some() || plan.recent || !shape_ok {
                return Err(unsupported(stage.at, "this sort"));
            }
            plan.recent = true;
            Ok(i + 4)
        }
        _ => Err(unsupported(stage.at, &stage.value)),
    }
}

fn bind_condition(t: &Token, find: bool, plan: &mut SessionQuery) -> Result<(), Failure> {
    if !t.quoted
        && (matches!(t.value.as_str(), "OR" | "NOT" | "where")
            || !find && COMMAND_WORDS.contains(&t.value.as_str())
            || t.value.starts_with("--")
            || t.value.contains(['(', ')', '|', '<', '>', '=']))
    {
        return Err(unsupported(t.at, &t.value));
    }
    let (field, value) = match t.value.split_once(':') {
        _ if t.starts_quoted => ("text", t.value.as_str()),
        Some((origin, _)) if ID_ORIGINS.contains(&origin) => ("id", t.value.as_str()),
        Some((field, value)) => (field, value),
        None => ("text", t.value.as_str()),
    };
    if value.is_empty() {
        return Err(invalid(t.at, "empty condition"));
    }
    match field {
        "origin" => {
            Origin::new(value).map_err(|_| invalid(t.at, "origin is not a valid origin name"))?;
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
        "text" if t.quoted => {
            if !value.chars().any(char::is_alphanumeric) || value.contains('*') {
                return Err(unsupported(t.at, "empty phrase or phrase wildcard"));
            }
            plan.text.push(TextTerm::Phrase(value.to_owned()));
        }
        "text" => {
            let (word, prefix) = match value.strip_suffix('*') {
                Some(word) => (word, true),
                None => (value, false),
            };
            if word.is_empty() || !word.chars().all(char::is_alphanumeric) {
                return Err(unsupported(
                    t.at,
                    "word with characters other than letters and digits",
                ));
            }
            plan.text.push(if prefix {
                TextTerm::Prefix(word.to_owned())
            } else {
                TextTerm::Word(word.to_owned())
            });
        }
        _ => return Err(unsupported(t.at, field)),
    }
    Ok(())
}

/// Binds `tool_calls` and `tool_items` queries, which need an exact
/// `session:` ID. Returns `None` for any other query.
pub fn bind_tools(text: &str) -> Result<Option<ToolQuery>, Failure> {
    let tokens = tokens(text)?;
    let mut i = usize::from(tokens.first().is_some_and(|t| t.is_keyword("find")));
    let Some(table) = tokens.get(i) else {
        return Ok(None);
    };
    let calls_only = table.is_keyword("tool_calls");
    if !calls_only && !table.is_keyword("tool_items") {
        return Ok(None);
    }
    i += 1;
    if tokens.get(i).is_some_and(|t| t.is_keyword("where")) {
        i += 1;
    }
    let mut session = None;
    let mut text = None;
    let mut limit = 100;
    while let Some(t) = tokens.get(i) {
        if t.is_keyword("|") {
            if tokens.get(i + 1).is_none_or(|t| !t.is_keyword("limit")) || tokens.len() != i + 3 {
                return Err(unsupported(
                    t.at,
                    "a tool query stage other than a final limit",
                ));
            }
            limit = tokens[i + 2]
                .value
                .parse::<usize>()
                .ok()
                .filter(|n| (1..=4096).contains(n))
                .ok_or_else(|| invalid(t.at, "tool query limit must be 1 to 4096"))?;
            break;
        }
        if let Some(v) = t.value.strip_prefix("session:").filter(|v| !v.is_empty()) {
            if session.replace(v.to_owned()).is_some() {
                return Err(invalid(t.at, "session: given twice"));
            }
        } else if let Some(v) = t.value.strip_prefix("text:").filter(|v| !v.is_empty()) {
            if text.replace(v.to_owned()).is_some() {
                return Err(invalid(t.at, "text: given twice"));
            }
        } else {
            return Err(unsupported(
                t.at,
                "this tool filter; use session:ID and optional text:",
            ));
        }
        i += 1;
    }
    Ok(Some(ToolQuery {
        calls_only,
        session: session.ok_or_else(|| invalid(0, "tool queries need session:ID"))?,
        text,
        limit,
    }))
}
