//! Shell completion: one request to the daemon per Tab for the archive's
//! values, bounded by the completion deadline; without an answer in time,
//! the grammar from the CLI's own tables and nothing else. The query keeps
//! its full argument context and byte position.
use clap_complete::CompletionCandidate;
use query_language::complete::Completion;
use std::{
    ffi::{OsStr, OsString},
    path::PathBuf,
    sync::OnceLock,
    time::Duration,
};

/// The callback argv follows `--` and includes the invoked wrapper name.
#[derive(Default)]
pub struct Context {
    pub arguments: Vec<OsString>,
    pub index: usize,
    pub prefix: Option<String>,
    /// The archive whose daemon is asked; `None` when the configuration
    /// does not load, and only the grammar is offered.
    pub archive: Option<PathBuf>,
}
static INVOCATION: OnceLock<Context> = OnceLock::new();
/// Main supplies its one captured process/configuration boundary.
pub fn capture(context: Context) {
    let _ = INVOCATION.set(context);
}

pub struct Query;
impl clap_complete::engine::ValueCompleter for Query {
    fn complete(&self, current: &OsStr) -> Vec<CompletionCandidate> {
        self.complete_at(0, current)
    }
    fn complete_at(&self, arg_index: usize, current: &OsStr) -> Vec<CompletionCandidate> {
        let Some(current) = current.to_str() else {
            return vec![];
        };
        let invocation = INVOCATION.get();
        let (text, position, start, end) = invocation
            .and_then(|call| {
                context(
                    &call.arguments,
                    call.index,
                    arg_index,
                    current,
                    call.prefix.as_deref(),
                )
            })
            .unwrap_or_else(|| (current.to_owned(), current.len(), 0, current.len()));
        let candidates = invocation
            .and_then(|call| call.archive.as_deref())
            .and_then(|archive| {
                ask(
                    archive,
                    crate::protocol::complete::Arguments {
                        text: text.clone(),
                        position: Some(position as u64),
                        ..Default::default()
                    },
                    Some(KEYSTROKE_WAIT),
                )
            })
            .unwrap_or_else(|| crate::completion::grammar(&text, position));
        candidates
            .into_iter()
            .filter_map(|completion| {
                if completion.replace.start < start || completion.replace.end > end {
                    return None;
                }
                let mut value = text[start..completion.replace.start].to_owned();
                value.push_str(&completion.value);
                value.push_str(&text[completion.replace.end..end]);
                candidate(value, completion.help)
            })
            .collect()
    }
}

/// How long one Tab waits for the daemon before offering the grammar alone.
/// The completion class's budget is 50 ms per keystroke at p95
/// (docs/latency.md); the wait is longer so that a loaded daemon still
/// answers with values, while a daemon that is down refuses at once.
pub(crate) const KEYSTROKE_WAIT: Duration = Duration::from_millis(250);

/// The daemon's candidates, or `None` when no daemon answered within the
/// budget or it refused; the caller then offers what needs no archive.
pub(crate) fn ask(
    archive: &std::path::Path,
    arguments: crate::protocol::complete::Arguments,
    wait: Option<Duration>,
) -> Option<Vec<Completion>> {
    let request = crate::client::request(
        "internal complete",
        serde_json::to_value(&arguments).ok()?,
        polylogue_store::user_writes::new_request_id(),
    );
    let socket = polylogue_store::Archive::at(archive).socket();
    let answer = match wait {
        Some(wait) => crate::client::call_with_budget(&socket, &request, wait),
        None => crate::client::call(&socket, &request),
    }
    .ok()?;
    let value = crate::protocol::into_value(answer).ok()?;
    serde_json::from_value(value).ok()
}

pub(super) fn candidate(value: String, help: String) -> Option<CompletionCandidate> {
    if value.chars().any(char::is_control) || polylogue_render::terminal::escape(&value) != value {
        return None;
    }
    let help = (!help.is_empty()).then(|| polylogue_render::terminal::escape(&help));
    Some(CompletionCandidate::new(value).help(help.map(Into::into)))
}

/// Whether a saved query is named `word`: completing `query:WORD` offers
/// the word itself. The answer decides how the line is read, so it waits
/// for the daemon as any request does; `false` when no daemon runs.
pub(crate) fn is_saved_name(archive: &std::path::Path, word: &str) -> bool {
    let text = format!("query:{word}");
    ask(
        archive,
        crate::protocol::complete::Arguments {
            position: Some(text.len() as u64),
            text,
            ..Default::default()
        },
        None,
    )
    .is_some_and(|candidates| candidates.iter().any(|candidate| candidate.value == word))
}

/// Shell arguments are already decoded. Their query spelling is the same join
/// used by CLI parsing; only the current shell word is replaced.
fn context(
    arguments: &[OsString],
    index: usize,
    arg_index: usize,
    current: &str,
    prefix: Option<&str>,
) -> Option<(String, usize, usize, usize)> {
    let first = index.checked_sub(arg_index)?;
    let words = arguments
        .get(first..)?
        .iter()
        .map(|word| word.to_str())
        .collect::<Option<Vec<_>>>()?;
    let relative = index.checked_sub(first)?;
    let start = words
        .get(..relative)?
        .iter()
        .map(|word| word.len() + 1)
        .sum::<usize>();
    let original = *words.get(relative)?;
    let prefix = prefix.unwrap_or(current);
    if !original.starts_with(prefix) {
        return None;
    }
    Some((
        words.join(" "),
        start + prefix.len(),
        start,
        start + original.len(),
    ))
}

#[cfg(test)]
mod tests;

pub struct Setting {
    pub key: bool,
}
impl clap_complete::engine::ValueCompleter for Setting {
    fn complete(&self, current: &OsStr) -> Vec<CompletionCandidate> {
        let Some(prefix) = current.to_str() else {
            return vec![];
        };
        let key = if self.key {
            None
        } else {
            let Some(call) = INVOCATION.get() else {
                return vec![];
            };
            let mut arguments = call.arguments.clone();
            if call.index >= arguments.len() {
                arguments.push("__completion__".into());
            } else {
                arguments[call.index] = "__completion__".into();
            }
            let Ok(matches) = super::command().try_get_matches_from(arguments) else {
                return vec![];
            };
            matches
                .subcommand_matches("set")
                .and_then(|setting| setting.get_one::<String>("key"))
                .cloned()
        };
        INVOCATION
            .get()
            .and_then(|call| call.archive.as_deref())
            .and_then(|archive| {
                ask(
                    archive,
                    crate::protocol::complete::Arguments {
                        text: prefix.to_owned(),
                        setting: true,
                        key: key.clone(),
                        ..Default::default()
                    },
                    Some(KEYSTROKE_WAIT),
                )
            })
            .unwrap_or_else(|| crate::completion::settings(key.as_deref(), prefix, &[], |_| None))
            .into_iter()
            .filter_map(|completion| candidate(completion.value, completion.help))
            .collect()
    }
}
