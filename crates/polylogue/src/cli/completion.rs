//! Shell query completion preserves the full argument context and byte position.
use clap_complete::CompletionCandidate;
use std::{
    ffi::{OsStr, OsString},
    sync::OnceLock,
};

#[derive(Default)]
struct Invocation {
    arguments: Vec<OsString>,
    index: usize,
    prefix: Option<String>,
}
static INVOCATION: OnceLock<Invocation> = OnceLock::new();

/// Capture before clap's completion entry point consumes its environment.
pub fn capture() {
    if std::env::var_os("COMPLETE").is_none() {
        return;
    }
    let arguments = std::env::args_os()
        .skip_while(|arg| arg != "--")
        .skip(1)
        .collect::<Vec<_>>();
    let index = std::env::var("_CLAP_COMPLETE_INDEX")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or_else(|| arguments.len().saturating_sub(1));
    let _ = INVOCATION.set(Invocation {
        arguments,
        index,
        prefix: std::env::var("_POLYLOGUE_COMPLETE_PREFIX").ok(),
    });
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
        crate::completion_catalog::complete(
            &polylogue_store::Archive::default_root(),
            &text,
            position,
        )
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

pub(super) fn candidate(value: String, help: String) -> Option<CompletionCandidate> {
    if value.chars().any(char::is_control) || polylogue_render::terminal::escape(&value) != value {
        return None;
    }
    let help = (!help.is_empty()).then(|| polylogue_render::terminal::escape(&help));
    Some(CompletionCandidate::new(value).help(help.map(Into::into)))
}

/// Shell arguments are already decoded. Their query spelling is the same join
/// used by ordinary CLI parsing; only the current shell word is replaced.
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
        let root = polylogue_store::Archive::default_root();
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
                .subcommand_matches("setting")
                .and_then(|setting| setting.get_one::<String>("key"))
                .cloned()
        };
        crate::completion_catalog::settings(&root, key.as_deref(), prefix)
            .into_iter()
            .filter_map(|(value, help)| candidate(value, help))
            .collect()
    }
}
