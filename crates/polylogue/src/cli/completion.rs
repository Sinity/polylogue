//! Shell query completion preserves the full argument context and byte position.
use clap_complete::CompletionCandidate;
use std::{
    ffi::{OsStr, OsString},
    path::PathBuf,
    sync::OnceLock,
};

/// The callback argv follows `--` and includes the invoked wrapper name.
#[derive(Default)]
pub struct Context {
    pub arguments: Vec<OsString>,
    pub index: usize,
    pub prefix: Option<String>,
    pub catalog: Option<Catalog>,
}
pub struct Catalog {
    pub path: PathBuf,
    pub archive_key: String,
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
        let candidates = match invocation.and_then(|call| call.catalog.as_ref()) {
            Some(catalog) => crate::completion_catalog::complete_file(
                &catalog.path,
                &catalog.archive_key,
                &text,
                position,
            ),
            None => crate::completion_catalog::grammar(&text, position),
        };
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
        let Some(catalog) = INVOCATION.get().and_then(|call| call.catalog.as_ref()) else {
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
                .subcommand_matches("setting")
                .and_then(|setting| setting.get_one::<String>("key"))
                .cloned()
        };
        crate::completion_catalog::settings_file(
            &catalog.path,
            &catalog.archive_key,
            key.as_deref(),
            prefix,
        )
        .into_iter()
        .filter_map(|(value, help)| candidate(value, help))
        .collect()
    }
}
