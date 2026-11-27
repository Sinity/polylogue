//! What is designed and not built yet. Each row names a verb or command,
//! the option or use the design gives it, and the section of `docs/` that
//! designs it.
//!
//! An option on this list has no field on its step or request type, so
//! help, completion and the MCP tool schemas offer only what the build does.
//! The line parser, the command parser and the argument decoder each consult
//! the list where they meet an option the type does not declare: a listed
//! option is answered with the typed `not_built` refusal naming its page,
//! any other with the ordinary unknown-option refusal. A use (a value or a
//! kind of selection a built option or verb does not take yet) is refused
//! by its handler through [`refusal`]. Building an option is adding its
//! field and deleting its row; building a use is deleting its row and the
//! handler's call.
use command_surface::Failure;
use polylogue_model::Code;
use serde_json::Value;

/// One designed option or use that the build does not have.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Unbuilt {
    /// The verb or command (`read`, `help`); `polylogue` for an option of
    /// every line.
    pub command: &'static str,
    /// An option as written on the command line (`--until`), or a use
    /// (`over several rows`, `--format citation`).
    pub word: &'static str,
    /// The section of `docs/` that designs it.
    pub page: &'static str,
}

const fn row(command: &'static str, word: &'static str, page: &'static str) -> Unbuilt {
    Unbuilt {
        command,
        word,
        page,
    }
}

/// Every designed option and use the build does not have, by verb in the
/// order of the verb table, then by command.
pub const UNBUILT: &[Unbuilt] = &[
    row("read", "--until", "reading.md#ranges"),
    row("read", "--from a message or a time", "reading.md#ranges"),
    row("read", "--beside", "reading.md#alternates"),
    row("read", "--blind", "reading.md#compare-and-blind"),
    row("read", "--seed", "reading.md#compare-and-blind"),
    row("read", "--align", "reading.md#compare-and-blind"),
    row("read", "--until-idle", "session-control.md#following"),
    row("read", "--format citation", "page-text.md#citation-lines"),
    row(
        "pick",
        "Ctrl-o (open in the browser)",
        "terminal.md#the-picker",
    ),
    row(
        "pick",
        "Ctrl-y (copy the reference)",
        "terminal.md#the-picker",
    ),
    row("brief", "of a page or other rows", "agent-context.md#brief"),
    row("note", "--parts", "alternates.md"),
    row("note", "--check", "annotations.md#checked-claims"),
    row("note", "--expect", "annotations.md#checked-claims"),
    row("note", "--page", "pages.md#writing"),
    row("note", "--section", "pages.md#writing"),
    row("note", "--level", "pages.md#writing"),
    row("note", "--template", "page-templates.md"),
    row("note", "--param", "page-templates.md"),
    row("note", "--layout", "pages.md#layouts"),
    row("note", "--about", "pages.md#writing"),
    row("note", "--role", "pages.md#members"),
    row("note", "--brief", "agent-context.md#pages-here"),
    row("note", "--rename", "annotations.md#note"),
    row(
        "note",
        "--as translation:LANG, transliteration:LANG, simplification or summary",
        "alternates.md",
    ),
    row("note", "--as heading", "pages.md#writing"),
    row("judge", "--on", "judging.md#the-better-of-several-rows"),
    row("judge", "--seed", "judging.md#blind"),
    row("judge", "defer", "judging.md#verdicts-on-annotations"),
    row(
        "judge",
        "over several rows",
        "judging.md#the-better-of-several-rows",
    ),
    row("save", "page:NAME", "saved-queries.md#saving-to-a-page"),
    row("save", "--section", "saved-queries.md#saving-to-a-page"),
    row("save", "--after", "saved-queries.md#saving-to-a-page"),
    row("save", "--fixed", "saved-queries.md#saving-to-a-page"),
    row("save", "--live", "saved-queries.md#saving-to-a-page"),
    row("save", "--kept", "saved-queries.md#saving-to-a-page"),
    row("save", "--template", "saved-queries.md#saving-to-a-page"),
    row("save", "--as", "pages.md#forms"),
    row("send", "--page", "session-control.md#send"),
    row("send", "--section", "session-control.md#send"),
    row(
        "send",
        "--attach to a harness session",
        "session-control.md#send",
    ),
    row("answer", "with text", "session-control.md#answer"),
    row("resume", "--fork", "session-control.md#resume"),
    row("resume", "--anyway", "session-control.md#resume"),
    row("help", "--to", "terminal.md#help"),
    row("help", "--format", "terminal.md#help"),
    row("help", "views", "terminal.md#help"),
    row("help", "kinds", "terminal.md#help"),
    row("help", "keys", "terminal.md#help"),
    row("help", "saved", "terminal.md#help"),
    row("help", "COMMAND", "terminal.md#help"),
    row(
        "start",
        "--submit in a harness session",
        "session-control.md#start",
    ),
    row(
        "start",
        "--model in a harness session",
        "session-control.md#start",
    ),
    row(
        "start",
        "--effort in a harness session",
        "session-control.md#start",
    ),
    row(
        "start",
        "--project in a harness session",
        "session-control.md#start",
    ),
    row(
        "start",
        "--attach in a harness session",
        "session-control.md#start",
    ),
    row(
        "import",
        "--withdraw-missing",
        "page-working-copy.md#check-in",
    ),
    row("install", "HARNESS --dry-run", "installing.md#installing"),
    row(
        "polylogue",
        "--to browser",
        "reading.md#formats-and-destinations",
    ),
    row(
        "polylogue",
        "--to clipboard",
        "reading.md#formats-and-destinations",
    ),
];

impl Unbuilt {
    /// An option a type leaves out, rather than a use of a declared one.
    pub fn is_option(&self) -> bool {
        self.word.starts_with("--") && !self.word.contains(' ')
    }

    /// The typed `not_built` refusal, naming the page that designs it.
    pub fn failure(&self) -> Failure {
        crate::protocol::failure(
            Code::NotBuilt,
            format!(
                "{} {} is not built yet; docs/{} designs it",
                self.command, self.word, self.page
            ),
        )
    }
}

/// The row of `word` under `command`, exactly as written in the list.
pub fn listed(command: &str, word: &str) -> Option<&'static Unbuilt> {
    UNBUILT
        .iter()
        .find(|row| row.command == command && row.word == word)
}

/// The refusal of an option that `command` does not declare when the list
/// names it. `name` is the option as the caller wrote it: `--until-idle`,
/// `--until-idle=true` on the command line, `until_idle` in JSON.
pub fn option(command: &str, name: &str) -> Option<Failure> {
    let name = name.split_once('=').map_or(name, |(name, _)| name);
    let name = format!("--{}", name.trim_start_matches('-').replace('_', "-"));
    listed(command, &name)
        .filter(|row| row.is_option())
        .map(Unbuilt::failure)
}

/// The refusal of a listed use, from the handler that meets it.
pub fn refusal(command: &str, word: &str) -> Failure {
    match listed(command, word) {
        Some(row) => row.failure(),
        None => crate::protocol::failure(
            Code::Internal,
            format!("{command} {word} is refused as not built but is not on the list"),
        ),
    }
}

/// The first listed option among the arguments of a request for `row` that
/// its type does not declare, at the top level or in a step.
pub fn in_arguments(row: &str, arguments: &Value) -> Option<Failure> {
    let object = arguments.as_object()?;
    // Most requests name no listed option; they are answered without
    // building the declarations.
    let named = |key: &String| {
        UNBUILT.iter().any(|row| {
            row.is_option() && row.word.trim_start_matches('-').replace('-', "_") == *key
        })
    };
    let steps = object
        .values()
        .filter_map(Value::as_array)
        .flatten()
        .filter_map(Value::as_object);
    if !object.keys().any(named) && !steps.flat_map(|step| step.keys()).any(named) {
        return None;
    }
    let declarations = crate::commands::declarations();
    let properties = declarations
        .descriptions()
        .iter()
        .find(|description| description.spec.name == row)
        .and_then(|description| description.input_schema["properties"].as_object());
    if let Some(properties) = properties
        && let Some(failure) = object
            .keys()
            .filter(|key| !properties.contains_key(*key))
            .find_map(|key| option(row, key))
    {
        return Some(failure);
    }
    for (argument, value) in object {
        let (Some(steps), Value::Array(values)) = (declarations.steps(row, argument), value) else {
            continue;
        };
        for step in values.iter().filter_map(Value::as_object) {
            let Some(verb) = step.get("verb").and_then(Value::as_str) else {
                continue;
            };
            let Some(declared) = steps
                .verbs()
                .iter()
                .find(|declared| declared.verb() == verb)
            else {
                continue;
            };
            let fields = &declared.input_schema()["properties"];
            if let Some(failure) = step
                .keys()
                .filter(|key| *key != "verb" && fields.get(key.as_str()).is_none())
                .find_map(|key| option(verb, key))
            {
                return Some(failure);
            }
        }
    }
    None
}

/// `polylogue help unbuilt`: the list, one row per line.
pub fn text() -> String {
    let mut text = String::from("Designed and not built yet, with the page that designs each:\n");
    for row in UNBUILT {
        text.push_str(&format!(
            "  {} {}: docs/{}\n",
            row.command, row.word, row.page
        ));
    }
    text
}

#[cfg(test)]
mod tests;
