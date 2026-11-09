//! Merge only the hook commands this installer owns, into Claude Code's
//! `settings.json` or Codex's `hooks.json` (the same layout). Hermes reads
//! its hooks from `config.yaml`; its entries are listed in the README.

use super::{EVENTS, Harness};
use serde_json::{Value, json};
use std::io;
use std::path::{Path, PathBuf};

mod parse;
mod spans;

/// The events Gemini CLI calls hooks on.
const GEMINI_EVENTS: &[&str] = &[
    "SessionStart",
    "BeforeAgent",
    "BeforeTool",
    "AfterTool",
    "AfterAgent",
    "Notification",
    "PreCompress",
    "SessionEnd",
];

/// The events Codex calls hooks on.
const CODEX_EVENTS: &[&str] = &[
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "Stop",
];

/// The harness's hook settings file in its configured directory.
pub fn settings_path(
    harness: Harness,
    config: &polylogue_engine::Config,
) -> Result<PathBuf, String> {
    match harness {
        Harness::ClaudeCode => Ok(config
            .root(polylogue_sources::claude_code::ROOT)
            .join("settings.json")),
        Harness::Codex => Ok(config.root(polylogue_sources::codex::ROOT).join("hooks.json")),
        Harness::GeminiCli => Ok(config.root(polylogue_sources::origin::Root::HOME).join(".gemini/settings.json")),
        Harness::Hermes => Err(
            "Hermes reads its hooks from config.yaml: add `hooks: {pre_llm_call: [{command: \"polylogue hook pre_llm_call --harness hermes\"}]}`"
                .into(),
        ),
    }
}

/// Claude Code events registered only from the version a probe confirmed
/// them in: a harness older than that may not know them, and they carry
/// the waits and stalls the live fold reads (permission requests, failed
/// stops, subagent starts, questions).
pub const CONFIRMED_FROM: &[(&str, &str)] = &[
    ("PermissionRequest", "2.1.316"),
    ("PermissionDenied", "2.1.316"),
    ("StopFailure", "2.1.316"),
    ("SubagentStart", "2.1.316"),
    ("Elicitation", "2.1.316"),
];

/// An installed harness's version, as `--version` prints it: dotted
/// numbers compared part by part.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct HarnessVersion(Vec<u64>);

impl HarnessVersion {
    /// The first word of `text` made of dotted numbers (`2.1.316 (Claude
    /// Code)`, `codex-cli 0.164.0`).
    pub fn parse(text: &str) -> Option<Self> {
        text.split_whitespace().find_map(|word| {
            let parts: Option<Vec<u64>> = word.split('.').map(|p| p.parse().ok()).collect();
            parts.filter(|p| p.len() >= 2).map(Self)
        })
    }
}

/// Asks the installed harness for its version; `None` when it is not
/// installed or prints none.
pub fn probe_version(harness: Harness) -> Option<HarnessVersion> {
    let program = match harness {
        Harness::ClaudeCode => "claude",
        Harness::Codex => "codex",
        Harness::GeminiCli => "gemini",
        Harness::Hermes => "hermes",
    };
    let output = std::process::Command::new(program)
        .arg("--version")
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    HarnessVersion::parse(&String::from_utf8_lossy(&output.stdout))
}

/// The events to register for `harness` at `version`: an event of
/// [`CONFIRMED_FROM`] only when the version is known and not older.
fn harness_events(harness: Harness, version: Option<&HarnessVersion>) -> Vec<&'static str> {
    match harness {
        Harness::Codex => CODEX_EVENTS.to_vec(),
        Harness::GeminiCli => GEMINI_EVENTS.to_vec(),
        _ => EVENTS
            .iter()
            .copied()
            .chain(CONFIRMED_FROM.iter().filter_map(|(event, from)| {
                version
                    .zip(HarnessVersion::parse(from))
                    .is_some_and(|(v, from)| *v >= from)
                    .then_some(*event)
            }))
            .collect(),
    }
}

/// Every event an install may have registered for `harness`, for removal.
fn every_event(harness: Harness) -> Vec<&'static str> {
    match harness {
        Harness::Codex | Harness::GeminiCli => harness_events(harness, None),
        _ => EVENTS
            .iter()
            .chain(CONFIRMED_FROM.iter().map(|(event, _)| event))
            .copied()
            .collect(),
    }
}

fn command(harness: Harness, event: &str) -> String {
    match harness {
        Harness::ClaudeCode => format!("polylogue hook {event}"),
        _ => format!("polylogue hook {event} --harness {}", harness.origin()),
    }
}

fn invalid(path: &str, expected: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("{path}: expected {expected}"),
    )
}

fn validate(settings: &Value) -> io::Result<()> {
    let settings = settings
        .as_object()
        .ok_or_else(|| invalid("settings", "an object"))?;
    let Some(hooks) = settings.get("hooks") else {
        return Ok(());
    };
    let events = hooks
        .as_object()
        .ok_or_else(|| invalid("hooks", "an object"))?;
    for (event, entries) in events {
        let path = format!("hooks.{event}");
        let entries = entries
            .as_array()
            .ok_or_else(|| invalid(&path, "an array"))?;
        for (index, entry) in entries.iter().enumerate() {
            let path = format!("{path}[{index}]");
            let entry = entry
                .as_object()
                .ok_or_else(|| invalid(&path, "an object"))?;
            let hooks = entry
                .get("hooks")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid(&format!("{path}.hooks"), "an array"))?;
            for (index, hook) in hooks.iter().enumerate() {
                let path = format!("{path}.hooks[{index}]");
                let hook = hook
                    .as_object()
                    .ok_or_else(|| invalid(&path, "an object"))?;
                let kind = hook
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid(&format!("{path}.type"), "a string"))?;
                if kind == "command" && !hook.get("command").is_some_and(Value::is_string) {
                    return Err(invalid(&format!("{path}.command"), "a string"));
                }
            }
        }
    }
    Ok(())
}

/// A command this installer wrote: `polylogue hook EVENT`, with
/// `--harness NAME` for a harness other than Claude Code, and nothing else.
fn ours(hook: &Value) -> bool {
    let Some(rest) = hook["command"]
        .as_str()
        .and_then(|command| command.strip_prefix("polylogue hook "))
    else {
        return false;
    };
    let words: Vec<&str> = rest.split(' ').collect();
    let event = words[0];
    let harness_ok = match words[1..] {
        [] => true,
        ["--harness", name] => Harness::parse(name).is_some(),
        _ => false,
    };
    hook["type"] == "command"
        && (EVENTS.contains(&event)
            || GEMINI_EVENTS.contains(&event)
            || CONFIRMED_FROM.iter().any(|(known, _)| *known == event))
        && harness_ok
}

/// One hook entry this installer writes for `event`.
fn entry(harness: Harness, event: &str) -> Value {
    let seconds = if matches!(event, "SessionStart" | "UserPromptSubmit" | "BeforeAgent") {
        10
    } else {
        2
    };
    let timeout = if harness == Harness::GeminiCli {
        seconds * 1000
    } else {
        seconds
    };
    json!({"hooks": [{"type": "command", "command": command(harness, event), "timeout": timeout}]})
}

/// Installs or removes Polylogue's commands in the settings `text`, editing
/// only their own bytes: other hooks, entry metadata, key order, numbers
/// and formatting stay as written. An entry, event or `hooks` object that
/// removing Polylogue's commands leaves empty goes too; one that was empty
/// before stays. Invalid containers are refused instead of replaced.
/// `version` is the installed harness's, which decides the events of
/// [`CONFIRMED_FROM`]; removal covers every event an install may have added.
fn edit(
    text: &str,
    harness: Harness,
    version: Option<&HarnessVersion>,
    install: bool,
) -> io::Result<String> {
    use spans::{Kind, splice};
    validate(&parse::settings(text)?)?;
    let root = spans::read(text)?;
    let Kind::Object(members) = &root.kind else {
        return Err(invalid("settings", "an object"));
    };
    let wanted = harness_events(harness, version);
    let mut edits = Vec::new();
    let Some(hooks) = root.member("hooks") else {
        if !install {
            return Ok(text.to_owned());
        }
        let events: serde_json::Map<String, Value> = wanted
            .iter()
            .map(|event| ((*event).to_owned(), json!([entry(harness, event)])))
            .collect();
        let spans: Vec<_> = members
            .iter()
            .map(|(_, start, value)| *start..value.end)
            .collect();
        edits.extend(splice(
            text,
            &root,
            &spans,
            &vec![false; spans.len()],
            |layout| vec![layout.member("hooks", &Value::Object(events.clone()))],
        ));
        return Ok(spans::apply(text, edits));
    };
    let Kind::Object(events) = &hooks.kind else {
        return Err(invalid("hooks", "an object"));
    };
    let is_ours = |hook: &spans::Spanned| {
        serde_json::from_str::<Value>(&text[hook.start..hook.end]).is_ok_and(|hook| ours(&hook))
    };
    let mut removed_events = vec![false; events.len()];
    let mut added_events = Vec::new();
    for event in every_event(harness) {
        let install = install && wanted.contains(&event);
        let Some(index) = events.iter().position(|(name, _, _)| name == event) else {
            if install {
                added_events.push(event);
            }
            continue;
        };
        let entries = &events[index].2;
        let mut removed_entries = Vec::new();
        let mut entry_edits = Vec::new();
        for entry in entries.items() {
            let hooks = entry.member("hooks").expect("validated hook array");
            let mine: Vec<bool> = hooks.items().iter().map(is_ours).collect();
            let all_mine = !mine.is_empty() && mine.iter().all(|m| *m);
            removed_entries.push(all_mine);
            if !all_mine && mine.contains(&true) {
                let spans: Vec<_> = hooks.items().iter().map(|h| h.start..h.end).collect();
                entry_edits.extend(splice(text, hooks, &spans, &mine, |_| Vec::new()));
            }
        }
        let emptied = !removed_entries.is_empty() && removed_entries.iter().all(|r| *r);
        if emptied && !install {
            removed_events[index] = true;
            continue;
        }
        edits.extend(entry_edits);
        let spans: Vec<_> = entries.items().iter().map(|e| e.start..e.end).collect();
        edits.extend(splice(text, entries, &spans, &removed_entries, |layout| {
            if install {
                vec![layout.render(&entry(harness, event))]
            } else {
                Vec::new()
            }
        }));
    }
    let every_event_removed = !events.is_empty() && removed_events.iter().all(|r| *r);
    if every_event_removed && added_events.is_empty() {
        let spans: Vec<_> = members
            .iter()
            .map(|(_, start, value)| *start..value.end)
            .collect();
        let removed: Vec<bool> = members.iter().map(|(name, _, _)| name == "hooks").collect();
        edits.extend(splice(text, &root, &spans, &removed, |_| Vec::new()));
        return Ok(spans::apply(text, edits));
    }
    let spans: Vec<_> = events
        .iter()
        .map(|(_, start, value)| *start..value.end)
        .collect();
    edits.extend(splice(text, hooks, &spans, &removed_events, |layout| {
        added_events
            .iter()
            .map(|event| layout.member(event, &json!([entry(harness, event)])))
            .collect()
    }));
    Ok(spans::apply(text, edits))
}

/// [`edit`] over a settings value, for callers that hold one.
#[cfg(test)]
pub(super) fn merge_hooks(
    settings: Value,
    harness: Harness,
    version: Option<&HarnessVersion>,
    install: bool,
) -> io::Result<Value> {
    let text = edit(&settings.to_string(), harness, version, install)?;
    serde_json::from_str(&text).map_err(io::Error::other)
}

/// The file the original settings bytes are kept in before the first edit.
pub(super) fn backup_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_owned();
    name.push(".polylogue-backup");
    path.with_file_name(name)
}

/// Installs or removes the hooks in a settings file, atomically, editing only
/// Polylogue's entries. Before the first edit of an existing file its bytes
/// are kept in [`backup_path`]; a removal restores them exactly when the file
/// is still what installing into them produced. A file that is a symlink
/// (managed by Home Manager or similar) is refused: the hooks belong in the
/// file's declared source. The version provider runs only for an install
/// after the existing file has passed structural validation.
pub fn install(
    path: &Path,
    harness: Harness,
    version: impl FnOnce() -> Option<HarnessVersion>,
    install: bool,
) -> std::io::Result<()> {
    if path.is_symlink() {
        return Err(std::io::Error::other(format!(
            "{} is a symlink to a managed file; add the hooks in its source instead",
            path.display()
        )));
    }
    let current = match std::fs::read_to_string(path) {
        Ok(text) => Some(text),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    if current.is_none() && !install {
        return Ok(());
    }
    let text = current.as_deref().unwrap_or("{}\n");
    // Refuse the settings before consulting an external harness launcher.
    validate(&parse::settings(text)?)?;
    let backup = backup_path(path);
    let saved = match std::fs::read_to_string(&backup) {
        Ok(saved) => Some(saved),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    let restore = !install
        && saved
            .as_deref()
            .is_some_and(|saved| installed_from(saved, harness, text));
    let new = if restore {
        saved.clone().expect("a restore has its backup")
    } else {
        let version = install.then(version).flatten();
        edit(text, harness, version.as_ref(), install)?
    };
    if current.as_deref() == Some(new.as_str()) {
        return Ok(());
    }
    let dir = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(dir)?;
    use std::os::unix::fs::PermissionsExt;
    let permissions = match std::fs::metadata(path) {
        Ok(meta) => meta.permissions(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            std::fs::Permissions::from_mode(0o600)
        }
        Err(e) => return Err(e),
    };
    if let Some(original) = &current
        && saved.is_none()
    {
        polylogue_model::atomic_write::write_atomic(
            &backup,
            original.as_bytes(),
            Some(permissions.clone()),
        )?;
    }
    polylogue_model::atomic_write::write_atomic(path, new.as_bytes(), Some(permissions))?;
    if restore {
        std::fs::remove_file(&backup)?;
    }
    Ok(())
}

/// Whether `current` is what an install into `saved` produces. The harness
/// version decides only which [`CONFIRMED_FROM`] events are added, so each
/// version that changes them is tried.
fn installed_from(saved: &str, harness: Harness, current: &str) -> bool {
    std::iter::once(None)
        .chain(
            CONFIRMED_FROM
                .iter()
                .map(|(_, from)| HarnessVersion::parse(from)),
        )
        .any(|version| {
            edit(saved, harness, version.as_ref(), true).is_ok_and(|installed| installed == current)
        })
}
