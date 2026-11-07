//! Merge only the hook commands this installer owns, into Claude Code's
//! `settings.json` or Codex's `hooks.json` (the same layout). Hermes reads
//! its hooks from `config.yaml`; its entries are listed in the README.

use super::{EVENTS, Harness};
use serde_json::{Value, json};
use std::io;
use std::path::{Path, PathBuf};

mod parse;

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

/// Merge Polylogue's commands, keeping other hooks and entry metadata.
/// Refuse invalid containers instead of replacing their contents.
/// `version` is the installed harness's, which decides the events of
/// [`CONFIRMED_FROM`]; removal covers every event an install may have added.
pub(super) fn merge_hooks(
    mut settings: Value,
    harness: Harness,
    version: Option<&HarnessVersion>,
    install: bool,
) -> io::Result<Value> {
    validate(&settings)?;
    let settings_object = settings.as_object_mut().expect("validated settings object");
    if !install && !settings_object.contains_key("hooks") {
        return Ok(settings);
    }
    let events = settings_object
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .expect("validated hooks object");
    let wanted = harness_events(harness, version);
    for event in &every_event(harness) {
        let install = install && wanted.contains(event);
        if let Some(entries) = events.get_mut(*event) {
            let entries = entries.as_array_mut().expect("validated event array");
            let had_entries = !entries.is_empty();
            entries.retain_mut(|entry| {
                let hooks = entry["hooks"].as_array_mut().expect("validated hook array");
                let before = hooks.len();
                hooks.retain(|hook| !ours(hook));
                // Keep entries that were already empty; only remove entries
                // made empty by removing our commands.
                before == hooks.len() || !hooks.is_empty()
            });
            if had_entries && entries.is_empty() && !install {
                events.remove(*event);
            }
        }
        if install {
            events.entry(*event).or_insert_with(|| json!([]))
                .as_array_mut().expect("validated event array").push(json!({
                    "hooks": [{"type":"command", "command":command(harness, event),
                        "timeout": (if *event=="SessionStart" || *event=="UserPromptSubmit" || *event=="BeforeAgent" {10} else {2}) * if harness == Harness::GeminiCli {1000} else {1}}]
                }));
        }
    }
    Ok(settings)
}

/// Installs or removes the hooks in a settings file, atomically. A file
/// that is a symlink (managed by Home Manager or similar) is refused: the
/// hooks belong in the file's declared source. The version provider runs only
/// for an install after the existing file has passed structural validation.
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
        Ok(text) => parse::settings(&text)?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => json!({}),
        Err(e) => return Err(e),
    };
    // Refuse the settings before consulting an external harness launcher.
    validate(&current)?;
    let version = install.then(version).flatten();
    let merged = merge_hooks(current, harness, version.as_ref(), install)?;
    let mut text = serde_json::to_string_pretty(&merged).map_err(std::io::Error::other)?;
    text.push('\n');
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
    polylogue_model::atomic_write::write_atomic(path, text.as_bytes(), Some(permissions))?;
    Ok(())
}
