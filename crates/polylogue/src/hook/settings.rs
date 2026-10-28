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
pub fn settings_path(harness: Harness) -> Result<PathBuf, String> {
    let config = polylogue_engine::Config::from_env();
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

fn harness_events(harness: Harness) -> &'static [&'static str] {
    match harness {
        Harness::Codex => CODEX_EVENTS,
        Harness::GeminiCli => GEMINI_EVENTS,
        _ => EVENTS,
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
        && (EVENTS.contains(&event) || GEMINI_EVENTS.contains(&event))
        && harness_ok
}

/// Merge Polylogue's commands, keeping other hooks and entry metadata.
/// Refuse invalid containers instead of replacing their contents.
pub(super) fn merge_hooks(
    mut settings: Value,
    harness: Harness,
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
    for event in harness_events(harness) {
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
/// hooks belong in the file's declared source.
pub fn install(path: &Path, harness: Harness, install: bool) -> std::io::Result<()> {
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
    let merged = merge_hooks(current, harness, install)?;
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
