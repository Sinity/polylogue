use super::*;
use std::collections::BTreeMap;

#[test]
fn a_synthetic_environment_replaces_every_inherited_directory() {
    let home = Path::new("/home/op");
    let mut command = Command::new("env");
    command
        .env("CLAUDE_CONFIG_DIR", "/synthetic/elsewhere/.claude")
        .env("CODEX_HOME", "/synthetic/elsewhere/.codex")
        .env("GEMINI_CLI", "1")
        .env("DBUS_SESSION_BUS_ADDRESS", "unix:path=/synthetic/bus");
    synthetic_environment(&mut command, home);
    let output = command.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    let values: BTreeMap<&str, &str> = text.lines().filter_map(|l| l.split_once('=')).collect();
    let path = std::env::var("PATH").unwrap();
    assert_eq!(
        values,
        BTreeMap::from([
            ("CLAUDE_CONFIG_DIR", "/home/op/.claude"),
            ("CODEX_HOME", "/home/op/.codex"),
            ("HERMES_HOME", "/home/op/.hermes"),
            ("HOME", "/home/op"),
            ("LANG", "C.UTF-8"),
            ("PATH", path.as_str()),
            ("XDG_CACHE_HOME", "/home/op/.cache"),
            ("XDG_CONFIG_HOME", "/home/op/.config"),
            ("XDG_DATA_HOME", "/home/op/.local/share"),
            ("XDG_STATE_HOME", "/home/op/.local/state"),
        ])
    );
}

/// Temporary: a panic whose backtrace shows whether file and line survive.
#[test]
#[ignore]
fn experiment_backtrace() {
    let value: Option<u8> = std::hint::black_box(None);
    value.expect("experiment panic");
}
