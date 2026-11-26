use super::*;
use std::collections::BTreeMap;

#[test]
fn a_sandbox_replaces_every_inherited_variable() {
    let sandbox = Sandbox::new(Path::new("/run/sandbox")).unwrap();
    let mut command = Command::new("env");
    command
        .env("CLAUDE_CONFIG_DIR", "/synthetic/elsewhere/.claude")
        .env("CODEX_HOME", "/synthetic/elsewhere/.codex")
        .env("XDG_STATE_HOME", "/synthetic/elsewhere/.local/state")
        .env("POLYLOGUE_LISTEN", "127.0.0.1:8765")
        .env("GEMINI_CLI", "1")
        .env("DBUS_SESSION_BUS_ADDRESS", "unix:path=/synthetic/bus");
    sandbox.apply(&mut command);
    let output = command.output().unwrap();
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    let values: BTreeMap<&str, &str> = text.lines().filter_map(|l| l.split_once('=')).collect();
    let path = std::env::var("PATH").unwrap();
    let listen = sandbox.listen().to_string();
    assert_eq!(
        values,
        BTreeMap::from([
            ("CLAUDE_CONFIG_DIR", "/run/sandbox/home/.claude"),
            ("CODEX_HOME", "/run/sandbox/home/.codex"),
            ("HERMES_HOME", "/run/sandbox/home/.hermes"),
            ("HOME", "/run/sandbox/home"),
            ("LANG", "C.UTF-8"),
            ("PATH", path.as_str()),
            ("POLYLOGUE_LISTEN", listen.as_str()),
            ("TMPDIR", "/run/sandbox/tmp"),
            ("XDG_CACHE_HOME", "/run/sandbox/cache"),
            ("XDG_CONFIG_HOME", "/run/sandbox/config"),
            ("XDG_DATA_HOME", "/run/sandbox/data"),
            ("XDG_RUNTIME_DIR", "/run/sandbox/runtime"),
            ("XDG_STATE_HOME", "/run/sandbox/state"),
        ])
    );
}

#[test]
fn a_sandbox_names_a_loopback_port_and_writes_one_deployment() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let sandbox = Sandbox::new(dir.path()).unwrap();
    assert!(sandbox.listen().ip().is_loopback());
    assert_ne!(sandbox.listen().port(), 0);
    sandbox.create_directories().unwrap();
    let runtime = std::fs::metadata(dir.path().join("runtime")).unwrap();
    assert_eq!(runtime.permissions().mode() & 0o777, 0o700);
    let archive = Path::new("/srv/an \"archive\"\\\u{7f}");
    sandbox.write_deployment(archive).unwrap();
    assert_eq!(
        std::fs::read_to_string(sandbox.deployment()).unwrap(),
        "archive = \"/srv/an \\\"archive\\\"\\\\\\u007F\"\n\n[day]\nzone = \"UTC\"\nstart = \"04:00:00\"\n"
    );
    // A deployment already there is the caller's.
    std::fs::write(sandbox.deployment(), "archive = \"/elsewhere\"\n").unwrap();
    sandbox.write_deployment(archive).unwrap();
    assert_eq!(
        std::fs::read_to_string(sandbox.deployment()).unwrap(),
        "archive = \"/elsewhere\"\n"
    );
}
