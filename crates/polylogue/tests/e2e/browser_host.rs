//! The browser host through the real binary: `install-agent chrome` writes
//! and removes the host manifest, and the binary started as Chrome starts
//! it (with the extension's origin) answers a native messaging exchange.

use std::io::Write;
use std::process::{Command, Stdio};

fn frame(v: &serde_json::Value) -> Vec<u8> {
    let bytes = serde_json::to_vec(v).unwrap();
    let mut out = (bytes.len() as u32).to_ne_bytes().to_vec();
    out.extend(bytes);
    out
}

fn frames(mut bytes: &[u8]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    while bytes.len() >= 4 {
        let len = u32::from_ne_bytes(bytes[..4].try_into().unwrap()) as usize;
        out.push(serde_json::from_slice(&bytes[4..4 + len]).unwrap());
        bytes = &bytes[4 + len..];
    }
    out
}

#[test]
fn install_agent_chrome_writes_and_removes_the_host_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config");
    std::fs::create_dir_all(config.join("chromium")).unwrap();
    let run = |args: &[&str]| {
        Command::new(crate::BIN)
            .env("XDG_CONFIG_HOME", &config)
            .args(args)
            .output()
            .unwrap()
    };
    let out = run(&["install-agent", "chrome"]);
    assert!(!out.status.success(), "an extension ID is required");
    let out = run(&[
        "install-agent",
        "chrome",
        "--extension-id",
        "abcdefghijklmnop",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let path = config.join("chromium/NativeMessagingHosts/com.polylogue.browser_capture.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    assert_eq!(
        manifest["allowed_origins"][0],
        "chrome-extension://abcdefghijklmnop/"
    );
    assert_eq!(manifest["type"], "stdio");
    assert!(std::path::Path::new(manifest["path"].as_str().unwrap()).is_absolute());
    // Only the existing browser's directory is written.
    assert!(!config.join("google-chrome").exists());
    let out = run(&["install-agent", "chrome", "--remove"]);
    assert!(out.status.success());
    assert!(!path.exists());
    let out = run(&["install-agent", "codex"]);
    assert!(!out.status.success());
}

#[test]
fn started_as_chrome_starts_it_the_binary_is_the_host() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(crate::BIN)
        .arg("chrome-extension://abcdefghijklmnop/")
        .env("POLYLOGUE_ARCHIVE", dir.path().join("archive"))
        .env("POLYLOGUE_HOME", dir.path().join("home"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut input = Vec::new();
    input.extend(frame(&serde_json::json!({
        "type": "request", "version": 1, "method": "GET", "path": "/v1/status", "headers": {}
    })));
    input.extend(frame(&serde_json::json!({ "type": "body_end" })));
    for _ in 0..4 {
        input.extend(frame(&serde_json::json!({ "type": "response_next" })));
    }
    child.stdin.take().unwrap().write_all(&input).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    let got = frames(&out.stdout);
    assert_eq!(got[0]["type"], "response");
    assert_eq!(got[0]["status"], 200);
    assert_eq!(got[0]["api_schema"], "polylogue-browser-capture/v1");
    assert_eq!(got.last().unwrap()["type"], "response_end");
}
