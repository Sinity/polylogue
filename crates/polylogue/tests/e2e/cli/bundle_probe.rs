//! Scratch probe (not committed): writes Morrow session bundles to
//! $BUNDLE_PROBE_OUT for a before/after comparison.
use super::*;

#[test]
fn bundle_probe_writes_morrow_bundles() {
    let out = std::path::PathBuf::from(
        "/realm/tmp/work/claude-1000/session-tmp/3fccd4f7-89c1-4126-b631-97976f8c1ec3/r2/after",
    );
    let temp = tempfile::tempdir().unwrap();
    let operator = crate::Fixture::new(
        &temp.path().join("operator-archive"),
        &temp.path().join("operator-home"),
        &temp.path().join("config"),
    );
    let directory = temp.path().join("morrow");
    let mut demo = Command::new(crate::BIN);
    operator.environment(&mut demo);
    let built = demo.arg("demo").arg(&directory).output().unwrap();
    assert!(built.status.success(), "{built:?}");
    let env = crate::Fixture::new(
        &directory.join("archive"),
        &directory.join("home"),
        &directory.join("config"),
    );
    let _daemon = start_daemon(&env);
    wait_status(&env, |status| status["store"]["sessions"] == 8);
    for (name, target, format) in [
        (
            "s2-json",
            "claude-code:7f3a2c10-0002-4000-8000-000000000002",
            "json",
        ),
        (
            "s4-jsonl",
            "claude-code:7f3a2c10-0004-4000-8000-000000000004",
            "jsonl",
        ),
        (
            "s5-markdown",
            "codex:01a20000-0005-7000-8000-000000000005",
            "markdown",
        ),
        (
            "s7-html",
            "chatgpt:68f0a1b2-0007-8000-8000-000000000007",
            "html",
        ),
    ] {
        let destination = out.join(name);
        let output = cli(
            &env,
            &[
                "--deadline",
                "none",
                &format!("session:{target}"),
                "read",
                "--to",
                &format!("{}/", destination.display()),
                "--format",
                format,
            ],
        );
        assert!(output.status.success(), "{name}: {output:?}");
    }
    // Report every written file's hash; the manifest's ran_at is the only
    // value that differs between runs.
    let mut lines = Vec::new();
    let mut stack = vec![out.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let mut bytes = std::fs::read(&path).unwrap();
            if path.file_name().unwrap() == "manifest.json"
                || path.extension().is_some_and(|e| e == "jsonl")
            {
                let text = String::from_utf8(bytes).unwrap();
                let re = |t: &str| {
                    let mut out = String::new();
                    let mut rest = t;
                    while let Some(at) = rest.find("\"ran_at\":\"") {
                        out.push_str(&rest[..at + 10]);
                        rest = &rest[at + 10..];
                        let end = rest.find('"').unwrap();
                        out.push_str("RAN_AT");
                        rest = &rest[end..];
                    }
                    out.push_str(rest);
                    out
                };
                bytes = re(&text).into_bytes();
                if path.file_name().unwrap() == "manifest.json" {
                    lines.push(format!("MANIFEST {}", String::from_utf8_lossy(&bytes)));
                }
            }
            lines.push(format!(
                "{} {} {}",
                path.strip_prefix(&out).unwrap().display(),
                bytes.len(),
                polylogue_model::BlobHash::of(&bytes)
            ));
        }
    }
    lines.sort();
    panic!("BUNDLE-PROBE\n{}\nBUNDLE-PROBE-END", lines.join("\n"));
}
