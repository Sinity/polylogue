//! Golden cases: `examples/sources/<origin>/<case>/` holds an input file
//! (`input.jsonl`, `input.json`),
//! the path it would have under the home directory (`path`), and the
//! expected rows and accounting (`expected.json`). A parser bug is fixed by
//! adding its input as a case. `POLYLOGUE_BLESS=1` rewrites the expected
//! files after a deliberate change.

use polylogue_sources::origin::by_name;
use std::fs;
use std::path::{Path, PathBuf};

fn examples() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/sources")
}

fn run_case(origin: &str, dir: &Path) -> Result<(), String> {
    let spec = by_name(origin).ok_or_else(|| format!("unknown origin {origin}"))?;
    let path = fs::read_to_string(dir.join("path")).map_err(|e| format!("path: {e}"))?;
    let path = Path::new(path.trim());
    let input_file = fs::read_dir(dir)
        .map_err(|e| e.to_string())?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| p.file_stem().is_some_and(|s| s == "input"))
        .ok_or("no input.* file")?;
    let input = fs::read(&input_file).map_err(|e| format!("input: {e}"))?;
    let head = &input[..input.len().min(64)];
    if !(spec.recognize)(path, head) {
        return Err(format!("{} does not recognize {}", origin, path.display()));
    }
    let parser = (spec.open)(path).ok_or("origin does not open this path")?;
    let (step, _carry) = parser.step(None, &input, 0).map_err(|e| e.to_string())?;
    if step.consumed != input.len() as u64 {
        return Err(format!(
            "consumed {} of {} bytes",
            step.consumed,
            input.len()
        ));
    }
    let value = sorted(serde_json::json!({"output": step.output, "accounting": step.accounting}));
    let got = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())? + "\n";
    let expected_path = dir.join("expected.json");
    let bless = std::env::var_os("POLYLOGUE_BLESS").is_some();
    match fs::read_to_string(&expected_path) {
        Ok(expected) if !bless => {
            // Compared as values: a workspace build can turn on serde_json's
            // preserve_order (through the Sinex crates), which changes key
            // order but not content.
            let expected: serde_json::Value =
                serde_json::from_str(&expected).map_err(|e| e.to_string())?;
            if expected == value {
                Ok(())
            } else {
                Err(format!(
                    "output differs from {}; rerun with POLYLOGUE_BLESS=1 if the change is intended\n{got}",
                    expected_path.display()
                ))
            }
        }
        _ => fs::write(&expected_path, got).map_err(|e| e.to_string()),
    }
}

#[test]
fn golden_cases() {
    let mut failures = Vec::new();
    let mut ran = 0;
    for origin in fs::read_dir(examples()).expect("examples/sources exists") {
        let origin = origin.unwrap().path();
        let name = origin.file_name().unwrap().to_str().unwrap().to_owned();
        for case in fs::read_dir(&origin).unwrap() {
            let case = case.unwrap().path();
            ran += 1;
            if let Err(e) = run_case(&name, &case) {
                failures.push(format!("{}: {e}", case.display()));
            }
        }
    }
    assert!(ran > 0, "no golden cases found");
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}

/// Objects with their keys in sorted order, however serde_json is built.
fn sorted(v: serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<_> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(entries.into_iter().map(|(k, v)| (k, sorted(v))).collect())
        }
        serde_json::Value::Array(items) => {
            serde_json::Value::Array(items.into_iter().map(sorted).collect())
        }
        other => other,
    }
}
