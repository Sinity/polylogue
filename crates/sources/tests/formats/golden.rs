//! Golden cases: `examples/sources/<origin>/<case>/` holds an input file
//! (`input.jsonl`, `input.json`),
//! the path it would have under the home directory (`path`), and the
//! expected rows and accounting (`expected.json`). A parser bug is fixed by
//! adding its input as a case. Expected rows are written from the format
//! and reviewed independently of the parser.

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
    let mut parser = (spec.open)(path).ok_or("origin does not open this path")?;
    let step = super::support::array_fixture::parse(&mut parser, &input)?;
    if step.consumed != input.len() as u64 {
        return Err(format!(
            "consumed {} of {} bytes",
            step.consumed,
            input.len()
        ));
    }
    let value = sorted(serde_json::json!({"output": step.output, "counts": step.counts}));
    let got = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())? + "\n";
    let expected_path = dir.join("expected.json");
    compare_expected(&expected_path, &value, &got)
}

fn compare_expected(
    expected_path: &Path,
    value: &serde_json::Value,
    got: &str,
) -> Result<(), String> {
    match fs::read_to_string(expected_path) {
        Ok(expected) => {
            // Compared as values: a workspace build can turn on serde_json's
            // preserve_order (through the Sinex crates), which changes key
            // order but not content.
            let expected: serde_json::Value =
                serde_json::from_str(&expected).map_err(|e| e.to_string())?;
            if &expected == value {
                Ok(())
            } else {
                Err(format!(
                    "output differs from {}; compare with the input format and review the expected rows\n{got}",
                    expected_path.display()
                ))
            }
        }
        Err(error) => Err(format!("read {}: {error}", expected_path.display())),
    }
}

#[test]
fn missing_expectation_fails_without_creating_it() {
    let dir = tempfile::tempdir().unwrap();
    let expected = dir.path().join("expected.json");
    let value = serde_json::json!({"output": []});
    assert!(compare_expected(&expected, &value, "{}\n").is_err());
    assert!(!expected.exists());
}

#[test]
fn invalid_expectation_is_preserved() {
    let dir = tempfile::tempdir().unwrap();
    let expected = dir.path().join("expected.json");
    fs::write(&expected, "invalid JSON").unwrap();
    let value = serde_json::json!({});
    assert!(compare_expected(&expected, &value, "{}\n").is_err());
    assert_eq!(fs::read_to_string(&expected).unwrap(), "invalid JSON");
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

#[test]
fn every_declaration_checks_its_reading_version() {
    use std::collections::BTreeMap;
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/formats/reading_versions.json");
    let saved: BTreeMap<String, (String, String)> =
        serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let mut current = BTreeMap::new();
    for origin in polylogue_sources::ORIGINS
        .iter()
        .filter(|o| o.support == polylogue_sources::Support::Parsed)
    {
        assert!(
            !origin.modules.is_empty(),
            "{} declares no modules",
            origin.name
        );
        let dir = examples().join(origin.name);
        let mut cases: Vec<_> = fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("{} needs synthetic fixtures: {e}", origin.name))
            .map(|e| e.unwrap().path())
            .collect();
        cases.sort();
        assert!(
            !cases.is_empty(),
            "{} has no synthetic fixtures",
            origin.name
        );
        let mut hash = blake3::Hasher::new();
        let mut parsed_rows = 0;
        for case in cases {
            let recorded = fs::read_to_string(case.join("path")).unwrap();
            let input = fs::read_dir(&case)
                .unwrap()
                .map(|e| e.unwrap().path())
                .find(|p| p.file_stem().is_some_and(|s| s == "input"))
                .unwrap();
            let bytes = fs::read(input).unwrap();
            let mut parser = (origin.open)(Path::new(recorded.trim())).unwrap();
            let step = super::support::array_fixture::parse(&mut parser, &bytes).unwrap();
            parsed_rows += step.output.sessions.len()
                + step.output.messages.len()
                + step.output.documents.len()
                + step.output.events.len()
                + step.output.links.len()
                + step.output.repo_lookups.len();
            let output = serde_json::to_vec(&sorted(serde_json::to_value(step).unwrap())).unwrap();
            hash.update(&(output.len() as u64).to_le_bytes());
            hash.update(&output);
        }
        assert!(parsed_rows > 0, "{} fixtures produce no rows", origin.name);
        let tuple = (
            origin.reading_version(),
            hash.finalize().to_hex().to_string(),
        );
        if let Some(previous) = saved.get(origin.name) {
            assert!(
                previous.1 == tuple.1 || previous.0 != tuple.0,
                "{} output changed without a parser module version bump",
                origin.name
            );
        }
        current.insert(origin.name.to_owned(), tuple);
    }
    assert_eq!(
        current, saved,
        "review changed output against the input format, bump its parser module version, and update reading_versions.json"
    );
}
