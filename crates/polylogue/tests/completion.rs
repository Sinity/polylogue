//! Real shell callbacks over an offline synthetic catalog.
use polylogue_model::ChangePosition;
use polylogue_store::completion::{CatalogWriter, Kept, Metadata, Saved, Session, Setting};
use query_language::catalog::{Field, FieldKind, OutputSchema};
use std::{
    collections::BTreeMap,
    path::Path,
    process::Command,
    time::{Duration, Instant},
};
const BIN: &str = env!("CARGO_BIN_EXE_polylogue");
fn seed(root: &Path, cache: &Path) {
    let key = blake3::hash(root.canonicalize().unwrap().as_os_str().as_encoded_bytes()).to_hex()
        [..12]
        .to_owned();
    let catalog = polylogue_query::catalog::catalog(&["codex".into()], &["status".into()]);
    CatalogWriter::build(
        &cache.join("polylogue").join(&key).join("catalog"),
        &Metadata {
            archive: key,
            position: ChangePosition {
                generation: "g".into(),
                index: 1,
                archive: 1,
            },
            written_at: 100,
        },
        |writer| {
            writer.put_catalog(&catalog)?;
            writer.put_session(&Session {
                id: "codex:abc123".into(),
                native_id: "abc123".into(),
                prefix: String::new(),
                title: "A neutral title".into(),
                origin: "codex".into(),
                repo: Some("two words".into()),
                last_activity: Some(100),
                first_prompt: Some("A neutral prompt".into()),
                values: BTreeMap::from([
                    (
                        "repo".into(),
                        BTreeMap::from([("two words".into(), 3), ("żółw".into(), 2)]),
                    ),
                    ("model".into(), BTreeMap::from([("alpha".into(), 2)])),
                ]),
            })?;
            let field = |name: &str, kind| Field {
                name: name.into(),
                kind,
                ..Default::default()
            };
            writer.put_saved(&Saved {
                family: "saved-family".into(),
                revision: "saved-revision".into(),
                name: "by-action".into(),
                text: "tool_calls | group by action | count".into(),
                parameters: vec![],
                help: "Count native actions".into(),
                schema: Some(OutputSchema {
                    unit: Some("tool_calls".into()),
                    fields: vec![
                        field("action", FieldKind::Text),
                        field("count", FieldKind::Number),
                    ],
                    main_time: None,
                }),
            })?;
            writer.put_kept(&Kept {
                id: "kept-id".into(),
                name: Some("Kept calls".into()),
                unit: Some("tool_calls".into()),
                row_count: Some(3),
                rows: Some(OutputSchema {
                    unit: Some("tool_calls".into()),
                    fields: vec![field("count", FieldKind::Number)],
                    main_time: None,
                }),
                matches: Some(OutputSchema {
                    unit: Some("tool_calls".into()),
                    fields: vec![field("tool", FieldKind::Text)],
                    main_time: None,
                }),
            })?;
            for (key, choices) in [
                ("day.zone", vec![]),
                ("briefing.session_start", vec!["on".into(), "off".into()]),
            ] {
                writer.put_setting(&Setting {
                    key: key.into(),
                    value: serde_json::Value::Null,
                    choices,
                })?;
            }
            Ok(())
        },
    )
    .unwrap();
}
fn callback(
    root: &Path,
    cache: &Path,
    shell: &str,
    words: &[&str],
    program: &str,
) -> (String, Duration) {
    let at = Instant::now();
    let output = Command::new(BIN)
        .env("COMPLETE", shell)
        .env("POLYLOGUE_ARCHIVE", root)
        .env("XDG_CACHE_HOME", cache)
        .env("_CLAP_COMPLETE_INDEX", words.len().to_string())
        .env("_CLAP_IFS", "\n")
        .arg("--")
        .arg(program)
        .args(words)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{shell} {words:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    (
        String::from_utf8(output.stdout)
            .unwrap()
            .replace("\\:", ":"),
        at.elapsed(),
    )
}
#[test]
fn completion_matrix_thirteen_positions_three_shells_empty_and_seeded() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let cache = directory.path().join("cache");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(
        root.join("index.db"),
        b"archive must never be opened by completion",
    )
    .unwrap();
    let cases: [(&str, &[&str], &str, bool); 13] = [
        ("commands", &["sta"], "status", false),
        (
            "session fields",
            &["sessions", "where", "ti"],
            "title",
            false,
        ),
        (
            "message fields",
            &["messages", "where", "mo"],
            "model",
            false,
        ),
        (
            "tool fields",
            &["tool_calls", "where", "ou"],
            "outcome",
            false,
        ),
        (
            "observed value",
            &["messages", "where", "model:alp"],
            "model:alpha",
            true,
        ),
        (
            "quoted value",
            &["sessions where repo:\"two"],
            "two words",
            true,
        ),
        ("pipe stage", &["messages", "|", "gr"], "group by", false),
        (
            "group alias",
            &["sessions | group by repo | sort by co"],
            "count",
            false,
        ),
        (
            "nested scope",
            &["sessions where exists (tool_calls where ou"],
            "outcome",
            false,
        ),
        (
            "saved source",
            &["query:by-action | where co"],
            "count",
            true,
        ),
        (
            "kept matches",
            &["result:kept-id.matches | where to"],
            "tool",
            true,
        ),
        ("session reference", &["abc"], "codex:a", true),
        ("setting key", &["setting", "day."], "day.zone", true),
    ];
    let mut timings = vec![];
    for seeded in [false, true] {
        if seeded {
            seed(&root, &cache);
        }
        for shell in ["bash", "zsh", "fish"] {
            for (name, words, expected, needs_catalog) in cases {
                let (output, took) = callback(&root, &cache, shell, words, "polylogue");
                timings.push(took);
                if seeded || !needs_catalog {
                    assert!(
                        output.contains(expected),
                        "{shell}/{name}/seeded={seeded}: {output:?}"
                    );
                } else {
                    assert!(
                        !output.contains(expected),
                        "{shell}/{name}: missing cache invented a value: {output:?}"
                    );
                }
            }
        }
    }
    assert_eq!(
        std::fs::read(root.join("index.db")).unwrap(),
        b"archive must never be opened by completion"
    );
    assert_eq!(
        std::fs::read_dir(&root).unwrap().count(),
        1,
        "completion wrote inside the archive"
    );
    timings.sort();
    eprintln!(
        "78 shell callbacks: median {:?}, p95 {:?}, maximum {:?}",
        timings[timings.len() / 2],
        timings[timings.len() * 95 / 100],
        timings.last().unwrap()
    );
}
#[test]
fn wrapper_unicode_and_setting_values_use_the_same_offline_provider() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let cache = directory.path().join("cache");
    std::fs::create_dir(&root).unwrap();
    seed(&root, &cache);
    for shell in ["bash", "zsh", "fish"] {
        let (output, _) = callback(&root, &cache, shell, &["sessions where repo:żó"], "plg");
        assert!(output.contains("żółw"), "{shell}: {output}");
        let (output, _) = callback(&root, &cache, shell, &["ntrl"], "plg");
        assert!(output.contains("codex:a"), "{shell}: {output}");
        let (output, _) = callback(
            &root,
            &cache,
            shell,
            &["setting", "briefing.session_start", "o"],
            "plg",
        );
        assert!(
            output.contains("on") && output.contains("off"),
            "{shell}: {output}"
        );
    }
}
