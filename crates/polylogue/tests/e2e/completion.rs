//! Real shell callbacks over an offline synthetic catalog.
use crate::{BIN, process};

use polylogue_model::ChangePosition;
use polylogue_store::completion::{CatalogWriter, Kept, Metadata, Saved, Session, Setting};
use query_language::catalog::{Field, FieldKind, OutputSchema};
use std::{
    collections::BTreeMap,
    path::Path,
    time::{Duration, Instant},
};
fn seed(root: &Path, cache: &Path, model: &str) {
    let key = blake3::hash(root.canonicalize().unwrap().as_os_str().as_encoded_bytes()).to_hex()
        [..12]
        .to_owned();
    let catalog = polylogue_query::catalog::catalog(&["codex".into()]);
    CatalogWriter::fixture(
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
                    ("model".into(), BTreeMap::from([(model.into(), 2)])),
                ]),
            })?;
            let field = |name: &str, kind| Field {
                name: name.into(),
                kind,
                ..Default::default()
            };
            let mut saved = Saved {
                relation_scope: None,
                family: "saved-family".into(),
                revision: "saved-revision".into(),
                name: "by-action".into(),
                text: "tool_calls | group by action | count".into(),
                parameters: vec![],
                help: "Count native actions".into(),
                schema: Some(OutputSchema {
                    match_units: None,
                    unit: Some("tool_calls".into()),
                    fields: vec![
                        field("action", FieldKind::Text),
                        field("count", FieldKind::Number),
                    ],
                    main_time: None,
                }),
            };
            saved.relation_scope = saved.schema.clone();
            writer.put_saved(&saved)?;
            writer.put_kept(&Kept {
                id: "kept-id".into(),
                name: Some("Kept calls".into()),
                unit: Some("tool_calls".into()),
                row_count: Some(3),
                rows: Some(OutputSchema {
                    match_units: None,
                    unit: Some("tool_calls".into()),
                    fields: vec![field("count", FieldKind::Number)],
                    main_time: None,
                }),
                matches: Some(OutputSchema {
                    match_units: None,
                    unit: Some("tool_calls".into()),
                    fields: vec![field("tool", FieldKind::Text)],
                    main_time: None,
                }),
            })?;
            for (key, choices) in [
                ("embed.model", vec![]),
                ("browser.actions", vec!["on".into(), "off".into()]),
            ] {
                writer.put_setting(&Setting {
                    key: key.into(),
                    value: {
                        use polylogue_model::settings::{self, Current, Owner};
                        let declaration = settings::declaration(key).unwrap();
                        let value = settings::current(key, None, Owner::Standalone).unwrap();
                        Current {
                            key: key.into(),
                            kind: declaration.kind,
                            default: value.clone(),
                            scope: declaration.scope,
                            owner: "polylogue".into(),
                            value,
                            authored: None,
                            conflicts: vec![],
                        }
                    },
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
    let output = process::Fixture::new(
        root,
        &root.parent().unwrap().join("home"),
        &root.parent().unwrap().join("config"),
    )
    .native_command()
    .env("COMPLETE", shell)
    .env("XDG_CACHE_HOME", cache)
    .env("_CLAP_COMPLETE_INDEX", (words.len() + 2).to_string())
    .env("_CLAP_IFS", "\n")
    .arg("--")
    .arg(program)
    .arg("--archive")
    .arg(root)
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
        (
            "setting key",
            &["setting", "browser."],
            "browser.actions",
            true,
        ),
    ];
    let mut timings = vec![];
    for seeded in [false, true] {
        if seeded {
            seed(&root, &cache, "alpha");
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
    seed(&root, &cache, "alpha");
    for shell in ["bash", "zsh", "fish"] {
        let (output, _) = callback(&root, &cache, shell, &["sessions where repo:żó"], "plg");
        assert!(output.contains("żółw"), "{shell}: {output}");
        let (output, _) = callback(&root, &cache, shell, &["ntrl"], "plg");
        assert!(output.contains("codex:a"), "{shell}: {output}");
        let (output, _) = callback(
            &root,
            &cache,
            shell,
            &["setting", "browser.actions", "o"],
            "plg",
        );
        assert!(
            output.contains("on") && output.contains("off"),
            "{shell}: {output}"
        );
    }
}

#[test]
fn named_wrappers_keep_their_registration_and_selected_offline_catalog() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let base = directory.path();
    let cache = base.join("cache");
    let default = base.join("default");
    let first = base.join("first archive");
    let second = base.join("selected config/second archive");
    for (archive, model) in [
        (&default, "default_value"),
        (&first, "first_value"),
        (&second, "second_value"),
    ] {
        std::fs::create_dir_all(archive).unwrap();
        std::fs::write(archive.join("index.db"), b"must remain unopened").unwrap();
        seed(archive, &cache, model);
    }
    let fixture = process::Fixture::new(&default, &base.join("home"), &base.join("config"));
    let first_config = base.join("first.toml");
    std::fs::write(
        &first_config,
        format!("archive={}\n", serde_json::to_string(&second).unwrap()),
    )
    .unwrap();
    let second_config = base.join("selected config/polylogue.toml");
    std::fs::write(&second_config, "archive='second archive'\n").unwrap();
    let quote = |value: &str| format!("'{}'", value.replace('\'', "'\\''"));
    let bash = std::env::split_paths(&std::env::var_os("PATH").unwrap())
        .map(|path| path.join("bash"))
        .find(|path| path.is_file())
        .expect("the development shell provides bash");
    let mut functions = Vec::new();
    for (name, flags, expected, other) in [
        (
            "polylogue-first",
            format!(
                "--archive {} --config {}",
                quote(first.to_str().unwrap()),
                quote(first_config.to_str().unwrap())
            ),
            "first_value",
            "second_value",
        ),
        (
            "polylogue_first",
            format!("--config {}", quote(second_config.to_str().unwrap())),
            "second_value",
            "first_value",
        ),
    ] {
        let wrapper = base.join(name);
        std::fs::write(
            &wrapper,
            format!(
                "#!{}\nexec -a \"$0\" {} {flags} \"$@\"\n",
                bash.display(),
                quote(BIN)
            ),
        )
        .unwrap();
        std::fs::set_permissions(&wrapper, std::fs::Permissions::from_mode(0o700)).unwrap();
        let command = |shell: &str| {
            let mut command = std::process::Command::new(&wrapper);
            fixture.environment(&mut command);
            command.env("COMPLETE", shell).env("XDG_CACHE_HOME", &cache);
            command
        };
        let registration = command("bash").output().unwrap();
        assert!(
            registration.status.success(),
            "{}",
            String::from_utf8_lossy(&registration.stderr)
        );
        let registration = String::from_utf8(registration.stdout).unwrap();
        assert!(
            registration.contains(wrapper.to_str().unwrap()),
            "{registration}"
        );
        let registrations: Vec<_> = registration
            .lines()
            .filter(|line| line.trim_start().starts_with("complete "))
            .collect();
        assert!(
            !registrations.is_empty()
                && registrations
                    .iter()
                    .all(|line| line.ends_with(&format!(" {name}"))),
            "{registration}"
        );
        functions.push(
            registration
                .lines()
                .find(|line| line.starts_with("_clap_complete_") && line.ends_with("() {"))
                .expect("Bash registration declares a completion function")
                .to_owned(),
        );
        for shell in ["bash", "zsh", "fish"] {
            let output = command(shell)
                .env("_CLAP_COMPLETE_INDEX", "3")
                .env("_CLAP_IFS", "\n")
                .args(["--", name, "messages", "where", "model:"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{shell}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let output = String::from_utf8(output.stdout).unwrap();
            assert!(output.contains(expected), "{name}/{shell}: {output}");
            assert!(
                !output.contains(other) && !output.contains("default_value"),
                "{name}/{shell}: {output}"
            );
        }
    }
    assert_ne!(
        functions[0], functions[1],
        "wrapper registrations must coexist"
    );
    for archive in [&default, &first, &second] {
        assert_eq!(
            std::fs::read(archive.join("index.db")).unwrap(),
            b"must remain unopened"
        );
        assert_eq!(std::fs::read_dir(archive).unwrap().count(), 1);
    }
}
