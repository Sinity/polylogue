//! Real shell callbacks: the archive's values come from the daemon, and
//! without it the grammar alone.
use crate::{BIN, process};

use polylogue_store::fixture::{CallFacts, Fixture, MessageFacts, SessionFacts};
use std::{
    path::Path,
    time::{Duration, Instant},
};

/// A synthetic archive: two sessions with titles, repositories, a model and
/// a tool call, and the saved query `by-action`.
fn seed(root: &Path, model: &str) {
    let mut fixture = Fixture::new(root).unwrap();
    let session = fixture
        .session(SessionFacts {
            id: "codex:abc123",
            title: Some("A neutral title"),
            repo: Some("two words"),
            last: Some(100),
            ..Default::default()
        })
        .unwrap();
    fixture
        .session(SessionFacts {
            id: "codex:other",
            repo: Some("żółw"),
            last: Some(50),
            ..Default::default()
        })
        .unwrap();
    let message = fixture
        .message(
            session,
            MessageFacts {
                id: "codex:m1",
                role: "assistant",
                written_by: "model",
                model: Some(model),
                ..Default::default()
            },
        )
        .unwrap();
    fixture
        .tool_call(
            session,
            message,
            CallFacts {
                tool: "Bash",
                action: "shell",
                outcome: "ok",
                ..Default::default()
            },
        )
        .unwrap();
    fixture.derive().unwrap();
    let plan = polylogue_query::plan::prepare(
        "tool_calls | group by action | count",
        &polylogue_query::Bindings::default(),
        |_| Ok(()),
    )
    .unwrap();
    fixture.saved_query("by-action", &plan).unwrap();
}

/// The fixture's environment for an archive in `base`, its own home beside it.
fn environment(base: &Path, root: &Path) -> process::Fixture {
    let name = root.file_name().unwrap().to_string_lossy();
    process::Fixture::new(
        root,
        &base.join(format!("home-{name}")),
        &base.join("config"),
    )
}

/// Keeps the rows of a query with match rows through the daemon, as
/// `QUERY save` does, and returns the kept result's ID.
fn keep(environment: &process::Fixture) -> String {
    let output = environment
        .command()
        .args([
            "--json",
            "--request-id",
            "kept",
            "sessions where exists tool_calls()",
            "save",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&output.stdout).unwrap();
    let reply = answer.value().unwrap();
    let kept = reply["reply"]["steps"][0]["outcome"]["reply"]["reply"]["kept"]["result"]
        .as_str()
        .unwrap_or_else(|| panic!("{reply}"));
    kept.strip_prefix("result:").unwrap().to_owned()
}

fn callback(
    environment: &process::Fixture,
    shell: &str,
    words: &[&str],
    program: &str,
) -> (String, Duration) {
    let at = Instant::now();
    let output = environment
        .native_command()
        .env("COMPLETE", shell)
        .env("_CLAP_COMPLETE_INDEX", (words.len() + 2).to_string())
        .env("_CLAP_IFS", "\n")
        .arg("--")
        .arg(program)
        .arg("--archive")
        .arg(environment.archive())
        .args(words)
        .output()
        .unwrap();
    assert!(
        output.status.success() && output.stderr.is_empty(),
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
fn completion_matrix_thirteen_positions_three_shells_daemon_down_and_up() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    seed(&root, "alpha");
    let environment = environment(directory.path(), &root);
    let mut timings = vec![];
    let mut kept = String::new();
    let mut daemon = None;
    for up in [false, true] {
        if up {
            daemon = Some(crate::cli::start_daemon(&environment));
            kept = keep(&environment);
        }
        let kept_matches = format!("result:{kept}.matches | where to");
        let cases: [(&str, Vec<&str>, &str, bool); 13] = [
            ("commands", vec!["sta"], "status", false),
            ("session fields", vec!["sessions where ti"], "title", false),
            ("message fields", vec!["messages where mo"], "model", false),
            ("tool fields", vec!["tool_calls where ou"], "outcome", false),
            (
                "observed value",
                vec!["messages where model:alp"],
                "model:alpha",
                true,
            ),
            (
                "quoted value",
                vec!["sessions where repo:\"two"],
                "two words",
                true,
            ),
            ("pipe stage", vec!["messages | gr"], "group by", false),
            (
                "group alias",
                vec!["sessions | group by repo | sort by co"],
                "count",
                false,
            ),
            (
                "nested scope",
                vec!["sessions where exists (tool_calls where ou"],
                "outcome",
                false,
            ),
            (
                "saved source",
                vec!["query:by-action | where co"],
                "count",
                true,
            ),
            ("kept matches", vec![&kept_matches], "tool", true),
            ("session reference", vec!["abc"], "codex:a", true),
            // Setting keys are the CLI's own table, offered without the daemon.
            (
                "setting key",
                vec!["set", "browser."],
                "browser.actions",
                false,
            ),
        ];
        for shell in ["bash", "zsh", "fish"] {
            for (name, words, expected, needs_daemon) in &cases {
                if !up && *name == "kept matches" {
                    continue;
                }
                let (output, took) = callback(&environment, shell, words, "polylogue");
                timings.push(took);
                if up || !needs_daemon {
                    assert!(
                        output.contains(expected),
                        "{shell}/{name}/daemon up={up}: {output:?}"
                    );
                } else {
                    assert!(
                        !output.contains(expected),
                        "{shell}/{name}: no daemon, yet a value was offered: {output:?}"
                    );
                }
            }
        }
    }
    drop(daemon);
    timings.sort();
    eprintln!(
        "{} shell callbacks: median {:?}, p95 {:?}, maximum {:?}",
        timings.len(),
        timings[timings.len() / 2],
        timings[timings.len() * 95 / 100],
        timings.last().unwrap()
    );
}

#[test]
fn wrapper_unicode_and_setting_values_come_from_the_daemon() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    seed(&root, "alpha");
    let environment = environment(directory.path(), &root);
    let _daemon = crate::cli::start_daemon(&environment);
    for shell in ["bash", "zsh", "fish"] {
        let (output, _) = callback(&environment, shell, &["sessions where repo:żó"], "plg");
        assert!(output.contains("żółw"), "{shell}: {output}");
        let (output, _) = callback(&environment, shell, &["ntrl"], "plg");
        assert!(output.contains("codex:a"), "{shell}: {output}");
        let (output, _) = callback(&environment, shell, &["set", "browser.actions", "o"], "plg");
        assert!(
            output.contains("on") && output.contains("off"),
            "{shell}: {output}"
        );
    }
}

#[test]
fn a_saved_name_as_first_word_runs_once_the_daemon_answers() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    seed(&root, "alpha");
    let environment = environment(directory.path(), &root);
    let run = |words: &[&str]| environment.command().args(words).output().unwrap();
    let daemon = crate::cli::start_daemon(&environment);
    let saved = run(&["--json", "by-action"]);
    assert!(saved.status.success(), "{saved:?}");
    // The saved query's one group: the tool call's action.
    let shown = String::from_utf8_lossy(&saved.stdout);
    assert!(shown.contains("\"shell\""), "{shown}");
    drop(daemon);
    // With the daemon down the word is a search, and the line fails as any
    // line does then.
    let down = run(&["by-action"]);
    let sessions = run(&["sessions"]);
    assert!(!down.status.success(), "{down:?}");
    assert_eq!(down.status.code(), sessions.status.code(), "{down:?}");
}

#[test]
fn named_wrappers_keep_their_registration_and_selected_archive() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().unwrap();
    let base = directory.path();
    let default = base.join("default");
    let first = base.join("first archive");
    let second = base.join("selected config/second archive");
    let mut daemons = Vec::new();
    for (archive, model) in [
        (&default, "default_value"),
        (&first, "first_value"),
        (&second, "second_value"),
    ] {
        std::fs::create_dir_all(archive.parent().unwrap()).unwrap();
        seed(archive, model);
        daemons.push(crate::cli::start_daemon(&environment(base, archive)));
    }
    let fixture = environment(base, &default);
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
            command.env("COMPLETE", shell);
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
                .env("_CLAP_COMPLETE_INDEX", "1")
                .env("_CLAP_IFS", "\n")
                .args(["--", name, "messages where model:"])
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
    drop(daemons);
}

#[test]
fn completion_latency_reports_real_shell_callbacks_with_daemon_up_and_down() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("archive");
    let mut source = Fixture::new(&root).unwrap();
    source
        .session(SessionFacts {
            id: "codex:completion-latency",
            repo: Some("neutral-repository"),
            ..Default::default()
        })
        .unwrap();
    drop(source);
    let environment = environment(directory.path(), &root);
    let mut daemon = Some(crate::cli::start_daemon(&environment));
    for mode in ["up", "down"] {
        if mode == "down" {
            drop(daemon.take());
        }
        for shell in ["bash", "zsh", "fish"] {
            let mut samples = Vec::new();
            for _ in 0..10 {
                let (output, elapsed) = callback(
                    &environment,
                    shell,
                    &["sessions where repo:neutral"],
                    "polylogue",
                );
                assert_eq!(
                    output.contains("neutral-repository"),
                    mode == "up",
                    "daemon {mode}, {shell}: {output}"
                );
                samples.push(elapsed.as_micros());
            }
            let mut sorted = samples.clone();
            sorted.sort_unstable();
            eprintln!(
                "completion daemon={mode} shell={shell} samples_us={samples:?} median_us={} p95_us={} maximum_us={}",
                sorted[sorted.len() / 2],
                sorted[sorted.len() * 95 / 100],
                sorted.last().unwrap()
            );
        }
    }
}

#[test]
fn normal_install_honors_captured_zdotdir_and_preserves_home_init() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    let home_init = root.join("home/.zshrc");
    std::fs::create_dir_all(home_init.parent().unwrap()).unwrap();
    std::fs::write(&home_init, b"# home initialization\n").unwrap();
    let zdotdir = root.join("zsh config");
    std::fs::create_dir_all(&zdotdir).unwrap();
    let init = zdotdir.join(".zshrc");
    let original = b"# custom initialization\n";
    std::fs::write(&init, original).unwrap();
    for extra in [vec![], vec!["--remove"]] {
        let output = crate::command(root)
            .env("ZDOTDIR", &zdotdir)
            .args(["install", "zsh"])
            .args(&extra)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            std::fs::read(&home_init).unwrap(),
            b"# home initialization\n"
        );
        if extra.is_empty() {
            assert!(
                std::fs::read_to_string(&init)
                    .unwrap()
                    .contains("polylogue")
            );
        }
    }
    assert_eq!(std::fs::read(init).unwrap(), original);
}

#[test]
fn normal_drive_install_without_client_names_the_setting_and_creates_no_token() {
    let temp = tempfile::tempdir().unwrap();
    let output = crate::command(temp.path())
        .args(["install", "drive", "--dry-run"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2), "{output:?}");
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains("polylogue set drive.client_id CLIENT_ID"),
        "{text}"
    );
    assert!(!temp.path().join("archive/drive-token.json").exists());
}
