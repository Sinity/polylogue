use super::*;

fn arguments(words: &[&str]) -> Vec<OsString> {
    words.iter().map(OsString::from).collect()
}

fn scanned(words: &[&str]) -> Invocation {
    scan(&arguments(words), &EnvSnapshot::from_values([]))
}

#[test]
fn hook_detected_after_valued_globals() {
    let invocation = scanned(&[
        "polylogue",
        "--archive",
        "/synthetic/archive",
        "--deadline",
        "100",
        "--request-id=r1",
        "--json",
        "internal",
        "hook",
        "--harness",
        "claude-code",
        "Stop",
    ]);
    assert!(
        matches!(invocation.shape, Shape::Hook),
        "{:?}",
        invocation.shape
    );
    assert_eq!(
        invocation.overrides.archive,
        Some("/synthetic/archive".into())
    );
    assert_eq!(
        invocation.words,
        [
            "--deadline",
            "100",
            "--request-id=r1",
            "--json",
            "internal",
            "hook",
            "--harness",
            "claude-code",
            "Stop"
        ]
    );
    assert!(invocation.json);
    assert_eq!(invocation.problem, None);
    // A selection word is not the hook, wherever `internal hook` follows.
    assert!(matches!(
        scanned(&["polylogue", "sessions", "internal", "hook"]).shape,
        Shape::Line
    ));
}

/// Every option the command tree declares global and valued is one the
/// scan steps over with its value, because both are built from
/// `GLOBAL_OPTIONS`.
#[test]
fn valued_globals_come_from_the_declarations() {
    let tree = crate::cli::command();
    let valued: Vec<&str> = tree
        .get_arguments()
        .filter(|argument| argument.is_global_set() && argument.get_action().takes_values())
        .filter_map(|argument| argument.get_long())
        .collect();
    assert_eq!(
        valued,
        [
            "request",
            "request-id",
            "deadline",
            "cursor",
            "budget",
            "archive",
            "config",
            "listen",
            "auth-token-file"
        ]
    );
    for name in valued {
        let value = if name == "listen" {
            "127.0.0.1:9"
        } else {
            "mcp"
        };
        for words in [
            vec!["polylogue".to_owned(), format!("--{name}"), value.into()],
            vec!["polylogue".to_owned(), format!("--{name}={value}")],
        ] {
            let mut words = words;
            words.extend(["internal".into(), "hook".into(), "Stop".into()]);
            let words: Vec<&str> = words.iter().map(String::as_str).collect();
            let invocation = scanned(&words);
            assert!(
                matches!(invocation.shape, Shape::Hook),
                "{words:?}: {:?}",
                invocation.shape
            );
            assert_eq!(invocation.problem, None, "{words:?}");
        }
    }
}

#[test]
fn deployment_options_leave_the_words_and_select_the_native_host() {
    let origin = "chrome-extension://aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/";
    let invocation = scanned(&[
        "polylogue",
        "--archive",
        "/synthetic/archive",
        "--config=/synthetic/config",
        "--listen",
        "127.0.0.1:8102",
        "--auth-token-file",
        "/synthetic/token",
        origin,
    ]);
    assert!(matches!(invocation.shape, Shape::NativeHost));
    assert_eq!(invocation.words, [origin]);
    let overrides = invocation.overrides;
    assert_eq!(overrides.archive, Some("/synthetic/archive".into()));
    assert_eq!(overrides.config_file, Some("/synthetic/config".into()));
    assert_eq!(overrides.listen, Some("127.0.0.1:8102".parse().unwrap()));
    assert_eq!(overrides.auth_token_file, Some("/synthetic/token".into()));
    assert!(matches!(
        scanned(&["polylogue", origin, "extra"]).shape,
        Shape::Line
    ));
}

#[test]
fn shapes_of_commands_help_and_lines() {
    let shape = |words: &[&str]| format!("{:?}", scanned(words).shape);
    assert_eq!(
        shape(&["polylogue", "--json", "mcp"]),
        "Command(Some(\"mcp\"))"
    );
    assert_eq!(
        shape(&["polylogue", "internal", "browser-host"]),
        "Command(Some(\"internal browser-host\"))"
    );
    assert_eq!(shape(&["polylogue", "--version"]), "Command(None)");
    assert_eq!(shape(&["polylogue", "--help"]), "Help");
    assert_eq!(shape(&["polylogue", "help", "read"]), "Help");
    assert_eq!(shape(&["polylogue"]), "Line");
    assert_eq!(shape(&["polylogue", "find", "mcp"]), "Line");
    // `--` makes the next argument a literal selection, even a command's
    // name or a deployment option's.
    assert_eq!(shape(&["polylogue", "--", "mcp"]), "Line");
    let literal = scanned(&["polylogue", "--", "--archive", "read"]);
    assert_eq!(literal.words, ["--", "--archive", "read"]);
    assert_eq!(literal.overrides.archive, None);
}

#[test]
fn an_unusable_deployment_option_is_a_problem() {
    assert_eq!(
        scanned(&["polylogue", "rows", "--archive"])
            .problem
            .as_deref(),
        Some("--archive needs a value")
    );
    assert!(
        scanned(&["polylogue", "--listen", "nowhere", "rows"])
            .problem
            .is_some_and(|problem| problem.starts_with("--listen: "))
    );
}

#[test]
fn completion_keeps_valid_deployment_flags_beside_an_invalid_one() {
    let environment = EnvSnapshot::from_values([
        ("COMPLETE".into(), "zsh".into()),
        ("_CLAP_COMPLETE_INDEX".into(), "2".into()),
    ]);
    let invocation = scan(
        &arguments(&[
            "polylogue",
            "--listen",
            "not-an-address",
            "--archive",
            "/synthetic/archive",
            "--",
            "polylogue",
            "rows",
            "--config",
        ]),
        &environment,
    );
    assert_eq!(
        invocation.overrides.archive,
        Some("/synthetic/archive".into())
    );
    assert_eq!(invocation.overrides.listen, None);
    assert_eq!(invocation.overrides.config_file, None);
    let Shape::Completion(completion) = invocation.shape else {
        panic!("a completion callback")
    };
    assert_eq!(completion.index, 2);
    assert_eq!(
        completion.arguments,
        arguments(&["polylogue", "rows", "--config"])
    );
}
