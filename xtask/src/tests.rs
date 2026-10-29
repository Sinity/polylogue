use super::*;
use std::{fs, io::Write};

#[test]
fn forbidden_internal_dependency_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    fs::write(
        root.path().join("Cargo.toml"),
        "[workspace]\nmembers=[\"query\",\"store\"]\nresolver=\"3\"\n",
    )
    .unwrap();
    for (dir, name) in [("query", "polylogue-query"), ("store", "polylogue-store")] {
        fs::create_dir_all(root.path().join(dir).join("src")).unwrap();
        fs::write(root.path().join(dir).join("src/lib.rs"), "").unwrap();
        fs::write(
            root.path().join(dir).join("Cargo.toml"),
            format!("[package]\nname=\"{name}\"\nversion=\"0.1.0\"\nedition=\"2024\"\n"),
        )
        .unwrap();
    }
    assert!(boundaries(&metadata(root.path()).unwrap()).is_ok());
    fs::OpenOptions::new()
        .append(true)
        .open(root.path().join("store/Cargo.toml"))
        .unwrap()
        .write_all(b"[dependencies]\npolylogue-query={path=\"../query\"}\n")
        .unwrap();
    assert!(boundaries(&metadata(root.path()).unwrap()).is_ok());
    // Remove the forward edge before asking Cargo to inspect the deliberate
    // reverse edge; Cargo rejects a cycle before the boundary checker runs.
    fs::write(
        root.path().join("store/Cargo.toml"),
        "[package]\nname=\"polylogue-store\"\nversion=\"0.1.0\"\nedition=\"2024\"\n",
    )
    .unwrap();

    fs::OpenOptions::new()
        .append(true)
        .open(root.path().join("query/Cargo.toml"))
        .unwrap()
        .write_all(b"[dependencies]\npolylogue-store={path=\"../store\"}\n")
        .unwrap();
    assert_eq!(
        boundaries(&metadata(root.path()).unwrap()).unwrap_err(),
        "forbidden dependency: polylogue-query -> polylogue-store"
    );
}

fn arguments(args: &[&str], required: Option<&str>) -> Result<Vec<String>, String> {
    arguments_at(Path::new("."), args, required)
}

fn arguments_at(root: &Path, args: &[&str], required: Option<&str>) -> Result<Vec<String>, String> {
    test_arguments(
        root,
        &args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(),
        required.map(str::to_owned),
        false,
    )
}

#[test]
fn package_selection_limits_the_build_with_the_required_fault_features() {
    let build = [
        "--locked",
        "--workspace",
        "--features",
        TEST_FEATURES,
        "--exclude",
        "polylogue-sinex",
    ];
    assert_eq!(arguments(&[], None).unwrap(), build);
    assert_eq!(
        arguments(
            &[
                "-p",
                "polylogue-store",
                "--package=polylogue-engine",
                "rows"
            ],
            None
        )
        .unwrap(),
        [
            "--locked",
            "-p",
            "polylogue-store",
            "-p",
            "polylogue-engine",
            "--features",
            TEST_FEATURES,
            "rows",
        ]
    );
}

#[test]
fn packages_filtersets_and_a_required_selection_intersect() {
    let selected = arguments(
        &[
            "-p",
            "polylogue-store",
            "-E",
            "test(a)",
            "-Etest(b)",
            "--no-capture",
        ],
        Some("rdeps(=polylogue-model)"),
    )
    .unwrap();
    assert_eq!(
        selected,
        [
            "--locked",
            "-p",
            "polylogue-store",
            "--features",
            "polylogue-store/faults",
            "-E",
            "(rdeps(=polylogue-model)) & ((test(a)) | (test(b)))",
            "--no-capture"
        ]
    );
}

#[test]
fn native_scopes_and_literal_test_arguments_keep_their_meaning() {
    assert!(
        arguments(&["-p", "polylogue*"], None)
            .unwrap()
            .iter()
            .any(|arg| arg.contains("polylogue/sinex"))
    );
    for flag in [
        "-ppolylogue-store",
        "-p=polylogue-store",
        "--package=polylogue-store",
    ] {
        assert_eq!(
            arguments(&[flag], None).unwrap(),
            [
                "--locked",
                "-p",
                "polylogue-store",
                "--features",
                "polylogue-store/faults"
            ]
        );
    }
    assert_eq!(
        arguments(
            &[
                "--workspace",
                "-p",
                "polylogue-store",
                "--exclude",
                "polylogue-engine"
            ],
            None
        )
        .unwrap(),
        [
            "--locked",
            "--workspace",
            "-p",
            "polylogue-store",
            "--features",
            TEST_FEATURES,
            "--exclude",
            "polylogue-sinex",
            "--exclude",
            "polylogue-engine"
        ]
    );
    assert_eq!(
        arguments(&["-E=test(rows)", "--", "-p", "--features"], None).unwrap(),
        [
            "--locked",
            "--workspace",
            "--features",
            TEST_FEATURES,
            "--exclude",
            "polylogue-sinex",
            "-E",
            "((test(rows)))",
            "--",
            "-p",
            "--features"
        ]
    );
}

#[test]
fn package_selection_reaches_the_real_runner_before_unrelated_compilation() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    fs::write(root.join("Cargo.toml"), "[workspace]\nmembers=['polylogue-store','polylogue-engine','polylogue-sinex','polylogue','helper']\nresolver='3'\n").unwrap();
    for (name, manifest, source) in [
        (
            "polylogue-store",
            "[features]\nfaults=[]\n[dependencies]\nhelper={path='../helper'}\n",
            "#[test] fn selected_read() { assert!(cfg!(feature=\"faults\")); assert_eq!(helper::value(),42); }\n#[test] fn selected_write() { assert!(cfg!(feature=\"faults\")); }\n#[test] #[ignore] fn selected_ignored() {}\n",
        ),
        (
            "polylogue-engine",
            "[features]\nfaults=[]\n",
            "#[cfg(test)] compile_error!(\"unrelated engine tests must not build\");\n",
        ),
        (
            "polylogue-sinex",
            "[dependencies]\npolylogue-store={path='../polylogue-store'}\npolylogue-engine={path='../polylogue-engine'}\n",
            "#[test] fn primary_mode_selected() {}\n",
        ),
        (
            "polylogue",
            "[features]\nsinex=[]\n[dependencies]\npolylogue-store={path='../polylogue-store'}\npolylogue-engine={path='../polylogue-engine'}\n",
            "",
        ),
        (
            "helper",
            "",
            "pub fn value() -> u32 { 42 }\n#[test] fn helper_test() {}\n",
        ),
    ] {
        fs::create_dir_all(root.join(name).join("src")).unwrap();
        fs::write(
            root.join(name).join("Cargo.toml"),
            format!("[package]\nname='{name}'\nversion='0.1.0'\nedition='2024'\n{manifest}"),
        )
        .unwrap();
        fs::write(root.join(name).join("src/lib.rs"), source).unwrap();
    }
    let lock = devkit::cargo_command()
        .args(["generate-lockfile", "--offline"])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        lock.status.success(),
        "{}",
        String::from_utf8_lossy(&lock.stderr)
    );
    let execute = |args: &[&str], required: Option<&str>| {
        let mut run = Run::unrecorded("selection-fixture", "test", &[]);
        run.root = root.into();
        let mut options = nextest(arguments_at(root, args, required).unwrap());
        options.env.push((
            "CARGO_TARGET_DIR".into(),
            root.join("build").display().to_string(),
        ));
        let result = run.nextest("test", &options);
        (run, result)
    };
    let outcomes = |run: &Run| {
        let mut results: Vec<_> = run
            .tests
            .iter()
            .map(|test| (test.name.clone(), test.status.as_str().to_owned()))
            .collect();
        results.sort();
        results
    };
    let expected = |pairs: &[(&str, &str)]| {
        pairs
            .iter()
            .map(|(name, status)| ((*name).to_owned(), (*status).to_owned()))
            .collect::<Vec<_>>()
    };
    let selected = expected(&[
        ("selected_ignored", "ignored"),
        ("selected_read", "passed"),
        ("selected_write", "passed"),
    ]);
    // Full checks still compile every standalone test, including the planted error.
    let (run, result) = execute(&[], None);
    assert!(result.is_err());
    assert!(run.diagnostics.iter().any(|diagnostic| {
        diagnostic.package == "polylogue-engine"
            && diagnostic
                .text
                .contains("unrelated engine tests must not build")
    }));
    assert!(run.tests.is_empty());
    // This calls the same argument conversion and shared runner as `xtask test -p`.
    // Before native package forwarding, the unrelated test compile error prevents it.
    let (run, result) = execute(&["-p", "polylogue-store"], None);
    result.unwrap();
    assert_eq!(outcomes(&run), selected);
    assert!(
        run.tests
            .iter()
            .all(|test| test.binary == "polylogue-store")
    );
    let (run, result) = execute(&["--exclude", "polylogue-engine"], None);
    result.unwrap();
    let mut included = expected(&[("helper_test", "passed")]);
    included.extend(selected);
    assert_eq!(outcomes(&run), included);
    let (run, result) = execute(&["--package=polylogue-sinex"], None);
    result.unwrap();
    assert_eq!(
        outcomes(&run),
        expected(&[("primary_mode_selected", "passed")])
    );

    fs::write(root.join(".gitignore"), "/build/\n/target/\n").unwrap();
    git(root, &["init", "-q"]);
    git(root, &["add", "."]);
    git(root, &["commit", "-qm", "baseline"]);
    fs::write(
        root.join("helper/src/lib.rs"),
        "pub fn value() -> u32 { 40 + 2 }\n#[test] fn helper_test() {}\n",
    )
    .unwrap();
    let selection = devkit::select::changed(root, "HEAD").unwrap();
    assert_eq!(selection, Selection::Packages(vec!["helper".into()]));
    let filter = selection.filterset().unwrap();
    let (run, result) = execute(
        &["-ppolylogue-store", "-E", "test(selected_read)"],
        Some(&filter),
    );
    result.unwrap();
    assert_eq!(
        outcomes(&run),
        expected(&[
            ("selected_ignored", "ignored"),
            ("selected_read", "passed"),
            ("selected_write", "filtered")
        ])
    );
}

#[test]
fn feature_flags_are_refused_rather_than_building_a_second_feature_set() {
    for flag in [
        &["--features", "faults"][..],
        &["-Ffaults"],
        &["--all-features"],
    ] {
        assert!(arguments(flag, None).unwrap_err().contains(TEST_FEATURES));
    }
    assert!(
        arguments(&["-p"], None)
            .unwrap_err()
            .contains("needs a value")
    );
}

fn git(root: &Path, args: &[&str]) {
    let output = std::process::Command::new("git")
        .current_dir(root)
        .args([
            "-c",
            "user.name=Example",
            "-c",
            "user.email=example@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn changed_client_dependencies_select_the_sinex_build() {
    let graph = serde_json::json!({"packages": [
        {"name": "polylogue", "dependencies": [{"name": "polylogue-sinex"}]},
        {"name": "polylogue-sinex", "dependencies": [{"name": "polylogue-store"}]},
        {"name": "polylogue-store", "dependencies": [{"name": "polylogue-model"}]},
        {"name": "polylogue-model", "dependencies": []},
        {"name": "polylogue-render", "dependencies": [{"name": "polylogue-model"}]}
    ]});
    for name in ["polylogue-sinex", "polylogue-store", "polylogue-model"] {
        let selection = Selection::Packages(vec![name.into()]);
        assert!(affected(&selection, &graph, "polylogue-sinex"));
        let args = test_arguments(
            Path::new("."),
            &[],
            selection.filterset(),
            affected(&selection, &graph, "polylogue-sinex"),
        )
        .unwrap();
        assert!(args.iter().any(|arg| arg.contains("polylogue/sinex")));
        assert!(!args.iter().any(|arg| arg == "--exclude"));
        assert!(
            args.iter()
                .any(|arg| arg.contains(&format!("rdeps(={name})")))
        );
    }
    assert!(!affected(
        &Selection::Packages(vec!["polylogue-render".into()]),
        &graph,
        "polylogue-sinex"
    ));
    assert!(!affected(&Selection::Nothing, &graph, "polylogue-sinex"));
    assert!(affected(&Selection::Workspace, &graph, "polylogue-sinex"));
    let tests = Selection::Tests {
        packages: vec![],
        binaries: vec![("polylogue-sinex".into(), "rebuild".into())],
    };
    assert!(affected(&tests, &graph, "polylogue-sinex"));
    assert!(!affected(&tests, &graph, "polylogue"));
}

#[test]
fn a_changed_sinex_package_builds_its_tests_in_primary_mode() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers=[\"client\"]\nresolver=\"3\"\n",
    )
    .unwrap();
    fs::create_dir_all(root.join("client/src")).unwrap();
    fs::write(
        root.join("client/Cargo.toml"),
        "[package]\nname=\"polylogue-sinex\"\nversion=\"0.1.0\"\nedition=\"2024\"\n",
    )
    .unwrap();
    fs::write(root.join("client/src/lib.rs"), "pub fn initial() {}\n").unwrap();
    fs::write(root.join(".gitignore"), "Cargo.lock\n").unwrap();
    git(root, &["init", "--initial-branch=main"]);
    git(root, &["add", "--", "Cargo.toml", "client", ".gitignore"]);
    git(root, &["commit", "-m", "base"]);
    git(root, &["update-ref", "refs/remotes/origin/master", "HEAD"]);
    fs::write(root.join("client/src/lib.rs"), "pub fn changed() {}\n").unwrap();
    let selection = devkit::select::changed(root, "origin/master").unwrap();
    let sinex = affected(&selection, &metadata(root).unwrap(), "polylogue-sinex");
    let args = test_arguments(root, &[], selection.filterset(), sinex).unwrap();
    assert!(args.iter().any(|arg| arg.contains("polylogue/sinex")));
    assert!(!args.iter().any(|arg| arg == "--exclude"));
    let scoped = arguments(&["-p", "polylogue-sinex"], None).unwrap();
    assert!(scoped.iter().any(|arg| arg == "polylogue-sinex"));
    assert!(scoped.iter().any(|arg| arg == TEST_FEATURES));
    assert!(!scoped.iter().any(|arg| arg.contains("polylogue/sinex")));
}
