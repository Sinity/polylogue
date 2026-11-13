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
    test_arguments(
        &args.iter().map(|arg| (*arg).to_owned()).collect::<Vec<_>>(),
        required.map(str::to_owned),
    )
}

const BUILD: [&str; 4] = ["--locked", "--workspace", "--features", FEATURES];

fn built_with(tail: &[&str]) -> Vec<String> {
    BUILD
        .iter()
        .chain(tail)
        .map(|arg| (*arg).to_owned())
        .collect()
}

#[test]
fn package_filters_keep_the_workspace_feature_graph() {
    assert_eq!(arguments(&[], None).unwrap(), built_with(&["--tests"]));
    for flag in [
        &["-p", "polylogue-store"][..],
        &["-ppolylogue-store"],
        &["-p=polylogue-store"],
        &["--package=polylogue-store"],
        &["--package", "polylogue-store"],
    ] {
        assert_eq!(
            arguments(flag, None).unwrap(),
            built_with(&["-E", "((package(=polylogue-store)))", "--tests"]),
            "{flag:?}"
        );
    }
    assert_eq!(
        arguments(&["-p", "polylogue*", "--workspace"], None).unwrap(),
        built_with(&["-E", "((package(polylogue*)))", "--tests"])
    );
    assert_eq!(
        arguments(&["--exclude", "polylogue-engine", "rows"], None).unwrap(),
        built_with(&[
            "-E",
            "(not ((package(=polylogue-engine))))",
            "--tests",
            "rows"
        ])
    );
}

#[test]
fn packages_exclusions_filtersets_and_a_required_selection_intersect() {
    assert_eq!(
        arguments(
            &[
                "-p",
                "polylogue-store",
                "-p",
                "polylogue-engine",
                "--exclude=polylogue-sinex",
                "-E",
                "test(a)",
                "-Etest(b)",
                "--no-capture",
            ],
            Some("rdeps(=polylogue-model)"),
        )
        .unwrap(),
        built_with(&[
            "-E",
            "(rdeps(=polylogue-model)) & ((package(=polylogue-store)) | (package(=polylogue-engine))) \
             & (not ((package(=polylogue-sinex)))) & ((test(a)) | (test(b)))",
            "--tests",
            "--no-capture",
        ])
    );
    assert_eq!(
        arguments(&["-E=test(rows)", "--", "-p", "--features"], None).unwrap(),
        built_with(&["-E", "((test(rows)))", "--tests", "--", "-p", "--features"])
    );
}

#[test]
fn explicit_target_selection_is_preserved_without_adding_the_default_tests() {
    for flags in [
        vec!["--lib"],
        vec!["--bins"],
        vec!["--bin", "polylogue"],
        vec!["--bin=polylogue"],
        vec!["--test", "e2e"],
        vec!["--test=e2e"],
        vec!["--example", "first_build"],
        vec!["--example=first_build"],
        vec!["--examples"],
        vec!["--bench", "index"],
        vec!["--bench=index"],
        vec!["--benches"],
        vec!["--all-targets"],
        vec!["--doc"],
    ] {
        let selected = arguments(&flags, None).unwrap();
        assert!(!selected.iter().any(|arg| arg == "--tests"), "{flags:?}");
        assert!(
            selected.ends_with(
                &flags
                    .iter()
                    .map(|arg| (*arg).to_owned())
                    .collect::<Vec<_>>()
            )
        );
    }
    assert_eq!(
        arguments(&["--tests"], None)
            .unwrap()
            .iter()
            .filter(|arg| *arg == "--tests")
            .count(),
        1
    );
    let selected = arguments(&["--test-threads", "2", "--", "--lib"], None).unwrap();
    assert!(selected.iter().any(|arg| arg == "--tests"));
    assert!(selected.ends_with(&[
        "--test-threads".into(),
        "2".into(),
        "--".into(),
        "--lib".into()
    ]));
}

#[test]
fn selections_share_features_and_native_targets_through_the_real_runner() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    fs::write(root.join("Cargo.toml"), "[workspace]\nmembers=['polylogue-store','polylogue-engine','polylogue-sinex','polylogue','helper']\nresolver='3'\n").unwrap();
    for (name, manifest, source) in [
        (
            "polylogue-store",
            "[features]\nfaults=[]\nfixture=[]\n[dependencies]\nhelper={path='../helper'}\n",
            "#[test] fn selected_read() { assert!(cfg!(feature=\"faults\")); assert_eq!(helper::value(),42); }\n#[test] fn selected_write() { assert!(cfg!(feature=\"fixture\")); }\n#[test] #[ignore] fn selected_ignored() {}\n",
        ),
        (
            "polylogue-engine",
            "[features]\nfaults=[]\n[dev-dependencies]\nhelper={path='../helper',features=['shared-test']}\n",
            "#[test] fn engine_test() { assert!(cfg!(feature=\"faults\")); }\n",
        ),
        (
            "polylogue-sinex",
            "[dependencies]\npolylogue-store={path='../polylogue-store'}\npolylogue-engine={path='../polylogue-engine'}\n",
            "#[test] fn primary_mode_selected() {}\n",
        ),
        (
            "polylogue",
            "[features]\nsinex=[]\n[dependencies]\npolylogue-store={path='../polylogue-store'}\npolylogue-engine={path='../polylogue-engine'}\n",
            "#[test] fn application_runs_in_primary_mode() { assert!(cfg!(feature=\"sinex\")); }\n",
        ),
        (
            "helper",
            "[features]\nshared-test=[]\n",
            "pub fn value() -> u32 { if cfg!(feature=\"shared-test\") { 42 } else { 0 } }\n#[test] fn helper_test() {}\n",
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
    let examples = root.join("polylogue-store/examples");
    fs::create_dir(&examples).unwrap();
    fs::write(
        examples.join("unselected.rs"),
        "compile_error!(\"a non-test example must not be compiled by default\");\nfn main() {}\n",
    )
    .unwrap();
    fs::write(
        examples.join("checked.rs"),
        "fn main() {}\n#[test] fn opted_in_example() { assert_eq!(helper::value(),42); }\n",
    )
    .unwrap();
    fs::OpenOptions::new()
        .append(true)
        .open(root.join("polylogue-store/Cargo.toml"))
        .unwrap()
        .write_all(b"\n[[example]]\nname='checked'\ntest=true\n")
        .unwrap();
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
        let mut options = nextest(arguments(args, required).unwrap());
        options.env.push((
            "CARGO_TARGET_DIR".into(),
            root.join("build").display().to_string(),
        ));
        let result = run.nextest("test", &options);
        (run, result)
    };
    let ran = |run: &Run| {
        let mut results: Vec<_> = run
            .tests
            .iter()
            .filter(|test| test.status.as_str() != "filtered")
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
    let store = [
        ("opted_in_example", "passed"),
        ("selected_ignored", "ignored"),
        ("selected_read", "passed"),
        ("selected_write", "passed"),
    ];
    // Every package's tests run with the fault, fixture and Sinex features.
    let (run, result) = execute(&[], None);
    result.unwrap();
    let mut all = expected(&store);
    all.extend(expected(&[
        ("application_runs_in_primary_mode", "passed"),
        ("engine_test", "passed"),
        ("helper_test", "passed"),
        ("primary_mode_selected", "passed"),
    ]));
    all.sort();
    assert_eq!(ran(&run), all);

    let (run, result) = execute(&["-p", "polylogue-store"], None);
    result.unwrap();
    assert_eq!(ran(&run), expected(&store));

    let (run, result) = execute(&["-p", "polylogue"], None);
    result.unwrap();
    assert_eq!(
        ran(&run),
        expected(&[("application_runs_in_primary_mode", "passed")])
    );

    let (run, result) = execute(&["-p", "polylogue-store", "-p", "polylogue-engine"], None);
    result.unwrap();
    let mut selected = expected(&store);
    selected.extend(expected(&[("engine_test", "passed")]));
    selected.sort();
    assert_eq!(ran(&run), selected);

    let (run, result) = execute(&["-p", "polylogue-store", "--example", "unselected"], None);
    assert!(result.is_err());
    assert!(run.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .text
            .contains("a non-test example must not be compiled by default")
    }));

    let (run, result) = execute(&["--exclude", "polylogue-engine"], None);
    result.unwrap();
    assert!(!ran(&run).iter().any(|(name, _)| name == "engine_test"));
    assert_eq!(ran(&run).len(), all.len() - 1);

    fs::write(root.join(".gitignore"), "/build/\n/target/\n").unwrap();
    git(root, &["init", "-q"]);
    git(root, &["add", "."]);
    git(root, &["commit", "-qm", "baseline"]);
    fs::write(
        root.join("helper/src/lib.rs"),
        "pub fn value() -> u32 { if cfg!(feature=\"shared-test\") { 40 + 2 } else { 0 } }\n#[test] fn helper_test() {}\n",
    )
    .unwrap();
    // devkit's selection reaches the store through its dependency on the
    // changed helper, intersected with the explicit package and filterset.
    let (run, result) = execute(
        &[
            "-ppolylogue-store",
            "-E",
            "test(selected_read)",
            "--changed-base=HEAD",
        ],
        None,
    );
    result.unwrap();
    assert_eq!(
        ran(&run),
        expected(&[("selected_ignored", "ignored"), ("selected_read", "passed")])
    );

    // Native library selection excludes integration binaries without dropping
    // another workspace member's test-only feature contribution.
    fs::create_dir(root.join("polylogue/tests")).unwrap();
    fs::write(
        root.join("polylogue/tests/unrelated.rs"),
        "compile_error!(\"unrelated application integration target\");\n",
    )
    .unwrap();
    fs::write(
        root.join("polylogue/src/main.rs"),
        "fn main() {}\n#[test] fn named_application_target() { assert!(cfg!(feature=\"sinex\")); }\n",
    )
    .unwrap();
    let (run, result) = execute(&["-p", "polylogue", "--bin", "polylogue"], None);
    result.unwrap();
    assert_eq!(
        ran(&run),
        expected(&[("named_application_target", "passed")])
    );
    let (run, result) = execute(&["-p", "polylogue-store", "--lib"], None);
    result.unwrap();
    assert_eq!(
        ran(&run),
        expected(&[
            ("selected_ignored", "ignored"),
            ("selected_read", "passed"),
            ("selected_write", "passed"),
        ])
    );
    let (run, result) = execute(&["-p", "polylogue-store"], None);
    assert!(result.is_err());
    assert!(run.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .text
            .contains("unrelated application integration target")
    }));
}

#[test]
fn feature_flags_are_refused_rather_than_building_a_second_feature_set() {
    for flag in [
        &["--features", "faults"][..],
        &["-Ffaults"],
        &["--all-features"],
        &["--no-default-features"],
    ] {
        assert!(arguments(flag, None).unwrap_err().contains(FEATURES));
    }
    for flag in ["-p", "--exclude", "-E"] {
        assert!(
            arguments(&[flag], None)
                .unwrap_err()
                .contains("needs a value")
        );
    }
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
