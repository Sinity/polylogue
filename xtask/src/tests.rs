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

/// The build every test command shares, followed by its filter and the rest.
fn shared_build(rest: &[&str]) -> Vec<String> {
    [
        "--locked",
        "--workspace",
        "--tests",
        "--features",
        TEST_FEATURES,
    ]
    .iter()
    .chain(rest)
    .map(|arg| (*arg).to_owned())
    .collect()
}

#[test]
fn package_selection_filters_the_shared_build() {
    assert_eq!(arguments(&[], None).unwrap(), shared_build(&[]));
    assert!(TEST_FEATURES.contains("polylogue/sinex"));
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
        shared_build(&[
            "-E",
            "((package(polylogue-store)) | (package(polylogue-engine)))",
            "rows",
        ])
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
        shared_build(&[
            "-E",
            "(rdeps(=polylogue-model)) & ((package(polylogue-store))) & ((test(a)) | (test(b)))",
            "--no-capture",
        ])
    );
}

#[test]
fn native_scopes_and_literal_test_arguments_keep_their_meaning() {
    for flag in [
        "-ppolylogue-store",
        "-p=polylogue-store",
        "--package=polylogue-store",
    ] {
        assert_eq!(
            arguments(&[flag], None).unwrap(),
            shared_build(&["-E", "((package(polylogue-store)))"])
        );
    }
    assert_eq!(
        arguments(&["-p", "polylogue*"], None).unwrap(),
        shared_build(&["-E", "((package(polylogue*)))"])
    );
    assert_eq!(
        arguments(
            &[
                "--workspace",
                "-p",
                "polylogue-store",
                "--exclude",
                "polylogue-engine",
                "--exclude=polylogue-sinex"
            ],
            None
        )
        .unwrap(),
        shared_build(&[
            "-E",
            "((package(polylogue-store))) & (not package(polylogue-engine)) & (not package(polylogue-sinex))",
        ])
    );
    assert_eq!(
        arguments(&["-E=test(rows)", "--", "-p", "--features"], None).unwrap(),
        shared_build(&["-E", "((test(rows)))", "--", "-p", "--features"])
    );
}

/// Every file in the test build's output directory with its modification time.
fn build_outputs(build: &Path) -> Vec<(std::ffi::OsString, std::time::SystemTime)> {
    let mut outputs: Vec<_> = fs::read_dir(build.join("debug/deps"))
        .unwrap()
        .map(|entry| {
            let entry = entry.unwrap();
            (
                entry.file_name(),
                entry.metadata().unwrap().modified().unwrap(),
            )
        })
        .collect();
    outputs.sort();
    outputs
}

#[test]
fn a_package_selection_runs_its_tests_from_the_full_build() {
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
            "#[test] fn primary_mode_built() { assert!(cfg!(feature=\"sinex\")); }\n",
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
    let build = root.join("build");
    let execute = |args: &[&str], required: Option<&str>| {
        let mut run = Run::unrecorded("selection-fixture", "test", &[]);
        run.root = root.into();
        let mut options = nextest(arguments(args, required).unwrap());
        options
            .env
            .push(("CARGO_TARGET_DIR".into(), build.display().to_string()));
        let result = run.nextest("test", &options);
        (run, result)
    };
    // What ran, leaving out tests the filter did not select.
    let outcomes = |run: &Run| {
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
        let mut pairs: Vec<_> = pairs
            .iter()
            .map(|(name, status)| ((*name).to_owned(), (*status).to_owned()))
            .collect();
        pairs.sort();
        pairs
    };
    let selected = [
        ("selected_ignored", "ignored"),
        ("selected_read", "passed"),
        ("selected_write", "passed"),
    ];
    let (run, result) = execute(&[], None);
    result.unwrap();
    let mut everything = selected.to_vec();
    everything.extend([
        ("engine_test", "passed"),
        ("primary_mode_selected", "passed"),
        ("primary_mode_built", "passed"),
        ("helper_test", "passed"),
    ]);
    assert_eq!(outcomes(&run), expected(&everything));
    let full_build = build_outputs(&build);

    // This calls the same argument conversion and shared runner as `xtask test -p`.
    let (run, result) = execute(&["-p", "polylogue-store"], None);
    result.unwrap();
    assert_eq!(outcomes(&run), expected(&selected));
    assert_eq!(
        build_outputs(&build),
        full_build,
        "a package selection rebuilt what the full run built"
    );
    let (run, result) = execute(&["--exclude", "polylogue-engine"], None);
    result.unwrap();
    let mut included = everything.clone();
    included.retain(|(name, _)| *name != "engine_test");
    assert_eq!(outcomes(&run), expected(&included));
    let (run, result) = execute(&["--package=polylogue-sinex"], None);
    result.unwrap();
    assert_eq!(
        outcomes(&run),
        expected(&[("primary_mode_selected", "passed")])
    );
    assert_eq!(build_outputs(&build), full_build);

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
        expected(&[("selected_ignored", "ignored"), ("selected_read", "passed")])
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
fn changed_dependencies_reach_the_application() {
    let graph = serde_json::json!({"packages": [
        {"name": "polylogue", "dependencies": [{"name": "polylogue-sinex"}]},
        {"name": "polylogue-sinex", "dependencies": [{"name": "polylogue-store"}]},
        {"name": "polylogue-store", "dependencies": [{"name": "polylogue-model"}]},
        {"name": "polylogue-model", "dependencies": []},
        {"name": "polylogue-render", "dependencies": []}
    ]});
    for name in ["polylogue-sinex", "polylogue-store", "polylogue-model"] {
        let selection = Selection::Packages(vec![name.into()]);
        assert!(affected(&selection, &graph, "polylogue"));
    }
    assert!(!affected(
        &Selection::Packages(vec!["polylogue-render".into()]),
        &graph,
        "polylogue"
    ));
    assert!(!affected(&Selection::Nothing, &graph, "polylogue"));
    assert!(affected(&Selection::Workspace, &graph, "polylogue"));
    let tests = Selection::Tests {
        packages: vec![],
        binaries: vec![("polylogue-sinex".into(), "rebuild".into())],
    };
    assert!(affected(&tests, &graph, "polylogue-sinex"));
    assert!(!affected(&tests, &graph, "polylogue"));
}

#[test]
fn a_changed_package_selects_its_dependents_within_the_shared_build() {
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
    let filter = selection.filterset().unwrap();
    assert!(filter.contains("rdeps(=polylogue-sinex)"), "{filter}");
    assert_eq!(
        test_arguments(&[], Some(filter.clone())).unwrap(),
        shared_build(&["-E", &format!("({filter})")])
    );
}
