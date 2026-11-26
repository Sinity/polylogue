use super::*;
use std::path::Path;
use std::{fs, io::Write};

fn parse(args: &[&str]) -> Result<(Base, Selection), String> {
    let args = ["xtask", "test"]
        .iter()
        .chain(args)
        .map(|arg| (*arg).to_owned());
    match crate::parse(args.collect()) {
        Ok(Cli::Test {
            base, selection, ..
        }) => Ok((base, *selection)),
        Ok(_) => unreachable!("the test command parses as itself"),
        Err(error) => Err(error.to_string()),
    }
}

/// The packages and direct workspace dependencies of the real workspace.
fn workspace() -> selection::Workspace {
    selection::Workspace::of(&[
        ("polylogue-model", &[]),
        ("polylogue-query", &["polylogue-model"]),
        ("polylogue-sources", &["polylogue-model"]),
        ("polylogue-render", &["polylogue-model"]),
        ("polylogue-store", &["polylogue-model", "polylogue-query"]),
        (
            "polylogue-engine",
            &[
                "polylogue-model",
                "polylogue-sources",
                "polylogue-store",
                "polylogue-testkit",
            ],
        ),
        (
            "polylogue-testkit",
            &[
                "polylogue-engine",
                "polylogue-model",
                "polylogue-sources",
                "polylogue-store",
            ],
        ),
        (
            "polylogue-sinex",
            &["polylogue-store", "polylogue-model", "polylogue-engine"],
        ),
        (
            "polylogue",
            &[
                "polylogue-sinex",
                "polylogue-model",
                "polylogue-sources",
                "polylogue-store",
                "polylogue-engine",
                "polylogue-query",
                "polylogue-render",
            ],
        ),
        ("xtask", &[]),
    ])
}

fn arguments(args: &[&str]) -> Result<Option<Vec<String>>, String> {
    let (base, selection) = parse(args)?;
    selection.nextest_arguments(&workspace(), base.reference(), None)
}

fn owned(args: &[&str]) -> Vec<String> {
    args.iter().map(|arg| (*arg).to_owned()).collect()
}

const STORE_FEATURES: &str = "polylogue-model/fixture,polylogue-query/fixture,polylogue-store/faults,polylogue-store/fixture";

#[test]
fn workspace_hack_verification_is_skipped_while_linked() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    assert_eq!(workspace_hack_skip_reason(root), None);

    fs::create_dir(root.join(".cargo")).unwrap();
    fs::write(
        root.join(".cargo/local.toml"),
        "[patch.'https://github.com/Sinity/polylogue-sinex-common']\ndevkit = { path = '../common/crates/devkit' }\n",
    )
    .unwrap();
    assert_eq!(
        workspace_hack_skip_reason(root),
        Some("Skipped workspace-hack verification because the shared crates are linked.")
    );
}

#[test]
fn a_package_selection_builds_that_package_with_the_test_features_it_reaches() {
    assert_eq!(
        arguments(&[]).unwrap().unwrap(),
        owned(&["--locked", "--workspace", "--features", FEATURES, "--tests"])
    );
    for args in [
        &["-p", "polylogue-store"][..],
        &["-ppolylogue-store"],
        &["-p=polylogue-store"],
        &["--package=polylogue-store"],
        &["--package", "polylogue-store", "--workspace"],
    ] {
        assert_eq!(
            arguments(args).unwrap().unwrap(),
            owned(&[
                "--locked",
                "-p",
                "polylogue-store",
                "--features",
                STORE_FEATURES,
                "--tests"
            ]),
            "{args:?}"
        );
    }
    // Direct dependencies' features only: Cargo refuses the others, and the
    // store's own `fixture` forwards the query catalog's.
    assert_eq!(
        arguments(&["-p", "polylogue-render", "--lib"])
            .unwrap()
            .unwrap(),
        owned(&[
            "--locked",
            "-p",
            "polylogue-render",
            "--features",
            "polylogue-model/fixture",
            "--lib"
        ])
    );
    assert_eq!(
        arguments(&["-p", "polylogue-s*", "--exclude", "polylogue-sinex", "rows"])
            .unwrap()
            .unwrap(),
        owned(&[
            "--locked",
            "-p",
            "polylogue-sources",
            "-p",
            "polylogue-store",
            "--features",
            "polylogue-model/fixture,polylogue-query/fixture,polylogue-sources/fixture,polylogue-store/faults,polylogue-store/fixture",
            "--tests",
            "rows"
        ])
    );
    let excluded = arguments(&["--exclude", "polylogue-engine"])
        .unwrap()
        .unwrap();
    assert!(!excluded.contains(&"polylogue-engine".to_owned()));
    assert!(excluded.contains(&"polylogue-testkit".to_owned()));
    assert!(
        arguments(&["-p", "polylogue-nothing"])
            .unwrap_err()
            .contains("no workspace package matches")
    );
}

#[test]
fn a_changed_selection_intersects_the_named_packages() {
    let (base, selection) = parse(&["-p", "polylogue-*", "--base", "HEAD^1"]).unwrap();
    let changed: std::collections::BTreeSet<String> =
        ["polylogue-store", "xtask"].map(String::from).into();
    assert_eq!(
        selection
            .nextest_arguments(&workspace(), base.reference(), Some(&changed))
            .unwrap()
            .unwrap(),
        owned(&[
            "--locked",
            "-p",
            "polylogue-store",
            "--features",
            STORE_FEATURES,
            "--tests",
            "--base=HEAD^1"
        ])
    );
    let nothing = std::collections::BTreeSet::new();
    assert_eq!(
        selection
            .nextest_arguments(&workspace(), base.reference(), Some(&nothing))
            .unwrap(),
        None
    );
}

#[test]
fn filtersets_partition_and_pass_through_arguments_reach_nextest() {
    assert_eq!(
        arguments(&[
            "-p",
            "polylogue-store",
            "-E",
            "test(a)",
            "-Etest(b)",
            "--changed",
            "--partition",
            "hash:2/4",
            "--no-capture",
            "-E",
            "test(c)",
        ])
        .unwrap()
        .unwrap(),
        owned(&[
            "--locked",
            "-p",
            "polylogue-store",
            "--features",
            STORE_FEATURES,
            "-E",
            "(test(a)) | (test(b))",
            "--tests",
            "--base=origin/master",
            "--partition",
            "hash:2/4",
            "--no-capture",
            "-E",
            "test(c)",
        ])
    );
    assert_eq!(
        arguments(&["--filterset=test(rows)", "--", "-p", "--features"])
            .unwrap()
            .unwrap(),
        owned(&[
            "--locked",
            "--workspace",
            "--features",
            FEATURES,
            "-E",
            "(test(rows))",
            "--tests",
            "--",
            "-p",
            "--features"
        ])
    );
    assert!(parse(&["--changed", "--base", "HEAD"]).is_err());
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
        vec!["--examples"],
        vec!["--bench", "index"],
        vec!["--benches"],
        vec!["--all-targets"],
    ] {
        let selected = arguments(&flags).unwrap().unwrap();
        assert!(!selected.iter().any(|arg| arg == "--tests"), "{flags:?}");
        let expected: Vec<String> = flags
            .iter()
            .flat_map(|flag| {
                flag.split_once('=')
                    .map_or(vec![*flag], |(a, b)| vec![a, b])
            })
            .map(str::to_owned)
            .collect();
        assert!(selected.ends_with(&expected), "{selected:?}");
    }
    let selected = arguments(&["--test-threads", "2", "--", "--lib"])
        .unwrap()
        .unwrap();
    assert!(selected.iter().any(|arg| arg == "--tests"));
    assert!(selected.ends_with(&owned(&["--test-threads", "2", "--", "--lib"])));
}

#[test]
fn selections_build_their_packages_with_their_test_features() {
    let fixture = tempfile::tempdir().unwrap();
    let root = fixture.path();
    fs::write(root.join("Cargo.toml"), "[workspace]\nmembers=['polylogue-model','polylogue-query','polylogue-sources','polylogue-store','polylogue-engine','polylogue-sinex','polylogue','helper']\nresolver='3'\n").unwrap();
    for (name, manifest, source) in [
        (
            "polylogue-model",
            "[features]\nfixture=[]\n",
            "pub fn assert_fixture() { assert!(cfg!(feature=\"fixture\")); }\n",
        ),
        (
            "polylogue-query",
            "[features]\nfixture=[]\n",
            "pub fn assert_fixture() { assert!(cfg!(feature=\"fixture\")); }\n",
        ),
        (
            "polylogue-sources",
            "[features]\nfixture=[]\n",
            "#[test] fn sources_fixture() { assert!(cfg!(feature=\"fixture\")); }\n",
        ),
        (
            "polylogue-store",
            "[features]\nfaults=[]\nfixture=[]\n[dependencies]\nhelper={path='../helper'}\npolylogue-model={path='../polylogue-model'}\npolylogue-query={path='../polylogue-query'}\n",
            "#[test] fn selected_read() { assert!(cfg!(feature=\"faults\")); assert_eq!(helper::value(),42); }\n#[test] fn selected_write() { assert!(cfg!(feature=\"fixture\")); polylogue_model::assert_fixture(); polylogue_query::assert_fixture(); }\n#[test] #[ignore] fn selected_ignored() {}\n",
        ),
        (
            "polylogue-engine",
            "[features]\nfaults=[]\n[dev-dependencies]\nhelper={path='../helper'}\n",
            "#[test] fn engine_test() { assert!(cfg!(feature=\"faults\")); }\n",
        ),
        (
            "polylogue-sinex",
            "[dependencies]\npolylogue-store={path='../polylogue-store'}\npolylogue-engine={path='../polylogue-engine'}\n",
            "#[test] fn sinex_kinds() {}\n",
        ),
        (
            "polylogue",
            "[features]\nsinex=[]\n[dependencies]\npolylogue-store={path='../polylogue-store'}\npolylogue-engine={path='../polylogue-engine'}\n",
            "#[test] fn application_runs_in_primary_mode() { assert!(cfg!(feature=\"sinex\")); }\n",
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
    let lockfile = devkit::cargo_command()
        .args(["generate-lockfile", "--offline"])
        .current_dir(root)
        .output()
        .unwrap();
    assert!(
        lockfile.status.success(),
        "{}",
        String::from_utf8_lossy(&lockfile.stderr)
    );
    let execute = |args: &[&str]| {
        let mut run = Run::unrecorded("selection-fixture", "test", &[]);
        run.root = root.into();
        let (base, selection) = parse(args).unwrap();
        let mut options = test_options(root, base.reference(), &selection)
            .unwrap()
            .expect("the selection holds packages");
        options.env.push((
            "CARGO_TARGET_DIR".into(),
            root.join("build").display().to_string(),
        ));
        let result = run.nextest("test", &options);
        (run, result)
    };
    let unfiltered = |run: &Run| {
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
    let (run, result) = execute(&[]);
    result.unwrap();
    let mut all = expected(&store);
    all.extend(expected(&[
        ("application_runs_in_primary_mode", "passed"),
        ("engine_test", "passed"),
        ("helper_test", "passed"),
        ("sinex_kinds", "passed"),
        ("sources_fixture", "passed"),
    ]));
    all.sort();
    assert_eq!(unfiltered(&run), all);

    let (run, result) = execute(&["-p", "polylogue-store"]);
    result.unwrap();
    assert_eq!(unfiltered(&run), expected(&store));

    let (run, result) = execute(&["-p", "polylogue"]);
    result.unwrap();
    assert_eq!(
        unfiltered(&run),
        expected(&[("application_runs_in_primary_mode", "passed")])
    );

    let (run, result) = execute(&["-p", "polylogue-store", "-p", "polylogue-engine"]);
    result.unwrap();
    let mut selected = expected(&store);
    selected.extend(expected(&[("engine_test", "passed")]));
    selected.sort();
    assert_eq!(unfiltered(&run), selected);

    let (run, result) = execute(&["-p", "polylogue-store", "--example", "unselected"]);
    assert!(result.is_err());
    assert!(run.diagnostics.iter().any(|diagnostic| {
        diagnostic
            .text
            .contains("a non-test example must not be compiled by default")
    }));

    let (run, result) = execute(&["--exclude", "polylogue-engine"]);
    result.unwrap();
    assert!(
        !unfiltered(&run)
            .iter()
            .any(|(name, _)| name == "engine_test")
    );
    assert_eq!(unfiltered(&run).len(), all.len() - 1);

    // Partitions divide the selection; each part runs, even an empty one.
    let passed = |run: &Run| {
        run.tests
            .iter()
            .filter(|test| test.status == devkit::TestStatus::Passed)
            .map(|test| test.name.clone())
            .collect::<Vec<_>>()
    };
    let mut parts = Vec::new();
    for part in ["hash:1/2", "hash:2/2", "count:20/20"] {
        let (run, result) = execute(&["--partition", part]);
        result.unwrap();
        parts.push(passed(&run));
    }
    let mut divided: Vec<_> = parts[..2].concat();
    divided.sort();
    let mut every: Vec<_> = all
        .iter()
        .filter(|(_, status)| status == "passed")
        .map(|(name, _)| name.clone())
        .collect();
    every.sort();
    assert_eq!(divided, every);
    assert!(parts[2].is_empty(), "{:?}", parts[2]);

    fs::write(root.join(".gitignore"), "/build/\n/target/\n").unwrap();
    git(root, &["init", "-q"]);
    git(root, &["add", "."]);
    git(root, &["commit", "-qm", "baseline"]);
    fs::write(
        root.join("helper/src/lib.rs"),
        "pub fn value() -> u32 { 40 + 2 }\n#[test] fn helper_test() {}\n",
    )
    .unwrap();
    // devkit's selection reaches the store through its dependency on the
    // changed helper, intersected with the explicit package and filterset.
    let (run, result) = execute(&[
        "-ppolylogue-store",
        "-E",
        "test(selected_read)",
        "--base",
        "HEAD",
    ]);
    result.unwrap();
    assert_eq!(
        unfiltered(&run),
        expected(&[("selected_ignored", "ignored"), ("selected_read", "passed")])
    );
    // The packages the change selects: the helper and everything that
    // depends on it, through normal and test-only dependencies.
    assert_eq!(
        selection::changed_packages(root, "HEAD")
            .unwrap()
            .unwrap()
            .into_iter()
            .collect::<Vec<_>>(),
        [
            "helper",
            "polylogue",
            "polylogue-engine",
            "polylogue-sinex",
            "polylogue-store"
        ]
    );

    // A selection builds its own packages' targets only: a broken
    // integration target of another package does not stop it.
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
    let (run, result) = execute(&["-p", "polylogue", "--bin", "polylogue"]);
    result.unwrap();
    assert_eq!(
        unfiltered(&run),
        expected(&[("named_application_target", "passed")])
    );
    let (run, result) = execute(&["-p", "polylogue-store", "--lib"]);
    result.unwrap();
    assert_eq!(
        unfiltered(&run),
        expected(&[
            ("selected_ignored", "ignored"),
            ("selected_read", "passed"),
            ("selected_write", "passed"),
        ])
    );
    let (run, result) = execute(&["-p", "polylogue-store"]);
    result.unwrap();
    assert_eq!(unfiltered(&run), expected(&store));
    let (run, result) = execute(&["-p", "polylogue"]);
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
        assert!(arguments(flag).unwrap_err().contains(FEATURES));
    }
    for flag in ["-p", "--exclude", "-E", "--base", "--partition"] {
        assert!(parse(&[flag]).is_err(), "{flag}");
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

#[test]
fn nix_checks_follow_the_files_changed_since_the_base() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path();
    fs::create_dir_all(root.join("nix/lib")).unwrap();
    fs::write(root.join("nix/lib/settings.nix"), "{ }\n").unwrap();
    fs::write(root.join("README.md"), "a\n").unwrap();
    git(root, &["init", "-q", "-b", "main"]);
    git(root, &["add", "."]);
    git(root, &["commit", "-qm", "baseline"]);
    assert_eq!(
        nix::changed_files(root, "HEAD").unwrap(),
        Vec::<String>::new()
    );

    fs::write(root.join("nix/lib/settings.nix"), "{ a = 1; }\n").unwrap();
    fs::write(root.join("flake.nix"), "{ }\n").unwrap();
    let mut files = nix::changed_files(root, "HEAD").unwrap();
    files.sort();
    assert_eq!(files, ["flake.nix", "nix/lib/settings.nix"]);
    assert!(files.iter().all(|file| nix::is_nix_input(file)));

    fs::write(root.join("README.md"), "b\n").unwrap();
    let files = nix::changed_files(root, "HEAD").unwrap();
    assert!(files.iter().any(|file| !nix::is_nix_input(file)));
}
