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

#[test]
fn every_test_selection_builds_the_whole_workspace_with_one_feature_set() {
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
            &build[..],
            &[
                "-E",
                "((package(=polylogue-store)) | (package(=polylogue-engine)))",
                "rows"
            ]
        ]
        .concat()
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
        selected[6..],
        [
            "-E",
            "(rdeps(=polylogue-model)) & ((package(=polylogue-store))) & ((test(a)) | (test(b)))",
            "--no-capture"
        ]
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

#[test]
fn sinex_checks_follow_client_and_feature_wiring_changes() {
    for path in [
        "crates/sinex/src/lib.rs",
        "crates/store/src/primary.rs",
        "crates/polylogue/Cargo.toml",
        "Cargo.toml",
        "Cargo.lock",
    ] {
        assert!(sinex_input(path), "{path}");
    }
    for path in [
        "crates/query/src/parse.rs",
        "crates/polylogue/src/cli.rs",
        "crates/store/src/read.rs",
    ] {
        assert!(!sinex_input(path), "{path}");
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
fn changed_files_select_only_the_needed_modes() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers=[\"crates/*\"]\nresolver=\"3\"\n",
    )
    .unwrap();
    for (dir, name) in [
        ("polylogue", "polylogue"),
        ("sinex", "polylogue-sinex"),
        ("store", "polylogue-store"),
        ("query", "polylogue-query"),
    ] {
        let path = root.join("crates").join(dir);
        fs::create_dir_all(path.join("src")).unwrap();
        fs::write(path.join("src/lib.rs"), "").unwrap();
        fs::write(
            path.join("Cargo.toml"),
            format!("[package]\nname=\"{name}\"\nversion=\"0.1.0\"\nedition=\"2024\"\n"),
        )
        .unwrap();
    }
    fs::write(root.join("crates/store/src/primary.rs"), "").unwrap();
    fs::create_dir_all(root.join("crates/polylogue/src/daemon/service/tests")).unwrap();
    fs::write(
        root.join("crates/polylogue/src/daemon/service/tests/mode.rs"),
        "",
    )
    .unwrap();
    git(root, &["init", "--initial-branch=main"]);
    git(root, &["add", "--", "Cargo.toml", "crates"]);
    git(root, &["commit", "-m", "base"]);
    git(root, &["update-ref", "refs/remotes/origin/master", "HEAD"]);
    // Cargo metadata may create a lockfile; it is a workspace build input.
    fs::write(root.join(".gitignore"), "Cargo.lock\n").unwrap();
    git(root, &["add", "--", ".gitignore"]);
    git(root, &["commit", "-m", "ignore generated lockfile"]);
    git(root, &["update-ref", "refs/remotes/origin/master", "HEAD"]);
    let changes = Changes::read(root).unwrap();
    assert!(!changes.application && !changes.sinex);
    for (path, application, sinex) in [
        ("crates/query/src/lib.rs", false, false),
        ("crates/store/src/lib.rs", false, false),
        ("crates/store/src/primary.rs", false, true),
        ("crates/sinex/src/lib.rs", false, true),
        ("crates/polylogue/src/lib.rs", true, false),
        ("crates/polylogue/Cargo.toml", true, true),
        (
            "crates/polylogue/src/daemon/service/tests/mode.rs",
            true,
            true,
        ),
    ] {
        let path = root.join(path);
        let before = fs::read_to_string(&path).unwrap();
        fs::write(
            &path,
            format!("{before}\n# change\n").replace(
                "# change",
                if path.extension().unwrap() == "rs" {
                    "// change"
                } else {
                    "# change"
                },
            ),
        )
        .unwrap();
        let changes = Changes::read(root).unwrap();
        assert_eq!(
            (changes.application, changes.sinex),
            (application, sinex),
            "{}",
            path.display()
        );
        fs::write(path, before).unwrap();
    }
    fs::write(root.join("extension.js"), "").unwrap();
    let changes = Changes::read(root).unwrap();
    assert!(!changes.application && !changes.sinex);
    fs::write(root.join("crates/sinex/src/new.rs"), "").unwrap();
    let changes = Changes::read(root).unwrap();
    assert!(!changes.application && changes.sinex);
}
