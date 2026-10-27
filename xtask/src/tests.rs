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
    let build = ["--locked", "--workspace", "--features", TEST_FEATURES];
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
        selected[4..],
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
fn changed_selection_reaches_devkit_with_explicit_filters() {
    let selected = arguments(
        &[
            "--changed",
            "--changed-base=HEAD~1",
            "-p",
            "polylogue-store",
        ],
        None,
    )
    .unwrap();
    assert_eq!(
        selected[4..],
        [
            "-E",
            "((package(=polylogue-store)))",
            "--changed",
            "--changed-base=HEAD~1"
        ]
    );
}
