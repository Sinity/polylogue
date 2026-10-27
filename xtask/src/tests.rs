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
        false,
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
    let args = test_arguments(&[], selection.filterset(), sinex).unwrap();
    assert!(args.iter().any(|arg| arg.contains("polylogue/sinex")));
    assert!(!args.iter().any(|arg| arg == "--exclude"));
    assert!(
        arguments(&["-p", "polylogue-sinex"], None)
            .unwrap()
            .iter()
            .any(|arg| arg.contains("polylogue/sinex"))
    );
}
