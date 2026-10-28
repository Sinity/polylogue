use super::*;
use serde_json::json;

fn workspace() -> Value {
    json!({
        "packages": [
            {
                "name": "app",
                "manifest_path": "/w/crates/app/Cargo.toml",
                "dependencies": [
                    { "name": "hashed-blobs", "source": "git+https://github.com/Sinity/polylogue-sinex-common?branch=master" },
                    { "name": "serde", "source": "registry+https://github.com/rust-lang/crates.io-index" },
                    { "name": "app-model", "source": null },
                ]
            },
            {
                "name": "xtask",
                "manifest_path": "/w/xtask/Cargo.toml",
                "dependencies": [
                    { "name": "devkit", "source": "git+https://github.com/Sinity/polylogue-sinex-common?branch=master" },
                    { "name": "hashed-blobs", "source": "git+https://github.com/Sinity/polylogue-sinex-common?branch=master" },
                    { "name": "fork", "source": "git+https://github.com/Sinity/polylogue-sinex-common-fork?branch=master" },
                ]
            }
        ]
    })
}

#[test]
fn shared_crates_are_the_direct_dependencies_from_the_shared_repository() {
    let names: Vec<String> = shared_crates(&workspace()).into_iter().collect();
    assert_eq!(names, ["devkit", "hashed-blobs"]);
}

#[test]
fn patch_points_each_shared_crate_into_the_checkout() {
    let checkout = json!({
        "packages": [
            { "name": "devkit", "manifest_path": "/c/crates/devkit/Cargo.toml" },
            { "name": "hashed-blobs", "manifest_path": "/c/crates/hashed-blobs/Cargo.toml" },
            { "name": "devkit-macros", "manifest_path": "/c/crates/devkit-macros/Cargo.toml" },
        ]
    });
    let config = patch_config(&shared_crates(&workspace()), &crate_directories(&checkout)).unwrap();
    assert_eq!(
        config,
        "# Written by `cargo xtask link-common`; `cargo xtask unlink-common` removes it.\n\
         [patch.\"https://github.com/Sinity/polylogue-sinex-common\"]\n\
         devkit = { path = \"/c/crates/devkit\" }\n\
         hashed-blobs = { path = \"/c/crates/hashed-blobs\" }\n"
    );
}

#[test]
fn a_crate_missing_from_the_checkout_is_refused() {
    let checkout = json!({ "packages": [
        { "name": "devkit", "manifest_path": "/c/crates/devkit/Cargo.toml" },
    ]});
    let error =
        patch_config(&shared_crates(&workspace()), &crate_directories(&checkout)).unwrap_err();
    assert_eq!(error, "the checkout has no crate hashed-blobs");
}

#[test]
fn a_workspace_without_shared_crates_is_refused() {
    let error = patch_config(&BTreeSet::new(), &BTreeMap::new()).unwrap_err();
    assert!(
        error.starts_with("no dependency of this workspace"),
        "{error}"
    );
}

#[test]
fn paths_are_written_as_toml_strings() {
    assert_eq!(toml_string(r#"/a "b"\c"#), r#""/a \"b\"\\c""#);
}
