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
    assert!(boundaries(root.path()).is_ok());
    fs::OpenOptions::new()
        .append(true)
        .open(root.path().join("query/Cargo.toml"))
        .unwrap()
        .write_all(b"[dependencies]\npolylogue-store={path=\"../store\"}\n")
        .unwrap();
    assert_eq!(
        boundaries(root.path()).unwrap_err(),
        "forbidden dependency: polylogue-query -> polylogue-store"
    );
}
