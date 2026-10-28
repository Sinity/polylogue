//! Product paths and CLI messages for local shared-crate development.
use devkit::local_link::LocalLink;
use std::{
    path::{Path, PathBuf},
    process::Stdio,
};

const REPOSITORY: &str = "https://github.com/Sinity/polylogue-sinex-common";

pub fn link(checkout: Option<PathBuf>) -> Result<(), String> {
    let root = workspace_root()?;
    let checkout = checkout
        .or_else(|| std::env::var_os("POLYLOGUE_SINEX_COMMON").map(PathBuf::from))
        .unwrap_or_else(|| root.join("../polylogue-sinex-common"));
    let names = with_link(&root, |link| link.link(&checkout))?;
    println!(
        "linked {} to {}; run `cargo xtask unlink-common` before committing the workspace locks",
        names.join(", "),
        checkout.display()
    );
    Ok(())
}

pub fn unlink() -> Result<(), String> {
    let root = workspace_root()?;
    let restored = with_link(&root, |link| link.unlink())?;
    if restored.is_empty() {
        println!("polylogue-sinex-common was not linked");
    } else {
        for path in restored {
            println!(
                "restored {} as it was before linking",
                path.strip_prefix(&root).unwrap_or(&path).display()
            );
        }
    }
    Ok(())
}

fn with_link<T>(
    root: &Path,
    operation: impl FnOnce(LocalLink<'_>) -> Result<T, String>,
) -> Result<T, String> {
    let patch = root.join(".cargo/local.toml");
    let disabled = root.join(".cargo/common-link.off");
    let lock = root.join("Cargo.lock");
    let backup = root.join(".cargo/common-link.lock");
    let fuzz_lock = root.join("fuzz/Cargo.lock");
    let fuzz_backup = root.join(".cargo/common-link.fuzz.lock");
    let mut locks = vec![(lock.as_path(), backup.as_path())];
    if fuzz_lock.exists() || fuzz_backup.exists() {
        locks.push((fuzz_lock.as_path(), fuzz_backup.as_path()));
    }
    operation(LocalLink {
        repository: REPOSITORY,
        patch: &patch,
        disabled: &disabled,
        locks: &locks,
    })
}

fn workspace_root() -> Result<PathBuf, String> {
    let output = devkit::cargo_command()
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .stdout(Stdio::piped())
        .output()
        .map_err(|error| format!("cargo locate-project: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    let manifest = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    manifest
        .parent()
        .map(Path::to_owned)
        .ok_or_else(|| format!("{} has no directory", manifest.display()))
}
