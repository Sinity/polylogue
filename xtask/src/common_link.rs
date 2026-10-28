//! Building against a local checkout of polylogue-sinex-common.
//!
//! `link-common` writes `.cargo/local.toml`, which `.cargo/config.toml`
//! includes: one `[patch]` entry per shared crate this workspace names,
//! pointing into the checkout. The file belongs to these two commands. Cargo then records those crates as local
//! packages in `Cargo.lock`, so the lock as it was is kept in
//! `.cargo/common-link.lock`, and `unlink-common` puts it back. Unlinking
//! also writes `.cargo/common-link.off`, which stops the development shell
//! from linking again on entry. Git ignores all three files.
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs, io,
    path::{Path, PathBuf},
    process::Stdio,
};

/// The repository the shared crates come from, as the manifests spell it.
const REPOSITORY: &str = "https://github.com/Sinity/polylogue-sinex-common";
const LINK: &str = ".cargo/local.toml";
const SAVED_LOCK: &str = ".cargo/common-link.lock";
const OFF: &str = ".cargo/common-link.off";

/// Names a checkout of the shared repository when it is not the sibling
/// `../polylogue-sinex-common`.
const CHECKOUT_VARIABLE: &str = "POLYLOGUE_SINEX_COMMON";

/// Points the shared crates at `checkout`, else at `$POLYLOGUE_SINEX_COMMON`,
/// else at the sibling `../polylogue-sinex-common`.
pub fn link(checkout: Option<PathBuf>) -> Result<(), String> {
    let root = workspace_root()?;
    let checkout = checkout
        .or_else(|| std::env::var_os(CHECKOUT_VARIABLE).map(PathBuf::from))
        .unwrap_or_else(|| root.join("../polylogue-sinex-common"));
    let checkout =
        fs::canonicalize(&checkout).map_err(|e| format!("{}: {e}", checkout.display()))?;
    remove(&root.join(LINK))?;
    let used = shared_crates(&metadata(&root)?);
    let available = crate_directories(&metadata(&checkout)?);
    let config = patch_config(&used, &available)?;
    let saved = root.join(SAVED_LOCK);
    if !saved.exists() {
        fs::copy(root.join("Cargo.lock"), &saved).map_err(|e| format!("saving Cargo.lock: {e}"))?;
    }
    fs::write(root.join(LINK), config).map_err(|e| format!("{LINK}: {e}"))?;
    remove(&root.join(OFF))?;
    // Records the local crates in Cargo.lock, so `--locked` builds accept them.
    let status = devkit::cargo_command()
        .args(["metadata", "--format-version", "1"])
        .current_dir(&root)
        .stdout(Stdio::null())
        .status()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !status.success() {
        return Err(format!(
            "updating Cargo.lock for the local crates failed: {status}"
        ));
    }
    println!(
        "linked {} to {}; Cargo.lock now names the local crates, so run \
         `cargo xtask unlink-common` before committing it",
        used.into_iter().collect::<Vec<_>>().join(", "),
        checkout.display()
    );
    Ok(())
}

/// Returns to the commit of the shared crates that Cargo.lock pinned before
/// linking.
pub fn unlink() -> Result<(), String> {
    let root = workspace_root()?;
    remove(&root.join(LINK))?;
    fs::write(
        root.join(OFF),
        "Written by `cargo xtask unlink-common`: entering the development shell \
         does not link polylogue-sinex-common while this file exists.\n",
    )
    .map_err(|e| format!("{OFF}: {e}"))?;
    match fs::rename(root.join(SAVED_LOCK), root.join("Cargo.lock")) {
        Ok(()) => println!("restored Cargo.lock as it was before linking"),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            println!("polylogue-sinex-common was not linked");
        }
        Err(error) => return Err(format!("restoring Cargo.lock: {error}")),
    }
    Ok(())
}

/// The workspace's direct dependencies that come from the shared repository.
fn shared_crates(metadata: &Value) -> BTreeSet<String> {
    let packages = metadata["packages"].as_array().into_iter().flatten();
    packages
        .flat_map(|package| package["dependencies"].as_array().into_iter().flatten())
        .filter(|dependency| {
            dependency["source"]
                .as_str()
                .and_then(|source| source.strip_prefix("git+"))
                .and_then(|source| source.split(['?', '#']).next())
                == Some(REPOSITORY)
        })
        .filter_map(|dependency| dependency["name"].as_str().map(str::to_owned))
        .collect()
}

/// Each package of a workspace and the directory holding its manifest.
fn crate_directories(metadata: &Value) -> BTreeMap<String, PathBuf> {
    let packages = metadata["packages"].as_array().into_iter().flatten();
    packages
        .filter_map(|package| {
            let name = package["name"].as_str()?;
            let manifest = Path::new(package["manifest_path"].as_str()?);
            Some((name.to_owned(), manifest.parent()?.to_owned()))
        })
        .collect()
}

fn patch_config(
    used: &BTreeSet<String>,
    available: &BTreeMap<String, PathBuf>,
) -> Result<String, String> {
    if used.is_empty() {
        return Err(format!(
            "no dependency of this workspace comes from {REPOSITORY}"
        ));
    }
    let mut config = format!(
        "# Written by `cargo xtask link-common`; `cargo xtask unlink-common` removes it.\n\
         [patch.{}]\n",
        toml_string(REPOSITORY)
    );
    for name in used {
        let directory = available
            .get(name)
            .ok_or_else(|| format!("the checkout has no crate {name}"))?;
        config += &format!(
            "{name} = {{ path = {} }}\n",
            toml_string(&directory.display().to_string())
        );
    }
    Ok(config)
}

fn toml_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\""))
}

fn metadata(workspace: &Path) -> Result<Value, String> {
    let output = devkit::cargo_command()
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(workspace)
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata in {}: {}",
            workspace.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())
}

fn workspace_root() -> Result<PathBuf, String> {
    let output = devkit::cargo_command()
        .args(["locate-project", "--workspace", "--message-format", "plain"])
        .output()
        .map_err(|e| format!("cargo locate-project: {e}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    let manifest = PathBuf::from(String::from_utf8_lossy(&output.stdout).trim());
    manifest
        .parent()
        .map(Path::to_owned)
        .ok_or_else(|| format!("{} has no directory", manifest.display()))
}

fn remove(path: &Path) -> Result<(), String> {
    match fs::remove_file(path) {
        Err(error) if error.kind() != io::ErrorKind::NotFound => {
            Err(format!("{}: {error}", path.display()))
        }
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests;
