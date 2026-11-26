//! Keeps master's saved build to the artifacts its builds use. Cargo never
//! removes an artifact it no longer needs, and each change of a dependency's
//! version names every unit above it anew, so a target directory restored
//! and saved on every push keeps a full copy of the workspace per change.
//! Access times cannot tell what a build read: the hosted runner restores
//! the target from an archive onto a `relatime` file system. So the builds
//! are asked again, with JSON messages: Cargo lists every unit's files,
//! fresh ones too, and what none of them names is removed.

use std::collections::BTreeSet;
use std::path::Path;

/// The directories of a profile's output that hold one entry per unit,
/// named `<name>-<hash>` with an optional extension.
const UNIT_DIRS: [&str; 4] = ["deps", "build", ".fingerprint", "examples"];

/// Removes from `target/debug` every unit artifact that none of `builds`
/// names, and reports what it kept and removed.
pub(crate) fn prune(root: &Path, target: &Path, builds: &[&[&str]]) -> Result<(), String> {
    let mut kept = BTreeSet::new();
    for args in builds {
        kept.extend(unit_hashes(root, args)?);
    }
    let profile = target.join("debug");
    let (mut removed, mut removed_bytes, mut kept_bytes) = (0u64, 0u64, 0u64);
    for dir in UNIT_DIRS {
        let dir = profile.join(dir);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries {
            let entry = entry.map_err(|e| format!("{}: {e}", dir.display()))?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let Some(hash) = unit_hash(&name) else {
                continue;
            };
            let path = entry.path();
            let bytes = size(&path);
            if kept.contains(hash) {
                kept_bytes += bytes;
                continue;
            }
            let result = if entry.file_type().is_ok_and(|t| t.is_dir()) {
                std::fs::remove_dir_all(&path)
            } else {
                std::fs::remove_file(&path)
            };
            result.map_err(|e| format!("{}: {e}", path.display()))?;
            removed += 1;
            removed_bytes += bytes;
        }
    }
    eprintln!(
        "prune-build: {} units named by {} builds; kept {:.2} GB, removed {removed} entries, {:.2} GB",
        kept.len(),
        builds.len(),
        kept_bytes as f64 / 1e9,
        removed_bytes as f64 / 1e9
    );
    Ok(())
}

/// The unit hashes in every path `cargo <args>` reports: each unit's
/// files and its build script's output directory.
fn unit_hashes(root: &Path, args: &[&str]) -> Result<BTreeSet<String>, String> {
    let (subcommand, rest) = args.split_first().ok_or("cargo needs a subcommand")?;
    let output = devkit::cargo_command()
        .arg(subcommand)
        .arg("--message-format=json")
        .args(rest)
        .current_dir(root)
        .stderr(std::process::Stdio::inherit())
        .output()
        .map_err(|e| format!("cargo {subcommand}: {e}"))?;
    if !output.status.success() {
        return Err(format!("cargo {}: {}", args.join(" "), output.status));
    }
    let mut hashes = BTreeSet::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let Ok(message) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let paths = message["filenames"]
            .as_array()
            .into_iter()
            .flatten()
            .chain([&message["executable"], &message["out_dir"]]);
        for path in paths.filter_map(serde_json::Value::as_str) {
            for component in Path::new(path).components() {
                if let Some(hash) = unit_hash(&component.as_os_str().to_string_lossy()) {
                    hashes.insert(hash.to_owned());
                }
            }
        }
    }
    Ok(hashes)
}

/// The hash in a unit entry's name, `<name>-<16 hex digits>[.<ext>]`.
fn unit_hash(name: &str) -> Option<&str> {
    let stem = name.split_once('.').map_or(name, |(stem, _)| stem);
    let (_, hash) = stem.rsplit_once('-')?;
    (hash.len() == 16 && hash.bytes().all(|b| b.is_ascii_hexdigit())).then_some(hash)
}

fn size(path: &Path) -> u64 {
    let Ok(meta) = std::fs::symlink_metadata(path) else {
        return 0;
    };
    if !meta.is_dir() {
        return meta.len();
    }
    std::fs::read_dir(path)
        .into_iter()
        .flatten()
        .flatten()
        .map(|entry| size(&entry.path()))
        .sum()
}
