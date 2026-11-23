//! The flake checks that cover the Nix modules and settings, selected from the
//! same diff as the Rust tests.

use devkit::Run;
use std::path::Path;
use std::process::Command;

/// The cheap flake checks: formatting, settings rendering and module evaluation.
const CHECKS: [&str; 3] = ["nix-fmt", "settings", "modules"];

/// Whether a changed file can alter what those checks evaluate.
pub(crate) fn is_nix_input(file: &str) -> bool {
    file.starts_with("nix/") || matches!(file, "flake.nix" | "flake.lock")
}

/// Files changed since the merge base with `base`, including staged, unstaged
/// and untracked files (the set devkit selects Rust tests from).
pub(crate) fn changed_files(root: &Path, base: &str) -> Result<Vec<String>, String> {
    let git = |args: &[&str]| -> Result<String, String> {
        let output = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .map_err(|error| format!("git {}: {error}", args.join(" ")))?;
        if !output.status.success() {
            return Err(format!(
                "git {}: {}",
                args.join(" "),
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    };
    let merge_base = git(&["merge-base", "HEAD", base])?;
    let mut files = git(&[
        "diff",
        "--no-renames",
        "--name-only",
        "-z",
        merge_base.trim(),
    ])?;
    files.push('\0');
    files.push_str(&git(&["ls-files", "--others", "--exclude-standard", "-z"])?);
    Ok(files
        .split('\0')
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect())
}

pub(crate) fn check(run: &mut Run) -> devkit::Result {
    let mut build = Command::new("nix");
    build.current_dir(&run.root).args(["build", "--no-link"]);
    build.args(
        CHECKS
            .iter()
            .map(|name| format!(".#checks.x86_64-linux.{name}")),
    );
    run.command("nix checks", &mut build)
}
