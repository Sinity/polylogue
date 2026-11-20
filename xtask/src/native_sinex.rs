use devkit::Run;
use std::path::{Path, PathBuf};
use std::process::Command;

const SINEX_LOCK_SOURCE: &str = "source = \"git+https://github.com/Sinity/sinex?rev=";

fn locked_revision(lock: &str) -> Result<&str, String> {
    let mut revision = None;
    for source in lock
        .lines()
        .filter_map(|line| line.strip_prefix(SINEX_LOCK_SOURCE))
    {
        let (requested, resolved) = source
            .strip_suffix('"')
            .and_then(|source| source.split_once('#'))
            .ok_or_else(|| "Cargo.lock has an invalid Sinex source".to_owned())?;
        if requested != resolved
            || resolved.len() != 40
            || !resolved.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            return Err("Cargo.lock does not pin an exact Sinex revision".into());
        }
        if revision.is_some_and(|prior| prior != resolved) {
            return Err("Cargo.lock mixes more than one Sinex revision".into());
        }
        revision = Some(resolved);
    }
    revision.ok_or_else(|| "Cargo.lock has no exact Sinex revision".to_owned())
}

fn cached_package(root: &Path, revision: &str, package: &str) -> Result<PathBuf, String> {
    // Match Sinex's publication input, including its source export rules.
    let installable = format!(
        "git+https://github.com/Sinity/sinex?rev={revision}&shallow=1&exportIgnore=1#{package}"
    );
    let output = Command::new("nix")
        .current_dir(root)
        .args(["build", "--no-link", "--print-out-paths", "--max-jobs", "0"])
        .arg(&installable)
        .output()
        .map_err(|error| format!("obtain {installable}: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "obtain {installable} from the package cache: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    let path = String::from_utf8(output.stdout).map_err(|error| error.to_string())?;
    let path = PathBuf::from(path.trim());
    if !path.is_absolute() || !path.join("bin").is_dir() {
        return Err(format!("{installable} returned no package path"));
    }
    Ok(path)
}

pub(crate) fn check(run: &mut Run) -> devkit::Result {
    let mut env = Vec::new();
    run.step("matching Sinex test packages", |run| {
        let lock = std::fs::read_to_string(run.root.join("Cargo.lock"))
            .map_err(|error| format!("read Cargo.lock: {error}"))?;
        let revision = locked_revision(&lock)?;
        let sinex = cached_package(&run.root, revision, "sinexd")?;
        let postgres = cached_package(&run.root, revision, "postgresql")?;
        let path = std::env::var_os("PATH").unwrap_or_default();
        let path = std::env::join_paths(
            std::iter::once(postgres.join("bin")).chain(std::env::split_paths(&path)),
        )
        .map_err(|error| error.to_string())?
        .into_string()
        .map_err(|_| "test PATH is not UTF-8".to_owned())?;
        env = vec![
            (
                "SINEXD".into(),
                sinex.join("bin/sinexd").display().to_string(),
            ),
            ("PATH".into(), path),
        ];
        Ok(())
    })?;
    let arguments = crate::test_arguments(
        &["--run-ignored".into(), "only".into()],
        Some(concat!(
            "package(=polylogue-sinex) & (",
            "test(the_morrow_example_rebuilt_from_sinex_gives_the_same_answers) | ",
            "test(paired_backup_restores_polylogue_pending_frames_after_live_confirmed_removal))",
        ).into()),
    )?;
    let mut options = crate::nextest(arguments);
    options.env = env;
    run.nextest("real Sinex rebuild and backup", &options)
}
