use serde_json::Value;
use std::{
    env,
    path::Path,
    process::{Command, ExitCode},
};

/// Packages rustfmt checks. The shared crates keep their upstream layout.
const FORMATTED: &[&str] = &[
    "polylogue-model",
    "polylogue-query",
    "polylogue-sources",
    "polylogue-engine",
    "polylogue-store",
    "polylogue",
    "xtask",
];

/// Internal dependencies each workspace package may use. Only the store may
/// depend on rusqlite.
fn allowed_dependencies(package: &str) -> Option<&'static [&'static str]> {
    Some(match package {
        "polylogue-model" | "hashed-blobs" | "credential-output" | "xtask" => &[],
        "polylogue-sources" | "polylogue-query" => &["polylogue-model"],
        "polylogue-store" => &["polylogue-model", "hashed-blobs"],
        "polylogue-engine" => &[
            "polylogue-model",
            "polylogue-sources",
            "polylogue-query",
            "polylogue-store",
        ],
        "polylogue" => &[
            "polylogue-model",
            "polylogue-store",
            "polylogue-sources",
            "polylogue-query",
            "polylogue-engine",
            "credential-output",
        ],
        _ => return None,
    })
}

fn boundaries(root: &Path) -> Result<(), String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    let metadata: Value = serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())?;
    let packages = metadata["packages"]
        .as_array()
        .ok_or("metadata packages missing")?;
    let workspace: Vec<&str> = packages.iter().filter_map(|p| p["name"].as_str()).collect();
    for package in packages {
        let name = package["name"].as_str().ok_or("package name missing")?;
        let allowed = allowed_dependencies(name)
            .ok_or_else(|| format!("unassigned workspace package {name}"))?;
        for dependency in package["dependencies"]
            .as_array()
            .ok_or("dependencies missing")?
        {
            let dependency = dependency["name"]
                .as_str()
                .ok_or("dependency name missing")?;
            if (workspace.contains(&dependency) && !allowed.contains(&dependency))
                || (dependency == "rusqlite" && name != "polylogue-store")
            {
                return Err(format!("forbidden dependency: {name} -> {dependency}"));
            }
        }
    }
    Ok(())
}

fn cargo(args: &[&str]) -> Result<(), ExitCode> {
    match Command::new("cargo").args(args).status() {
        Ok(status) if status.success() => Ok(()),
        Ok(status) => Err(ExitCode::from(status.code().unwrap_or(1) as u8)),
        Err(error) => {
            eprintln!("cargo: {error}");
            Err(ExitCode::FAILURE)
        }
    }
}

fn fmt_check() -> Result<(), ExitCode> {
    let mut args = vec!["fmt"];
    for package in FORMATTED {
        args.extend(["-p", package]);
    }
    args.extend(["--", "--check"]);
    cargo(&args)
}

fn run(command: &str) -> Result<(), ExitCode> {
    if let Err(error) = boundaries(Path::new(".")) {
        eprintln!("{error}");
        return Err(ExitCode::FAILURE);
    }
    match command {
        "boundaries" => Ok(()),
        "fast" => {
            fmt_check()?;
            cargo(&["check", "--locked", "--workspace", "--all-targets"])
        }
        "check" => {
            fmt_check()?;
            cargo(&["test", "--locked", "--workspace", "--no-fail-fast"])
        }
        _ => unreachable!(),
    }
}

fn usage() -> ExitCode {
    eprintln!("usage: cargo run -p xtask -- <fast|check|boundaries>");
    ExitCode::from(2)
}

fn main() -> ExitCode {
    let args: Vec<_> = env::args().skip(1).collect();
    let [command] = args.as_slice() else {
        return usage();
    };
    if !matches!(command.as_str(), "fast" | "check" | "boundaries") {
        return usage();
    }
    match run(command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests;
