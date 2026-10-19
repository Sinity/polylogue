mod bench;

use clap::Parser;
use devkit::{Nextest, Run, select::Selection};
use serde_json::Value;
use std::{
    path::Path,
    process::{Command, ExitCode},
};

/// Packages rustfmt checks.
const FORMATTED: &[&str] = &[
    "polylogue-model",
    "polylogue-sources",
    "polylogue-store",
    "polylogue-engine",
    "polylogue-query",
    "polylogue-render",
    "polylogue-sinex",
    "polylogue-testkit",
    "polylogue",
    "xtask",
];

#[derive(Parser)]
#[command(name = "xtask", about = "Polylogue development checks")]
enum Cli {
    /// Formatting, crate boundaries and clippy (warnings denied) of every
    /// target.
    Fast,
    /// Formatting, crate boundaries, clippy and the full test suite.
    Check,
    /// The end-to-end tests: the real daemon, CLI and MCP server.
    Scenarios,
    /// Tests only: everything, or what changed since the merge base.
    Test {
        /// Only packages changed since the merge base with origin/master, and
        /// what depends on them.
        #[arg(long)]
        changed: bool,
        /// Further `cargo nextest run` arguments: packages, filters, names.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Crate dependency boundaries only.
    Boundaries,
    /// The first build on a frozen representative sample of the corpus.
    #[command(subcommand)]
    Bench(bench::Bench),
    #[command(flatten)]
    Shared(devkit::cli::Command),
}

/// Internal dependencies each workspace package may use (architecture 3).
/// Only the store may depend on rusqlite.
fn allowed_dependencies(package: &str) -> Option<&'static [&'static str]> {
    Some(match package {
        "polylogue-model" | "xtask" => &[],
        "polylogue-sources" | "polylogue-query" | "polylogue-store" | "polylogue-render" => {
            &["polylogue-model"]
        }
        "polylogue-engine" => &["polylogue-model", "polylogue-sources", "polylogue-store"],
        // Test worlds: built through the engine, read through the store.
        "polylogue-testkit" => &["polylogue-model", "polylogue-store", "polylogue-engine"],
        // The Sinex adapter publishes and restores through the store; its
        // rebuild proof drives the engine.
        "polylogue-sinex" => &["polylogue-model", "polylogue-store", "polylogue-engine"],
        "polylogue" => &[
            "polylogue-model",
            "polylogue-sources",
            "polylogue-store",
            "polylogue-engine",
            "polylogue-query",
            "polylogue-render",
            "polylogue-sinex",
        ],
        _ => return None,
    })
}

fn metadata(root: &Path) -> Result<Value, String> {
    let output = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(root)
        .output()
        .map_err(|e| e.to_string())?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).into_owned());
    }
    serde_json::from_slice(&output.stdout).map_err(|e| e.to_string())
}

fn boundaries(metadata: &Value) -> Result<(), String> {
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
            let test_only = dependency["kind"].as_str() == Some("dev");
            let dependency = dependency["name"]
                .as_str()
                .ok_or("dependency name missing")?;
            // Tests may build worlds with the testkit and open SQLite to
            // break an archive on purpose (law 22); product code may not.
            if test_only && matches!(dependency, "polylogue-testkit" | "rusqlite") {
                continue;
            }
            if (workspace.contains(&dependency) && !allowed.contains(&dependency))
                || (dependency == "rusqlite" && name != "polylogue-store")
            {
                return Err(format!("forbidden dependency: {name} -> {dependency}"));
            }
        }
    }
    Ok(())
}

fn fmt_check(run: &mut Run) -> devkit::Result {
    let mut fmt = Command::new("cargo");
    fmt.arg("fmt");
    for package in FORMATTED {
        fmt.args(["-p", package]);
    }
    fmt.args(["--", "--check"]);
    run.command("fmt", &mut fmt)
}

fn boundaries_step(run: &mut Run) -> devkit::Result {
    run.step("boundaries", |run| {
        Ok(metadata(&run.root).and_then(|metadata| boundaries(&metadata))?)
    })
}

/// Clippy over every target, warnings denied: it also compiles everything,
/// so it replaces a separate `cargo check`.
fn clippy(run: &mut Run) -> devkit::Result {
    run.cargo(
        "clippy",
        &[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--",
            "-D",
            "warnings",
        ],
    )
}

fn nextest(args: Vec<String>) -> Nextest {
    Nextest {
        args,
        tmpfs: true,
        ..Nextest::default()
    }
}

fn run(command: &str, args: &[String], steps: impl FnOnce(&mut Run) -> devkit::Result) -> ExitCode {
    let mut run = Run::start("polylogue", command, args);
    let _ = steps(&mut run);
    run.finish()
}

fn main() -> ExitCode {
    match Cli::parse() {
        Cli::Fast => run("fast", &[], |run| {
            fmt_check(run)?;
            boundaries_step(run)?;
            clippy(run)
        }),
        Cli::Check => run("check", &[], |run| {
            fmt_check(run)?;
            boundaries_step(run)?;
            clippy(run)?;
            run.nextest(
                "test",
                &nextest(vec!["--locked".into(), "--workspace".into()]),
            )?;
            // The laws that inject failures, built with the test-only feature.
            run.nextest(
                "faults",
                &nextest(
                    [
                        "--locked",
                        "-p",
                        "polylogue-engine",
                        "--features",
                        "faults",
                        "--test",
                        "faults",
                    ]
                    .map(String::from)
                    .to_vec(),
                ),
            )
        }),
        Cli::Scenarios => run("scenarios", &[], |run| {
            run.nextest(
                "test",
                &nextest(
                    ["--locked", "-p", "polylogue", "-E", "kind(test)"]
                        .map(String::from)
                        .to_vec(),
                ),
            )
        }),
        Cli::Boundaries => run("boundaries", &[], boundaries_step),
        Cli::Test { changed, args } => {
            let mut nextest_args = vec!["--locked".to_owned()];
            if !args.iter().any(|a| a == "-p" || a.starts_with("--package")) {
                nextest_args.push("--workspace".to_owned());
            }
            if changed {
                let root = Path::new(".");
                match devkit::select::changed(root, "origin/master") {
                    Ok(Selection::Nothing) => {
                        eprintln!("no package changed since the merge base; nothing to test");
                        return ExitCode::SUCCESS;
                    }
                    Ok(selection) => {
                        if let Some(filter) = selection.filterset() {
                            nextest_args.extend(["-E".to_owned(), filter]);
                        }
                    }
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            nextest_args.extend(args.iter().cloned());
            run("test", &args, |run| {
                run.nextest("test", &nextest(nextest_args))
            })
        }
        Cli::Bench(command) => bench::execute(command),
        Cli::Shared(command) => devkit::cli::execute("polylogue", command),
    }
}

#[cfg(test)]
mod tests;
