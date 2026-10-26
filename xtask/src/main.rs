mod bench;

use clap::Parser;
use devkit::{Nextest, Run, select::Selection};
use serde_json::Value;
use std::{path::Path, process::ExitCode};

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

/// The one feature set every test build uses, with the whole workspace
/// selected, so `test`, `scenarios` and `check` build the same artifacts and
/// each reuses what the others built. Package selection narrows which tests
/// run, never what is built.
const TEST_FEATURES: &str = "polylogue/sinex,polylogue-engine/faults,polylogue-store/faults";

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
        /// Further `cargo nextest run` arguments: `-p` packages and `-E`
        /// filtersets (both select which tests run), names and runner flags.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Crate dependency boundaries only.
    Boundaries,
    /// Corpus builds and native interactive read measurements.
    #[command(subcommand)]
    Bench(bench::Bench),
    #[command(flatten)]
    Shared(devkit::cli::Command),
}

/// Internal dependencies each workspace package may use.
/// Only the store may depend on rusqlite.
fn allowed_dependencies(package: &str) -> Option<&'static [&'static str]> {
    Some(match package {
        "polylogue-model" | "xtask" => &[],
        "polylogue-sources" | "polylogue-query" | "polylogue-store" | "polylogue-render" => {
            &["polylogue-model"]
        }
        "polylogue-engine" => &["polylogue-model", "polylogue-sources", "polylogue-store"],
        // Test archives: built through the engine, read through the store.
        "polylogue-testkit" => &["polylogue-model", "polylogue-store", "polylogue-engine"],
        // The Sinex client sends inputs and register changes and rebuilds the
        // register through the store; its rebuild test drives the engine.
        "polylogue-sinex" => &[
            "polylogue-model",
            "polylogue-query",
            "polylogue-store",
            "polylogue-engine",
        ],
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
    let output = devkit::cargo_command()
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
            // Tests may build archives with the testkit and open SQLite to
            // break an archive on purpose; product code may not.
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
    let mut fmt = devkit::cargo_command();
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

/// Clippy over every target, warnings denied: once with `polylogue/sinex`,
/// which also compiles everything, so it replaces a separate `cargo check`;
/// then the application without it, so neither build has dead code.
fn clippy(run: &mut Run) -> devkit::Result {
    run.cargo(
        "clippy",
        &[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--features",
            "polylogue/sinex",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run.cargo(
        "clippy-standalone",
        &[
            "clippy",
            "--locked",
            "-p",
            "polylogue",
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

/// `cargo nextest run` arguments for the shared test build that run the
/// tests `args` select. `-p` packages and `-E` filtersets become one
/// filterset, intersected with `required` (a changed-package selection, or
/// the scenarios' own). Feature flags are refused: there is one feature set.
fn test_arguments(args: &[String], required: Option<String>) -> Result<Vec<String>, String> {
    let mut packages = Vec::new();
    let mut filtersets = Vec::new();
    let mut rest = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        let mut value = |flag: &str| {
            args.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "-p" | "--package" => packages.push(value(arg)?),
            "-E" | "--filterset" | "--filter-expr" => filtersets.push(value(arg)?),
            "--workspace" | "--all" => {}
            "-F" | "--features" | "--all-features" | "--no-default-features" => {
                return Err(format!(
                    "{arg}: tests always build with {TEST_FEATURES}; select tests with -p, -E or names"
                ));
            }
            _ => {
                if let Some(package) = arg.strip_prefix("--package=").or(arg.strip_prefix("-p")) {
                    packages.push(package.to_owned());
                } else if let Some(filterset) = arg
                    .strip_prefix("--filterset=")
                    .or(arg.strip_prefix("--filter-expr="))
                    .or(arg.strip_prefix("-E"))
                {
                    filtersets.push(filterset.to_owned());
                } else if arg.starts_with("--features=") || arg.starts_with("-F") {
                    return Err(format!(
                        "{arg}: tests always build with {TEST_FEATURES}; select tests with -p, -E or names"
                    ));
                } else {
                    rest.push(arg.clone());
                }
            }
        }
    }
    let either = |sets: Vec<String>| {
        sets.into_iter()
            .map(|set| format!("({set})"))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let terms: Vec<String> = [
        required,
        (!packages.is_empty()).then(|| {
            either(
                packages
                    .into_iter()
                    .map(|package| format!("package(={package})"))
                    .collect(),
            )
        }),
        (!filtersets.is_empty()).then(|| either(filtersets)),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut arguments = ["--locked", "--workspace", "--features", TEST_FEATURES]
        .map(String::from)
        .to_vec();
    if !terms.is_empty() {
        arguments.push("-E".to_owned());
        arguments.push(
            terms
                .iter()
                .map(|term| format!("({term})"))
                .collect::<Vec<_>>()
                .join(" & "),
        );
    }
    arguments.extend(rest);
    Ok(arguments)
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
            let arguments = test_arguments(&[], None).expect("the full selection is valid");
            run.nextest("test", &nextest(arguments))
        }),
        Cli::Scenarios => run("scenarios", &[], |run| {
            let arguments = test_arguments(&[], Some("package(=polylogue) & kind(test)".into()))
                .expect("the scenario selection is valid");
            run.nextest("test", &nextest(arguments))
        }),
        Cli::Boundaries => run("boundaries", &[], boundaries_step),
        Cli::Test { changed, args } => {
            let mut required = None;
            if changed {
                let root = Path::new(".");
                match devkit::select::changed(root, "origin/master") {
                    Ok(Selection::Nothing) => {
                        eprintln!("no package changed since the merge base; nothing to test");
                        return ExitCode::SUCCESS;
                    }
                    Ok(selection) => required = selection.filterset(),
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            let arguments = match test_arguments(&args, required) {
                Ok(arguments) => arguments,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            run("test", &args, |run| {
                run.nextest("test", &nextest(arguments))
            })
        }
        Cli::Bench(command) => bench::execute(command),
        Cli::Shared(command) => devkit::cli::execute("polylogue", command),
    }
}

#[cfg(test)]
mod tests;
