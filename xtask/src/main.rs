mod bench;
mod common_link;

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

/// The one feature set every test build uses: primary mode (the Sinex
/// client) and the store's and engine's fault points. Every test command
/// builds the whole workspace with it, so `check`, `test` and `scenarios`
/// resolve features alike and reuse each other's test binaries; a package
/// selection only filters what runs.
const TEST_FEATURES: &str = "polylogue/sinex,polylogue-engine/faults,polylogue-store/faults";

#[derive(Parser)]
#[command(name = "xtask", about = "Polylogue development checks")]
enum Cli {
    /// Formatting, crate boundaries and clippy (warnings denied) of every
    /// target.
    Fast,
    /// Formatting, crate boundaries, both clippy configurations and all tests.
    Check,
    /// The end-to-end tests: the real daemon, CLI and MCP server.
    Scenarios,
    /// Tests only: everything, or what changed since the merge base.
    Test {
        /// Only packages changed since the merge base with origin/master, and
        /// what depends on them.
        #[arg(long)]
        changed: bool,
        /// Build every test binary without running any (a cloud
        /// environment's warm build); selection does not apply.
        #[arg(long, conflicts_with_all = ["changed", "args"])]
        no_run: bool,
        /// Further `cargo nextest run` arguments: `-p` packages and `-E`
        /// filtersets, names and runner flags. Packages select which tests
        /// run; the build is always the whole workspace's.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Crate dependency boundaries only.
    Boundaries,
    /// Offline schema commands from the application's command table.
    Schema {
        /// The Polylogue binary to run, built by the normal compile operation;
        /// by default `debug/polylogue` in Cargo's target directory.
        #[arg(long)]
        binary: Option<std::path::PathBuf>,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Corpus builds and native interactive read measurements.
    Bench(bench::Bench),
    /// Build against a local checkout of polylogue-sinex-common instead of
    /// the commit Cargo.lock pins.
    LinkCommon {
        /// The checkout; by default `$POLYLOGUE_SINEX_COMMON`, else
        /// `../polylogue-sinex-common`.
        checkout: Option<std::path::PathBuf>,
    },
    /// Return to the commit of polylogue-sinex-common that Cargo.lock pinned.
    UnlinkCommon,
    #[command(flatten)]
    Shared(devkit::cli::Command),
}

/// Internal dependencies each workspace package may use.
/// Only the store may depend on rusqlite.
fn allowed_dependencies(package: &str) -> Option<&'static [&'static str]> {
    Some(match package {
        "polylogue-model" | "xtask" => &[],
        "polylogue-sources" | "polylogue-query" | "polylogue-render" => &["polylogue-model"],
        "polylogue-store" => &["polylogue-model", "polylogue-query"],
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

/// Whether Cargo's dependency graph makes a package affected by the selection.
fn affected(selection: &Selection, metadata: &Value, target: &str) -> bool {
    let (changed, binaries) = match selection {
        Selection::Nothing => return false,
        // Function filters are already narrowed by devkit; treating them as
        // reaching every package keeps this independent of the filter syntax.
        Selection::Workspace | Selection::Filter(_) => return true,
        Selection::Packages(packages) => (packages.as_slice(), &[][..]),
        Selection::Tests { packages, binaries } => (packages.as_slice(), binaries.as_slice()),
    };
    if binaries.iter().any(|(package, _)| package == target) {
        return true;
    }
    let mut dependencies = std::collections::BTreeSet::new();
    let mut pending = vec![target.to_owned()];
    while let Some(name) = pending.pop() {
        if !dependencies.insert(name.clone()) {
            continue;
        }
        if let Some(package) = metadata["packages"].as_array().and_then(|packages| {
            packages
                .iter()
                .find(|package| package["name"].as_str() == Some(&name))
        }) {
            pending.extend(
                package["dependencies"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|dependency| dependency["name"].as_str().map(str::to_owned)),
            );
        }
    }
    changed.iter().any(|package| dependencies.contains(package))
}

/// Clippy over every target with the test feature set, then the application
/// without primary mode, which only this compile check covers. `fast`
/// skips the second when the change cannot reach the application.
fn clippy(run: &mut Run, full: bool) -> devkit::Result {
    let standalone = full || {
        let selection = devkit::select::changed(&run.root, "origin/master")?;
        affected(&selection, &metadata(&run.root)?, "polylogue")
    };
    run.cargo(
        "clippy",
        &[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--features",
            TEST_FEATURES,
            "--",
            "-D",
            "warnings",
        ],
    )?;
    if standalone {
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
        )?;
    }
    Ok(())
}

fn build_tests(run: &mut Run) -> devkit::Result {
    let mut command = vec!["test".to_owned(), "--no-run".to_owned()];
    command.extend(test_arguments(&[], None).expect("the full selection is valid"));
    let arguments: Vec<&str> = command.iter().map(String::as_str).collect();
    run.cargo("test build", &arguments)
}

fn nextest(args: Vec<String>) -> Nextest {
    Nextest {
        args,
        tmpfs: true,
        ..Nextest::default()
    }
}

/// One `cargo nextest run` over the whole workspace with [`TEST_FEATURES`].
/// Package selections (`-p`, `--exclude`) become nextest filters,
/// intersected with the filtersets and with `required` (a changed-package
/// selection, or the scenarios' own), so every selection builds the same test
/// binaries and a focused run reuses a full one's. `--tests` leaves out the
/// examples that do not test anything; Clippy still compiles them.
/// Feature flags are refused: a second feature set is a second build.
fn test_arguments(args: &[String], required: Option<String>) -> Result<Vec<String>, String> {
    let refused = |arg: &str| {
        format!("{arg}: tests always build with {TEST_FEATURES}; select tests with -p, -E or names")
    };
    let mut packages = Vec::new();
    let mut excluded = Vec::new();
    let mut filtersets = Vec::new();
    let mut rest = Vec::new();
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--" {
            rest.push(arg.clone());
            rest.extend(args.cloned());
            break;
        }
        let mut value = |flag: &str| {
            args.next()
                .cloned()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match arg.as_str() {
            "-p" | "--package" => packages.push(value(arg)?),
            "--exclude" => excluded.push(value(arg)?),
            "-E" | "--filterset" | "--filter-expr" => filtersets.push(value(arg)?),
            "--workspace" | "--all" => {}
            "-F" | "--features" | "--all-features" | "--no-default-features" => {
                return Err(refused(arg));
            }
            _ => {
                if let Some(package) = arg.strip_prefix("--package=").or(arg.strip_prefix("-p")) {
                    packages.push(package.strip_prefix('=').unwrap_or(package).to_owned());
                } else if let Some(package) = arg.strip_prefix("--exclude=") {
                    excluded.push(package.to_owned());
                } else if let Some(filterset) = arg
                    .strip_prefix("--filterset=")
                    .or(arg.strip_prefix("--filter-expr="))
                    .or(arg.strip_prefix("-E"))
                {
                    filtersets.push(filterset.strip_prefix('=').unwrap_or(filterset).to_owned());
                } else if arg.starts_with("--features=") || arg.starts_with("-F") {
                    return Err(refused(arg));
                } else {
                    rest.push(arg.clone());
                }
            }
        }
    }
    let either = |sets: Vec<String>| {
        (!sets.is_empty()).then(|| {
            sets.into_iter()
                .map(|set| format!("({set})"))
                .collect::<Vec<_>>()
                .join(" | ")
        })
    };
    let package = |name: String| format!("package({name})");
    let terms: Vec<String> = [
        required,
        either(packages.into_iter().map(package).collect()),
        either(filtersets),
    ]
    .into_iter()
    .flatten()
    .chain(
        excluded
            .into_iter()
            .map(|name| format!("not {}", package(name))),
    )
    .collect();
    let mut arguments: Vec<String> = [
        "--locked",
        "--workspace",
        "--tests",
        "--features",
        TEST_FEATURES,
    ]
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

fn exit_code(result: Result<(), String>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn main() -> ExitCode {
    match Cli::parse() {
        Cli::Fast => run("fast", &[], |run| {
            fmt_check(run)?;
            boundaries_step(run)?;
            clippy(run, false)
        }),
        Cli::Check => run("check", &[], |run| {
            fmt_check(run)?;
            boundaries_step(run)?;
            clippy(run, true)?;
            let arguments = test_arguments(&[], None).expect("the full selection is valid");
            run.nextest("test", &nextest(arguments))
        }),
        Cli::Scenarios => run("scenarios", &[], |run| {
            let arguments = test_arguments(&[], Some("package(=polylogue) & kind(test)".into()))
                .expect("the scenario selection is valid");
            run.nextest("test", &nextest(arguments))
        }),
        Cli::Boundaries => run("boundaries", &[], boundaries_step),
        Cli::Test { no_run: true, .. } => run("test", &["--no-run".into()], build_tests),
        Cli::Test { changed, args, .. } => {
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
        Cli::Schema { binary, args } => {
            let binary = binary.unwrap_or_else(|| {
                std::env::var_os("CARGO_TARGET_DIR")
                    .map_or_else(|| "target".into(), std::path::PathBuf::from)
                    .join("debug/polylogue")
            });
            match std::process::Command::new(binary)
                .arg("schema")
                .args(args)
                .status()
            {
                Ok(status) => ExitCode::from(
                    status
                        .code()
                        .and_then(|n| u8::try_from(n).ok())
                        .unwrap_or(1),
                ),
                Err(error) => {
                    eprintln!("schema command: {error}");
                    ExitCode::FAILURE
                }
            }
        }
        Cli::Bench(command) => bench::execute(command),
        Cli::LinkCommon { checkout } => exit_code(common_link::link(checkout)),
        Cli::UnlinkCommon => exit_code(common_link::unlink()),
        Cli::Shared(command) => devkit::cli::execute("polylogue", command),
    }
}

#[cfg(test)]
mod tests;
