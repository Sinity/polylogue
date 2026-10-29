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

/// The feature set default test builds use, with the Sinex client excluded.
/// `test`, `scenarios` and `check` reuse these feature-enabled artifacts.
/// Native package selection limits which test targets Cargo builds.
const TEST_FEATURES: &str = "polylogue-engine/faults,polylogue-store/faults";

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
        /// filtersets, names and runner flags. Packages limit the build too.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Build and test the Sinex client and the application in primary mode.
    SinexClient,
    /// Crate dependency boundaries only.
    Boundaries,
    /// Offline schema commands from the application's command table.
    Schema {
        /// The Polylogue binary to run, built by the normal compile operation.
        #[arg(long, default_value = "target/debug/polylogue")]
        binary: std::path::PathBuf,
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Corpus builds and native interactive read measurements.
    #[command(subcommand)]
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

/// Whether Cargo's dependency graph makes a package affected by the selection.
fn affected(selection: &Selection, metadata: &Value, target: &str) -> bool {
    let (changed, binaries) = match selection {
        Selection::Nothing => return false,
        // Function filters are already narrowed by devkit. Checking both modes
        // keeps their configuration independent of the filter's text syntax.
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

fn clippy(run: &mut Run, full: bool) -> devkit::Result {
    let (application, sinex) = if full {
        (true, true)
    } else {
        let selection = devkit::select::changed(&run.root, "origin/master")?;
        let metadata = metadata(&run.root)?;
        (
            affected(&selection, &metadata, "polylogue"),
            affected(&selection, &metadata, "polylogue-sinex"),
        )
    };
    let mut args = vec!["clippy", "--locked", "--workspace", "--all-targets"];
    if sinex {
        args.extend(["--features", "polylogue/sinex"]);
    } else {
        args.extend(["--exclude", "polylogue-sinex"]);
    }
    args.extend(["--", "-D", "warnings"]);
    run.cargo("clippy", &args)?;
    if sinex && application {
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

fn sinex_client_arguments() -> Vec<String> {
    [
        "--locked",
        "-p",
        "polylogue-sinex",
        "-p",
        "polylogue",
        "--features",
        "polylogue/sinex,polylogue-engine/faults,polylogue-store/faults",
    ]
    .map(String::from)
    .to_vec()
}

fn sinex_client(run: &mut Run) -> devkit::Result {
    run.nextest("sinex-client", &nextest(sinex_client_arguments()))
}

fn build_tests(run: &mut Run) -> devkit::Result {
    let standalone =
        test_arguments(&run.root, &[], None, false).expect("the full selection is valid");
    for (stage, arguments) in [
        ("test build", standalone),
        ("sinex-client test build", sinex_client_arguments()),
    ] {
        let mut command = vec!["test".to_owned(), "--no-run".to_owned()];
        command.extend(arguments);
        let arguments: Vec<&str> = command.iter().map(String::as_str).collect();
        run.cargo(stage, &arguments)?;
    }
    Ok(())
}

fn nextest(args: Vec<String>) -> Nextest {
    Nextest {
        args,
        tmpfs: true,
        ..Nextest::default()
    }
}

/// Native Cargo package selection with nextest filters intersected with
/// `required` (a changed-package selection, or the scenarios' own).
/// Feature flags are refused so each mode keeps its fault-enabled build.
fn test_arguments(
    root: &Path,
    args: &[String],
    required: Option<String>,
    mut sinex: bool,
) -> Result<Vec<String>, String> {
    let mut packages = Vec::new();
    let mut workspace = false;
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
            "-E" | "--filterset" | "--filter-expr" => filtersets.push(value(arg)?),
            "--workspace" | "--all" => workspace = true,
            "-F" | "--features" | "--all-features" | "--no-default-features" => {
                return Err(format!(
                    "{arg}: tests always build with {TEST_FEATURES}; select tests with -p, -E or names"
                ));
            }
            _ => {
                if let Some(package) = arg.strip_prefix("--package=").or(arg.strip_prefix("-p")) {
                    packages.push(package.strip_prefix('=').unwrap_or(package).to_owned());
                } else if let Some(filterset) = arg
                    .strip_prefix("--filterset=")
                    .or(arg.strip_prefix("--filter-expr="))
                    .or(arg.strip_prefix("-E"))
                {
                    filtersets.push(filterset.strip_prefix('=').unwrap_or(filterset).to_owned());
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
    if packages.iter().any(|package| package == "polylogue-sinex") {
        sinex = true;
    }
    let either = |sets: Vec<String>| {
        sets.into_iter()
            .map(|set| format!("({set})"))
            .collect::<Vec<_>>()
            .join(" | ")
    };
    let terms: Vec<String> = [
        required,
        (!filtersets.is_empty()).then(|| either(filtersets)),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut arguments = vec!["--locked".into()];
    workspace |= packages.is_empty();
    if workspace {
        arguments.push("--workspace".into());
    }
    for package in packages {
        arguments.extend(["-p".into(), package]);
    }
    let features = if workspace {
        if sinex {
            format!("polylogue/sinex,{TEST_FEATURES}")
        } else {
            TEST_FEATURES.into()
        }
    } else {
        // Cargo accepts dependency features only when their provider is selected
        // or directly visible to a selected package. Ask Cargo for that scope.
        let output = devkit::cargo_command()
            .args([
                "tree", "--depth", "1", "--prefix", "none", "--format", "{p}",
            ])
            .args(&arguments)
            .current_dir(root)
            .output()
            .map_err(|error| format!("cargo tree: {error}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        let visible: std::collections::BTreeSet<_> = String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| line.split_whitespace().next().map(str::to_owned))
            .collect();
        let mut features: Vec<_> = TEST_FEATURES
            .split(',')
            .filter(|feature| visible.contains(feature.split_once('/').unwrap().0))
            .collect();
        if sinex && visible.contains("polylogue") {
            features.insert(0, "polylogue/sinex");
        }
        features.join(",")
    };
    if !features.is_empty() {
        arguments.extend(["--features".into(), features]);
    }
    if workspace && !sinex {
        arguments.extend(["--exclude".into(), "polylogue-sinex".into()]);
    }
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
            let arguments =
                test_arguments(&run.root, &[], None, false).expect("the full selection is valid");
            run.nextest("test", &nextest(arguments))?;
            sinex_client(run)?;
            Ok(())
        }),
        Cli::SinexClient => run("sinex-client", &[], sinex_client),
        Cli::Scenarios => run("scenarios", &[], |run| {
            let arguments = test_arguments(
                &run.root,
                &[],
                Some("package(=polylogue) & kind(test)".into()),
                false,
            )
            .expect("the scenario selection is valid");
            run.nextest("test", &nextest(arguments))
        }),
        Cli::Boundaries => run("boundaries", &[], boundaries_step),
        Cli::Test { no_run: true, .. } => run("test", &["--no-run".into()], build_tests),
        Cli::Test { changed, args, .. } => {
            let mut required = None;
            let mut sinex = false;
            if changed {
                let root = Path::new(".");
                match devkit::select::changed(root, "origin/master") {
                    Ok(Selection::Nothing) => {
                        eprintln!("no package changed since the merge base; nothing to test");
                        return ExitCode::SUCCESS;
                    }
                    Ok(selection) => {
                        let graph = match metadata(root) {
                            Ok(graph) => graph,
                            Err(error) => {
                                eprintln!("{error}");
                                return ExitCode::FAILURE;
                            }
                        };
                        sinex = affected(&selection, &graph, "polylogue-sinex");
                        required = selection.filterset();
                    }
                    Err(error) => {
                        eprintln!("{error}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            let arguments = match test_arguments(Path::new("."), &args, required, sinex) {
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
