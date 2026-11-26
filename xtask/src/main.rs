mod bench;
mod common_link;
mod nix;

use clap::Parser;
use devkit::{Nextest, Run};
use std::process::ExitCode;

/// Checks, warm builds and focused tests share the primary workspace graph.
/// Package predicates select execution; Cargo target flags bound compilation.
/// The standalone Clippy pass also covers `cfg(not(feature = "sinex"))`.
const FEATURES: &str = "polylogue/sinex,polylogue-model/fixture,polylogue-query/fixture,polylogue-engine/faults,polylogue-store/faults,polylogue-store/fixture";
/// The standalone Clippy pass: the same features without `sinex`.
const STANDALONE_FEATURES: &str = "polylogue-model/fixture,polylogue-query/fixture,polylogue-engine/faults,polylogue-store/faults,polylogue-store/fixture";

#[derive(Parser)]
#[command(name = "xtask", about = "Polylogue development checks")]
enum Cli {
    /// Formatting and clippy (warnings denied) of every
    /// target.
    Fast,
    /// Formatting, both clippy configurations and every
    /// test, or with `--changed` the tests the changes since the merge base
    /// with origin/master can affect (devkit's selection; `--changed-base`
    /// names another base). Clippy reuses every unchanged crate's result, so
    /// `check --changed` costs what the change touches: the loop while editing.
    Check {
        #[arg(long)]
        changed: bool,
        #[arg(long, value_name = "REF", conflicts_with = "changed")]
        changed_base: Option<String>,
    },
    /// Build every test binary and both clippy configurations without
    /// running any test: the warm target that new checkouts copy.
    Warm,
    /// The end-to-end tests: the real daemon, CLI and MCP server.
    Scenarios,
    /// Tests only: everything, or a selection.
    Test {
        /// Build every test binary without running any (a cloud
        /// environment's warm build); selection does not apply.
        #[arg(long, conflicts_with = "args")]
        no_run: bool,
        /// Further `cargo nextest run` arguments. Packages, exclusions,
        /// filtersets and changed-package filters select which tests run.
        /// Cargo target flags such as `--lib` bound compilation while
        /// keeping the workspace feature graph. Other flags pass through.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Corpus builds and interactive read measurements of the real binary.
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
    /// Write the browser extension's provider table
    /// (`extension/src/background/providers.js`) from the origin registry.
    ExtensionProviders,
    #[command(flatten)]
    Shared(devkit::cli::Command),
}

fn fmt_check(run: &mut Run) -> devkit::Result {
    let mut fmt = devkit::cargo_command();
    fmt.args(["fmt", "--all"]);
    fmt.args(["--", "--check"]);
    run.command("fmt", &mut fmt)
}

fn clippy(run: &mut Run) -> devkit::Result {
    run.cargo(
        "clippy",
        &[
            "clippy",
            "--locked",
            "--workspace",
            "--all-targets",
            "--features",
            FEATURES,
            "--",
            "-D",
            "warnings",
        ],
    )?;
    // Only the code behind the `sinex` feature differs; the units below
    // the application are the ones the first pass checked.
    run.cargo(
        "clippy-standalone",
        &[
            "clippy",
            "--locked",
            "--workspace",
            "--exclude",
            "polylogue-sinex",
            "--all-targets",
            "--features",
            STANDALONE_FEATURES,
            "--",
            "-D",
            "warnings",
        ],
    )
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

fn selects_test_targets(arg: &str) -> bool {
    matches!(
        arg,
        "--lib" | "--bins" | "--tests" | "--examples" | "--benches" | "--all-targets" | "--doc"
    ) || ["--bin", "--test", "--example", "--bench"]
        .iter()
        .any(|flag| {
            arg == *flag
                || arg
                    .strip_prefix(*flag)
                    .is_some_and(|rest| rest.starts_with('='))
        })
}

/// Every selection keeps the workspace graph so changing the executed tests
/// does not create new dependency compiler units. Target flags pass through;
/// package, exclusion and filterset predicates intersect with `required`.
fn test_arguments(args: &[String], required: Option<String>) -> Result<Vec<String>, String> {
    let explicit_targets = args
        .iter()
        .take_while(|arg| arg.as_str() != "--")
        .any(|arg| selects_test_targets(arg));
    let package = |name: String| {
        if name.contains(['*', '?', '[']) {
            format!("package({name})")
        } else {
            format!("package(={name})")
        }
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
        let refused = || format!("{arg}: all test selections use {FEATURES}");
        match arg.as_str() {
            "-p" | "--package" => packages.push(value(arg)?),
            "--exclude" => excluded.push(value(arg)?),
            "-E" | "--filterset" | "--filter-expr" => filtersets.push(value(arg)?),
            "--workspace" | "--all" => (),
            "-F" | "--features" | "--all-features" | "--no-default-features" => {
                return Err(refused());
            }
            _ => {
                if let Some(name) = arg.strip_prefix("--package=").or(arg.strip_prefix("-p")) {
                    packages.push(name.to_owned());
                } else if let Some(name) = arg.strip_prefix("--exclude=") {
                    excluded.push(name.to_owned());
                } else if let Some(filterset) = arg
                    .strip_prefix("--filterset=")
                    .or(arg.strip_prefix("--filter-expr="))
                    .or(arg.strip_prefix("-E"))
                {
                    filtersets.push(filterset.strip_prefix('=').unwrap_or(filterset).to_owned());
                } else if arg.starts_with("--features=") || arg.starts_with("-F") {
                    return Err(refused());
                } else {
                    rest.push(arg.clone());
                }
            }
        }
    }
    for name in packages.iter_mut().chain(excluded.iter_mut()) {
        if let Some(spec) = name.strip_prefix('=') {
            *name = spec.to_owned();
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
        (!packages.is_empty()).then(|| either(packages.iter().cloned().map(package).collect())),
        (!excluded.is_empty()).then(|| {
            format!(
                "not ({})",
                either(excluded.into_iter().map(package).collect())
            )
        }),
        (!filtersets.is_empty()).then(|| either(filtersets)),
    ]
    .into_iter()
    .flatten()
    .collect();
    let mut arguments = ["--locked", "--workspace", "--features", FEATURES]
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
    if !explicit_targets {
        arguments.push("--tests".into());
    }
    arguments.extend(rest);
    Ok(arguments)
}

fn run(command: &str, args: &[String], steps: impl FnOnce(&mut Run) -> devkit::Result) -> ExitCode {
    let mut run = Run::start("polylogue", command, args);
    let _ = steps(&mut run);
    run.finish()
}

/// The extension module is the registry example's output, written whole.
fn extension_providers() -> Result<(), String> {
    let output = devkit::cargo_command()
        .args([
            "run",
            "--locked",
            "--quiet",
            "-p",
            "polylogue-sources",
            "--example",
            "extension_providers",
        ])
        .output()
        .map_err(|error| format!("cargo run: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "the registry example failed ({}):\n{}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../extension/src/background/providers.js");
    std::fs::write(&path, &output.stdout).map_err(|error| format!("{}: {error}", path.display()))
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
            clippy(run)
        }),
        Cli::Check {
            changed,
            changed_base,
        } => {
            let base = changed_base;
            let args: Vec<String> = match (changed, &base) {
                (_, Some(base)) => vec![format!("--changed-base={base}")],
                (true, None) => vec!["--changed".into()],
                (false, None) => Vec::new(),
            };
            run("check", &args, |run| {
                // The Nix checks run for every full check, and for a diff
                // that touches their inputs; a diff confined to them has no
                // Rust to build or test.
                let files = match (changed, &base) {
                    (false, None) => None,
                    (_, base) => Some(nix::changed_files(
                        &run.root,
                        base.as_deref().unwrap_or("origin/master"),
                    )?),
                };
                let nix_changed = files
                    .as_ref()
                    .is_none_or(|files| files.iter().any(|file| nix::is_nix_input(file)));
                let nix_only = files.as_ref().is_some_and(|files| {
                    !files.is_empty() && files.iter().all(|file| nix::is_nix_input(file))
                });
                fmt_check(run)?;
                if nix_changed {
                    nix::check(run)?;
                }
                if !nix_only {
                    clippy(run)?;
                    let arguments = test_arguments(&args, None).expect("the selection is valid");
                    run.nextest("test", &nextest(arguments))?;
                }
                Ok(())
            })
        }
        Cli::Warm => run("warm", &[], |run| {
            // A lint failure on master must not leave the test builds cold.
            let tests = build_tests(run);
            clippy(run).and(tests)
        }),
        Cli::Scenarios => run("scenarios", &[], |run| {
            let arguments = test_arguments(&[], Some("package(=polylogue) & kind(test)".into()))
                .expect("the scenario selection is valid");
            run.nextest("test", &nextest(arguments))
        }),
        Cli::Test { no_run: true, .. } => run("test", &["--no-run".into()], build_tests),
        Cli::Test { args, .. } => {
            let arguments = match test_arguments(&args, None) {
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
        Cli::LinkCommon { checkout } => exit_code(common_link::link(checkout)),
        Cli::UnlinkCommon => exit_code(common_link::unlink()),
        Cli::ExtensionProviders => exit_code(extension_providers()),
        Cli::Shared(command) => devkit::cli::execute("polylogue", command),
    }
}

#[cfg(test)]
mod tests;
