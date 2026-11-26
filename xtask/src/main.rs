mod bench;
mod common_link;
mod development;
mod nix;
mod prune;
mod selection;

use clap::Parser;
use devkit::{Nextest, Run, TestStatus};
use selection::{Base, Selection};
use std::process::ExitCode;

/// Every workspace package's test features: the whole workspace's test build
/// enables all of them, and a build of selected packages the ones Cargo
/// reaches from them (`selection`), which compiles the same units.
/// The standalone Clippy pass also covers `cfg(not(feature = "sinex"))`.
const FEATURES: &str = "polylogue/sinex,polylogue-model/fixture,polylogue-query/fixture,polylogue-sources/fixture,polylogue-engine/faults,polylogue-store/faults,polylogue-store/fixture";
/// The standalone Clippy pass: the same features without `sinex`.
const STANDALONE_FEATURES: &str = "polylogue-model/fixture,polylogue-query/fixture,polylogue-engine/faults,polylogue-store/faults,polylogue-store/fixture";

/// Clippy over every target with every test feature.
const CLIPPY: &[&str] = &[
    "clippy",
    "--locked",
    "--workspace",
    "--all-targets",
    "--features",
    FEATURES,
    "--",
    "-D",
    "warnings",
];
/// Only the code behind the `sinex` feature differs; the units below the
/// application are the ones the first pass checked.
const CLIPPY_STANDALONE: &[&str] = &[
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
];
/// Every test binary of the workspace feature graph.
const BUILD_TESTS: &[&str] = &[
    "test",
    "--no-run",
    "--locked",
    "--workspace",
    "--features",
    FEATURES,
    "--tests",
];
/// The xtask binary `cargo xtask` runs.
const BUILD_XTASK: &[&str] = &["build", "--locked", "-p", "xtask"];

#[derive(Parser)]
#[command(name = "xtask", about = "Polylogue development checks")]
enum Cli {
    /// Formatting, the workspace-hack crate's verification, and Clippy
    /// (warnings denied) of every target, with and without the `sinex` feature.
    Fast,
    /// Every part of the hosted check in sequence: formatting, the
    /// workspace-hack verification, the Nix checks, both Clippy
    /// configurations and the tests. With `--changed` or `--base` it runs the
    /// tests those changes can affect and the Nix checks when their inputs
    /// changed. Clippy reuses
    /// every unchanged crate's result, so `check --changed` costs what the
    /// change touches.
    Check {
        #[command(flatten)]
        base: Base,
    },
    /// Build every test binary and both clippy configurations without
    /// running any test: the warm target that new checkouts copy.
    Warm,
    /// The end-to-end tests: the real daemon, CLI and MCP server.
    Scenarios,
    /// The checkout's daemon, standalone, in a sandbox on a scratch root:
    /// nothing of the caller's environment reaches it, and its web reader
    /// has a free port. Prints where it serves and the CLI script that
    /// reaches it; stopping the command stops the daemon.
    Run(development::Development),
    /// Tests only: everything, the tests changes can affect, or a selection.
    Test {
        /// Build every test binary without running any (a cloud
        /// environment's warm build); selection does not apply.
        #[arg(long, conflicts_with_all = ["rest", "partition", "packages", "filtersets"])]
        no_run: bool,
        #[command(flatten)]
        base: Base,
        #[command(flatten)]
        selection: Box<Selection>,
    },
    /// The flake's formatting, settings and module checks. With a base, they
    /// run only when the changes touch `nix/`, `flake.nix` or `flake.lock`.
    Nix {
        #[command(flatten)]
        base: Base,
    },
    /// Remove from the target directory every artifact that Clippy's two
    /// configurations, the test build and xtask's own build do not use:
    /// master's saved build, before it is saved.
    PruneBuild,
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

/// The workspace-hack crate matches the dependency graph
/// (`cargo hakari generate` rewrites it).
fn workspace_hack(run: &mut Run) -> devkit::Result {
    let mut verify = devkit::cargo_command();
    verify.args(["hakari", "verify"]);
    run.command("workspace-hack", &mut verify)
}

fn clippy(run: &mut Run) -> devkit::Result {
    run.cargo("clippy", CLIPPY)?;
    run.cargo("clippy-standalone", CLIPPY_STANDALONE)
}

fn build_tests(run: &mut Run) -> devkit::Result {
    run.cargo("test build", BUILD_TESTS)
}

fn nextest(args: Vec<String>) -> Nextest {
    Nextest {
        args,
        tmpfs: true,
        ..Nextest::default()
    }
}

/// How nextest runs a selection, or `None` when it selects no package. A
/// partition may hold none of the selected tests.
fn test_options(
    root: &std::path::Path,
    base: Option<&str>,
    selection: &Selection,
) -> Result<Option<Nextest>, String> {
    let workspace = selection::Workspace::load(root)?;
    let changed = match base {
        Some(base) => selection::changed_packages(root, base)?,
        None => None,
    };
    let Some(arguments) = selection.nextest_arguments(&workspace, base, changed.as_ref())? else {
        return Ok(None);
    };
    let mut options = nextest(arguments);
    options.allow_no_tests = selection.is_partitioned();
    options.junit = selection.junit();
    Ok(Some(options))
}

/// The selected tests under nextest; their packages' test targets and what
/// those depend on are all that builds.
fn test(run: &mut Run, base: Option<&str>, selection: &Selection) -> devkit::Result {
    let options = run.step("test packages", |run| {
        Ok(test_options(&run.root, base, selection)?)
    })?;
    match options {
        Some(options) => run.nextest("test", &options),
        None => Ok(()),
    }
}

/// The Nix checks, unless a base is given and the changes since it leave
/// their inputs alone. Returns the changed files when a base was given.
fn nix_checks(run: &mut Run, base: Option<&str>) -> devkit::Result<Option<Vec<String>>> {
    let files = match base {
        Some(base) => Some(run.step("nix selection", |run| {
            Ok(nix::changed_files(&run.root, base)?)
        })?),
        None => None,
    };
    if files
        .as_ref()
        .is_none_or(|files| files.iter().any(|file| nix::is_nix_input(file)))
    {
        nix::check(run)?;
    }
    Ok(files)
}

/// Appends failed and slow tests to the GitHub Actions job summary, so a
/// hosted run's failures and slow tests can be read without its log.
fn write_job_summary(run: &Run) {
    let Some(path) = std::env::var_os("GITHUB_STEP_SUMMARY") else {
        return;
    };
    if run.tests.is_empty() {
        return;
    }
    let mut text = format!(
        "### {} {} {}\n\n",
        run.project,
        run.command,
        run.args.join(" ")
    );
    let failed: Vec<_> = run
        .tests
        .iter()
        .filter(|test| test.status == TestStatus::Failed)
        .collect();
    let ran = run
        .tests
        .iter()
        .filter(|test| matches!(test.status, TestStatus::Passed | TestStatus::Failed))
        .count();
    text += &format!("{ran} tests ran, {} failed.\n\n", failed.len());
    for test in &failed {
        text += &format!("- failed: `{}` `{}`\n", test.binary, test.name);
    }
    let mut slow: Vec<_> = run
        .tests
        .iter()
        .filter(|test| test.duration >= SLOW_TEST)
        .collect();
    slow.sort_by_key(|test| std::cmp::Reverse(test.duration));
    if !slow.is_empty() {
        text += &format!(
            "\nTests that took {} s or longer:\n\n| seconds | binary | test |\n|---:|---|---|\n",
            SLOW_TEST.as_secs()
        );
        for test in slow {
            text += &format!(
                "| {:.1} | `{}` | `{}` |\n",
                test.duration.as_secs_f64(),
                test.binary,
                test.name
            );
        }
    }
    let written = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(&path)
        .and_then(|mut file| std::io::Write::write_all(&mut file, text.as_bytes()));
    if let Err(error) = written {
        eprintln!(
            "job summary {}: {error}",
            std::path::Path::new(&path).display()
        );
    }
}

/// The duration from which a test is listed as slow in the job summary.
const SLOW_TEST: std::time::Duration = std::time::Duration::from_secs(10);

fn run(command: &str, args: &[String], steps: impl FnOnce(&mut Run) -> devkit::Result) -> ExitCode {
    let compiler_cache = CompilerCache::start();
    let mut run = Run::start("polylogue", command, args);
    let _ = steps(&mut run);
    write_job_summary(&run);
    if let Some(cache) = compiler_cache {
        cache.report();
    }
    run.finish()
}

/// The sccache server the development shell routes compilations through,
/// when it does. A run starts its own server, which outlives pauses between
/// compilations, zeroes its counters, and at the end reports how much of the
/// run's compiling the cache answered and stops it. The server's start
/// creates a socket in its temporary directory, so it gets the short runtime
/// directory; the compilations keep the run's.
struct CompilerCache;

impl CompilerCache {
    fn start() -> Option<Self> {
        let wrapper = std::env::var_os("RUSTC_WRAPPER")?;
        if !wrapper.to_string_lossy().contains("sccache") {
            return None;
        }
        let sccache = |args: &[&str]| {
            let mut command = std::process::Command::new("sccache");
            command
                .args(args)
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null());
            command
        };
        // The server `cargo xtask` itself started stops after 10 idle
        // seconds; replace it with one that waits through the test run.
        let _ = sccache(&["--stop-server"]).status();
        let mut start = sccache(&["--start-server"]);
        start.env("SCCACHE_IDLE_TIMEOUT", "0");
        if let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR") {
            start.env("TMPDIR", runtime);
        }
        let _ = start.status();
        let zeroed = sccache(&["--zero-stats"]).status();
        zeroed.is_ok_and(|status| status.success()).then_some(Self)
    }

    fn report(self) {
        let output = std::process::Command::new("sccache")
            .args(["--show-stats", "--stats-format", "json"])
            .output();
        let _ = std::process::Command::new("sccache")
            .arg("--stop-server")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status();
        let Some(stats) = output
            .ok()
            .and_then(|output| serde_json::from_slice::<serde_json::Value>(&output.stdout).ok())
        else {
            return;
        };
        let stats = &stats["stats"];
        let total = |counts: &serde_json::Value| {
            counts["counts"].as_object().map_or(0, |counts| {
                counts.values().filter_map(|n| n.as_u64()).sum::<u64>()
            })
        };
        eprintln!(
            "compiler cache: {} hits, {} misses, {} not cacheable",
            total(&stats["cache_hits"]),
            total(&stats["cache_misses"]),
            stats["requests_not_cacheable"].as_u64().unwrap_or(0)
                + stats["non_cacheable_compilations"].as_u64().unwrap_or(0)
        );
    }
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

fn parse(mut args: Vec<String>) -> Result<Cli, clap::Error> {
    if args.get(1).is_none_or(|command| command != "test") {
        return Cli::try_parse_from(args);
    }
    // clap consumes the `--` that hands what follows to the test binaries;
    // that part passes to nextest as written.
    let after = args
        .iter()
        .position(|arg| arg == "--")
        .map(|separator| args.split_off(separator));
    let mut cli = Cli::try_parse_from(selection::separate_short_values(args))?;
    if let (Cli::Test { selection, .. }, Some(after)) = (&mut cli, after) {
        selection.pass_through(after);
    }
    Ok(cli)
}

fn main() -> ExitCode {
    let cli = parse(std::env::args().collect()).unwrap_or_else(|error| error.exit());
    match cli {
        Cli::Fast => run("fast", &[], |run| {
            fmt_check(run)?;
            workspace_hack(run)?;
            clippy(run)
        }),
        Cli::Check { base } => run("check", &base.arguments(), |run| {
            let reference = base.reference();
            fmt_check(run)?;
            workspace_hack(run)?;
            // A diff confined to the Nix inputs has no Rust to build or test.
            let files = nix_checks(run, reference)?;
            if !files.is_some_and(|files| {
                !files.is_empty() && files.iter().all(|file| nix::is_nix_input(file))
            }) {
                clippy(run)?;
                test(run, reference, &Selection::default())?;
            }
            Ok(())
        }),
        Cli::Warm => run("warm", &[], |run| {
            // A lint failure on master must not leave the test builds cold.
            let tests = build_tests(run);
            clippy(run).and(tests)
        }),
        Cli::Scenarios => run("scenarios", &[], |run| {
            test(
                run,
                None,
                &Selection::of(&["polylogue"], "kind(test)", &[]).with_test_target("e2e"),
            )
        }),
        Cli::Test { no_run: true, .. } => run("test", &["--no-run".into()], build_tests),
        Cli::Test {
            base, selection, ..
        } => {
            let args: Vec<String> = std::env::args().skip(2).collect();
            run("test", &args, |run| test(run, base.reference(), &selection))
        }
        Cli::Nix { base } => run("nix", &base.arguments(), |run| {
            nix_checks(run, base.reference()).map(drop)
        }),
        Cli::PruneBuild => run("prune-build", &[], |run| {
            let target = run.target_dir()?;
            run.step("prune", |run| {
                Ok(prune::prune(
                    &run.root,
                    &target,
                    &[CLIPPY, CLIPPY_STANDALONE, BUILD_TESTS, BUILD_XTASK],
                )?)
            })
        }),
        Cli::Run(options) => exit_code(development::execute(options)),
        Cli::Bench(command) => bench::execute(command),
        Cli::LinkCommon { checkout } => exit_code(common_link::link(checkout)),
        Cli::UnlinkCommon => exit_code(common_link::unlink()),
        Cli::ExtensionProviders => exit_code(extension_providers()),
        Cli::Shared(command) => devkit::cli::execute("polylogue", command),
    }
}

#[cfg(test)]
mod tests;
