//! What a test command selects: the base its changes are measured from, the
//! packages, exclusions and filtersets that choose which tests run, the Cargo
//! target flags that bound compilation, and a partition of the result.
//!
//! A run builds only the selected packages' test targets and what they
//! depend on. The units are the ones the workspace's test build compiles:
//! the workspace-hack crate gives every third-party crate the workspace's
//! features whatever is selected, and each workspace package gets its test
//! features (`FEATURES`) through the selected packages, their direct
//! dependencies, and the features those forward.

use crate::FEATURES;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The changes a run selects tests from; neither flag runs every test.
#[derive(clap::Args, Debug, Default, Clone)]
pub(crate) struct Base {
    /// Select the tests the changes since the merge base with origin/master
    /// can affect.
    #[arg(long, conflicts_with = "base")]
    changed: bool,
    /// Select the tests the changes since the merge base with REF can affect.
    #[arg(long, value_name = "REF")]
    base: Option<String>,
}

impl Base {
    pub(crate) fn reference(&self) -> Option<&str> {
        match (&self.base, self.changed) {
            (Some(base), _) => Some(base),
            (None, true) => Some("origin/master"),
            (None, false) => None,
        }
    }

    pub(crate) fn arguments(&self) -> Vec<String> {
        match (&self.base, self.changed) {
            (Some(base), _) => vec!["--base".into(), base.clone()],
            (None, true) => vec!["--changed".into()],
            (None, false) => Vec::new(),
        }
    }
}

/// Cargo's target flags. Without any, the targets marked `test = true` build.
#[derive(clap::Args, Debug, Default, Clone)]
pub(crate) struct Targets {
    /// Library unit tests.
    #[arg(long)]
    lib: bool,
    /// Every binary's unit tests.
    #[arg(long)]
    bins: bool,
    /// The named binary's unit tests.
    #[arg(long, value_name = "NAME")]
    bin: Vec<String>,
    /// Every target marked `test = true` (the default).
    #[arg(long)]
    tests: bool,
    /// The named integration test target.
    #[arg(long, value_name = "NAME")]
    test: Vec<String>,
    #[arg(long)]
    examples: bool,
    #[arg(long, value_name = "NAME")]
    example: Vec<String>,
    #[arg(long)]
    benches: bool,
    #[arg(long, value_name = "NAME")]
    bench: Vec<String>,
    #[arg(long)]
    all_targets: bool,
}

impl Targets {
    fn arguments(&self) -> Vec<String> {
        let mut arguments = Vec::new();
        for (set, flag) in [
            (self.lib, "--lib"),
            (self.bins, "--bins"),
            (self.tests, "--tests"),
            (self.examples, "--examples"),
            (self.benches, "--benches"),
            (self.all_targets, "--all-targets"),
        ] {
            if set {
                arguments.push(flag.to_owned());
            }
        }
        for (names, flag) in [
            (&self.bin, "--bin"),
            (&self.test, "--test"),
            (&self.example, "--example"),
            (&self.bench, "--bench"),
        ] {
            for name in names {
                arguments.extend([flag.to_owned(), name.clone()]);
            }
        }
        if arguments.is_empty() {
            arguments.push("--tests".into());
        }
        arguments
    }
}

/// Which tests run. Packages, exclusions and filtersets intersect; the
/// arguments xtask does not interpret follow and pass to nextest unchanged,
/// so xtask's own options come first.
#[derive(clap::Args, Debug, Default, Clone)]
pub(crate) struct Selection {
    /// Run this package's tests (a name, or a glob such as `polylogue-*`).
    #[arg(short = 'p', long = "package", value_name = "SPEC")]
    packages: Vec<String>,
    /// Leave out this package's tests.
    #[arg(long, value_name = "SPEC")]
    exclude: Vec<String>,
    /// Run the tests this nextest filterset matches; several are alternatives.
    #[arg(
        short = 'E',
        long = "filterset",
        visible_alias = "filter-expr",
        value_name = "EXPR"
    )]
    filtersets: Vec<String>,
    /// Run one part of the selected tests: nextest's `hash:K/N`, `slice:K/N`
    /// or `count:K/N`. A part that holds no tests passes.
    #[arg(long, value_name = "PART")]
    partition: Option<String>,
    #[command(flatten)]
    targets: Targets,
    /// Cargo's spelling of the default: every package's tests.
    #[arg(long, visible_alias = "all", hide = true)]
    workspace: bool,
    /// Refused: every selection builds with the workspace's features.
    #[arg(short = 'F', long, hide = true)]
    features: Vec<String>,
    #[arg(long, hide = true)]
    all_features: bool,
    #[arg(long, hide = true)]
    no_default_features: bool,
    /// Test-name filters and further `cargo nextest run` options.
    #[arg(
        trailing_var_arg = true,
        allow_hyphen_values = true,
        value_name = "NEXTEST"
    )]
    rest: Vec<String>,
}

impl Selection {
    /// A selection made in code: packages, a filterset and run options.
    pub(crate) fn of(packages: &[&str], filterset: &str, rest: &[&str]) -> Self {
        Self {
            packages: packages.iter().map(|name| (*name).to_owned()).collect(),
            filtersets: vec![filterset.to_owned()],
            rest: rest.iter().map(|arg| (*arg).to_owned()).collect(),
            ..Self::default()
        }
    }

    /// One named integration test target of one package.
    pub(crate) fn with_test_target(mut self, name: &str) -> Self {
        self.targets.test.push(name.to_owned());
        self
    }

    pub(crate) fn is_partitioned(&self) -> bool {
        self.partition.is_some()
    }

    /// The packages whose tests run, or `None` for every package. `changed`
    /// holds the packages the changes since a base select, when one is given.
    fn packages(
        &self,
        workspace: &Workspace,
        changed: Option<&BTreeSet<String>>,
    ) -> Result<Option<BTreeSet<String>>, String> {
        let mut selected = if self.packages.is_empty() {
            workspace.members.keys().cloned().collect()
        } else {
            let mut selected = BTreeSet::new();
            for spec in &self.packages {
                selected.extend(workspace.matching(spec)?);
            }
            selected
        };
        for spec in &self.exclude {
            for name in workspace.matching(spec)? {
                selected.remove(&name);
            }
        }
        if let Some(changed) = changed {
            selected.retain(|name| changed.contains(name));
        }
        Ok((selected.len() < workspace.members.len()).then_some(selected))
    }

    /// The arguments of `cargo nextest run`, or `None` when the selection
    /// holds no package. `base` reaches devkit, which narrows the changed
    /// packages' tests to the changed test functions where it can;
    /// `changed` is the package selection of the same base.
    pub(crate) fn nextest_arguments(
        &self,
        workspace: &Workspace,
        base: Option<&str>,
        changed: Option<&BTreeSet<String>>,
    ) -> Result<Option<Vec<String>>, String> {
        if !self.features.is_empty() || self.all_features || self.no_default_features {
            return Err(format!(
                "feature flags are refused: all test selections use {FEATURES}"
            ));
        }
        let mut arguments = vec!["--locked".to_owned()];
        match self.packages(workspace, changed)? {
            None => arguments.extend(["--workspace", "--features", FEATURES].map(String::from)),
            Some(packages) if packages.is_empty() => return Ok(None),
            Some(packages) => {
                let features = workspace.test_features(&packages);
                for package in packages {
                    arguments.extend(["-p".to_owned(), package]);
                }
                if !features.is_empty() {
                    arguments.extend(["--features".to_owned(), features.join(",")]);
                }
            }
        }
        if !self.filtersets.is_empty() {
            arguments.push("-E".to_owned());
            arguments.push(
                self.filtersets
                    .iter()
                    .map(|set| format!("({set})"))
                    .collect::<Vec<_>>()
                    .join(" | "),
            );
        }
        arguments.extend(self.targets.arguments());
        if let Some(base) = base {
            arguments.push(format!("--changed-base={base}"));
        }
        if let Some(partition) = &self.partition {
            arguments.extend(["--partition".to_owned(), partition.clone()]);
        }
        arguments.extend(self.rest.iter().cloned());
        Ok(Some(arguments))
    }
}

/// The workspace's packages and the workspace packages each one depends on
/// directly, through any kind of dependency.
pub(crate) struct Workspace {
    members: BTreeMap<String, BTreeSet<String>>,
}

impl Workspace {
    pub(crate) fn load(root: &Path) -> Result<Self, String> {
        let output = devkit::cargo_command()
            .args(["metadata", "--locked", "--format-version", "1", "--no-deps"])
            .current_dir(root)
            .output()
            .map_err(|error| format!("cargo metadata: {error}"))?;
        if !output.status.success() {
            return Err(String::from_utf8_lossy(&output.stderr).into_owned());
        }
        let metadata: serde_json::Value = serde_json::from_slice(&output.stdout)
            .map_err(|error| format!("cargo metadata: {error}"))?;
        let packages = metadata["packages"]
            .as_array()
            .ok_or("cargo metadata has no packages")?;
        let names: BTreeSet<&str> = packages
            .iter()
            .filter_map(|package| package["name"].as_str())
            .collect();
        let members = packages
            .iter()
            .filter_map(|package| {
                let dependencies = package["dependencies"]
                    .as_array()?
                    .iter()
                    .filter_map(|dependency| dependency["name"].as_str())
                    .filter(|name| names.contains(name))
                    .map(str::to_owned)
                    .collect();
                Some((package["name"].as_str()?.to_owned(), dependencies))
            })
            .collect();
        Ok(Self { members })
    }

    #[cfg(test)]
    pub(crate) fn of(members: &[(&str, &[&str])]) -> Self {
        Self {
            members: members
                .iter()
                .map(|(name, dependencies)| {
                    (
                        (*name).to_owned(),
                        dependencies.iter().map(|name| (*name).to_owned()).collect(),
                    )
                })
                .collect(),
        }
    }

    /// The members a package specification names: a name, or a glob of `*`
    /// and `?`.
    fn matching(&self, spec: &str) -> Result<Vec<String>, String> {
        let spec = spec.strip_prefix('=').unwrap_or(spec);
        let matching: Vec<String> = self
            .members
            .keys()
            .filter(|name| glob_matches(spec.as_bytes(), name.as_bytes()))
            .cloned()
            .collect();
        if matching.is_empty() {
            return Err(format!("no workspace package matches `{spec}`"));
        }
        Ok(matching)
    }

    /// The entries of `FEATURES` that Cargo accepts for a build of these
    /// packages: their own, and those of their direct dependencies.
    fn test_features(&self, packages: &BTreeSet<String>) -> Vec<&'static str> {
        let reachable: BTreeSet<&str> = packages
            .iter()
            .flat_map(|name| {
                std::iter::once(name.as_str()).chain(self.members[name].iter().map(String::as_str))
            })
            .collect();
        FEATURES
            .split(',')
            .filter(|feature| {
                feature
                    .split_once('/')
                    .is_some_and(|(package, _)| reachable.contains(package))
            })
            .collect()
    }

    /// The packages that depend on any of these, with these.
    fn with_dependents(&self, packages: impl IntoIterator<Item = String>) -> BTreeSet<String> {
        let mut selected = BTreeSet::new();
        let mut pending: Vec<String> = packages.into_iter().collect();
        while let Some(package) = pending.pop() {
            if selected.insert(package.clone()) {
                pending.extend(
                    self.members
                        .iter()
                        .filter(|(_, dependencies)| dependencies.contains(&package))
                        .map(|(name, _)| name.clone()),
                );
            }
        }
        selected
    }
}

fn glob_matches(pattern: &[u8], name: &[u8]) -> bool {
    match (pattern.split_first(), name.split_first()) {
        (None, None) => true,
        (Some((b'*', rest)), _) => {
            glob_matches(rest, name) || (!name.is_empty() && glob_matches(pattern, &name[1..]))
        }
        (Some((b'?', rest)), Some((_, name))) => glob_matches(rest, name),
        (Some((expected, rest)), Some((actual, name))) => {
            expected == actual && glob_matches(rest, name)
        }
        _ => false,
    }
}

/// Spells `-pNAME`, `-EEXPR` and `-FFEATURES` as two words. With pass-through
/// arguments allowed, clap reads a short option with an attached value as
/// the first argument it does not interpret.
pub(crate) fn separate_short_values(args: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut separated = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg == "--" {
            separated.push(arg);
            separated.extend(args);
            break;
        }
        match ["-p", "-E", "-F"].iter().find_map(|flag| {
            Some((
                *flag,
                arg.strip_prefix(flag).filter(|value| !value.is_empty())?,
            ))
        }) {
            Some((flag, value)) => {
                separated.push(flag.to_owned());
                separated.push(value.strip_prefix('=').unwrap_or(value).to_owned());
            }
            None => separated.push(arg),
        }
    }
    separated
}

/// The workspace packages whose tests the changes since `base` select, or
/// `None` for every package: the changed packages and their reverse
/// dependencies, as devkit selects them.
pub(crate) fn changed_packages(
    root: &Path,
    workspace: &Workspace,
    base: &str,
) -> Result<Option<BTreeSet<String>>, String> {
    use devkit::select::Selection as Changed;
    let (changed, binaries) = match devkit::select::changed(root, base)? {
        Changed::Workspace => return Ok(None),
        Changed::Nothing => return Ok(Some(BTreeSet::new())),
        Changed::Packages(packages) => (packages, Vec::new()),
        Changed::Tests { packages, binaries } => (
            packages,
            binaries.into_iter().map(|(package, _)| package).collect(),
        ),
        // devkit narrows a change confined to test functions to a filterset
        // of `package(=P) & ...` and `rdeps(=P)` terms.
        Changed::Filter(filter) => (
            names_after(&filter, "rdeps(="),
            names_after(&filter, "package(="),
        ),
    };
    let mut selected = workspace.with_dependents(changed);
    selected.extend(binaries);
    Ok(Some(selected))
}

fn names_after(filter: &str, prefix: &str) -> Vec<String> {
    filter
        .split(prefix)
        .skip(1)
        .filter_map(|rest| rest.split_once(')').map(|(name, _)| name.to_owned()))
        .collect()
}
