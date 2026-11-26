use super::*;

#[test]
fn names_are_unique_and_valid_origins() {
    let mut names: Vec<_> = ORIGINS.iter().map(|o| o.name).collect();
    for n in &names {
        polylogue_model::Origin::new(n).unwrap();
    }
    names.sort();
    names.dedup();
    assert_eq!(names.len(), ORIGINS.len());
}

#[test]
fn imports_have_no_live_locations() {
    for o in ORIGINS.iter().filter(|o| o.arrival == Arrival::Import) {
        assert!(o.locations.is_empty(), "{}", o.name);
    }
}

#[test]
fn every_spool_lies_in_polylogue_s_state_directory() {
    let spools: Vec<(&str, PathBuf)> = ORIGINS
        .iter()
        .filter_map(|o| Some((o.name, o.spool_dir()?)))
        .collect();
    assert_eq!(
        spools,
        [
            (
                "antigravity-exports",
                PathBuf::from("polylogue/antigravity")
            ),
            (
                "polylogue-repo-lookup",
                PathBuf::from("polylogue/repo-lookup")
            ),
            ("hook-spool", PathBuf::from("polylogue/hooks")),
            (
                "browser-capture",
                PathBuf::from("polylogue/browser-capture")
            ),
            ("ai-studio", PathBuf::from("polylogue/drive/ai-studio")),
        ]
    );
    for o in ORIGINS.iter().filter(|o| o.arrival == Arrival::Spool) {
        assert_eq!(o.root, Root::STATE, "{}", o.name);
        assert!(o.spool_dir().unwrap().starts_with(STATE_DIR), "{}", o.name);
    }
}

#[test]
fn only_session_tools_resume_and_take_bare_ids() {
    let resumes: Vec<&str> = ORIGINS
        .iter()
        .filter(|o| o.resume.is_some())
        .map(|o| o.name)
        .collect();
    assert_eq!(resumes, ["claude-code", "codex", "gemini-cli"]);
    let bare: Vec<&str> = ORIGINS
        .iter()
        .filter(|o| o.bare_uuid_ids)
        .map(|o| o.name)
        .collect();
    assert_eq!(bare, ["claude-code", "codex"]);
}

#[test]
fn capture_only_database_reading_tracks_metadata_parser_and_shared_readers() {
    for name in ["llm", "tool-database"] {
        let origin = by_name(name).unwrap();
        assert_eq!(origin.support, Support::NoParser);
        let reading = origin.parse_version();
        assert!(reading.contains(&format!(
            "crate::row_dump::VERSION={}",
            crate::row_dump::VERSION
        )));
        assert!(reading.contains(&format!("fields={}", crate::fields::VERSION)));
        assert!(reading.contains(&format!("parse={}", crate::parse::VERSION)));
    }
    assert!(by_name("tool-context").unwrap().parse_version().is_empty());
}

/// The extension's checked-in provider table is the registry's current output;
/// `cargo xtask extension-providers` rewrites it.
#[test]
fn extension_origin_table_is_current() {
    assert_eq!(
        include_str!("../../../../extension/src/background/providers.js"),
        extension_providers_module(),
        "run `cargo xtask extension-providers`"
    );
}

#[test]
fn a_synthetic_home_moves_every_declared_root_under_it() {
    let home = Path::new("/home/op");
    let environment = synthetic_home_environment(home);
    assert_eq!(
        environment,
        [
            ("HOME", PathBuf::from("/home/op")),
            ("CLAUDE_CONFIG_DIR", PathBuf::from("/home/op/.claude")),
            ("CODEX_HOME", PathBuf::from("/home/op/.codex")),
            ("HERMES_HOME", PathBuf::from("/home/op/.hermes")),
            ("XDG_STATE_HOME", PathBuf::from("/home/op/.local/state")),
        ]
    );
}

#[test]
fn parser_versions_include_referenced_modules() {
    use std::collections::BTreeSet;

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let opens: Vec<_> = include_str!("../origin.rs")
        .lines()
        .filter_map(|line| line.trim().strip_prefix("open: "))
        .collect();
    assert_eq!(opens.len(), ORIGINS.len());
    for (origin, open) in ORIGINS.iter().zip(opens) {
        let version = origin.parse_version();
        let declared: BTreeSet<_> = version
            .split(',')
            .filter_map(|entry| entry.split_once('='))
            .map(|(name, _)| {
                name.strip_prefix("crate::")
                    .or_else(|| name.strip_prefix("polylogue_sources::"))
                    .unwrap_or(name)
                    .trim_end_matches("::VERSION")
                    .to_owned()
            })
            .collect();
        if let Some(open) = open.strip_prefix("crate::") {
            let (module, _) = open.rsplit_once("::").unwrap();
            assert!(
                declared.contains(module),
                "{}: parser version is missing module {module}",
                origin.name
            );
        } else {
            assert_eq!(
                open, "opens_nothing,",
                "{}: read the parser's module path",
                origin.name
            );
        }
        let mut files: Vec<_> = declared
            .iter()
            .filter(|name| !name.starts_with("polylogue_model::") && *name != "command")
            .map(|name| root.join(name.replace("::", "/")).with_extension("rs"))
            .collect();
        let mut read = BTreeSet::new();
        while let Some(path) = files.pop() {
            if !read.insert(path.clone()) {
                continue;
            }
            let source = std::fs::read_to_string(&path).unwrap();
            let mut test_only = false;
            let mut references = String::new();
            for line in source.lines().map(str::trim) {
                if line == "#[cfg(test)]" {
                    test_only = true;
                    continue;
                }
                if std::mem::take(&mut test_only) {
                    continue;
                }
                if line.starts_with("//") {
                    continue;
                }
                // A declaration outside the macro must not omit a child.
                if let Some(child) = line.strip_prefix("mod ").and_then(|s| s.strip_suffix(';'))
                    && child != "tests"
                {
                    let child_path = path.with_extension("").join(child);
                    let module = child_path
                        .strip_prefix(&root)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .replace('/', "::");
                    assert!(
                        declared.contains(&module),
                        "{}: parser version is missing module {module}",
                        origin.name
                    );
                    files.push(child_path.with_extension("rs"));
                }
                references.push_str(line);
            }
            for child in source
                .lines()
                .filter_map(|line| line.trim().strip_prefix("pub mod ")?.strip_suffix(';'))
            {
                if references.contains(&format!("{child}::")) {
                    let child_path = path.with_extension("").join(child);
                    let module = child_path
                        .strip_prefix(&root)
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .replace('/', "::");
                    assert!(
                        declared.contains(&module),
                        "{}: parser version is missing module {module}",
                        origin.name
                    );
                    files.push(child_path.with_extension("rs"));
                }
            }
            for reference in references.split("crate::").skip(1) {
                let names = reference.strip_prefix('{').map_or_else(
                    || vec![reference],
                    |group| group.split('}').next().unwrap().split(',').collect(),
                );
                for reference in names {
                    let reference = reference.trim();
                    let module: String = reference
                        .chars()
                        .take_while(|c| c.is_ascii_lowercase() || *c == '_')
                        .collect();
                    // Metadata readers use only the shared line interface.
                    if origin.support != Support::Parsed
                        && module == "export"
                        && path == root.join("parse.rs")
                    {
                        continue;
                    }
                    // The registry describes origins, rather than parsing rows.
                    if module.is_empty()
                        || module == "origin"
                        || !root.join(&module).with_extension("rs").exists()
                    {
                        continue;
                    }
                    assert!(
                        declared.contains(&module),
                        "{}: parser version is missing module {module} (referenced by {})",
                        origin.name,
                        path.strip_prefix(&root).unwrap().display()
                    );
                }
            }
        }
    }
}
