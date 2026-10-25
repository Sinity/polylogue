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
