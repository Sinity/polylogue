use polylogue_model::{DocumentAuthorship, DocumentId, DocumentPurpose};
use polylogue_testkit::ClaudeHome;

fn project() -> Vec<u8> {
    // The native project/docs contract is also exercised directly by the
    // claude.ai sidecar parser; an account conversations file is not required.
    serde_json::to_vec(&serde_json::json!({
        "uuid":"project-a", "name":"Synthetic project",
        "docs":[{"uuid":"doc-a", "filename":"Guide λ🙂",
                 "content":"Supplied context λ🙂", "created_at":"2026-05-02T00:00:00Z"}]
    }))
    .unwrap()
}

#[test]
fn a_project_sidecar_declares_only_its_native_member() {
    let home = ClaudeHome::new();
    let source = home.dir.path().join("project-export");
    std::fs::create_dir_all(source.join("projects")).unwrap();
    let native = source.join("projects/project-a.json");
    std::fs::write(&native, project()).unwrap();
    let unrelated = source.join("projects/unknown.json");
    std::fs::write(&unrelated, br#"{"uuid":"other","unrelated":true}"#).unwrap();
    let outside_layout = source.join("unrelated.json");
    std::fs::write(&outside_layout, project()).unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
    let report = engine
        .import_with_options(
            &[source],
            &crate::ImportOptions {
                explain: true,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert!(report.imported.is_empty());
    let declared = report
        .members
        .iter()
        .find(|member| member.path == native)
        .unwrap();
    assert_eq!(declared.origin.as_deref(), Some("claude-ai"));
    assert_eq!(declared.class, "parsed");
    for path in [unrelated, outside_layout] {
        assert!(
            report.refused.iter().any(|refusal| refusal.path == path),
            "{report:?}"
        );
        let member = report
            .members
            .iter()
            .find(|member| member.path == path)
            .unwrap();
        assert!(member.origin.is_none());
        assert_eq!(member.class, "refused");
    }
    assert_eq!(report.refused.len(), 2, "{report:?}");
    assert_eq!(home.count("SELECT count(*) AS n FROM a.input"), 0);
}

#[test]
fn standalone_project_documents_keep_exact_text_identity_and_receiving_source() {
    let home = ClaudeHome::new();
    let source = home.dir.path().join("project-export");
    std::fs::create_dir_all(source.join("projects")).unwrap();
    let path = source.join("projects/project-a.json");
    let raw = project();
    std::fs::write(&path, &raw).unwrap();
    let (mut engine, _lock) = crate::import::tests::engine(&home, 64);
    let report = engine.import(&[source]).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(report.imported, std::slice::from_ref(&path));
    assert_eq!(report.members.len(), 1);
    assert_eq!(report.members[0].origin.as_deref(), Some("claude-ai"));
    assert_eq!(report.members[0].class, "parsed");
    assert_eq!(report.packages[0].documents, 1);
    assert_eq!(report.packages[0].sessions, 0);
    let id = DocumentId::parse("claude-ai:project:project-a:doc:doc-a").unwrap();
    let reader = engine.reads.reader().unwrap();
    let selected = reader.select_document(&id).unwrap().unwrap();
    let document = selected.document().unwrap();
    assert_eq!(document.id, id);
    assert_eq!(document.purpose, DocumentPurpose::Context);
    assert_eq!(document.authorship, DocumentAuthorship::Unknown);
    assert_eq!(document.title.as_deref(), Some("Guide λ🙂"));
    assert_eq!(
        document
            .blocks
            .iter()
            .filter_map(polylogue_model::Block::text)
            .collect::<String>(),
        "Supplied context λ🙂"
    );
    let sources = selected.sources().unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].input, blake3::hash(&raw).to_hex().as_str());
    assert!(
        sources[0]
            .locations
            .iter()
            .any(|location| location["path"] == path.to_string_lossy().as_ref())
    );
    let supplied: serde_json::Value =
        serde_json::from_slice(&raw[sources[0].start as usize..sources[0].end as usize]).unwrap();
    assert_eq!(supplied["uuid"], "doc-a");
    assert_eq!(supplied["content"], "Supplied context λ🙂");
}
