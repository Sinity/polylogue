use super::*;

#[test]
fn separate_immutable_exports_keep_older_text_when_new_collections_are_null_or_empty() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    std::fs::create_dir_all(&home).unwrap();
    let source = dir.path().join("projects/p.json");
    std::fs::create_dir_all(source.parent().unwrap()).unwrap();
    let env = [
        ("POLYLOGUE_HOME", home.as_path()),
        ("POLYLOGUE_ARCHIVE", archive.as_path()),
        ("XDG_CONFIG_HOME", config.as_path()),
    ];
    let daemon = start_daemon(&env);
    let context = "document:claude-ai:project:p:doc:d";
    let instruction = "document:claude-ai:project:p:instructions";
    let mut initial = None;
    let mut input = None;
    for (time, body, docs) in [
        (
            "2026-01-01T00:00:00Z",
            "First instruction",
            json!([{"uuid":"d","content":"Exact older retained text λ🙂"}]),
        ),
        (
            "2026-01-02T00:00:00Z",
            "Instruction beside unknown docs",
            json!(null),
        ),
        (
            "2026-01-03T00:00:00Z",
            "Instruction beside known empty docs",
            json!([]),
        ),
    ] {
        std::fs::write(
            &source,
            serde_json::to_vec(
                &json!({"uuid":"p","updated_at":time,"prompt_template":body,"docs":docs}),
            )
            .unwrap(),
        )
        .unwrap();
        let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
        assert!(imported.status.success(), "{imported:?}");
        let out = cli(&env, &["--json", "read", instruction]);
        assert!(out.status.success(), "{out:?}");
        assert_eq!(json(&out)["document"]["blocks"][0]["text"], body);
        let out = cli(&env, &["--json", "read", context]);
        assert!(out.status.success(), "{out:?}");
        let value = json(&out);
        if initial.is_none() {
            initial = Some(value["document"].clone());
            input = Some(value["sources"][0]["input"].clone());
        }
        assert_eq!(&value["document"], initial.as_ref().unwrap());
        assert_eq!(value["sources"][0]["input"], *input.as_ref().unwrap());
        assert_eq!(value["sources"].as_array().unwrap().len(), 1);
    }
    let inputs = cli(
        &env,
        &[
            "--json",
            "sql",
            "SELECT count(*) AS n FROM a.input WHERE format='claude-ai'",
        ],
    );
    assert!(inputs.status.success(), "{inputs:?}");
    assert_eq!(json(&inputs)[0]["n"], 3);
    let status = cli(&env, &["--json", "status"]);
    assert!(status.status.success());
    assert!(
        json(&status)["store"]["unrecognized"]
            .to_string()
            .contains("project_document")
    );
    let basis_sql = "SELECT d.id,r.id AS input,s.chain,s.source_start,s.source_end,hex(s.version_hash) AS version FROM document_source s JOIN document d ON d.pk=s.document_pk JOIN input_ref r ON r.pk=s.input_pk ORDER BY d.id,r.id,s.chain,s.source_start";
    let basis = cli(&env, &["--json", "sql", basis_sql]);
    assert!(basis.status.success(), "{basis:?}");
    let basis = json(&basis);
    assert_eq!(basis.as_array().unwrap().len(), 4);
    let ids_sql = "SELECT id FROM a.input WHERE format='claude-ai' ORDER BY id";
    let ids = cli(&env, &["--json", "sql", ids_sql]);
    assert!(ids.status.success(), "{ids:?}");
    let ids = json(&ids);
    drop(daemon);
    std::fs::remove_file(&source).unwrap();
    polylogue_testkit::delete_index(&archive);
    let _daemon = start_daemon(&env);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let out = cli(&env, &["--json", "status"]);
        assert!(out.status.success(), "{out:?}");
        let status = json(&out);
        if status["store"]["documents"] == 2 && status["store"]["search_pending"] == 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "retained rebuild did not restore both documents: {status}; rows {:?}",
            cli(&env, &["--json", "documents"])
        );
        std::thread::sleep(Duration::from_millis(10));
    }
    let after = cli(&env, &["--json", "sql", basis_sql]);
    assert!(after.status.success(), "{after:?}");
    assert_eq!(json(&after), basis);
    let after = cli(&env, &["--json", "sql", ids_sql]);
    assert!(after.status.success(), "{after:?}");
    assert_eq!(json(&after), ids);
    let out = cli(&env, &["--json", "read", context]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(&json(&out)["document"], initial.as_ref().unwrap());
    let out = cli(&env, &["--json", "read", instruction]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(
        json(&out)["document"]["blocks"][0]["text"],
        "Instruction beside known empty docs"
    );
}
