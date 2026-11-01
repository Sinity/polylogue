use super::*;
use std::io::Write;

#[test]
fn provider_zip_and_directory_shapes_import_and_rebuild_from_stored_bytes() {
    for zipped in [false, true] {
        for provider in ["chatgpt", "claude-ai"] {
            let w = ClaudeHome::new();
            let mut members = Vec::new();
            if provider == "chatgpt" {
                // The first key in the April sharded export precedes the
                // conversation ID and mapping by hundreds of bytes.
                let body = format!(
                    "[{{\"async_status\":null,\"atlas_mode_enabled\":false,\"context_scopes\":[],\"conversation_id\":\"one\",\"mapping\":{}}}]",
                    conversation("one", 1, &[("u1", "First")])["mapping"]
                );
                members.push(("export/conversations-000.json", body));
                members.push((
                    "export/conversations-001.json",
                    format!("[{}]", conversation("two", 2, &[("u2", "Second")])),
                ));
            } else {
                members.push(("export/conversations.json", r#"[{"uuid":"one","name":"Example","summary":"","chat_messages":[{"uuid":"u1","sender":"human","text":"First"}]}]"#.to_owned()));
                members.push(("export/projects.json", "[]".to_owned()));
            }
            let source = if zipped {
                let path = w.dir.path().join("provider.zip");
                let mut zip = zip::ZipWriter::new(std::fs::File::create(&path).unwrap());
                for (name, body) in &members {
                    zip.start_file(*name, zip::write::SimpleFileOptions::default())
                        .unwrap();
                    zip.write_all(body.as_bytes()).unwrap();
                }
                zip.finish().unwrap();
                path
            } else {
                let directory = w.dir.path().join("provider");
                for (name, body) in &members {
                    let path = directory.join(name);
                    fs::create_dir_all(path.parent().unwrap()).unwrap();
                    fs::write(path, body).unwrap();
                }
                directory
            };
            {
                let (mut e, _lock) = engine(&w, 64);
                let report = e.import(std::slice::from_ref(&source)).unwrap();
                assert!(report.pass.errors.is_empty(), "{:?}", report.pass.errors);
                assert_eq!(texts(&w, &format!("{provider}:one")), ["First"]);
                if provider == "chatgpt" {
                    assert_eq!(texts(&w, "chatgpt:two"), ["Second"]);
                }
                if zipped {
                    assert_eq!(
                        w.count("SELECT count(*) AS n FROM a.observation_zip"),
                        members.len() as u64
                    );
                }
            }
            if zipped {
                fs::remove_file(&source).unwrap();
            } else {
                fs::remove_dir_all(&source).unwrap();
            }
            polylogue_testkit::delete_index(&w.root);
            let (mut e, _lock) = engine(&w, 64);
            let pass = e.run_pass().unwrap();
            assert!(pass.errors.is_empty(), "{:?}", pass.errors);
            assert_eq!(texts(&w, &format!("{provider}:one")), ["First"]);
        }
    }
}
