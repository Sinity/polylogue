use super::*;
use serde_json::json;

use polylogue_model::bundle::Record;
use polylogue_model::read::ReadPage;

/// The report of an import that succeeded. Its inputs are readable when it
/// answers (docs/capture.md, Import), so what follows reads them at once.
pub(super) fn import_report(out: &Output) -> polylogue_engine::ImportReport {
    assert!(out.status.success(), "{out:?}");
    let report: polylogue_engine::ImportReport = serde_json::from_value(json(out)).unwrap();
    assert!(report.refused.is_empty(), "{report:?}");
    assert!(report.pass.errors.is_empty(), "{report:?}");
    assert_eq!(
        report.readable,
        polylogue_engine::Readable::Now,
        "{report:?}"
    );
    report
}
pub(super) fn target(reference: &str) -> String {
    if reference.starts_with("session:")
        || reference.starts_with("document:")
        || reference.starts_with("event:")
    {
        reference.into()
    } else {
        format!("session:{reference}")
    }
}
pub(super) fn reading(env: &crate::Fixture, reference: &str, options: &[&str]) -> ReadPage {
    let target = target(reference);
    let mut args = vec!["--json", target.as_str(), "read"];
    args.extend_from_slice(options);
    wait_caught_up(env);
    let out = cli(env, &args);
    let answer: command_surface::Answer<serde_json::Value> = serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|error| panic!("invalid reading: {error}: {out:?}"));
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    match answer.outcome {
        command_surface::Outcome::Ok(page) => read_page(page),
        other => panic!("a current reading was expected: {other:?}: {out:?}"),
    }
}
pub(super) fn portable(env: &crate::Fixture, reference: &str, path: &Path) -> Vec<Record> {
    std::fs::create_dir_all(path).unwrap();
    let reference = target(reference);
    wait_caught_up(env);
    let out = cli(
        env,
        &[
            "--json",
            &reference,
            "read",
            "--format",
            "jsonl",
            "--to",
            path.to_str().unwrap(),
        ],
    );
    assert_eq!(out.status.code(), Some(0), "{out:?}");
    let records: Vec<Record> = std::fs::read_to_string(path.join("records.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(matches!(records.first(), Some(Record::Manifest(_))));
    for record in &records {
        record.validate().unwrap();
    }
    records
}
pub(super) fn records(env: &crate::Fixture, reference: &str) -> Vec<Record> {
    let output = tempfile::tempdir().unwrap();
    portable(env, reference, output.path())
}
pub(super) fn messages(env: &crate::Fixture, reference: &str) -> Vec<serde_json::Value> {
    let page = reading(env, reference, &[]);
    assert!(!page.document.items.is_empty());
    let mut messages: Vec<_> = records(env, reference)
        .into_iter()
        .filter_map(|record| match record {
            Record::Message(record) => Some((record.order, record.message)),
            _ => None,
        })
        .collect();
    messages.sort_by_key(|(order, _)| *order);
    messages
        .into_iter()
        .map(|(_, message)| serde_json::to_value(message).unwrap())
        .collect()
}
pub(super) fn document(env: &crate::Fixture, reference: &str) -> serde_json::Value {
    let page = reading(env, reference, &[]);
    assert!(!page.document.items.is_empty());
    let documents: Vec<_> = records(env, reference)
        .into_iter()
        .filter_map(|record| match record {
            Record::Document(record) => Some(record.document),
            _ => None,
        })
        .collect();
    assert_eq!(documents.len(), 1);
    serde_json::to_value(&documents[0]).unwrap()
}
pub(super) fn joined_items(
    env: &crate::Fixture,
    reference: &str,
    budget: &str,
) -> Vec<polylogue_model::read::ReadItem> {
    joined_items_with_options(env, reference, budget, &[])
}
pub(super) fn joined_items_with_options(
    env: &crate::Fixture,
    reference: &str,
    budget: &str,
    options: &[&str],
) -> Vec<polylogue_model::read::ReadItem> {
    assert!(!options.contains(&"--budget"));
    let mut whole_options = options.to_vec();
    whole_options.extend(["--budget", "100000"]);
    let whole = reading(env, reference, &whole_options);
    assert!(whole.document.footer.next.is_none());
    let bound = serde_json::to_vec(&whole.document.items).unwrap().len();
    let mut page_options = options.to_vec();
    page_options.extend(["--budget", budget]);
    let mut page = reading(env, reference, &page_options);
    let mut items = super::read_current::Items::default();
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..=bound {
        items.push(&page);
        let Some(next) = page.document.footer.next else {
            return items.into_values();
        };
        assert!(
            seen.insert(serde_json::to_string(&next.request).unwrap()),
            "repeated continuation"
        );
        page = followed_page(env, &next);
    }
    panic!("finite reading did not terminate");
}

#[test]
fn account_and_project_text_import_without_turns_or_invented_document_authorship() {
    let dir = tempfile::tempdir().unwrap();
    let home = dir.path().join("home");
    let source = dir.path().join("export");
    std::fs::create_dir_all(&home).unwrap();
    std::fs::create_dir_all(source.join("projects")).unwrap();
    let memories = json!([{"account_uuid":"account-a","conversations_memory":"Provider account summary.","project_memories":{"project-a":"Provider project summary."}}]);
    let project = json!({"uuid":"project-a","name":"Synthetic knowledge","description":"Supplied project context","is_starter_project":false,"prompt_template":"Use supplied records.","created_at":"2026-05-01T00:00:00Z","docs":[{"uuid":"doc-a","filename":"manual.md","content":"Vendor document λ🙂","created_at":"2026-05-02T00:00:00Z"}]});
    for (name, value) in [
        ("memories.json", memories),
        ("projects/project-a.json", project),
    ] {
        std::fs::write(source.join(name), serde_json::to_vec(&value).unwrap()).unwrap();
    }
    std::fs::write(source.join("conversations.json"),r#"[{"uuid":"old-conversation","name":"Existing conversation","chat_messages":[{"uuid":"old-message","sender":"human","text":"Existing conversation text."}]}]"#).unwrap();
    let archive = dir.path().join("archive");
    let config = dir.path().join("config");
    let env = crate::Fixture::new(archive.as_path(), home.as_path(), config.as_path());
    let daemon = start_daemon(&env);
    let imported = cli(&env, &["--json", "import", source.to_str().unwrap()]);
    let _ = import_report(&imported);
    let old = messages(&env, "claude-ai:old-conversation");
    assert_eq!(old[0]["blocks"][0]["text"], "Existing conversation text.");
    let documents = cli(
        &env,
        &[
            "--json",
            "documents where origin:claude-ai \"Vendor document\"",
        ],
    );
    assert!(documents.status.success(), "{documents:?}");
    assert_eq!(
        listing(json(&documents))["count"],
        1,
        "project text must be queryable without pretending it is a human prompt"
    );
    let authored = cli(
        &env,
        &[
            "--json",
            "documents where origin:claude-ai authorship:human \"Vendor document\"",
        ],
    );
    assert_eq!(authored.status.code(), Some(1), "{authored:?}");
    let answer: command_surface::Answer<serde_json::Value> =
        serde_json::from_slice(&authored.stdout).unwrap();
    let command_surface::Answer {
        outcome: command_surface::Outcome::Empty(value),
        ..
    } = answer
    else {
        panic!("expected typed empty query: {answer:?}");
    };
    assert_eq!(listing(value)["count"], 0);
    let reference = "document:claude-ai:project:project-a:doc:doc-a";
    let read = cli(&env, &["--json", reference, "read"]);
    assert!(read.status.success(), "{read:?}");
    let before = document(&env, reference);
    assert_eq!(before["blocks"][0]["text"], "Vendor document λ🙂");
    assert_eq!(before["authorship"], "unknown");
    assert_eq!(before["source_time"], "2026-05-02T00:00:00Z");
    assert_eq!(
        records(&env, reference)
            .iter()
            .filter(|r| matches!(r, Record::Document(_)))
            .count(),
        1
    );
    let sources = cli(
        &env,
        &[
            "--json",
            "dev",
            "sql",
            "SELECT count(*) AS n FROM document_parse s WHERE s.document_id='claude-ai:project:project-a:doc:doc-a'",
        ],
    );
    assert!(sources.status.success(), "{sources:?}");
    assert_eq!(json(&sources)[0]["n"], 1);
    let whole = reading(&env, reference, &["--budget", "100000"]);
    assert_eq!(joined_items(&env, reference, "3"), whole.document.items);
    assert!(
        serde_json::to_string(&whole)
            .unwrap()
            .contains("Vendor document λ🙂")
    );
    for format in ["markdown", "html", "terminal"] {
        let destination = dir.path().join(format!("reading.{format}"));
        let out = cli(
            &env,
            &[
                "--json",
                reference,
                "read",
                "--format",
                format,
                "--to",
                destination.to_str().unwrap(),
            ],
        );
        assert_eq!(out.status.code(), Some(0), "{out:?}");
        assert!(
            std::fs::read_to_string(destination)
                .unwrap()
                .contains("Vendor document λ🙂")
        );
    }
    let copy = source.join("portable-docs");
    let exported = portable(&env, reference, &copy);
    let exported_documents: Vec<_> = exported
        .iter()
        .filter_map(|record| match record {
            Record::Document(record) => Some(serde_json::to_value(&record.document).unwrap()),
            _ => None,
        })
        .collect();
    assert_eq!(exported_documents.as_slice(), std::slice::from_ref(&before));
    assert!(
        !exported
            .iter()
            .any(|record| matches!(record, Record::Session(_)))
    );
    let imported = cli(&env, &["--json", "import", copy.to_str().unwrap()]);
    let _ = import_report(&imported);
    assert_eq!(document(&env, reference), before);
    let note = cli(
        &env,
        &["--json", reference, "note", "Synthetic document note"],
    );
    assert!(note.status.success(), "{note:?}");
    drop(daemon);
    // Rebuild this test archive from its stored bytes after the source file is gone.
    std::fs::remove_dir_all(&source).unwrap();
    polylogue_testkit::delete_index(&archive);
    let daemon = start_daemon(&env);
    wait_status(&env, |s| s["store"]["documents"].as_u64() == Some(4));
    let after = cli(&env, &["--json", reference, "read"]);
    assert!(after.status.success(), "{after:?}");
    assert_eq!(document(&env, reference), before);
    drop(daemon);
}

mod pages;

mod history;
