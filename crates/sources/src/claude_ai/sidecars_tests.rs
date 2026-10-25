use super::*;
use polylogue_model::{DocumentAuthorship, DocumentPurpose};
use serde_json::json;

fn parse_bound(parser: &mut Parse, input: &str, bytes: &[u8]) -> crate::parse::Step {
    crate::export::fixture::parse_input(parser, input, bytes).unwrap()
}

#[test]
fn memory_and_project_text_are_documents_with_unknown_authorship_and_exact_sources() {
    let bytes=br#"[{"project_memories":{"p":"project memory"},"conversations_memory":"account memory","account_uuid":"a"}]"#;
    let mut parser = open(Path::new("memories.json")).unwrap();
    let step = parse_bound(&mut parser, "input-a", bytes);
    assert!(
        step.output.sessions.is_empty()
            && step.output.messages.is_empty()
            && step.output.events.is_empty()
    );
    assert_eq!(step.output.documents.len(), 2);
    for row in &step.output.documents {
        assert_eq!(row.document.purpose, DocumentPurpose::Memory);
        assert_eq!(row.document.authorship, DocumentAuthorship::Unknown);
        assert!(row.document.source_time.is_none());
        let raw: &str =
            serde_json::from_slice(&bytes[row.offset as usize..row.end as usize]).unwrap();
        assert_eq!(row.document.blocks, [Block::Text { text: raw.into() }]);
        assert!(!row.document.metadata.to_string().contains("account memory"));
        assert!(!row.document.metadata.to_string().contains("project memory"));
    }
    let project = json!({"uuid":"p","name":"Native project","is_starter_project":true,"prompt_template":"Instruction λ🙂","created_at":"2026-05-01T00:00:00Z","docs":[{"uuid":"d","filename":"manual.md","content":"Supplied context","created_at":"2026-05-02T00:00:00Z"}]});
    let bytes = serde_json::to_vec(&project).unwrap();
    let mut parser = open(Path::new("projects/p.json")).unwrap();
    let step = parse_bound(&mut parser, "project-input", &bytes);
    assert_eq!(step.output.documents.len(), 2);
    assert!(
        step.output
            .documents
            .iter()
            .all(|r| r.document.authorship == DocumentAuthorship::Unknown)
    );
    let instruction = step
        .output
        .documents
        .iter()
        .find(|r| r.document.purpose == DocumentPurpose::Instruction)
        .unwrap();
    assert_eq!(
        serde_json::from_slice::<String>(
            &bytes[instruction.offset as usize..instruction.end as usize]
        )
        .unwrap(),
        "Instruction λ🙂"
    );
    let doc = step
        .output
        .documents
        .iter()
        .find(|r| r.document.purpose == DocumentPurpose::Context)
        .unwrap();
    let original: Json =
        serde_json::from_slice(&bytes[doc.offset as usize..doc.end as usize]).unwrap();
    assert_eq!(original["uuid"], "d");
    assert_eq!(doc.document.title.as_deref(), Some("manual.md"));
    assert_eq!(doc.document.metadata["text_kind"], "extracted_text");
}

#[test]
fn reused_parser_binds_each_input_and_missing_ids_use_physical_source_coordinates() {
    let original = br#"[{"conversations_memory":"first"}, {"conversations_memory":"second"}]"#;
    let mut parser = open(Path::new("memories.json")).unwrap();
    let a = parse_bound(&mut parser, "input-a", original);
    let b = parse_bound(&mut parser, "input-b", original);
    assert_ne!(
        a.output.documents[0].document.id,
        b.output.documents[0].document.id
    );
    let first_end = original.iter().position(|b| *b == b'}').unwrap() + 1;
    let comma = original[first_end..]
        .iter()
        .position(|b| *b == b',')
        .unwrap()
        + first_end;
    let mut erased = original.to_vec();
    for byte in &mut erased[1..=comma] {
        *byte = b' ';
    }
    let after = parse_bound(&mut parser, "input-a", &erased);
    assert_eq!(after.output.documents.len(), 1);
    assert_eq!(
        after.output.documents[0].document.id,
        a.output.documents[1].document.id
    );
    assert!(
        a.output.documents[0]
            .document
            .id
            .native()
            .as_str()
            .contains("~input:")
    );
    assert!(crate::export::fixture::parse_input(&mut parser, "", original).is_err());
}

#[test]
fn empty_text_is_a_version_but_absent_null_and_binary_references_are_not_invented_text() {
    let bytes=br#"{"uuid":"p","prompt_template":"","docs":[{"uuid":"empty","content":""},{"uuid":"null","content":null},{"uuid":"reference","filename":"unresolved.pdf","url":"https://invalid.example/unresolved.pdf"},{"uuid":"unknown","content":7,"future":false}]}"#;
    let mut parser = open(Path::new("projects/p.json")).unwrap();
    let step = parse_bound(&mut parser, "input", bytes);
    assert_eq!(step.output.documents.len(), 2);
    assert!(step.output.documents.iter().all(|r| r.document.blocks
        == [Block::Text {
            text: String::new()
        }]));
    assert!(!step.counts.skipped.is_empty());
    assert!(!step.counts.unread_fields.is_empty());
}

#[test]
fn native_ids_survive_reordering_and_edits_and_equal_text_remains_distinct_records() {
    let project = json!({"uuid":"p","docs":[{"uuid":"one","filename":"a.txt","content":"same"},{"uuid":"two","filename":"b.txt","content":"same"}]});
    let mut parser = open(Path::new("projects/p.json")).unwrap();
    let a = parse_bound(&mut parser, "a", &serde_json::to_vec(&project).unwrap());
    let mut changed = project;
    changed["docs"].as_array_mut().unwrap().reverse();
    changed["docs"][0]["content"] = json!("edited");
    let b = parse_bound(&mut parser, "b", &serde_json::to_vec(&changed).unwrap());
    assert_ne!(
        a.output.documents[0].document.id,
        a.output.documents[1].document.id
    );
    assert_eq!(
        a.output.documents[0].document.id,
        b.output.documents[1].document.id
    );
    assert_eq!(
        a.output.documents[1].document.id,
        b.output.documents[0].document.id
    );
}

#[test]
fn project_arrays_frame_one_document_and_resolve_metadata_that_follows_them() {
    let docs:Vec<_>=(0..256).map(|i|json!({"uuid":format!("d{i}"),"filename":format!("doc-{i}.md"),"content":"supplied body λ🙂 ".repeat(32)})).collect();
    let bytes=format!("  {{\"docs\":{},\"prompt_template\":\"instruction\",\"uuid\":\"late-project\",\"name\":\"After documents\"}}",serde_json::to_string(&docs).unwrap()).into_bytes();
    let Some(Parse::Export(mut parser)) = open(Path::new("projects/late.json")) else {
        panic!("not an export parser")
    };
    let mut records = 0;
    let mut largest = 0;
    let stats = parser
        .scan(
            "input",
            &mut std::io::Cursor::new(&bytes),
            &mut |range: std::ops::Range<u64>, _: Vec<u8>, _: Option<&str>, raw: &[u8]| {
                assert_eq!(&bytes[range.start as usize..range.end as usize], raw);
                records += 1;
                largest = largest.max(raw.len());
                Ok(())
            },
        )
        .unwrap();
    assert_eq!(records, 257);
    assert!(stats.peak_element_bytes <= largest * 2 + 2);
    assert!(stats.peak_element_bytes < bytes.len() / 32);
}

#[test]
fn explicit_null_collections_keep_sibling_text_and_parse_diagnostics() {
    for (path,raw,text) in [
        ("projects/p.json",br#"{"uuid":"p","prompt_template":"Keep instructions","docs":null}"#.as_slice(),"Keep instructions"),
        ("memories.json",br#"[{"account_uuid":"a","conversations_memory":"Keep memory","project_memories":null}]"#.as_slice(),"Keep memory"),
    ] {
        let mut parser=open(Path::new(path)).unwrap();let step=parse_bound(&mut parser,"input",raw);
        assert_eq!(step.output.documents.len(),1);assert_eq!(step.output.documents[0].document.blocks,[Block::Text{text:text.into()}]);
        assert!(!step.counts.skipped.is_empty());
        assert_eq!(&raw[step.output.documents[0].offset as usize..step.output.documents[0].end as usize],serde_json::to_string(text).unwrap().as_bytes());
    }
}
