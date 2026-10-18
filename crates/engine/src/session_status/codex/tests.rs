use super::*;
#[test]
fn flat_event_reads_check_cancellation_and_cumulative_admission_before_lookup() {
    use polylogue_store::{ImportObservation, InputFormat, Writer};
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(dir.path()).unwrap();
    let blob = writer.put_file_blob(b"{}\n").unwrap();
    writer
        .store_immutable_input(
            blob,
            InputFormat::CodexRollout,
            &[],
            &ImportObservation {
                id: "flat-event-controls",
                machine: "synthetic",
                host: "synthetic",
                path: "synthetic.jsonl",
                seen_at: 0,
            },
            4096,
        )
        .unwrap();
    let reader = writer.read_handle().unwrap();
    let catalog = reader.session_catalog(65536).unwrap();
    let (_, dependency) = catalog.next_dependency(None, 65536).unwrap().unwrap();
    let mut cancelled = || true;
    let mut reads = FlatReads {
        catalog: &catalog,
        dependency: &dependency,
        until: Instant::now() + std::time::Duration::from_secs(5),
        cancelled: &mut cancelled,
        allowance: 65536,
        used: 0,
    };
    assert!(matches!(reads.cell(b"key", 256), Err(Error::Cancelled)));
    assert_eq!(reads.used, 0);
    let mut continuing = || false;
    let mut reads = FlatReads {
        catalog: &catalog,
        dependency: &dependency,
        until: Instant::now() + std::time::Duration::from_secs(5),
        cancelled: &mut continuing,
        allowance: 1,
        used: 0,
    };
    assert!(matches!(
        reads.cell(b"key", 256),
        Err(Error::ParserAdmission {
            resource: "Codex active event proof",
            ..
        })
    ));
    assert!(reads.used > reads.allowance);
    reads.allowance = 65536;
    reads.used = 65530;
    assert!(matches!(
        reads.cell(b"key", 256),
        Err(Error::ParserAdmission { .. })
    ));
}

#[test]
fn benign_events_do_not_supply_an_absent_current_selected_message_projection() {
    use polylogue_model::{
        query::SessionQuery,
        settlement::{SourceRowScope, SourceRowState, SourceRows},
    };
    use polylogue_sources::{
        codex::{
            PARSER_VERSION,
            keyed::{KeyedRecords, MemoryReadView, Resume},
        },
        frame::drive_jsonl,
    };
    use polylogue_store::{
        ImportObservation, InputFormat, Writer,
        parser_state::{Accounting, Basis, Budget, Phase, Progress},
    };
    let raw=concat!(
        "{\"type\":\"session_meta\",\"payload\":{\"id\":\"s\",\"source\":\"cli\",\"base_instructions\":{\"text\":\"Neutral\"}}}\n",
        "{\"type\":\"response_item\",\"payload\":{\"type\":\"message\",\"id\":\"answer\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Neutral answer\"}]}}\n",
        "{\"type\":\"event_msg\",\"payload\":{\"type\":\"token_count\",\"info\":null}}\n"
    ).as_bytes();
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(dir.path()).unwrap();
    let blob = writer.put_file_blob(raw).unwrap();
    let retained = writer
        .store_immutable_input(
            blob,
            InputFormat::CodexRollout,
            &[],
            &ImportObservation {
                id: "no-selected-body",
                machine: "synthetic",
                host: "synthetic",
                path: "synthetic.jsonl",
                seen_at: 0,
            },
            4096,
        )
        .unwrap();
    let input = writer.stored_source(&retained.id).unwrap().unwrap();
    let head = writer
        .begin_parser(&Basis {
            input_pk: retained.input_pk,
            chain: retained.chain,
            parser_version: PARSER_VERSION,
            digest: owner::parser_identity(&input),
            acquired_upto: raw.len() as u64,
            companions_json: None,
            excisions_seen: 0,
        })
        .unwrap();
    let memory = MemoryReadView::default();
    let framed = drive_jsonl(KeyedRecords::new(&memory, Resume::default()), raw, 0);
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    assert!(!delta.resume.normalization_incomplete());
    let budget = Budget {
        cells: 65536,
        encoded_bytes: 1024 * 1024,
    };
    let encoded = crate::state::codex::encode_delta(&delta, 1, budget).unwrap();
    let mut work = writer
        .prepare_parser(
            &head,
            Progress {
                inspected_upto: raw.len() as u64,
                parsed_upto: raw.len() as u64,
                phase: Phase::Finalizing,
                finalization_pending: true,
                counts: Accounting {
                    framed: 3,
                    used: 3,
                    ignored: 0,
                    unrecognized: 0,
                    undecodable: 0,
                },
            },
        )
        .unwrap();
    work = writer
        .stage_parser(&work.token, &encoded.changes, budget)
        .unwrap();
    work = writer.seal_parser(&work.token).unwrap();
    while work.token.applied < work.token.staged {
        work = writer.apply_parser(&work.token, budget).unwrap();
    }
    let head = writer.publish_parser(&work.token).unwrap();
    let receipt = crate::source_rows::Receipt {
        predecessor: head.token.clone(),
        dispositions: [3, 3, 0, 0, 0],
        unsupported_effects: true,
        rows: SourceRows {
            scope: SourceRowScope::ImportedSourceRows,
            basis: source_rows::computed_from(&input),
            state: SourceRowState::Unavailable {
                usable_result: true,
                notes: vec!["Event product remains unsupported".into()],
                record_errors: 0,
            },
        },
    };
    writer
        .publish_source_rows(&head, &serde_json::to_vec(&receipt).unwrap(), false)
        .unwrap();
    let reader = writer.read_handle().unwrap();
    let catalog = reader.session_catalog(1024 * 1024).unwrap();
    let (_, dependency) = catalog.next_dependency(None, 1024 * 1024).unwrap().unwrap();
    let mut result = SessionStatus {
        exact: true,
        known_rows_readable: true,
        ..Default::default()
    };
    let mut cancel = || false;
    check(
        &catalog,
        &dependency,
        &SessionQuery {
            id: Some("codex:s".into()),
            ..Default::default()
        },
        Some("codex:s"),
        Instant::now() + std::time::Duration::from_secs(10),
        1024 * 1024,
        &mut result,
        &mut cancel,
    )
    .unwrap();
    assert_eq!(result.pending_inputs, 0);
    assert_eq!(result.unresolved_inputs, 0);
    assert_eq!(result.expected.len(), 1);
    assert!(!result.known_rows_readable());
    assert!(!result.active_path_known());
    assert!(
        result
            .unsupported
            .iter()
            .any(|n| n.contains("lacks its current selected source projection"))
    );
}
