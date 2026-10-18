use super::*;
#[test]
fn point_proof_checks_cancellation_and_admission_before_accessing_unprepared_state() {
    use polylogue_store::{ImportObservation, InputFormat, Writer};
    let dir = tempfile::tempdir().unwrap();
    let mut writer = Writer::open(dir.path()).unwrap();
    let blob = writer.put_file_blob(b"{}\n").unwrap();
    writer
        .retain_immutable_input(
            blob,
            InputFormat::ClaudeCodeJsonl,
            &["claude-main", "s"],
            &ImportObservation {
                id: "proof-controls",
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
    assert!(dependency.head().is_none());
    let mut cancelled = || true;
    let mut reads = Reads {
        catalog: &catalog,
        dependency: &dependency,
        allowance: 65536,
        used: 0,
        until: Instant::now() + std::time::Duration::from_secs(10),
        cancelled: &mut cancelled,
    };
    assert!(matches!(
        reads.cell::<RecordAddress>(claude::RECORD, &("s", "uuid")),
        Err(Error::Cancelled)
    ));
    assert_eq!(reads.used, 0);
    let mut continuing = || false;
    let mut reads = Reads {
        catalog: &catalog,
        dependency: &dependency,
        allowance: 1,
        used: 0,
        until: Instant::now() + std::time::Duration::from_secs(10),
        cancelled: &mut continuing,
    };
    assert!(matches!(
        reads.cell::<RecordAddress>(claude::RECORD, &("s", "uuid")),
        Err(Error::ParserAdmission {
            resource: "Claude active leaf proof",
            ..
        })
    ));
    assert!(reads.used > reads.allowance);
}
