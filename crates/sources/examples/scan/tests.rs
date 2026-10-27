use super::*;

fn parser() -> Parse {
    (polylogue_sources::origin::by_name("claude-code")
        .unwrap()
        .open)(Path::new("synthetic.jsonl"))
    .unwrap()
}

const UNKNOWN: &[u8] = b"{\"type\":\"future-synthetic-kind\",\"text\":\"PRIVATE_BODY_MARKER\"}\n";

#[test]
fn read_failure_preserves_successful_prefix_bytes_and_counts() {
    struct BreakAfterPrefix(bool);
    impl Read for BreakAfterPrefix {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            if self.0 {
                return Err(std::io::Error::from_raw_os_error(5));
            }
            self.0 = true;
            bytes[..UNKNOWN.len()].copy_from_slice(UNKNOWN);
            Ok(UNKNOWN.len())
        }
    }
    let scanned = scan_reader(&parser(), BreakAfterPrefix(false));
    assert_eq!(scanned.counts.bytes, UNKNOWN.len() as u64);
    assert_eq!(scanned.counts.counts.records, 1);
    assert!(matches!(
        scanned.outcome,
        Outcome::Failed {
            stage: "read",
            error: Failure::Io {
                os_code: Some(5),
                ..
            }
        }
    ));
}

#[test]
fn interrupted_reads_retry_but_do_not_become_eof() {
    // One interrupted operation followed by a finite ordinary reader.
    let reader = std::io::Read::chain(InterruptedOnce(false), std::io::Cursor::new(UNKNOWN));
    let scanned = scan_reader(&parser(), reader);
    assert!(matches!(scanned.outcome, Outcome::Complete));
    assert_eq!(scanned.counts.bytes, UNKNOWN.len() as u64);
}

struct InterruptedOnce(bool);
impl Read for InterruptedOnce {
    fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
        if self.0 {
            Ok(0)
        } else {
            self.0 = true;
            Err(std::io::ErrorKind::Interrupted.into())
        }
    }
}

#[test]
fn unconsumed_tail_is_incomplete_not_an_extra_skipped_record() {
    let mut bytes = UNKNOWN.to_vec();
    bytes.extend_from_slice(b"{\"type\":");
    let scanned = scan_reader(&parser(), std::io::Cursor::new(bytes));
    assert!(matches!(
        scanned.outcome,
        Outcome::Incomplete {
            reason: "unconsumed_tail"
        }
    ));
    assert_eq!(scanned.counts.counts.records, 1);
    assert_eq!(scanned.counts.counts.skipped.values().sum::<u64>(), 1);
    assert_eq!(scanned.remaining, 8);
}

#[test]
fn failed_open_is_one_outcome_and_does_not_hide_other_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let valid = dir.path().join("synthetic.jsonl");
    std::fs::write(&valid, UNKNOWN).unwrap();
    let (report, success) = run(input::Selection {
        origin: "claude-code".into(),
        threads: 2,
        files: vec![dir.path().join("absent.jsonl"), valid.clone(), valid],
    });
    assert!(!success);
    assert_eq!(report["selected_inputs"], 3);
    assert_eq!(report["inputs"].as_array().unwrap().len(), 3);
    assert_eq!(report["failed_inputs"], 1);
    assert_eq!(report["files"], 2);
    assert_eq!(report["bytes_read"], UNKNOWN.len() * 2);
    assert_eq!(report["inputs"][0]["outcome"]["stage"], "open");
    assert!(!report.to_string().contains("PRIVATE_BODY_MARKER"));
}

#[test]
fn invalid_selection_fails_before_any_scan() {
    for args in [
        vec![],
        vec!["unknown-origin", "synthetic"],
        vec!["claude-code"],
        vec!["claude-code", "--threads", "0", "synthetic.jsonl"],
        vec!["claude-code", "--threads", "no", "synthetic.jsonl"],
    ] {
        assert!(input::selection(args.into_iter().map(str::to_owned)).is_err());
    }
}

struct Document;
impl polylogue_sources::DocumentParser for Document {
    fn document(
        &self,
        bytes: &[u8],
        out: &mut polylogue_sources::Output,
        records: &mut polylogue_sources::Records,
    ) {
        assert_eq!(bytes, b"{}", "the document must be complete");
        let mut document = polylogue_model::Document::new(
            polylogue_model::DocumentId::parse("claude-ai:synthetic-account-memory").unwrap(),
            polylogue_model::DocumentPurpose::Memory,
        );
        document.blocks.push(polylogue_model::Block::Text {
            text: "PRIVATE_BODY_MARKER".into(),
        });
        out.documents.push(polylogue_model::DocumentRow {
            offset: 0,
            end: 0,
            document,
        });
        records.record("document", &polylogue_sources::RecordOutcome::Used);
    }
}

#[test]
fn whole_documents_are_not_parsed_at_short_read_boundaries() {
    struct Bytewise(std::io::Cursor<&'static [u8]>);
    impl Read for Bytewise {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            self.0.read(&mut bytes[..1])
        }
    }
    let scanned = scan_reader(
        &Parse::Document(Box::new(Document)),
        Bytewise(std::io::Cursor::new(b"{}")),
    );
    assert!(matches!(scanned.outcome, Outcome::Complete));
    assert_eq!(scanned.counts.counts.records, 1);
    assert_eq!(scanned.counts.bytes, 2);
    assert_eq!(scanned.counts.documents, 1);
    assert_eq!(scanned.counts.messages, 0);
    assert_eq!(scanned.counts.sessions, 0);
}

#[test]
fn coalesced_captures_use_end_offsets_not_a_sum_of_ends() {
    use polylogue_model::capture_log::{Header, MAGIC};
    let mut bytes = MAGIC.to_vec();
    for _ in 0..2 {
        bytes.extend(Header::of(b"{}").encode());
        bytes.extend(b"{}");
    }
    let scanned = scan_reader(
        &Parse::CaptureLog(Box::new(Document)),
        std::io::Cursor::new(&bytes),
    );
    assert!(matches!(scanned.outcome, Outcome::Complete));
    assert_eq!(scanned.counts.counts.records, 2);
    assert_eq!(scanned.counts.documents, 2);
    assert_eq!(scanned.counts.bytes, bytes.len() as u64);
}

#[test]
fn parser_failures_are_not_skipped_records_or_source_diagnostics() {
    let parser = (polylogue_sources::origin::by_name("chatgpt").unwrap().open)(Path::new(
        "conversations.json",
    ))
    .unwrap();
    let scanned = scan_reader(&parser, std::io::Cursor::new(b"PRIVATE_BODY_MARKER"));
    assert!(matches!(
        scanned.outcome,
        Outcome::Failed {
            stage: "parse",
            error: Failure::Parse {
                kind: "export_reader_required"
            }
        }
    ));
    assert_eq!(scanned.counts.bytes, 19);
    assert_eq!(scanned.counts.counts.records, 0);
    let diagnostic = Failure::parse(ParseError::CaptureLog("PRIVATE_BODY_MARKER".into()));
    assert!(
        !serde_json::to_string(&diagnostic)
            .unwrap()
            .contains("PRIVATE_BODY_MARKER")
    );
}

fn command_output(args: &[String]) -> (ExitCode, serde_json::Value) {
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let status = command(args.iter().cloned(), &mut output, &mut errors);
    for bytes in [&output, &errors] {
        assert!(!String::from_utf8_lossy(bytes).contains("PRIVATE_BODY_MARKER"));
    }
    let report = if output.is_empty() {
        assert!(!errors.is_empty());
        serde_json::Value::Null
    } else {
        serde_json::from_slice(&output).unwrap()
    };
    (status, report)
}

#[test]
fn command_rejects_invalid_arguments_and_path_lists() {
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty.txt");
    std::fs::write(&empty, "\n").unwrap();
    let missing = format!("@{}", dir.path().join("absent.txt").display());
    let empty = format!("@{}", empty.display());
    for args in [
        vec!["claude-code"],
        vec!["claude-code", "--threads", "0", "synthetic.jsonl"],
        vec!["claude-code", "--threads", "no", "synthetic.jsonl"],
        vec!["unknown-origin", "synthetic.jsonl"],
        vec!["claude-code", &empty],
        vec!["claude-code", &missing],
    ] {
        let (status, report) =
            command_output(&args.into_iter().map(str::to_owned).collect::<Vec<_>>());
        assert_eq!(status, ExitCode::from(2));
        assert!(report.is_null());
    }
}

#[test]
fn command_preserves_list_order_duplicates_and_failed_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let valid = dir.path().join("synthetic.jsonl");
    let missing = dir.path().join("absent.jsonl");
    let listing = dir.path().join("inputs.txt");
    std::fs::write(&valid, UNKNOWN).unwrap();
    std::fs::write(
        &listing,
        format!(
            "{} {}\n{}\n{}\n",
            UNKNOWN.len(),
            valid.display(),
            missing.display(),
            valid.display()
        ),
    )
    .unwrap();
    for threads in [1, 3] {
        let (status, report) = command_output(&[
            "claude-code".into(),
            "--threads".into(),
            threads.to_string(),
            format!("@{}", listing.display()),
        ]);
        assert_eq!(status, ExitCode::FAILURE);
        assert_eq!(report["selected_inputs"], 3);
        let inputs = report["inputs"].as_array().unwrap();
        assert_eq!(inputs.len(), 3);
        for (index, input) in inputs.iter().enumerate() {
            assert_eq!(input["input"], index);
        }
        assert_eq!(report["files"], 2);
        assert_eq!(report["failed_inputs"], 1);
        assert_eq!(report["bytes_read"], UNKNOWN.len() * 2);
        assert_eq!(inputs[1]["outcome"]["stage"], "open");
    }
}

#[test]
fn command_reports_read_errors_incomplete_unreadable_and_skipped_records() {
    let dir = tempfile::tempdir().unwrap();
    let directory = dir.path().join("directory.jsonl");
    std::fs::create_dir(&directory).unwrap();
    let (status, report) = command_output(&["claude-code".into(), directory.display().to_string()]);
    assert_eq!(status, ExitCode::FAILURE);
    assert_eq!(report["inputs"][0]["outcome"]["stage"], "read");
    assert_eq!(report["inputs"][0]["outcome"]["error"]["type"], "io");
    assert_eq!(report["inputs"][0]["bytes_read"], 0);

    let tail = dir.path().join("tail.jsonl");
    let mut bytes = UNKNOWN.to_vec();
    bytes.extend_from_slice(b"{\"type\":");
    std::fs::write(&tail, bytes).unwrap();
    let (status, report) = command_output(&["claude-code".into(), tail.display().to_string()]);
    assert_eq!(status, ExitCode::FAILURE);
    assert_eq!(report["incomplete_inputs"], 1);
    assert_eq!(report["records"], 1);
    assert_eq!(report["skipped"], 1);
    assert_eq!(report["inputs"][0]["unconsumed_bytes"], 8);

    let unreadable = dir.path().join("unreadable.jsonl");
    std::fs::write(&unreadable, "PRIVATE_BODY_MARKER\n").unwrap();
    let (status, report) =
        command_output(&["claude-code".into(), unreadable.display().to_string()]);
    assert_eq!(status, ExitCode::SUCCESS);
    assert_eq!(report["unreadable"], 1);
    assert_eq!(report["skipped"], 0);
    assert_eq!(report["failed_inputs"], 0);

    let valid = dir.path().join("synthetic.jsonl");
    std::fs::write(&valid, UNKNOWN).unwrap();
    let (status, report) = command_output(&["claude-code".into(), valid.display().to_string()]);
    assert_eq!(status, ExitCode::SUCCESS);
    assert_eq!(report["files"], 1);
    assert_eq!(report["records"], 1);
    assert_eq!(report["skipped"], 1);
    assert_eq!(report["failed_inputs"], 0);
    assert_eq!(report["incomplete_inputs"], 0);
}
