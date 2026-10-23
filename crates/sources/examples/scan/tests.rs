use super::*;

fn parser() -> Parse {
    (polylogue_sources::origin::by_name("claude-code").unwrap().open)(Path::new("synthetic.jsonl")).unwrap()
}

const UNKNOWN: &[u8] = b"{\"type\":\"future-synthetic-kind\",\"text\":\"PRIVATE_BODY_MARKER\"}\n";

#[test]
fn read_failure_preserves_successful_prefix_bytes_and_counts() {
    struct BreakAfterPrefix(bool);
    impl Read for BreakAfterPrefix {
        fn read(&mut self, bytes: &mut [u8]) -> std::io::Result<usize> {
            if self.0 { return Err(std::io::Error::from_raw_os_error(5)); }
            self.0 = true;
            bytes[..UNKNOWN.len()].copy_from_slice(UNKNOWN);
            Ok(UNKNOWN.len())
        }
    }
    let scanned = scan_reader(&parser(), BreakAfterPrefix(false));
    assert_eq!(scanned.counts.bytes, UNKNOWN.len() as u64);
    assert_eq!(scanned.counts.accounting.framed, 1);
    assert!(matches!(scanned.outcome, Outcome::Failed { stage: "read",
        error: Failure::Io { os_code: Some(5), .. } }));
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
        if self.0 { Ok(0) } else { self.0 = true; Err(std::io::ErrorKind::Interrupted.into()) }
    }
}

#[test]
fn unconsumed_tail_is_incomplete_not_an_extra_unknown_record() {
    let mut bytes = UNKNOWN.to_vec();
    bytes.extend_from_slice(b"{\"type\":");
    let scanned = scan_reader(&parser(), std::io::Cursor::new(bytes));
    assert!(matches!(scanned.outcome, Outcome::Incomplete { reason: "unconsumed_tail" }));
    assert_eq!(scanned.counts.accounting.framed, 1);
    assert_eq!(scanned.counts.accounting.unknown.values().sum::<u64>(), 1);
    assert_eq!(scanned.remaining, 8);
}

#[test]
fn failed_open_is_one_outcome_and_does_not_hide_other_inputs() {
    let dir = tempfile::tempdir().unwrap();
    let valid = dir.path().join("synthetic.jsonl");
    std::fs::write(&valid, UNKNOWN).unwrap();
    let (report, success) = run(input::Selection { origin: "claude-code".into(), threads: 2,
        files: vec![dir.path().join("absent.jsonl"), valid.clone(), valid] });
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
    for args in [vec![], vec!["unknown-origin", "synthetic"],
        vec!["claude-code"], vec!["claude-code", "--threads", "0", "synthetic.jsonl"],
        vec!["claude-code", "--threads", "no", "synthetic.jsonl"]] {
        assert!(input::selection(args.into_iter().map(str::to_owned)).is_err());
    }
}

struct Document;
impl polylogue_sources::DocumentParser for Document {
    fn document(&self, bytes: &[u8], _: &mut polylogue_sources::Output,
        records: &mut polylogue_sources::Records) {
        assert_eq!(bytes, b"{}", "the document must be complete");
        records.record("document", &polylogue_sources::Disposition::Materialized);
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
    let scanned = scan_reader(&Parse::Document(Box::new(Document)),
        Bytewise(std::io::Cursor::new(b"{}")));
    assert!(matches!(scanned.outcome, Outcome::Complete));
    assert_eq!(scanned.counts.accounting.framed, 1);
    assert_eq!(scanned.counts.bytes, 2);
}

#[test]
fn coalesced_snapshots_use_end_offsets_not_a_sum_of_ends() {
    use polylogue_model::snapshot_log::{Header, MAGIC};
    let mut bytes = MAGIC.to_vec();
    for _ in 0..2 {
        bytes.extend(Header::of(b"{}").encode());
        bytes.extend(b"{}");
    }
    let scanned = scan_reader(&Parse::Snapshots(Box::new(Document)), std::io::Cursor::new(&bytes));
    assert!(matches!(scanned.outcome, Outcome::Complete));
    assert_eq!(scanned.counts.accounting.framed, 2);
    assert_eq!(scanned.counts.bytes, bytes.len() as u64);
}
