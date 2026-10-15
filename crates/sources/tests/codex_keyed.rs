//! Provider-cell equivalence and access discrimination. These tests do not
//! qualify a durable SQLite adapter or the daemon's bounded owner route.
use polylogue_sources::codex::keyed::{
    Cell, Delta, Key, KeyedRecords, Limits, MemoryReadView, ReadView, Resume,
};
use polylogue_sources::frame::{JsonlDriver, StepBudget, drive_jsonl};
use serde_json::{Value, json};
use std::cell::Cell as Counter;

fn line(kind: &str, payload: Value) -> String {
    json!({"timestamp":"2026-10-09T14:30:00Z","type":kind,"payload":payload}).to_string() + "\n"
}
fn header() -> String {
    line(
        "session_meta",
        json!({"id":"01a20000-0202-7000-8000-000000000202","source":"cli"}),
    )
}
fn run(view: MemoryReadView, resume: Resume, bytes: &[u8]) -> (MemoryReadView, Resume, Delta) {
    let start = resume.parsed_upto;
    let framed = drive_jsonl(KeyedRecords::new(&view, resume), bytes, start);
    let mut delta = framed.output.unwrap();
    delta.resume.parsed_upto = framed.parsed_upto;
    assert!(framed.counts.balanced());
    let mut view = view;
    view.apply(&delta);
    let next = delta.resume.clone();
    (view, next, delta)
}
fn projected(view: &MemoryReadView, resume: &Resume) -> Value {
    let mut messages = Vec::new();
    for i in 0..resume.message_count() {
        messages.push(view.projected_message(i).unwrap().unwrap());
    }
    let session = resume.session().unwrap().clone();
    json!({"session":session,"messages":messages})
}
#[test]
fn keyed_steps_and_whole_parse_preserve_stateful_and_legacy_rows() {
    let fixture = include_bytes!("data/codex-stateful.jsonl").to_vec();
    let legacy=[json!({"id":"01a20000-0202-7000-8000-000000000202","timestamp":"2025-08-11T12:00:00Z"}),json!({"type":"message","role":"user","content":[{"text":"hello"}]}),json!({"type":"function_call","name":"shell","call_id":"c","arguments":"{\"command\":\"true\"}"}),json!({"type":"function_call_output","call_id":"c","output":"Exit code: 0"}),json!({"type":"message","role":"assistant","content":[{"text":"done"}]})].iter().map(|v|v.to_string()+"\n").collect::<String>().into_bytes();
    for bytes in [fixture, legacy] {
        let (whole_view, whole_resume, whole_delta) =
            run(MemoryReadView::default(), Resume::default(), &bytes);
        let whole = projected(&whole_view, &whole_resume);
        let mut view = MemoryReadView::default();
        let mut resume = Resume::default();
        let mut events = Vec::new();
        let mut links = Vec::new();
        for record in bytes.split_inclusive(|b| *b == b'\n') {
            let (next, state, delta) = run(view, resume, record);
            assert!(!delta.reached_back);
            events.extend(delta.events);
            links.extend(delta.links);
            view = next;
            resume = state;
        }
        assert_eq!(projected(&view, &resume), whole);
        assert_eq!(json!(events), json!(whole_delta.events));
        assert_eq!(json!(links), json!(whole_delta.links));
        // The cells and scalar continuation survive a process serialization boundary.
        let entries: Vec<_> = view.cells.into_iter().collect();
        let entries: Vec<(Key, Cell)> =
            serde_json::from_slice(&serde_json::to_vec(&entries).unwrap()).unwrap();
        let restored = MemoryReadView {
            cells: entries.into_iter().collect(),
        };
        let restored_resume: Resume =
            serde_json::from_slice(&serde_json::to_vec(&resume).unwrap()).unwrap();
        assert_eq!(projected(&restored, &restored_resume), whole);
    }
}
struct Meter<'a> {
    base: &'a MemoryReadView,
    reads: Counter<usize>,
    neighbors: Counter<usize>,
    fail: bool,
}
impl ReadView for Meter<'_> {
    fn read(&self, k: &Key) -> Result<Option<Cell>, String> {
        self.reads.set(self.reads.get() + 1);
        if self.fail {
            return Err("injected Codex state read failure".into());
        }
        self.base.read(k)
    }
    fn next(&self, k: &Key) -> Result<Option<(Key, Cell)>, String> {
        self.neighbors.set(self.neighbors.get() + 1);
        if self.fail {
            return Err("injected Codex ordered read failure".into());
        }
        self.base.next(k)
    }
    fn previous(&self, k: &Key) -> Result<Option<(Key, Cell)>, String> {
        self.neighbors.set(self.neighbors.get() + 1);
        if self.fail {
            return Err("injected Codex ordered read failure".into());
        }
        self.base.previous(k)
    }
}
fn prefix(n: usize) -> (MemoryReadView, Resume) {
    let mut bytes = header();
    // One early running result keeps the old contiguous suffix unsettled.
    bytes += &line(
        "response_item",
        json!({"type":"function_call","id":"early","name":"exec_command","call_id":"early","arguments":"{}"}),
    );
    bytes += &line(
        "response_item",
        json!({"type":"function_call_output","call_id":"early","output":"Process running with session ID 7"}),
    );
    bytes += &line("event_msg", json!({"type":"token_count","info":null}));
    for i in 0..n {
        bytes += &line(
            "response_item",
            json!({"type":"message","id":format!("u{i}"),"role":"user","content":[{"text":format!("prompt {i}")}]}),
        );
        bytes += &line(
            "response_item",
            json!({"type":"function_call","id":format!("a{i}"),"name":"exec_command","call_id":format!("c{i}"),"arguments":"{}"}),
        );
        bytes += &line(
            "response_item",
            json!({"type":"function_call_output","call_id":format!("c{i}"),"output":"Exit code: 0"}),
        );
        bytes += &line("event_msg", json!({"type":"token_count","info":null}));
    }
    let mut view = MemoryReadView::default();
    let mut resume = Resume::default();
    // Build the fixture through admitted provider steps; constructing a large
    // fixture in one overlay must not disable the production admission limit.
    for record in bytes.as_bytes().split_inclusive(|b| *b == b'\n') {
        let (next, state, _) = run(view, resume, record);
        view = next;
        resume = state;
    }
    (view, resume)
}
#[test]
fn one_record_after_large_unique_prefix_uses_only_point_and_neighbor_reads() {
    let mut measurements = Vec::new();
    for n in [3, 3000] {
        let (view, resume) = prefix(n);
        assert_eq!(resume.settled_before(), 1);
        let meter = Meter {
            base: &view,
            reads: Counter::new(0),
            neighbors: Counter::new(0),
            fail: false,
        };
        let bytes = line(
            "response_item",
            json!({"type":"function_call_output","call_id":"c0","output":"Exit code: 0"}),
        );
        let out = drive_jsonl(KeyedRecords::new(&meter, resume), bytes.as_bytes(), 0)
            .output
            .unwrap();
        let touched = out.cells.len();
        assert!(touched <= 12, "{touched} touched cells");
        assert!(!out.reached_back);
        measurements.push((meter.reads.get(), meter.neighbors.get(), touched));
    }
    assert_eq!(measurements[0], measurements[1]);
    assert!(
        measurements[1].0 + measurements[1].1 <= 32,
        "{measurements:?}"
    );
}
#[test]
fn lookup_failure_fuses_driver_and_never_acknowledges_attempted_step() {
    let (view, resume) = prefix(3);
    let meter = Meter {
        base: &view,
        reads: Counter::new(0),
        neighbors: Counter::new(0),
        fail: true,
    };
    let bytes = line(
        "response_item",
        json!({"type":"function_call_output","call_id":"c0","output":"Exit code: 0"}),
    );
    let mut driver = JsonlDriver::new(KeyedRecords::new(&meter, resume), 123);
    let budget = StepBudget {
        bytes: usize::MAX,
        records: u64::MAX,
        record_bytes: 1024 * 1024,
    };
    assert!(driver.advance(bytes.as_bytes(), budget).is_err());
    let reads = meter.reads.get() + meter.neighbors.get();
    assert!(driver.advance(bytes.as_bytes(), budget).is_err());
    assert_eq!(meter.reads.get() + meter.neighbors.get(), reads);
    let framed = driver.finish();
    assert_eq!(framed.parsed_upto, 123);
    let failure = framed.output.unwrap_err();
    assert!(failure.error.contains("injected Codex"));
    assert_eq!(framed.counts.used, 0);
}
#[test]
fn cell_corruption_and_overlay_admission_are_errors_not_missing_provider_records() {
    let (mut view, resume) = prefix(1);
    view.cells.insert(Key::Call("c0".into()), Cell::Present);
    let bytes = line(
        "response_item",
        json!({"type":"function_call_output","call_id":"c0","output":"Exit code: 0"}),
    );
    let e = drive_jsonl(KeyedRecords::new(view, resume), bytes.as_bytes(), 0)
        .output
        .unwrap_err();
    assert!(e.error.contains("invalid Codex cell"));
    let p = KeyedRecords::with_limits(
        MemoryReadView::default(),
        Resume::default(),
        Limits {
            changed_cells: 0,
            encoded_bytes: 0,
        },
    );
    let bytes = header()
        + &line(
            "response_item",
            json!({"type":"message","role":"user","content":[{"text":"owed"}]}),
        );
    let e = drive_jsonl(p, bytes.as_bytes(), 0).output.unwrap_err();
    assert!(e.error.contains("admission exceeded"));
}
#[test]
fn a_long_open_message_does_not_materialize_blocks_to_replace_one_reasoning_item() {
    let mut measurements = Vec::new();
    for n in [3, 3000] {
        let mut bytes = header()
            + &line(
                "response_item",
                json!({"type":"reasoning","id":"reasoning","summary":[{"text":"first"}]}),
            );
        for i in 0..n {
            bytes += &line(
                "response_item",
                json!({"type":"function_call","id":format!("anchor{i}"),"name":"tool","call_id":format!("c{i}"),"arguments":"{}"}),
            );
        }
        let (view, resume, _) = run(
            MemoryReadView::default(),
            Resume::default(),
            bytes.as_bytes(),
        );
        let meter = Meter {
            base: &view,
            reads: Counter::new(0),
            neighbors: Counter::new(0),
            fail: false,
        };
        let bytes = line(
            "response_item",
            json!({"type":"reasoning","id":"reasoning","summary":[{"text":"replacement"}]}),
        );
        let delta = drive_jsonl(KeyedRecords::new(&meter, resume), bytes.as_bytes(), 0)
            .output
            .unwrap();
        assert!(!delta.reached_back);
        assert_eq!(delta.cells.len(), 4);
        assert!(delta.cells.contains_key(&Key::Block(0, 0)));
        measurements.push((meter.reads.get(), meter.neighbors.get()));
    }
    assert_eq!(measurements[0], measurements[1]);
    assert!(measurements[1].0 + measurements[1].1 <= 16);
}

#[test]
fn ordered_neighbors_never_load_unrelated_family_payloads() {
    let mut view = MemoryReadView::default();
    view.cells.insert(
        Key::Prompt,
        Cell::Text("unrelated retained text".repeat(100)),
    );
    assert!(
        view.next(&Key::Mutable(0, String::new()))
            .unwrap()
            .is_none()
    );
    assert!(view.previous(&Key::Model(usize::MAX)).unwrap().is_none());
}

#[test]
fn provider_key_codec_roundtrips_and_preserves_derived_key_order() {
    let mut keys = Vec::new();
    for n in [0, 1, 9, 10, 255, 256, 65536, usize::MAX] {
        keys.extend([
            Key::Message(n),
            Key::Block(n, 0),
            Key::Block(n, usize::MAX),
            Key::Wrapper(n),
            Key::Model(n),
        ]);
        for s in ["", "\0", "a", "a\0", "aa", "é", "中", "🦀"] {
            keys.extend([
                Key::Call(s.into()),
                Key::CallBlock(s.into(), n, 0),
                Key::CallBlock(s.into(), n, usize::MAX),
                Key::Reasoning(s.into()),
                Key::Running(s.into()),
                Key::Outcome(s.into()),
                Key::Continuation(s.into()),
                Key::PendingStdin(s.into()),
                Key::WrapperCall(s.into(), n),
                Key::Child(s.into(), n),
                Key::ChildCount(s.into()),
                Key::Occurrence(s.into()),
                Key::Mutable(n, s.into()),
            ]);
        }
    }
    keys.extend([
        Key::Prompt,
        Key::BlockOrder(8),
        Key::Evidence(8, 0, 3),
        Key::Evidence(8, 1, 4),
        Key::EvidenceHead(8),
        Key::EvidenceItem(8, "id\0x".into()),
        Key::Context(1),
        Key::CallWitness("id".into()),
        Key::StatusWitness("id".into()),
        Key::Result("id".into()),
    ]);
    keys.sort();
    keys.dedup();
    for k in &keys {
        assert_eq!(Key::from_ordered_bytes(&k.ordered_bytes()).unwrap(), *k);
        assert_eq!(k.ordered_bytes().len(), k.ordered_len());
        let mut streamed = Vec::new();
        k.write_ordered(&mut streamed).unwrap();
        assert_eq!(streamed, k.ordered_bytes());
        assert!(!k.family().is_empty());
    }
    let mut encoded = keys.iter().map(Key::ordered_bytes).collect::<Vec<_>>();
    encoded.sort();
    assert_eq!(
        encoded
            .into_iter()
            .map(|v| Key::from_ordered_bytes(&v).unwrap())
            .collect::<Vec<_>>(),
        keys
    );
    // Negative control: serde JSON lexical order misorders numeric ordinals.
    let numeric = vec![Key::Wrapper(9), Key::Wrapper(10)];
    let mut wrong = numeric.clone();
    wrong.sort_by_key(|k| serde_json::to_string(k).unwrap());
    assert_ne!(wrong, numeric);
    for bad in [
        vec![],
        vec![255],
        vec![0, 1],
        vec![2, 0, 2],
        vec![2, 255, 0, 0],
        vec![16, 0],
    ] {
        assert!(Key::from_ordered_bytes(&bad).is_err(), "{bad:?}");
    }
}

#[test]
fn missing_referenced_cells_fail_instead_of_becoming_provider_unknown_records() {
    let cases = [
        (
            line(
                "response_item",
                json!({"type":"function_call","id":"exec","name":"exec","call_id":"exec","arguments":"{}"}),
            ),
            Key::Call("exec".into()),
            line(
                "event_msg",
                json!({"type":"item_completed","item":{"type":"CommandExecution","id":"child","status":"completed"}}),
            ),
            "wrapper call",
        ),
        (
            line(
                "response_item",
                json!({"type":"message","id":"prompt","role":"user","content":[{"text":"echo"}]}),
            ),
            Key::Prompt,
            line("event_msg", json!({"type":"user_message","message":"echo"})),
            "pending prompt text",
        ),
        (
            line(
                "response_item",
                json!({"type":"function_call","id":"stdin","name":"write_stdin","call_id":"stdin","arguments":"{\"session_id\":7}"}),
            ),
            Key::Call("stdin".into()),
            line(
                "response_item",
                json!({"type":"function_call_output","call_id":"stdin","output":"Exit code: 0"}),
            ),
            "pending stdin call",
        ),
    ];
    for (before, missing, after, expected) in cases {
        let (mut view, resume, _) = run(
            MemoryReadView::default(),
            Resume::default(),
            (header() + &before).as_bytes(),
        );
        view.cells.remove(&missing);
        let framed = drive_jsonl(KeyedRecords::new(view, resume), after.as_bytes(), 0);
        let error = framed.output.unwrap_err();
        assert!(error.error.contains(expected), "{error:?}");
        assert_eq!(framed.counts.used, 0);
        assert!(framed.counts.unrecognized.is_empty());
    }
    // A provider can genuinely record a result without a known call. Absence
    // is an error only when another retained state cell promises the call.
    let (view, resume, _) = run(
        MemoryReadView::default(),
        Resume::default(),
        header().as_bytes(),
    );
    let orphan = line(
        "response_item",
        json!({"type":"function_call_output","call_id":"not-recorded","output":"Exit code: 0"}),
    );
    let (view, resume, _) = run(view, resume, orphan.as_bytes());
    assert_eq!(resume.message_count(), 1);
    assert!(view.message(0).unwrap().is_some());
}

#[test]
fn key_sizing_rejects_native_length_before_counting_escapes() {
    let key = Key::Call("\0".repeat(1_000_000));
    // Returning the unescaped lower bound distinguishes the early refusal
    // from completing the escape-count pass (which requires 2_000_003 bytes).
    assert_eq!(key.ordered_len_limited(64), Err(1_000_003));
    assert_eq!(key.ordered_len_limited(1_000_003), Err(2_000_003));
    assert_eq!(key.ordered_len_limited(2_000_003), Ok(2_000_003));
    for k in [
        Key::Block(1, 2),
        Key::Mutable(1, "a\0é".into()),
        Key::Prompt,
    ] {
        let needed = k.ordered_len();
        assert_eq!(k.ordered_len_limited(needed), Ok(needed));
        assert!(k.ordered_len_limited(needed - 1).is_err());
    }
    struct Capped {
        written: usize,
        limit: usize,
    }
    impl std::io::Write for Capped {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            if b.len() > self.limit - self.written {
                return Err(std::io::Error::other("capped"));
            }
            self.written += b.len();
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let admitted = Key::Call("a\0b".into());
    let mut capped = Capped {
        written: 0,
        limit: admitted.ordered_len() - 1,
    };
    assert!(admitted.write_ordered(&mut capped).is_err());
    assert!(capped.written <= capped.limit);
}
