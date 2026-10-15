//! Provider output projection, independent of storage and canonical publication.
use polylogue_model::*;
use polylogue_sources::codex::keyed::{Cell, Key, KeyedRecords, MemoryReadView, ReadView, Resume};
use polylogue_sources::codex::projection::{self, Cursor, Mode, Step, Unit};
use polylogue_sources::frame::drive_jsonl;
use std::cell::Cell as Counter;

fn fixture(children: usize) -> MemoryReadView {
    let origin = Origin::new("codex").unwrap();
    let session = SessionId::new(&origin, "opaque:session").unwrap();
    let id = MessageId::new(session.clone(), "anchored:request").unwrap();
    let old = MessageId::new(session, "provisional:before-anchor").unwrap();
    let message = Message::new(id, IdKind::Provider, Role::Assistant, WrittenBy::Model);
    let mut view = MemoryReadView::default();
    view.cells
        .insert(Key::Message(0), Cell::Header { message, blocks: 2 });
    view.cells.insert(
        Key::Block(0, 0),
        Cell::Block(Block {
            id: BlockId::reasoning(old.clone(), 7),
            content: BlockContent::ToolCall(ToolCall {
                tool: Some("exec".into()),
                call_id: Some("wrapper:opaque".into()),
                input: None,
                command: None,
            }),
        }),
    );
    view.cells.insert(
        Key::Block(0, 1),
        Cell::Block(Block {
            id: BlockId::new(old, 9),
            content: BlockContent::ToolResult(ToolResult {
                call_id: Some("wrapper:opaque".into()),
                outcome: ToolOutcome::Unknown {
                    reason: UnknownReason::ResultWithoutStatus,
                },
                exit_code: None,
                duration_ms: None,
                output: Some(ToolOutput::Text("superseded".into())),
            }),
        }),
    );
    view.cells
        .insert(Key::Continuation("wrapper:opaque".into()), Cell::Present);
    view.cells.insert(
        Key::ChildCount("wrapper:opaque".into()),
        Cell::Number(children),
    );
    for i in 0..children {
        view.cells.insert(
            Key::Child("wrapper:opaque".into(), i),
            Cell::Text(format!("child:{i}")),
        );
    }
    view.cells.insert(Key::PendingResult(0, 1), Cell::Present);
    view.cells
        .insert(Key::OutputChunk(0, 1, 0), Cell::Text(String::new()));
    view.cells
        .insert(Key::OutputChunk(0, 1, 7), Cell::Text("second".into()));
    view
}
struct Meter<'a> {
    view: &'a MemoryReadView,
    reads: Counter<usize>,
    fail: bool,
}
impl ReadView for Meter<'_> {
    fn read(&self, key: &Key) -> Result<Option<Cell>, String> {
        self.reads.set(self.reads.get() + 1);
        if self.fail {
            return Err("injected projection read".into());
        }
        self.view.read(key)
    }
    fn next(&self, key: &Key) -> Result<Option<(Key, Cell)>, String> {
        self.reads.set(self.reads.get() + 1);
        if self.fail {
            return Err("injected projection neighbor".into());
        }
        self.view.next(key)
    }
    fn previous(&self, key: &Key) -> Result<Option<(Key, Cell)>, String> {
        self.reads.set(self.reads.get() + 1);
        self.view.previous(key)
    }
}
#[test]
fn point_projection_and_memory_collector_share_reanchor_chunks_and_annotations() {
    let view = fixture(3);
    let meter = Meter {
        view: &view,
        reads: Counter::new(0),
        fail: false,
    };
    assert_eq!(
        projection::collect(&meter, 0, Mode::Annotated).unwrap(),
        view.projected_message(0).unwrap()
    );
    let m = view.projected_message(0).unwrap().unwrap();
    assert_eq!(m.blocks[0].id, BlockId::reasoning(m.id.clone(), 7));
    assert_eq!(m.blocks[1].id, BlockId::new(m.id.clone(), 9));
    assert!(
        matches!(&m.blocks[1].content,BlockContent::ToolResult(r) if r.output==Some(ToolOutput::Text("second".into())))
    );
    assert_eq!(
        m.attrs.as_ref().unwrap()["continuation_calls"],
        serde_json::json!(["wrapper:opaque"])
    );
    assert_eq!(
        m.attrs.as_ref().unwrap()["code_children"]["wrapper:opaque"],
        serde_json::json!(["child:0", "child:1", "child:2"])
    );
    let raw = view.message(0).unwrap().unwrap();
    assert!(raw.attrs.is_none());
    assert!(raw.version_hash.is_none());
}
#[test]
fn cursor_roundtrip_and_single_lookup_steps_match_fixture_collection() {
    let bytes = include_bytes!("data/codex-stateful.jsonl");
    let delta = drive_jsonl(
        KeyedRecords::new(MemoryReadView::default(), Resume::default()),
        bytes,
        0,
    )
    .output
    .unwrap();
    let mut view = MemoryReadView::default();
    view.apply(&delta);
    for i in 0..delta.resume.message_count() {
        let mut cursor = Cursor::new(i, Mode::Annotated);
        let meter = Meter {
            view: &view,
            reads: Counter::new(0),
            fail: false,
        };
        let mut blocks = 0;
        let mut headers = 0;
        loop {
            cursor = serde_json::from_slice(&serde_json::to_vec(&cursor).unwrap()).unwrap();
            let before = meter.reads.get();
            let step = cursor.advance(&meter).unwrap();
            assert!(meter.reads.get() - before <= 1);
            match step {
                Step::Done => break,
                Step::Unit(Unit::Header(_)) => headers += 1,
                Step::Unit(Unit::Block { .. }) => blocks += 1,
                Step::Absent => panic!("missing fixture message"),
                _ => {}
            }
        }
        assert_eq!(headers, 1);
        assert_eq!(blocks, view.message(i).unwrap().unwrap().blocks.len());
        assert_eq!(
            projection::collect(&meter, i, Mode::Annotated).unwrap(),
            view.projected_message(i).unwrap()
        );
    }
}
#[test]
fn large_children_and_unrelated_prefix_do_not_expand_one_cursor_step() {
    let mut samples = Vec::new();
    for n in [3, 10000] {
        let mut view = fixture(n);
        for i in 1..n {
            view.cells.insert(
                Key::Child(format!("unrelated:{i}"), 0),
                Cell::Text("ignored".into()),
            );
        }
        let meter = Meter {
            view: &view,
            reads: Counter::new(0),
            fail: false,
        };
        let mut cursor = Cursor::new(0, Mode::Annotated);
        let mut steps = 0;
        loop {
            let before = meter.reads.get();
            let step = cursor.advance(&meter).unwrap();
            steps += 1;
            assert!(meter.reads.get() - before <= 1);
            if let Step::Unit(Unit::WrapperChild { index, .. }) = step {
                assert_eq!(index, 0);
                break;
            }
            assert!(steps < 12);
        }
        // Cursor stores scalar position/call identity, never all children.
        assert!(serde_json::to_vec(&cursor).unwrap().len() < 512);
        samples.push((steps, meter.reads.get()));
    }
    assert_eq!(samples[0], samples[1]);
}
#[test]
fn pending_result_absence_empty_chunks_and_neighbor_boundaries_are_distinct() {
    let mut view = fixture(0);
    view.cells.remove(&Key::OutputChunk(0, 1, 7));
    let m = view.message(0).unwrap().unwrap();
    assert!(
        matches!(&m.blocks[1].content,BlockContent::ToolResult(r) if r.output==Some(ToolOutput::Text(String::new())))
    );
    view.cells.remove(&Key::OutputChunk(0, 1, 0));
    view.cells.insert(
        Key::OutputChunk(1, 0, 0),
        Cell::Text("other message".into()),
    );
    let m = view.message(0).unwrap().unwrap();
    assert!(matches!(&m.blocks[1].content,BlockContent::ToolResult(r) if r.output.is_none()));
    view.cells.remove(&Key::PendingResult(0, 1));
    let m = view.message(0).unwrap().unwrap();
    assert!(
        matches!(&m.blocks[1].content,BlockContent::ToolResult(r) if r.output==Some(ToolOutput::Text("superseded".into())))
    );
}
#[test]
fn corrupt_state_or_read_failure_fuses_without_retrying() {
    let view = fixture(1);
    let meter = Meter {
        view: &view,
        reads: Counter::new(0),
        fail: true,
    };
    let mut cursor = Cursor::new(0, Mode::Annotated);
    assert!(cursor.advance(&meter).is_err());
    let reads = meter.reads.get();
    assert!(cursor.advance(&meter).is_err());
    assert_eq!(reads, meter.reads.get());
    let mut view = fixture(1);
    view.cells.remove(&Key::Child("wrapper:opaque".into(), 0));
    assert!(view.projected_message(0).is_err());
    view.cells.insert(Key::Message(0), Cell::Present);
    assert!(view.message(0).is_err());
    assert!(view.message(99).unwrap().is_none());
}
