//! Synthetic source-core tests, not runtime/canonical-writer qualification.
use polylogue_model::*;
use polylogue_sources::claude_code::keyed::*;
use polylogue_sources::claude_code::{FileCtx, KeyedRecords, StepOutput};
use polylogue_sources::frame::{Counts, FailurePhase, JsonlDriver, StepBudget, drive_jsonl};
use serde_json::{Value, json};
use std::cell::Cell;

fn line(value: Value) -> String {
    value.to_string() + "\n"
}
fn assistant(uuid: &str, response: &str, request: &str) -> Value {
    json!({"type":"assistant","uuid":uuid,"sessionId":"s","requestId":request,
        "timestamp":"2026-10-01T10:00:00Z","message":{"id":response,"model":"synthetic",
        "content":[{"type":"text","text":uuid}]}})
}
fn user(uuid: &str) -> Value {
    json!({"type":"user","uuid":uuid,"sessionId":"s","promptSource":"typed","message":{"content":"synthetic user"}})
}
fn step(
    memory: &mut MemoryReadView,
    ctx: FileCtx,
    bytes: &[u8],
    base: u64,
) -> (StepOutput, Counts) {
    let framed = drive_jsonl(KeyedRecords::new(ctx, &*memory).unwrap(), bytes, base);
    let mut out = framed.output.unwrap();
    out.header.parsed_upto = framed.parsed_upto;
    memory.apply(out.header.clone(), out.delta.clone());
    (out, framed.counts)
}
fn response(memory: &MemoryReadView, native: &str, key: &ResponseKey) -> ResponseCell {
    memory.response(native, key).unwrap().unwrap()
}
fn contribution(memory: &MemoryReadView, response: &ResponseCell, item: u64) -> Contribution {
    memory
        .contribution(&response.message.id, item)
        .unwrap()
        .unwrap()
}

#[test]
fn line_steps_equal_whole_core_with_nonadjacent_response_and_isolated_side_session() {
    let mut side = user("side1");
    side["isSidechain"] = json!(true);
    side["parentUuid"] = json!("u");
    let mut child = assistant("side2", "m1", "r1");
    child["isSidechain"] = json!(true);
    child["parentUuid"] = json!("side1");
    let records = vec![
        user("u"),
        assistant("a1", "m1", "r1"),
        side,
        child,
        user("u2"),
        assistant("a2", "m2", "r2"),
        assistant("a3", "m1", "r1"),
        json!({"type":"custom-title","customTitle":"Kept title"}),
        json!({"type":"system","subtype":"compact_boundary"}),
        json!({"type":"continued-in","continuedInSessionId":"next"}),
    ];
    let text: String = records.iter().cloned().map(line).collect();
    let mut whole = MemoryReadView::default();
    let (whole_out, counts) = step(&mut whole, FileCtx::main("s"), text.as_bytes(), 0);
    assert!(counts.balanced());
    for group in [1, 2, 3, records.len()] {
        let mut memory = MemoryReadView::default();
        let mut offset = 0;
        let mut used = 0;
        let mut events = Vec::new();
        let mut links = Vec::new();
        for chunk in records.chunks(group) {
            let text: String = chunk.iter().cloned().map(line).collect();
            let (out, count) = step(&mut memory, FileCtx::main("s"), text.as_bytes(), offset);
            offset += text.len() as u64;
            used += count.used;
            events.extend(out.events);
            links.extend(out.links);
        }
        assert_eq!(used, counts.used);
        assert_eq!(events, whole_out.events);
        assert_eq!(links, whole_out.links);
        assert_eq!(memory.header().unwrap(), whole.header().unwrap());
        for native in ["s", "s~side-side1"] {
            assert_eq!(
                memory.session(native).unwrap(),
                whole.session(native).unwrap()
            );
        }
        for (native, key) in [
            ("s", ResponseKey::new("m1", "r1")),
            ("s", ResponseKey::new("m2", "r2")),
            ("s~side-side1", ResponseKey::new("m1", "r1")),
        ] {
            let actual = response(&memory, native, &key);
            let expected = response(&whole, native, &key);
            assert_eq!(actual, expected);
            for item in 0..actual.items {
                assert_eq!(
                    contribution(&memory, &actual, item),
                    contribution(&whole, &expected, item)
                );
            }
        }
        assert_eq!(
            response(&memory, "s", &ResponseKey::new("m1", "r1")).items,
            2
        );
        assert_eq!(
            response(&memory, "s~side-side1", &ResponseKey::new("m1", "r1")).items,
            1
        );
    }
}

#[test]
fn uuid_replacement_keeps_item_slot_and_metadata_supplements_replace_fields_atomically() {
    let mut memory = MemoryReadView::default();
    let key = ResponseKey::new("m", "r");
    let mut first = assistant("a", "m", "r");
    first["message"]["usage"] = json!({"input_tokens":10,"output_tokens":5});
    step(
        &mut memory,
        FileCtx::main("s"),
        line(first.clone()).as_bytes(),
        0,
    );
    let before = response(&memory, "s", &key);
    let first_raw = contribution(&memory, &before, 0).raw;
    step(
        &mut memory,
        FileCtx::main("s"),
        line(first.clone()).as_bytes(),
        0,
    );
    assert_eq!(
        response(&memory, "s", &key).revision,
        before.revision,
        "identical duplicate does not advance selected metadata"
    );
    let supplement = json!({"type":"assistant","uuid":"a","message":{"usage":{"output_tokens":7}}});
    let (out, _) = step(
        &mut memory,
        FileCtx::main("s"),
        line(supplement).as_bytes(),
        0,
    );
    assert_eq!(out.delta.contributions.len(), 1);
    let revised = response(&memory, "s", &key);
    assert_eq!(revised.items, 1);
    assert_eq!(revised.message.usage.as_ref().unwrap().input, None);
    assert_eq!(revised.message.usage.as_ref().unwrap().output, Some(7));
    assert_eq!(
        contribution(&memory, &revised, 0).blocks.len(),
        1,
        "metadata-only supplement keeps blocks"
    );
    let mut changed = first.clone();
    changed["message"]["content"] = json!([]);
    step(&mut memory, FileCtx::main("s"), line(changed).as_bytes(), 0);
    assert!(
        contribution(&memory, &response(&memory, "s", &key), 0)
            .blocks
            .is_empty()
    );
    step(&mut memory, FileCtx::main("s"), line(first).as_bytes(), 0);
    let reverted = response(&memory, "s", &key);
    assert_eq!(reverted.items, 1);
    assert_eq!(
        contribution(&memory, &reverted, 0).raw,
        first_raw,
        "A,B,A uses current effective A evidence, not transition history"
    );
    assert!(
        reverted.message.blocks.is_empty() && reverted.message.version_hash.is_none(),
        "reflow/hash remain explicit preparation, not fabricated canonical output"
    );
}

#[test]
fn selected_metadata_evidence_corrects_base_slot_order_without_coordinate_salt() {
    let mut memory = MemoryReadView::default();
    let key = ResponseKey::new("m", "r");
    let mut a = assistant("a", "m", "r");
    a["message"]["usage"] = json!({"output_tokens":1});
    let mut b = assistant("b", "m", "r");
    b["message"]["usage"] = json!({"output_tokens":2});
    step(
        &mut memory,
        FileCtx::main("s"),
        (line(a.clone()) + &line(b)).as_bytes(),
        0,
    );
    step(
        &mut memory,
        FileCtx::main("s"),
        line(json!({"type":"assistant","uuid":"a","message":{"usage":{"output_tokens":3}}}))
            .as_bytes(),
        999,
    );
    let cell = response(&memory, "s", &key);
    let mut baseline: [Option<Vec<u8>>; 5] = Default::default();
    let mut cursor = ContributionCursor::default();
    let mut n = 0;
    while let Some((message, item)) = cell.next_contribution(&mut cursor) {
        let c = memory.contribution(&message, item).unwrap().unwrap();
        for (i, field) in c.final_fields().unwrap().into_iter().enumerate() {
            if field.is_some() {
                baseline[i] = field;
            }
        }
        n += 1;
    }
    assert_eq!(n, 2);
    let units = cell.metadata_units(&baseline);
    assert_eq!(units.len(), 1);
    assert!(
        matches!(&units[0],EvidenceUnit::Json{tag,canonical,whole_record:false} if tag=="response/usage" && canonical==br#"{"output_tokens":3}"#)
    );
    let before = cell.revision;
    step(&mut memory, FileCtx::main("s"), line(a).as_bytes(), 0);
    assert_eq!(response(&memory, "s", &key).revision, before + 1);
}

#[test]
fn response_tuple_is_typed_and_copies_keep_separate_opaque_relations() {
    let mut memory = MemoryReadView::default();
    let mut a = assistant("a", "x\u{1f}y", "z");
    a["sessionId"] = json!("source:colon");
    let text = line(a) + &line(assistant("b", "x", "y\u{1f}z"));
    let (out, _) = step(&mut memory, FileCtx::main("dest:colon"), text.as_bytes(), 0);
    assert_eq!(out.delta.responses.len(), 2);
    let copied = response(&memory, "dest:colon", &ResponseKey::new("x\u{1f}y", "z"));
    assert_eq!(
        copied.message.id.session(),
        &SessionId::new(&Origin::new("claude-code").unwrap(), "dest:colon").unwrap()
    );
    assert_eq!(
        copied.message.inherited_from.unwrap().session(),
        &SessionId::new(&Origin::new("claude-code").unwrap(), "source:colon").unwrap()
    );
}

#[test]
fn required_companions_leave_missing_work_explicit_and_optional_replicas_do_not_join_evidence() {
    let value = json!({"type":"user","uuid":"u","sessionId":"s","toolUseResult":{"persistedOutputPath":"/source/tool-results/c.txt"},"message":{"content":[{"type":"tool_result","tool_use_id":"c","content":"preview"}]}});
    let mut memory = MemoryReadView::default();
    let (out, _) = step(
        &mut memory,
        FileCtx::main("s"),
        line(value.clone()).as_bytes(),
        0,
    );
    assert!(out.delta.pending_messages[0].incomplete);
    let key = ResponseKey::user("u");
    let cell = response(&memory, "s", &key);
    let c = contribution(&memory, &cell, 0);
    assert!(c.body_unit(0, 0).is_err());
    assert!(
        matches!(&c.blocks[0],BlockContent::ToolResult(result) if result.outcome==ToolOutcome::Unknown{reason:UnknownReason::OutputNotRecorded})
    );
    let payload = BodyPayload {
        blob: BlobHash::of(b"complete bytes"),
        length: 14,
    };
    memory.set_body("c.txt", Some(payload.clone()));
    let parser = KeyedRecords::new(FileCtx::main("s"), &memory).unwrap();
    let framed = drive_jsonl(parser, line(value).as_bytes(), 0);
    let out = framed.output.unwrap();
    assert!(!out.delta.pending_messages[0].incomplete);
    let contribution = &out.delta.contributions[0].2;
    assert!(
        matches!(contribution.body_unit(0,0).unwrap(),Some(EvidenceUnit::Body{payload:p,..}) if p==payload)
    );
    let optional = user("optional");
    memory.set_body("optional.txt", Some(payload));
    let parser = KeyedRecords::new(FileCtx::main("s"), &memory).unwrap();
    let framed = drive_jsonl(parser, line(optional).as_bytes(), 0);
    assert!(
        framed.output.unwrap().delta.contributions[0]
            .2
            .bodies
            .is_empty()
    );
}

#[test]
fn complete_inline_duplicate_does_not_add_required_body_version_evidence() {
    let value = json!({"type":"user","uuid":"u","sessionId":"s","toolUseResult":{"persistedOutputPath":"c.txt"},"message":{"content":[{"type":"tool_result","tool_use_id":"c","content":"complete inline"}]}});
    let mut memory = MemoryReadView::default();
    memory.set_body(
        "c.txt",
        Some(BodyPayload {
            blob: BlobHash::of(b"complete inline"),
            length: 15,
        }),
    );
    let output = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &memory).unwrap(),
        line(value).as_bytes(),
        0,
    )
    .output
    .unwrap();
    let contribution = &output.delta.contributions[0].2;
    assert!(contribution.bodies.is_empty());
    assert!(
        matches!(&contribution.blocks[0],BlockContent::ToolResult(result) if result.output==Some(polylogue_model::ToolOutput::Text("complete inline".into())))
    );
}

struct Metered<'a> {
    memory: &'a MemoryReadView,
    reads: Cell<usize>,
    fail: Option<&'static str>,
}
impl Metered<'_> {
    fn read(&self, name: &'static str) -> Result<(), &'static str> {
        self.reads.set(self.reads.get() + 1);
        if self.fail == Some(name) {
            Err(name)
        } else {
            Ok(())
        }
    }
}
impl ReadView for Metered<'_> {
    type Error = &'static str;
    fn header(&self) -> Result<Header, Self::Error> {
        self.read("header")?;
        Ok(self.memory.header().unwrap())
    }
    fn session(&self, native: &str) -> Result<Option<SessionCell>, Self::Error> {
        self.read("session")?;
        let mut cell = self.memory.session(native).unwrap();
        if self.fail == Some("session-corrupt") {
            if let Some(cell) = &mut cell {
                cell.session.native_id = "different".into();
            }
        }
        Ok(cell)
    }
    fn response(
        &self,
        session: &str,
        key: &ResponseKey,
    ) -> Result<Option<ResponseCell>, Self::Error> {
        self.read("response")?;
        if self.fail == Some("response-missing") {
            return Ok(None);
        }
        Ok(self.memory.response(session, key).unwrap())
    }
    fn record(&self, session: &str, uuid: &str) -> Result<Option<RecordAddress>, Self::Error> {
        self.read("record")?;
        Ok(self.memory.record(session, uuid).unwrap())
    }
    fn contribution(
        &self,
        message: &MessageId,
        item: u64,
    ) -> Result<Option<Contribution>, Self::Error> {
        self.read("contribution")?;
        if self.fail == Some("contribution-missing") {
            return Ok(None);
        }
        Ok(self.memory.contribution(message, item).unwrap())
    }
    fn side(&self, uuid: &str) -> Result<Option<String>, Self::Error> {
        self.read("side")?;
        Ok(self.memory.side(uuid).unwrap())
    }
    fn has_link(&self, key: &LinkKey) -> Result<bool, Self::Error> {
        self.read("link")?;
        Ok(self.memory.has_link(key).unwrap())
    }
    fn body(&self, path: &str) -> Result<Option<BodyPayload>, Self::Error> {
        self.read("body")?;
        Ok(self.memory.body(path).unwrap())
    }
}
fn append_cost(prefix: usize) -> (usize, usize) {
    let mut memory = MemoryReadView::default();
    let mut text = line(user("u"));
    for i in 0..prefix {
        text += &line(assistant(&format!("a{i:05}"), &format!("m{i:05}"), "r"));
    }
    step(&mut memory, FileCtx::main("s"), text.as_bytes(), 0);
    let meter = Metered {
        memory: &memory,
        reads: Cell::new(0),
        fail: None,
    };
    let base = memory.header().unwrap().parsed_upto;
    assert_eq!(base, text.len() as u64);
    let text = line(assistant("continued", "m00000", "r"));
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &meter).unwrap(),
        text.as_bytes(),
        base,
    );
    let mut out = framed.output.unwrap();
    assert_eq!(out.delta.sessions.len(), 1);
    assert_eq!(out.delta.responses.len(), 1);
    assert_eq!(out.delta.contributions.len(), 1);
    assert_eq!(out.delta.pending_messages[0].items, 2);
    assert_eq!(out.delta.addresses.len(), 1);
    assert_eq!(out.delta.pending_messages.len(), 1);
    assert_eq!(out.delta.responses[0].2.ordinal, 1);
    assert_eq!(meter.reads.get(), 5);
    let encoded_bytes = serde_json::to_vec(&out.delta).unwrap().len();
    // The u64 counter and absolute original-record coordinates are bounded
    // scalars, not historical collections. Account for their decimal widths
    // exactly, retaining the rest of the serialized delta unchanged.
    fn width(value: u64) -> usize {
        let digits = serde_json::to_vec(&value).unwrap().len();
        assert!((1..=20).contains(&digits));
        digits
    }
    fn relative(
        range: &mut polylogue_sources::selection::RecordRange,
        base: u64,
        length: u64,
    ) -> usize {
        assert_eq!(range.start, base);
        assert_eq!(range.end, base.checked_add(length).unwrap());
        let absolute_width = width(range.start) + width(range.end);
        range.start = 0;
        range.end = length;
        absolute_width - width(range.start) - width(range.end)
    }
    let counter = &mut out.delta.sessions[0].next_message_ordinal;
    assert_eq!(*counter, u64::try_from(prefix).unwrap() + 1);
    let mut scalar_width = width(*counter) - width(0);
    *counter = 0;
    let record_length = text.trim_end_matches('\n').len() as u64;
    let origins = &mut out.delta.contributions[0].2.origins;
    scalar_width += relative(origins.envelope.as_mut().unwrap(), base, record_length);
    for origin in origins.fields.values_mut() {
        scalar_width += relative(&mut origin.record, base, record_length);
    }
    for origin in out.delta.responses[0]
        .2
        .selected_origins
        .iter_mut()
        .flatten()
    {
        scalar_width += relative(&mut origin.record, base, record_length);
    }
    let projection = &mut out.delta.contributions[0].2.projection;
    for fact in projection.facts.iter_mut().chain(&mut projection.retained) {
        if let Some(selection) = &mut fact.selection {
            for input in &mut selection.inputs {
                scalar_width += relative(&mut input.record, base, record_length);
            }
        }
    }
    for fact in out.delta.responses[0].2.selected_facts.iter_mut().flatten() {
        if let Some(selection) = &mut fact.selection {
            for input in &mut selection.inputs {
                scalar_width += relative(&mut input.record, base, record_length);
            }
        }
    }
    let normalized_bytes = serde_json::to_vec(&out.delta).unwrap().len();
    assert_eq!(encoded_bytes - scalar_width, normalized_bytes);
    (meter.reads.get(), normalized_bytes)
}
#[test]
fn single_record_append_point_work_is_independent_of_unique_history() {
    assert_eq!(append_cost(2), append_cost(2000));
}
#[test]
fn failed_point_read_fuses_driver_and_never_produces_checkpoint_output() {
    let mut memory = MemoryReadView::default();
    step(
        &mut memory,
        FileCtx::main("s"),
        line(assistant("a", "m", "r")).as_bytes(),
        0,
    );
    for failure in [
        "header",
        "session",
        "response",
        "record",
        "contribution",
        "side",
        "link",
    ] {
        let meter = Metered {
            memory: &memory,
            reads: Cell::new(0),
            fail: Some(failure),
        };
        let parser = KeyedRecords::new(FileCtx::main("s"), &meter);
        if failure == "header" {
            assert_eq!(parser.err(), Some("header".into()));
            continue;
        }
        let mut value = assistant("a", "m", "r");
        value["message"]["content"] = json!([{"type":"text","text":"changed"}]);
        if failure == "link" {
            value = user("fresh-side");
            value["isSidechain"] = json!(true);
        }
        let text = line(value);
        let mut driver = JsonlDriver::new(parser.unwrap(), 0);
        let budget = StepBudget {
            bytes: usize::MAX,
            records: u64::MAX,
            record_bytes: usize::MAX,
        };
        assert!(driver.advance(text.as_bytes(), budget).is_err());
        let reads = meter.reads.get();
        assert!(driver.advance(text.as_bytes(), budget).is_err());
        assert_eq!(meter.reads.get(), reads);
        match driver.checkpoint() {
            Ok(_) => panic!("failed point read exposed checkpoint"),
            Err(out) => {
                let err = out.output.err().unwrap();
                assert_eq!(err.phase, FailurePhase::Record);
                assert_eq!(err.error.to_string(), failure);
                assert_eq!(out.parsed_upto, 0);
            }
        }
    }
}
#[test]
fn metadata_only_steps_keep_their_owner_and_link_header_only_steps_do_not_invent_messages() {
    let mut memory = MemoryReadView::default();
    let (empty, _) = step(
        &mut memory,
        FileCtx::main("s"),
        line(json!({"type":"custom-title","customTitle":"Before"})).as_bytes(),
        0,
    );
    assert_eq!(empty.delta.sessions.len(), 1);
    assert_eq!(
        empty.delta.sessions[0].session.title.as_deref(),
        Some("Before")
    );
    assert_eq!(empty.delta.sessions[0].next_message_ordinal, 0);
    assert!(empty.delta.responses.is_empty());
    assert!(empty.delta.pending_messages.is_empty());
    step(
        &mut memory,
        FileCtx::main("s"),
        line(user("u")).as_bytes(),
        0,
    );
    let (title, _) = step(
        &mut memory,
        FileCtx::main("s"),
        line(json!({"type":"custom-title","customTitle":"After"})).as_bytes(),
        0,
    );
    assert_eq!(title.delta.sessions.len(), 1);
    assert_eq!(
        title.delta.sessions[0].session.title.as_deref(),
        Some("After")
    );
    assert!(title.delta.pending_messages.is_empty());
    let (link, _) = step(
        &mut memory,
        FileCtx::main("s"),
        line(json!({"type":"continued-in","continuedInSessionId":"next"})).as_bytes(),
        0,
    );
    assert_eq!(link.links.len(), 1);
    assert!(link.delta.responses.is_empty());
}

#[test]
fn side_uuid_supplement_keeps_first_membership_and_conflicting_response_is_counted() {
    let mut memory = MemoryReadView::default();
    let mut value = assistant("side", "m", "r");
    value["isSidechain"] = json!(true);
    step(&mut memory, FileCtx::main("s"), line(value).as_bytes(), 0);
    let update = json!({"type":"assistant","uuid":"side","message":{"content":[]}});
    let (out, _) = step(&mut memory, FileCtx::main("s"), line(update).as_bytes(), 0);
    assert_eq!(out.delta.sessions[0].session.native_id, "s~side-side");
    let old = response(&memory, "s~side-side", &ResponseKey::new("m", "r"));
    assert!(contribution(&memory, &old, 0).blocks.is_empty());
    let conflict = json!({"type":"assistant","uuid":"side","requestId":"different","message":{"content":"bad"}});
    let (out, count) = step(
        &mut memory,
        FileCtx::main("s"),
        line(conflict).as_bytes(),
        0,
    );
    assert_eq!(count.unrecognized.len(), 1);
    assert!(out.delta.responses.is_empty());
    assert_eq!(
        response(&memory, "s~side-side", &ResponseKey::new("m", "r")),
        old
    );
}

#[test]
fn required_body_read_failure_is_not_an_unavailable_payload() {
    let memory = MemoryReadView::default();
    let meter = Metered {
        memory: &memory,
        reads: Cell::new(0),
        fail: Some("body"),
    };
    let value = json!({"type":"user","uuid":"u","toolUseResult":{"persistedOutputPath":"c.txt"},"message":{"content":[{"type":"tool_result","content":"preview"}]}});
    let framed = drive_jsonl(
        KeyedRecords::new(FileCtx::main("s"), &meter).unwrap(),
        line(value).as_bytes(),
        0,
    );
    assert_eq!(framed.output.err().unwrap().error.to_string(), "body");
    assert_eq!(framed.parsed_upto, 0);
}

#[test]
fn missing_or_misidentified_committed_cells_cannot_restart_a_referenced_response() {
    let mut memory = MemoryReadView::default();
    step(
        &mut memory,
        FileCtx::main("s"),
        line(assistant("a", "m", "r")).as_bytes(),
        0,
    );
    for corruption in [
        "response-missing",
        "contribution-missing",
        "session-corrupt",
    ] {
        for changed in [false, true] {
            let view = Metered {
                memory: &memory,
                reads: Cell::new(0),
                fail: Some(corruption),
            };
            let mut update = assistant("a", "m", "r");
            if changed {
                update["message"]["content"] = json!([]);
            }
            let framed = drive_jsonl(
                KeyedRecords::new(FileCtx::main("s"), &view).unwrap(),
                line(update).as_bytes(),
                0,
            );
            assert!(
                framed.output.is_err(),
                "{corruption} must remain a state error"
            );
            assert_eq!(framed.parsed_upto, 0);
        }
    }
}
