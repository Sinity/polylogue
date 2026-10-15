//! Actual Owner-driven selected Session documents; synthetic retained sources only.
use super::*;
use polylogue_engine::owner::{self, ImportRequest, Observation, Owner};
use polylogue_model::{MessageId, Origin, SessionId};
use polylogue_store::{
    InputFormat, ReadHandle, RetainedInput, Writer, contributions::SessionCoverage,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
    thread,
    time::{Duration, Instant},
};

const SID: &str = "7f3a2c10-0001-4000-8000-000000000001";
const MORROW: &[u8] = include_bytes!("../../../sources/tests/fixtures/morrow-main.jsonl");
const WAIT: Duration = Duration::from_secs(40);
struct World {
    owner: Option<Owner>,
    reader: ReadHandle,
    _dir: tempfile::TempDir,
    path: PathBuf,
}
impl World {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let writer = Writer::open(&dir.path().join("archive")).unwrap();
        let reader = writer.read_handle().unwrap();
        let path = dir.path().join("source.jsonl");
        Self {
            owner: Some(Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap()),
            reader,
            _dir: dir,
            path,
        }
    }
    fn import(&self, bytes: &[u8], id: &str) -> RetainedInput {
        fs::write(&self.path, bytes).unwrap();
        let request = ImportRequest {
            format: InputFormat::ClaudeCodeJsonl,
            context: vec!["claude-main".into(), SID.into()],
            path: self.path.clone(),
            observation: Observation {
                id: id.into(),
                machine: "synthetic-machine".into(),
                host: "synthetic-host".into(),
                path: self.path.to_str().unwrap().into(),
                seen_at: 0,
            },
            metadata_allowance: 65536,
        };
        let handle = self.owner.as_ref().unwrap().handle();
        let deadline = Instant::now() + WAIT;
        loop {
            match handle.try_import(request.clone()) {
                Ok(reply) => return reply.recv_timeout(WAIT).unwrap().unwrap(),
                Err(owner::Error::Busy) if Instant::now() < deadline => thread::yield_now(),
                Err(e) => panic!("import admission: {e}"),
            }
        }
    }
    fn progress(&self, input: &RetainedInput) -> owner::Progress {
        let deadline = Instant::now() + WAIT;
        loop {
            match self
                .owner
                .as_ref()
                .unwrap()
                .handle()
                .try_progress(&input.id)
            {
                Ok(reply) => return reply.recv_timeout(WAIT).unwrap().unwrap(),
                Err(owner::Error::Busy) if Instant::now() < deadline => thread::yield_now(),
                Err(e) => panic!("progress admission: {e}"),
            }
        }
    }
    fn session(&self) -> SessionId {
        SessionId::new(&Origin::new("claude-code").unwrap(), SID).unwrap()
    }
    fn wait_selected(&self, input: &RetainedInput, native_ids: &[&str]) {
        let deadline = Instant::now() + WAIT;
        loop {
            let progress = self.progress(input);
            assert!(progress.failure.is_none(), "{:?}", progress.failure);
            assert!(
                progress.recovery_error.is_none(),
                "{:?}",
                progress.recovery_error
            );
            assert!(!progress.canonical_complete);
            if native_ids.iter().all(|native| {
                self.reader
                    .selected_message(&MessageId::new(self.session(), native).unwrap())
                    .unwrap()
                    .is_some()
            }) {
                assert!(progress.parser.unwrap().finalization_pending);
                return;
            }
            assert!(Instant::now() < deadline, "selected Message deadline");
            thread::yield_now();
        }
    }
    fn document(&self) -> (Value, Coverage) {
        let session = self
            .reader
            .selected_session(&self.session(), 65536)
            .unwrap()
            .unwrap();
        let mut bytes = Vec::new();
        let coverage = write(&session, &mut bytes, Instant::now() + WAIT).unwrap();
        (serde_json::from_slice(&bytes).unwrap(), coverage)
    }
}
impl Drop for World {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            let _ = owner.shutdown();
        }
    }
}
fn records() -> Vec<Value> {
    MORROW
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .filter(|v: &Value| v.get("uuid").is_some())
        .collect()
}
#[test]
fn retained_morrow_document_preserves_twelve_messages_payloads_and_exact_origins() {
    let world = World::new();
    let input = world.import(MORROW, "morrow-session-document");
    let records = records();
    let ids = records
        .iter()
        .map(|r| r["uuid"].as_str().unwrap())
        .collect::<Vec<_>>();
    world.wait_selected(&input, &ids);
    fs::remove_file(&world.path).unwrap();
    let (document, coverage) = world.document();
    assert_eq!(coverage.messages, 12);
    assert!(matches!(
        coverage.catalog,
        SessionCoverage::KnownCatalogOnly
    ));
    assert_eq!(document["format"], "polylogue-session");
    assert_eq!(document["version"], 1);
    assert_eq!(document["session"]["id"], world.session().as_str());
    assert_eq!(document["session"]["native_id"], SID);
    assert_eq!(document["session"]["origin"], "claude-code");
    assert_eq!(document["session"]["kind"], "main");
    let messages = document["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 12);
    for (message, raw) in messages.iter().zip(&records) {
        let id = MessageId::new(world.session(), raw["uuid"].as_str().unwrap()).unwrap();
        assert_eq!(message["id"], id.as_str());
        assert_eq!(message["role"], raw["type"]);
        assert_eq!(message["at"], raw["timestamp"]);
        let content = &raw["message"]["content"];
        let expected: Vec<Value> = if let Some(text) = content.as_str() {
            vec![json!({"type":"text","text":text})]
        } else {
            content.as_array().unwrap().clone()
        };
        let blocks = message["blocks"].as_array().unwrap();
        assert_eq!(blocks.len(), expected.len());
        assert_eq!(
            message["written_by"],
            if raw["type"] == "assistant" {
                "model"
            } else if expected.iter().any(|b| b["type"] == "tool_result") {
                "harness"
            } else {
                "human"
            }
        );
        for (index, (block, source)) in blocks.iter().zip(&expected).enumerate() {
            assert_eq!(block["id"], format!("{}:{index}", id.as_str()));
            let kind = source["type"].as_str().unwrap();
            match kind {
                "text" | "thinking" => {
                    assert_eq!(block["kind"], kind);
                    assert_eq!(
                        block["text"],
                        source[if kind == "text" { "text" } else { "thinking" }]
                    );
                    if let Some(signature) = source.get("signature") {
                        assert_eq!(&block["signature"], signature);
                    }
                }
                "tool_use" => {
                    assert_eq!(block["kind"], "tool_call");
                    assert_eq!(block["tool"], source["name"]);
                    assert_eq!(block["input"], source["input"]);
                    assert_eq!(block["call_id"], source["id"]);
                }
                "tool_result" => {
                    assert_eq!(block["kind"], "tool_result");
                    assert_eq!(block["text"], source["content"]);
                    assert_eq!(block["call_id"], source["tool_use_id"]);
                    assert_eq!(block["outcome"], "ok");
                }
                other => panic!("unexpected synthetic fixture block {other}"),
            }
        }
        if raw["type"] == "assistant" {
            assert_eq!(
                message["usage"]["input"],
                raw["message"]["usage"]["input_tokens"]
            );
            assert_eq!(
                message["usage"]["output"],
                raw["message"]["usage"]["output_tokens"]
            );
            assert_eq!(message["usage"]["service_tier"], "standard");
            assert!(
                message["attrs"]
                    .as_object()
                    .unwrap()
                    .contains_key("stop_sequence")
            );
            assert!(message["attrs"]["stop_sequence"].is_null());
        }
        let spans = message["sources"].as_array().unwrap();
        let mut offset = 0u64;
        let mut original_range = None;
        for line in MORROW.split_inclusive(|b| *b == b'\n') {
            let payload = line.strip_suffix(b"\n").unwrap_or(line);
            let original: Value = serde_json::from_slice(payload).unwrap();
            if original.get("uuid") == raw.get("uuid") {
                original_range = Some((offset, offset + payload.len() as u64));
            }
            offset += line.len() as u64;
        }
        let (start, end) = original_range.unwrap();
        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0]["start"], start);
        assert_eq!(spans[0]["end"], end);
        assert!(spans.iter().all(|span| span["input"] == input.id
            && span["preferred"] == true
            && span["start"].as_u64().unwrap() < span["end"].as_u64().unwrap()));
        let origins = message["field_origins"].as_array().unwrap();
        assert!(!origins.is_empty());
        assert!(
            origins
                .iter()
                .all(|fact| fact["origin"]["binding"]["input"] == input.id
                    && fact["origin"]["format"] == "claude-code-field-origin")
        );
    }
    let result = &messages[4]["blocks"][0];
    assert_eq!(
        result["data"]["stdout"],
        records[4]["toolUseResult"]["stdout"]
    );
    assert_eq!(result["data"]["stderr"], "");
    assert_eq!(result["data"]["interrupted"], false);
    assert!(!world.progress(&input).canonical_complete);
}

fn simple(uuid: &str, content: Value) -> Vec<u8> {
    let mut raw=serde_json::to_vec(&json!({"type":"user","uuid":uuid,"sessionId":SID,"timestamp":"2026-10-01T08:00:00.000Z","message":{"role":"user","content":content}})).unwrap();
    raw.push(b'\n');
    raw
}
#[test]
fn response_masking_preserves_dense_nested_strings_and_unicode_without_raw_credentials() {
    let world = World::new();
    let token = format!("ghp_{}", "A".repeat(36));
    let dense = (0..80)
        .map(|_| token.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    // A token straddles the renderer's ordinary 4096-byte boundary. The dense
    // tool input independently exercises many short JSON string values.
    let text = format!("{} {} suffix λ", "x".repeat(4089), token);
    let mut raw=serde_json::to_vec(&json!({"type":"assistant","uuid":"masked-message","sessionId":SID,"requestId":"masked-request","message":{"id":"masked-response","content":[{"type":"text","text":text},{"type":"tool_use","id":"masked-call","name":"Bash","input":{"command":"true","dense":dense,"nested":(0..80).map(|_|json!({"short":token})).collect::<Vec<_>>()}}],"stop_reason":"tool_use"}})).unwrap();
    raw.push(b'\n');
    let input = world.import(&raw, "masked-session-document");
    world.wait_selected(&input, &["masked-message"]);
    fs::remove_file(&world.path).unwrap();
    let (document, coverage) = world.document();
    assert_eq!(coverage.messages, 1);
    let encoded = serde_json::to_string(&document).unwrap();
    assert!(!encoded.contains(&token));
    assert!(encoded.contains("[credential:"));
    let text = document["messages"][0]["blocks"][0]["text"]
        .as_str()
        .unwrap();
    assert!(text.starts_with(&format!("{} ", "x".repeat(4089))));
    assert!(text.ends_with(" suffix λ"));
    assert_eq!(
        document["messages"][0]["blocks"][1]["input"]["nested"]
            .as_array()
            .unwrap()
            .len(),
        80
    );
    assert!(matches!(
        coverage.catalog,
        SessionCoverage::KnownCatalogOnly
    ));
}
#[test]
fn known_pending_member_refuses_before_emitting_a_complete_looking_subset() {
    let world = World::new();
    let mut raw = simple("ready-member", json!("ready text"));
    let first = world.import(&raw, "ready-session-document");
    world.wait_selected(&first, &["ready-member"]);
    let ready = world
        .reader
        .selected_message(&MessageId::new(world.session(), "ready-member").unwrap())
        .unwrap()
        .unwrap();
    let ready_body = ready.body_pk();
    // This is a second immutable input, not a proved revision of the first.
    // Preserve the already selected Message while the new member stays pending.
    raw.extend(simple(
        "pending-member",
        json!([{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]),
    ));
    let input = world.import(&raw, "pending-session-document");
    let deadline = Instant::now() + WAIT;
    loop {
        let p = world.progress(&input);
        assert!(!p.canonical_complete);
        if p.failure.is_some() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "unsupported image did not remain explicit"
        );
        thread::yield_now();
    }
    assert!(
        world
            .reader
            .selected_message(&MessageId::new(world.session(), "ready-member").unwrap())
            .unwrap()
            .is_some()
    );
    let view = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let polylogue_store::contributions::SessionStep::Ready {
        cursor, message, ..
    } = view.next(None, 65536).unwrap()
    else {
        panic!("previous selected Message must remain ready")
    };
    assert_eq!(message.body_pk(), ready_body);
    assert_eq!(ready.id(), message.id());
    let polylogue_store::contributions::SessionStep::Pending { id, sequence, .. } =
        view.next(Some(&cursor), 65536).unwrap()
    else {
        panic!("unsupported member must remain a known pending row")
    };
    assert_eq!(
        id,
        MessageId::new(world.session(), "pending-member").unwrap()
    );
    assert_eq!(sequence, 1);
    let mut bytes = Vec::new();
    let error = write(&view, &mut bytes, Instant::now() + WAIT)
        .err()
        .expect("pending member must refuse");
    assert!(error.to_string().contains("pending Message"), "{error}");
    assert!(bytes.is_empty());
}
#[test]
fn duplicate_source_sequence_refuses_instead_of_sorting_messages_by_primary_key() {
    let world = World::new();
    let first = world.import(
        &simple("first-import", json!("first")),
        "first-session-document",
    );
    world.wait_selected(&first, &["first-import"]);
    let second = world.import(
        &simple("independent-import", json!("second")),
        "second-session-document",
    );
    world.wait_selected(&second, &["first-import", "independent-import"]);
    let view = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let mut bytes = Vec::new();
    let error = write(&view, &mut bytes, Instant::now() + WAIT)
        .err()
        .expect("ambiguous order must refuse");
    assert!(error.to_string().contains("order is unresolved"), "{error}");
    assert!(bytes.is_empty());
}
