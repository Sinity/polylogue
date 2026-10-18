//! Session documents built by a real owner from synthetic source files.
use super::*;
use polylogue_engine::owner::{self, ImportRequest, Observation, Owner};
use polylogue_model::{MessageId, Origin, SessionId};
use polylogue_store::parser_state::Phase;
use polylogue_store::{InputFormat, ReadHandle, StoredInput, Writer};
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
// Debug test runs start many owners at once, so source processing gets a
// longer wait than document writes.
const SOURCE_WAIT: Duration = Duration::from_secs(120);
const POLL: Duration = Duration::from_millis(5);
/// Calls an owner request again while the owner is busy, then waits for the reply.
macro_rules! retry_busy {
    ($call:expr) => {{
        let deadline = Instant::now() + WAIT;
        loop {
            match $call {
                Ok(reply) => break reply.recv_timeout(WAIT).unwrap().unwrap(),
                Err(owner::Error::Busy) if Instant::now() < deadline => thread::sleep(POLL),
                Err(e) => panic!("owner request: {e}"),
            }
        }
    }};
}
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
    fn import(&self, bytes: &[u8], id: &str) -> StoredInput {
        self.import_as(bytes, id, SID)
    }
    fn import_as(&self, bytes: &[u8], id: &str, native: &str) -> StoredInput {
        self.import_context(bytes, id, vec!["claude-main".into(), native.into()])
    }
    fn import_context(&self, bytes: &[u8], id: &str, context: Vec<String>) -> StoredInput {
        fs::write(&self.path, bytes).unwrap();
        let request = ImportRequest {
            format: InputFormat::ClaudeCodeJsonl,
            context,
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
        retry_busy!(handle.try_import(request.clone()))
    }
    fn progress(&self, input: &StoredInput) -> owner::Progress {
        let handle = self.owner.as_ref().unwrap().handle();
        retry_busy!(handle.try_progress(&input.id))
    }
    fn session(&self) -> SessionId {
        SessionId::new(&Origin::new("claude-code").unwrap(), SID).unwrap()
    }
    fn wait_selected(&self, input: &StoredInput, native_ids: &[&str]) {
        let deadline = Instant::now() + SOURCE_WAIT;
        loop {
            let progress = self.progress(input);
            assert!(progress.failure.is_none(), "{:?}", progress.failure);
            assert_eq!(progress.source_rows.parsed_from.input, input.id);
            if native_ids.iter().all(|native| {
                self.reader
                    .selected_message(&MessageId::new(self.session(), native).unwrap())
                    .unwrap()
                    .is_some()
            }) {
                assert_eq!(
                    progress.parser.unwrap().finalization_pending,
                    !matches!(
                        progress.source_rows.state,
                        polylogue_model::source_rows::SourceRowState::Complete
                    )
                );
                return;
            }
            // A missing companion file may be reported after these rows are
            // selected; the document's completion status covers that case.
            assert!(
                progress.recovery_error.is_none(),
                "{:?}",
                progress.recovery_error
            );
            assert!(
                Instant::now() < deadline,
                "selected Message deadline: {progress:?}"
            );
            thread::sleep(POLL);
        }
    }
    fn wait_rows(&self, input: &StoredInput, unavailable: bool) {
        let deadline = Instant::now() + SOURCE_WAIT;
        loop {
            let progress = self.progress(input);
            assert!(progress.recovery_error.is_none(), "{progress:?}");
            if !unavailable {
                assert!(progress.failure.is_none(), "{progress:?}");
            }
            let done = if unavailable {
                matches!(
                    progress.source_rows.state,
                    polylogue_model::source_rows::SourceRowState::Unavailable { .. }
                )
            } else {
                matches!(
                    progress.source_rows.state,
                    polylogue_model::source_rows::SourceRowState::Complete
                )
            };
            if done {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "source row deadline: {progress:?}"
            );
            thread::sleep(POLL);
        }
    }
    fn document(&self) -> (Value, Summary) {
        let session = self
            .reader
            .selected_session(&self.session(), 65536)
            .unwrap()
            .unwrap();
        let mut bytes = Vec::new();
        let summary = write(&session, &mut bytes, Instant::now() + WAIT).unwrap();
        (serde_json::from_slice(&bytes).unwrap(), summary)
    }
}
/// The document writer without cancellation.
fn write(session: &SessionView, out: &mut impl Write, deadline: Instant) -> io::Result<Summary> {
    super::write(session, out, deadline, &mut || false)
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
fn morrow_document_keeps_all_twelve_messages_payloads_and_origins() {
    let world = World::new();
    let input = world.import(MORROW, "morrow-session-document");
    let records = records();
    let ids = records
        .iter()
        .map(|r| r["uuid"].as_str().unwrap())
        .collect::<Vec<_>>();
    world.wait_rows(&input, false);
    world.wait_selected(&input, &ids);
    fs::remove_file(&world.path).unwrap();
    let (document, summary) = world.document();
    assert_eq!(summary.messages, 12);
    assert_eq!(document["result"]["outcome"], "ok");
    assert_eq!(document["result"]["completion"]["state"], "complete");
    assert_eq!(document["session"]["title"], "Project skeleton");
    assert_eq!(document["session"]["display_title"], "Project skeleton");
    assert_eq!(document["session"]["title_source"], "provider");
    assert_eq!(document["session"]["cwd"], "/home/op/morrow");
    assert_eq!(document["session"]["branch"], "main");
    assert_eq!(document["session"]["harness_version"], "2.1.220");
    assert_eq!(document["session"]["models"], json!({"claude-fable-5-1":6}));
    assert_eq!(
        document["session"]["started_at"],
        "2026-10-01T07:00:00.000Z"
    );
    assert_eq!(
        document["session"]["last_activity_at"],
        "2026-10-01T07:42:00.000Z"
    );
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
    assert_eq!(
        world.progress(&input).source_rows.parsed_from.input,
        input.id
    );
}

fn simple(uuid: &str, content: Value) -> Vec<u8> {
    let mut raw=serde_json::to_vec(&json!({"type":"user","uuid":uuid,"sessionId":SID,"timestamp":"2026-10-01T08:00:00.000Z","message":{"role":"user","content":content}})).unwrap();
    raw.push(b'\n');
    raw
}
#[test]
fn document_masks_dense_nested_credentials_and_keeps_unicode() {
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
    let (document, summary) = world.document();
    assert_eq!(summary.messages, 1);
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
}
#[test]
fn pending_message_fails_the_write_before_any_output() {
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
    // A second input, not a revision of the first: the selected message stays
    // while the new one is pending.
    raw.extend(simple(
        "pending-member",
        json!([{"type":"image","source":{"type":"base64","media_type":"image/png","data":"AA=="}}]),
    ));
    let input = world.import(&raw, "pending-session-document");
    let deadline = Instant::now() + WAIT;
    loop {
        let p = world.progress(&input);
        assert_eq!(p.source_rows.parsed_from.input, input.id);
        if p.failure.is_some() {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "unsupported image was not reported as a failure"
        );
        thread::sleep(POLL);
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
    let error =
        write(&view, &mut bytes, Instant::now() + WAIT).expect_err("pending member must refuse");
    assert!(error.to_string().contains("pending Message"), "{error}");
    assert!(bytes.is_empty());
}
#[test]
fn duplicate_sequence_fails_instead_of_guessing_message_order() {
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
    let error =
        write(&view, &mut bytes, Instant::now() + WAIT).expect_err("ambiguous order must refuse");
    assert!(error.to_string().contains("order is unresolved"), "{error}");
    assert!(bytes.is_empty());
}

#[test]
fn pinned_view_ignores_later_inputs_and_cancel_stops_the_status_check() {
    let world = World::new();
    let raw=(json!({"type":"user","uuid":"pinned-row","sessionId":SID,"timestamp":"2026-10-01T08:00:00Z","message":{"role":"user","content":"pinned context"}}).to_string()+"\n").into_bytes();
    let input = world.import(&raw, "01900000-0000-7000-8000-000000000601");
    world.wait_selected(&input, &["pinned-row"]);
    let until = Instant::now() + WAIT;
    while !matches!(
        world.progress(&input).source_rows.state,
        polylogue_model::source_rows::SourceRowState::Complete
    ) {
        assert!(Instant::now() < until);
        thread::sleep(POLL);
    }
    let pinned = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let old_counter = pinned.source_archive_counter();
    let mut first = Vec::new();
    write(&pinned, &mut first, Instant::now() + WAIT).unwrap();
    let first: Value = serde_json::from_slice(&first).unwrap();
    assert!(first["result"]["unknown"].get("record_errors").is_none());
    let later = world.import(b"{not json}\n", "01900000-0000-7000-8000-000000000602");
    let until = Instant::now() + WAIT;
    while !matches!(
        world.progress(&later).source_rows.state,
        polylogue_model::source_rows::SourceRowState::Unavailable { .. }
    ) {
        assert!(Instant::now() < until);
        thread::sleep(POLL);
    }
    let fresh = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    assert!(fresh.source_archive_counter() > old_counter);
    let mut second = Vec::new();
    write(&fresh, &mut second, Instant::now() + WAIT).unwrap();
    let second: Value = serde_json::from_slice(&second).unwrap();
    assert_eq!(second["result"]["unknown"]["record_errors"], 1);
    // An undecodable newer record may hide a branch, so active-path flags go
    // away; content, IDs and origins stay the same.
    let mut original = first["messages"].clone();
    let mut current = second["messages"].clone();
    for messages in [&mut original, &mut current] {
        for message in messages.as_array_mut().unwrap() {
            message.as_object_mut().unwrap().remove("on_active_path");
        }
    }
    assert_eq!(current, original);
    let mut old_again = Vec::new();
    write(&pinned, &mut old_again, Instant::now() + WAIT).unwrap();
    assert_eq!(serde_json::from_slice::<Value>(&old_again).unwrap(), first);
    let mut attempts = 0;
    let cancelled = polylogue_engine::session_status::check_with_cancel(
        &fresh,
        Instant::now() + WAIT,
        65536,
        &mut || {
            attempts += 1;
            attempts > 1
        },
    );
    assert!(matches!(cancelled, Err(polylogue_store::Error::Cancelled)));
    drop(fresh);
    drop(pinned);
    // Cancelling a read does not affect the owner.
    assert!(matches!(
        world.progress(&input).source_rows.state,
        polylogue_model::source_rows::SourceRowState::Complete
    ));
    assert_eq!(world.document().0["result"]["unknown"]["record_errors"], 1);
}

fn titled(uuid: &str) -> Vec<u8> {
    let mut raw =
        (json!({"type":"ai-title","aiTitle":"Synthetic Session"}).to_string() + "\n").into_bytes();
    raw.extend((json!({"type":"user","uuid":uuid,"sessionId":SID,"promptSource":"typed","timestamp":"2026-10-01T08:00:00.000Z","message":{"role":"user","content":"Synthetic prompt"}}).to_string()+"\n").as_bytes());
    raw
}
#[test]
fn query_can_be_complete_while_document_waits_on_possible_foreign_links() {
    let world = World::new();
    let input = world.import(&titled("exact-row"), "exact-good");
    world.wait_rows(&input, false);
    let pinned = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let mut old = Vec::new();
    write(&pinned, &mut old, Instant::now() + WAIT).unwrap();
    assert_eq!(
        serde_json::from_slice::<Value>(&old).unwrap()["result"]["outcome"],
        "ok"
    );
    let unrelated = world.import_as(b"{", "unrelated-tail", "other-main");
    let deadline = Instant::now() + WAIT;
    while world
        .progress(&unrelated)
        .parser
        .is_none_or(|h| h.status != Phase::WaitingTail)
    {
        assert!(Instant::now() < deadline);
        thread::sleep(POLL);
    }
    let catalog = world.reader.session_catalog(65536).unwrap();
    let query = polylogue_model::query::SessionQuery {
        id: Some(world.session().as_str().into()),
        origin: Some("claude-code".into()),
        ..Default::default()
    };
    let selected = polylogue_engine::session_query::execute(
        &catalog,
        &query,
        Instant::now() + WAIT,
        &mut || false,
    )
    .unwrap();
    assert_eq!(selected["outcome"], "ok");
    assert_eq!(selected["up_to_date"], false);
    assert_eq!(selected["completion"]["state"], "complete");
    let fresh = world.document().0;
    assert_eq!(fresh["result"]["outcome"], "degraded");
    assert_eq!(fresh["result"]["completion"]["state"], "pending");
    assert_eq!(
        fresh["messages"],
        serde_json::from_slice::<Value>(&old).unwrap()["messages"]
    );
    let mut again = Vec::new();
    write(&pinned, &mut again, Instant::now() + WAIT).unwrap();
    assert_eq!(old, again);
}
#[test]
fn malformed_input_without_rows_makes_document_degraded() {
    let world = World::new();
    let input = world.import(&titled("known-row"), "malformed-control-good");
    world.wait_rows(&input, false);
    assert_eq!(world.document().0["result"]["outcome"], "ok");
    let malformed = world.import(b"{not json}\n", "malformed-control-only");
    world.wait_rows(&malformed, true);
    let document = world.document().0;
    assert_eq!(document["result"]["outcome"], "degraded");
    assert_eq!(document["result"]["unknown"]["record_errors"], 1);
    assert_eq!(document["messages"].as_array().unwrap().len(), 1);
}
#[test]
fn reading_to_eof_with_missing_companion_is_not_complete() {
    let world = World::new();
    let mut raw = titled("known-row");
    // The tool result's companion file does not exist, so reaching the end of
    // the source file does not finish its rows.
    raw.extend((json!({"type":"user","uuid":"required-result","sessionId":SID,"parentUuid":"known-row","timestamp":"2026-10-01T08:01:00Z","toolUseResult":{"persistedOutputPath":"/absent/tool-results/result.txt"},"message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"missing-companion","content":"<persisted-output>\n/absent/tool-results/result.txt\n</persisted-output>"}]}}).to_string()+"\n").as_bytes());
    let input = world.import(&raw, "required-companion-eof");
    // Wait for the known row first so the expected companion failure cannot
    // race this check.
    let selected = MessageId::new(world.session(), "known-row").unwrap();
    let deadline = Instant::now() + WAIT;
    while world.reader.selected_message(&selected).unwrap().is_none() {
        let progress = world.progress(&input);
        assert!(progress.recovery_error.is_none(), "{progress:?}");
        assert!(
            Instant::now() < deadline,
            "known row not selected: {progress:?}"
        );
        thread::sleep(POLL);
    }
    let deadline = Instant::now() + WAIT;
    loop {
        let p = world.progress(&input);
        assert!(p.recovery_error.is_none(), "{p:?}");
        if p.parser
            .as_ref()
            .is_some_and(|h| h.parsed_upto == input.length)
        {
            if let Some(failure) = &p.failure {
                assert_eq!(
                    failure,
                    "canonical compute: Claude pending locator no longer matches complete source evidence"
                );
            }
            assert!(p.parser.unwrap().finalization_pending);
            break;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(POLL);
    }
    let view = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let q = polylogue_engine::session_status::check(&view, Instant::now() + WAIT, 65536).unwrap();
    assert!(!q.complete());
    assert!(q.pending_inputs > 0 || q.unavailable_inputs > 0);
}

#[test]
fn foreign_link_copy_or_sidechain_keeps_document_incomplete() {
    let world = World::new();
    let input = world.import(&titled("known-row"), "lineage-good");
    world.wait_rows(&input, false);
    let link =
        (json!({"type":"continued-in","continuedInSessionId":SID}).to_string() + "\n").into_bytes();
    let foreign = world.import_as(&link, "lineage-foreign", "foreign-owner");
    world.wait_rows(&foreign, true);
    let document = world.document().0;
    assert_eq!(document["result"]["outcome"], "degraded");
    assert!(
        document["result"]["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|v| v.as_str().unwrap().contains("Session links"))
    );
    for (label, record) in [
        (
            "copy",
            json!({"type":"user","uuid":"known-row","sessionId":"copied-parent","message":{"role":"user","content":"copied record"}}),
        ),
        (
            "sidechain",
            json!({"type":"user","uuid":"side-row","isSidechain":true,"sessionId":SID,"message":{"role":"user","content":"side instruction"}}),
        ),
    ] {
        let input = world.import(&(record.to_string() + "\n").into_bytes(), label);
        let deadline = Instant::now() + WAIT;
        while world
            .progress(&input)
            .parser
            .is_none_or(|h| h.parsed_upto != input.length)
        {
            assert!(Instant::now() < deadline);
            thread::sleep(POLL);
        }
        let view = world
            .reader
            .selected_session(&world.session(), 65536)
            .unwrap()
            .unwrap();
        let q =
            polylogue_engine::session_status::check(&view, Instant::now() + WAIT, 65536).unwrap();
        assert!(!q.complete(), "{label} must keep the session incomplete");
    }
}

#[test]
fn foreign_copy_or_fork_keeps_target_document_incomplete() {
    for how in ["copied-sessionId", "forkedFrom"] {
        let world = World::new();
        let base = world.import(&titled("known-row"), "foreign-copy-base");
        world.wait_rows(&base, false);
        assert_eq!(world.document().0["result"]["outcome"], "ok");
        let mut record = json!({"type":"user","uuid":"foreign-row","sessionId":SID,"promptSource":"typed","message":{"role":"user","content":"Neutral foreign copy text"}});
        if how == "forkedFrom" {
            record["sessionId"] = json!("foreign-owner");
            record["forkedFrom"] = json!({"sessionId":SID,"messageUuid":"known-row"});
        }
        let foreign = world.import_as(
            &(record.to_string() + "\n").into_bytes(),
            how,
            "foreign-owner",
        );
        world.wait_rows(&foreign, true);
        let document = world.document().0;
        assert_eq!(document["result"]["outcome"], "degraded", "{how}");
        assert!(
            document["result"]["notes"]
                .as_array()
                .unwrap()
                .iter()
                .any(|note| note.as_str().unwrap().contains("Session links")),
            "{how}"
        );
    }
}
fn two_row_input(reverse: bool, branched: bool) -> Vec<u8> {
    let mut records = vec![json!({"type":"ai-title","aiTitle":"Synthetic Session"})];
    let ids = if reverse { ["v", "u"] } else { ["u", "v"] };
    for (ordinal, id) in ids.into_iter().enumerate() {
        let parent = if ordinal == 0 || branched {
            None
        } else {
            Some(ids[0])
        };
        records.push(json!({"type":"user","uuid":id,"sessionId":SID,"parentUuid":parent,"promptSource":"typed","message":{"role":"user","content":id}}));
    }
    records
        .into_iter()
        .flat_map(|record| (record.to_string() + "\n").into_bytes())
        .collect()
}
#[test]
fn later_branch_or_reversed_input_drops_active_path_flags() {
    for (reverse, branched) in [(false, true), (true, false)] {
        let world = World::new();
        let base = world.import(&two_row_input(false, false), "topology-base");
        world.wait_rows(&base, false);
        let baseline = world.document().0;
        assert_eq!(baseline["result"]["outcome"], "ok");
        assert!(
            baseline["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|message| message["on_active_path"] == true)
        );
        let later = world.import(&two_row_input(reverse, branched), "topology-later");
        world.wait_rows(&later, true);
        let view = world
            .reader
            .selected_session(&world.session(), 65536)
            .unwrap()
            .unwrap();
        let status =
            polylogue_engine::session_status::check(&view, Instant::now() + WAIT, 65536).unwrap();
        assert!(!status.complete());
        assert!(status.active_messages.is_empty());
        let document = world.document().0;
        assert!(
            document["messages"]
                .as_array()
                .unwrap()
                .iter()
                .all(|message| message.get("on_active_path").is_none())
        );
    }
}
#[test]
fn inputs_of_one_session_share_one_membership_limit() {
    fn cohort(prefix: &str) -> Vec<u8> {
        let mut raw = (json!({"type":"ai-title","aiTitle":"Synthetic Session"}).to_string() + "\n")
            .into_bytes();
        for i in 0..24 {
            let uuid = format!("{prefix}-{i:03}");
            let parent = (i > 0).then(|| format!("{prefix}-{:03}", i - 1));
            raw.extend((json!({"type":"user","uuid":uuid,"sessionId":SID,"parentUuid":parent,"promptSource":"typed","message":{"role":"user","content":"Synthetic prompt"}}).to_string()+"\n").as_bytes());
        }
        raw
    }
    let world = World::new();
    let first = world.import(&cohort("a"), "membership-first");
    world.wait_rows(&first, false);
    let view = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let first = polylogue_engine::session_status::check(&view, Instant::now() + WAIT, 8000);
    assert!(first.is_ok(), "first input must fit: {first:?}");
    drop(view);
    let second = world.import(&cohort("b"), "membership-second");
    world.wait_rows(&second, false);
    let view = world
        .reader
        .selected_session(&world.session(), 65536)
        .unwrap()
        .unwrap();
    let second = polylogue_engine::session_status::check(&view, Instant::now() + WAIT, 8000);
    assert!(
        matches!(
            second,
            Err(polylogue_store::Error::ParserAdmission {
                resource: "Session cumulative Message membership",
                ..
            })
        ),
        "two inputs must exceed the membership limit: {second:?}"
    );
}

#[test]
fn model_metadata_wraps_opaque_values_after_masking() {
    let original = polylogue_model::RawJson::parse(br#"{"n":1e+02,"s":"\ud800"}"#).unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    let result = metadata(
        |e| {
            e.syntax(b",")?;
            e.key("attrs")?;
            opaque_value(e, &original, true)
        },
        deadline,
    )
    .unwrap();
    assert_eq!(result["attrs"]["masked"], false);
    assert_eq!(result["attrs"]["encoding"], "canonical-json-v1");
    assert_eq!(
        polylogue_model::RawJson::parse(result["attrs"]["json"].as_str().unwrap().as_bytes())
            .unwrap(),
        original
    );
    let mut encoder = Encoder::new(Vec::new(), deadline);
    opaque_value(&mut encoder, &original, false).unwrap();
    assert_eq!(encoder.finish().unwrap(), original.canonical());
}

mod canonical_opaque;
mod configured_claude;
mod large_string;
mod native_id_privacy;
mod reference_guard;
