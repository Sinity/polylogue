//! Connected owner/read protocol controls against an independently fixed example.
use super::*;
use polylogue_engine::owner::{self, ImportRequest, Observation, Owner};
use polylogue_model::{
    BlobHash,
    authored::{NoteKind, Relation, Target, WriteNote},
};
use polylogue_store::{InputFormat, RetainedInput, Writer};
use std::{
    fs,
    path::PathBuf,
    sync::{Arc, Mutex, atomic::AtomicBool},
    thread,
    time::Duration,
};
const WAIT: Duration = Duration::from_secs(120);
struct World {
    owner: Option<Owner>,
    reader: Option<Mutex<ReadHandle>>,
    dir: tempfile::TempDir,
    archive: PathBuf,
}
impl World {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("archive");
        let writer = Writer::open(&archive).unwrap();
        let reader = Some(Mutex::new(writer.read_handle().unwrap()));
        let owner = Some(Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap());
        Self {
            owner,
            reader,
            dir,
            archive,
        }
    }
    fn handle(&self) -> owner::Handle {
        self.owner.as_ref().unwrap().handle()
    }
    fn write(&self, request: &WriteNote) -> Value {
        let mut request = serde_json::to_value(request).unwrap();
        request["command"] = json!("note");
        crate::protocol::handle(&request, &self.handle())
    }
    fn read(&self, reference: &str, budget: u64) -> Value {
        crate::model_read::execute(
            self.reader.as_ref().unwrap(),
            &json!({"command":"model-read","reference":reference,"budget_tokens":budget}),
            Instant::now() + Duration::from_secs(40),
            &mut || false,
        )
    }
    fn wait_rows(&self, input: &RetainedInput) {
        let until = Instant::now() + WAIT;
        loop {
            let p = self
                .handle()
                .try_progress(&input.id)
                .unwrap()
                .recv_timeout(WAIT)
                .unwrap()
                .unwrap();
            assert!(p.failure.is_none(), "{p:?}");
            if matches!(
                p.source_rows.state,
                polylogue_model::settlement::SourceRowState::Settled
            ) {
                return;
            }
            assert!(Instant::now() < until, "{p:?}");
            thread::sleep(Duration::from_millis(5));
        }
    }
    fn import(&self, native: &str, content: Value) -> RetainedInput {
        let path = self.dir.path().join(format!("{native}.jsonl"));
        let raw = format!(
            "{}\n{}\n",
            json!({"type":"ai-title","sessionId":native,"aiTitle":"Neutral authored tracer"}),
            json!({"type":"user","uuid":"answer","sessionId":native,"parentUuid":null,"promptSource":"typed","timestamp":"2001-01-01T00:00:00.000Z","message":{"role":"user","content":content}})
        );
        fs::write(&path, raw).unwrap();
        let input = self
            .handle()
            .try_import(ImportRequest {
                format: InputFormat::ClaudeCodeJsonl,
                context: vec!["claude-main".into(), native.into()],
                path: path.clone(),
                observation: Observation {
                    id: format!("authored-{native}"),
                    machine: "synthetic-machine".into(),
                    host: "synthetic-host".into(),
                    path: path.to_str().unwrap().into(),
                    seen_at: 0,
                },
                metadata_allowance: 65536,
            })
            .unwrap()
            .recv_timeout(WAIT)
            .unwrap()
            .unwrap();
        self.wait_rows(&input);
        fs::remove_file(path).unwrap();
        input
    }
    fn restart_without_index(&mut self) {
        self.owner.take().unwrap().shutdown().unwrap();
        drop(self.reader.take());
        for name in ["index.db", "index.db-wal", "index.db-shm"] {
            let p = self.archive.join(name);
            if p.exists() {
                fs::remove_file(p).unwrap();
            }
        }
        let writer = Writer::open(&self.archive).unwrap();
        self.reader = Some(Mutex::new(writer.read_handle().unwrap()));
        self.owner = Some(Owner::start(writer, Arc::new(AtomicBool::new(false))).unwrap());
    }
}
impl Drop for World {
    fn drop(&mut self) {
        if let Some(owner) = self.owner.take() {
            owner.shutdown().unwrap();
        }
    }
}
fn example() -> Value {
    serde_json::from_str(include_str!(
        "../../tests/fixtures/authored-correction.json"
    ))
    .unwrap()
}
fn request(e: &Value, key: &str) -> WriteNote {
    WriteNote {
        request_id: e["ids"][key].as_str().unwrap().into(),
        author: e["author"].as_str().unwrap().into(),
        machine: e["machine"].as_str().unwrap().into(),
        created_at: Some(900),
        kind: Some(NoteKind::Correction),
        value: e["values"][key]
            .as_str()
            .unwrap_or("withdrawal reason")
            .into(),
        targets: Some(vec![Target {
            reference: e["target"].as_str().unwrap().into(),
            seen_hash: None,
        }]),
        relations: vec![],
    }
}
fn pin(e: &Value, root: &str, revision: &str) -> String {
    format!(
        "note:{}@{}",
        e["ids"][root].as_str().unwrap(),
        e["ids"][revision].as_str().unwrap()
    )
}
fn root(e: &Value, key: &str) -> String {
    format!("note:{}", e["ids"][key].as_str().unwrap())
}
fn shown(read: &Value) -> &Value {
    &read["family"]["shown"][0]
}
// This checker derives expectations from the fixed example, not native effect IDs.
// Mutated captures must not turn a duplicate action, substituted basis or erased
// history into accepted evidence.
fn check_example(e: &Value, capture: &Value) -> Result<(), &'static str> {
    if capture["retry"]["note"] != e["ids"]["first"]
        || capture["retry"]["revision"]["targets"][0]["seen_hash"] != e["collected_seen_hash"]
    {
        return Err("retry changed original effect");
    }
    if capture["prepared"]["family"]["shown"][0]["revision"]["targets"][0]["seen_hash"]
        != e["prepared_seen_hash"]
        || capture["prepared"]["family"]["shown"][0]["stale"] != true
    {
        return Err("supplied basis substituted");
    }
    let expected = e["expected_history"]
        .as_array()
        .unwrap()
        .iter()
        .map(|id| {
            format!(
                "note:{}@{}",
                e["ids"]["first"].as_str().unwrap(),
                id.as_str().unwrap()
            )
        })
        .collect::<Vec<_>>();
    if capture["final"]["family"]["history"] != json!(expected) {
        return Err("history changed");
    }
    let trail = capture["final"]["family"]["withdrawal_trail"]
        .as_array()
        .ok_or("missing trail")?
        .iter()
        .map(|n| n["revision"]["id"].clone())
        .collect::<Vec<_>>();
    if json!(trail) != e["expected_withdrawal_trail"] {
        return Err("withdrawal trail changed");
    }
    if capture["target"]["authored_notes"]["families"]
        .as_array()
        .ok_or("missing backlinks")?
        .len()
        != 3
    {
        return Err("duplicate action family");
    }
    Ok(())
}
#[test]
fn authored_correction_example_retains_basis_history_and_original_retry_effect() {
    let e = example();
    let mut world = World::new();
    let target = e["target"].as_str().unwrap();
    let a = e["prepared_seen_hash"].as_str().unwrap();
    let b = e["collected_seen_hash"].as_str().unwrap();
    // The preparation is independent of native acquisition. No earlier native
    // read or existing Message rewrite is asserted by this discriminator.
    let mut prepared = request(&e, "prepared");
    prepared.targets.as_mut().unwrap()[0].seen_hash =
        Some(BlobHash::try_from(a.to_owned()).unwrap());
    assert_eq!(world.write(&prepared)["outcome"], "ok");
    let omitted = request(&e, "omitted");
    assert_eq!(world.write(&omitted)["outcome"], "ok");
    let before = world.read(&root(&e, "prepared"), 4000);
    assert!(shown(&before)["stale"].is_null());
    assert_eq!(shown(&before)["target_status"][0]["attachment"], "waiting");
    let input = world.import("authored-example", e["collected_text"].clone());
    let after = world.read(&root(&e, "prepared"), 4000);
    assert_eq!(shown(&after)["stale"], true);
    assert_eq!(shown(&after)["revision"]["targets"][0]["seen_hash"], a);
    let unknown = world.read(&root(&e, "omitted"), 4000);
    assert!(shown(&unknown)["stale"].is_null());
    assert!(
        shown(&unknown)["revision"]["targets"][0]
            .get("seen_hash")
            .is_none()
    );
    let read = world.read(target, 4000);
    assert_ne!(read["outcome"], "error", "{read}");
    assert_eq!(read["shown"][0]["text"], "abc");
    assert_eq!(read["authored_basis"]["seen_hash"], b);
    assert_eq!(read["authored_basis"]["covers"], "whole_target");
    assert_eq!(
        world.read(e["message"].as_str().unwrap(), 4000)["authored_basis"]["seen_hash"],
        b
    );
    let mut first = request(&e, "first");
    first.targets.as_mut().unwrap()[0].seen_hash =
        Some(serde_json::from_value(read["authored_basis"]["seen_hash"].clone()).unwrap());
    // Lose the original reply, then retry the exact same client action.
    drop(
        world
            .handle()
            .try_note(first.clone(), Instant::now() + WAIT)
            .unwrap(),
    );
    let retry = world.write(&first);
    assert_eq!(retry["outcome"], "ok", "{retry}");
    assert_eq!(retry["note"], e["ids"]["first"]);
    assert_eq!(retry["revision"]["author"], e["author"]);
    let mut changed = first.clone();
    changed.targets.as_mut().unwrap()[0].seen_hash = Some(BlobHash::of(b"substituted basis"));
    assert_eq!(world.write(&changed)["error"]["kind"], "conflict");
    let mut second = request(&e, "second");
    second.created_at = Some(100);
    second.kind = None;
    second.targets = None;
    second.relations = vec![Relation {
        kind: "revises".into(),
        other: pin(&e, "first", "first"),
    }];
    let revised = world.write(&second);
    assert_eq!(revised["outcome"], "ok", "{revised}");
    assert_eq!(revised["revision"]["targets"], retry["revision"]["targets"]);
    let mut stale = second.clone();
    stale.request_id = e["ids"]["stale_attempt"].as_str().unwrap().into();
    let refusal = world.write(&stale);
    assert_eq!(refusal["error"]["kind"], "conflict");
    assert!(
        refusal["error"]["message"]
            .as_str()
            .unwrap()
            .contains(e["ids"]["second"].as_str().unwrap())
    );
    let withdraw = |key: &str, reference: String| {
        let mut r = request(&e, key);
        r.kind = Some(NoteKind::Withdrawal);
        r.targets = None;
        r.relations = vec![Relation {
            kind: "withdraws".into(),
            other: reference,
        }];
        r
    };
    assert_eq!(
        world.write(&withdraw("withdraw_second", pin(&e, "first", "second")))["outcome"],
        "ok"
    );
    assert_eq!(
        world.read(&root(&e, "first"), 4000)["family"]["current"],
        json!([pin(&e, "first", "first")])
    );
    assert_eq!(
        world.write(&withdraw(
            "restore_second",
            pin(&e, "withdraw_second", "withdraw_second")
        ))["outcome"],
        "ok"
    );
    assert_eq!(
        world.write(&withdraw("withdraw_root", root(&e, "first")))["outcome"],
        "ok"
    );
    assert_eq!(
        world.read(&root(&e, "first"), 4000)["family"]["current"],
        json!([])
    );
    assert_eq!(
        world.write(&withdraw(
            "restore_root",
            pin(&e, "withdraw_root", "withdraw_root")
        ))["outcome"],
        "ok"
    );
    let final_read = world.read(&root(&e, "first"), 12000);
    assert_eq!(
        final_read["family"]["current"],
        json!([pin(&e, "first", "second")])
    );
    assert_eq!(
        final_read["family"]["history"],
        json!([pin(&e, "first", "first"), pin(&e, "first", "second")])
    );
    assert_eq!(
        final_read["family"]["withdrawal_trail"]
            .as_array()
            .unwrap()
            .len(),
        4
    );
    assert_eq!(
        shown(&world.read(&pin(&e, "first", "first"), 12000))["revision"]["value"],
        e["values"]["first"]
    );
    assert_eq!(
        world.write(&first)["revision"],
        retry["revision"],
        "retry after later revisions still returns the original effect"
    );
    let target_read = world.read(target, 24000);
    assert_eq!(
        target_read["authored_notes"]["families"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    let capture = json!({"retry":retry,"prepared":after,"final":final_read,"target":target_read});
    check_example(&e, &capture).unwrap();
    let mut duplicate = capture.clone();
    let families = duplicate["target"]["authored_notes"]["families"]
        .as_array_mut()
        .unwrap();
    families.push(families[0].clone());
    assert!(check_example(&e, &duplicate).is_err());
    let mut substituted = capture.clone();
    substituted["prepared"]["family"]["shown"][0]["revision"]["targets"][0]["seen_hash"] =
        e["collected_seen_hash"].clone();
    assert!(check_example(&e, &substituted).is_err());
    let mut erased = capture.clone();
    erased["final"]["family"]["history"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    assert!(check_example(&e, &erased).is_err());
    world.restart_without_index();
    world.wait_rows(&input);
    assert_eq!(
        world.read(&root(&e, "first"), 12000)["family"],
        final_read["family"]
    );
    assert_eq!(
        world.read(target, 24000)["shown"],
        target_read["shown"],
        "exact retained content/citations survive provider removal and index reconstruction"
    );
}
#[test]
fn authored_basis_is_raw_whole_target_incremental_and_budgeted() {
    let world = World::new();
    let raw = format!(
        "{} {}",
        "abc ".repeat(2500),
        format!("ghp_{}", "A".repeat(36))
    );
    world.import("authored-long", json!(raw));
    let reference = "block:claude-code:authored-long:answer:0";
    let read = world.read(reference, 2200);
    assert_ne!(read["outcome"], "error", "{read}");
    assert!(!read["next"].is_null());
    assert_eq!(
        read["authored_basis"]["seen_hash"],
        BlobHash::of(raw.as_bytes()).to_string()
    );
    assert_eq!(read["authored_basis"]["projection"], "decoded-utf8-text-v1");
    assert!((serde_json::to_vec(&read).unwrap().len() as u64).div_ceil(4) <= 2200);
    let tiny = world.read(reference, 1);
    assert_eq!(tiny["error"]["reason"], "insufficient_budget");
    assert!(tiny.get("shown").is_none());
    let secret = format!("ghp_{}", "A".repeat(36));
    world.import("authored-credential", json!(secret));
    let credential = world.read("block:claude-code:authored-credential:answer:0", 4000);
    assert_ne!(credential["shown"][0]["text"], secret);
    assert_eq!(
        credential["authored_basis"]["seen_hash"],
        BlobHash::of(secret.as_bytes()).to_string()
    );
    world.import(
        "authored-thinking",
        json!([{"type":"thinking","thinking":"abc"}]),
    );
    let thinking = world.read("block:claude-code:authored-thinking:answer:0", 4000);
    assert_eq!(
        thinking["authored_basis"]["seen_hash"],
        example()["collected_seen_hash"],
        "{thinking}"
    );
    world.import(
        "authored-composite",
        json!([{"type":"text","text":"abc"},{"type":"text","text":"second"}]),
    );
    let composite = world.read("message:claude-code:authored-composite:answer", 4000);
    assert_eq!(composite["authored_basis"]["status"], "unknown");
    assert!(composite["authored_basis"].get("seen_hash").is_none());
    assert_eq!(
        world.read("block:claude-code:authored-composite:answer:0", 4000)["authored_basis"]["seen_hash"],
        example()["collected_seen_hash"]
    );
}
#[test]
fn expired_queued_authored_intent_is_not_committed() {
    let world = World::new();
    let req = request(&example(), "first");
    let reply = world
        .handle()
        .try_note(req.clone(), Instant::now() - Duration::from_secs(1))
        .unwrap()
        .recv_timeout(WAIT)
        .unwrap();
    assert!(reply.is_err());
    assert_eq!(
        world.read(&root(&example(), "first"), 4000)["error"]["kind"],
        "not_found"
    );
    assert_eq!(world.write(&req)["outcome"], "ok");
    let cancelled = crate::model_read::execute(
        world.reader.as_ref().unwrap(),
        &json!({"command":"model-read","reference":root(&example(),"first"),"budget_tokens":4000}),
        Instant::now() + WAIT,
        &mut || true,
    );
    assert_eq!(cancelled["error"]["kind"], "timeout");
    assert_eq!(
        world.read(&root(&example(), "first"), 1)["error"]["reason"],
        "authored_presentation_admission"
    );
}
