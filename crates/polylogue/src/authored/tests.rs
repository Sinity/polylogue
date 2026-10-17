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

#[test]
fn authored_committed_inheritance_or_expansion_keeps_safe_retry_identity() {
    for shape in ["target", "corrects", "capacity"] {
        let world = World::new();
        let e = example();
        let mut seed = request(&e, "first");
        let credential = format!("message:ghp_{}", "A".repeat(36));
        match shape {
            "target" => seed.targets.as_mut().unwrap()[0].reference = credential.clone(),
            "corrects" => seed.relations.push(Relation {
                kind: "corrects".into(),
                other: credential.clone(),
            }),
            _ => {
                seed.targets = Some(
                    (0..8)
                        .map(|n| Target {
                            reference: format!("message:{n}{}", "x".repeat(7000)),
                            seen_hash: None,
                        })
                        .collect(),
                )
            }
        }
        // Trusted retained typed data is deliberately outside surface preflight.
        // The ordinary revision inherits it, rather than submitting it again.
        world
            .handle()
            .try_note(seed, Instant::now() + WAIT)
            .unwrap()
            .recv_timeout(WAIT)
            .unwrap()
            .unwrap();
        let mut next = request(&e, "second");
        next.kind = None;
        next.targets = None;
        next.relations = vec![Relation {
            kind: "revises".into(),
            other: pin(&e, "first", "first"),
        }];
        if shape == "capacity" {
            next.value = "v".repeat(16000);
        }
        let before = world
            .reader
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .status()
            .unwrap()
            .write_counter;
        let reply = world.write(&next);
        assert_eq!(reply["outcome"], "degraded", "{shape}: {reply}");
        assert_eq!(reply["commit_state"], "committed");
        assert_eq!(reply["presentation"], "retained_but_unpresentable");
        assert_eq!(reply["request_id"], e["ids"]["second"]);
        assert!(reply.get("revision").is_none());
        assert!(!reply.to_string().contains("ghp_"));
        assert_eq!(world.write(&next), reply);
        let reader = world.reader.as_ref().unwrap().lock().unwrap();
        assert_eq!(reader.status().unwrap().write_counter, before + 1);
        let family = reader
            .authored_snapshot()
            .unwrap()
            .family(&root(&e, "first"))
            .unwrap();
        assert_eq!(family.current, [pin(&e, "first", "second")]);
        assert_eq!(family.history.len(), 2);
        assert_eq!(family.shown[0].revision.value, next.value);
        drop(reader);
        if shape != "capacity" {
            let exact = world.read(&pin(&e, "first", "second"), 12000);
            assert_eq!(exact["error"]["kind"], "not_yet_available");
            assert!(!exact.to_string().contains("ghp_"));
        }
    }
}

#[test]
fn authored_success_must_bind_effect_identity_for_both_public_clients() {
    let request = serde_json::to_value(request(&example(), "first")).unwrap();
    let id = request["request_id"].as_str().unwrap();
    let reference = format!("note:{id}@{id}");
    let reply = json!({"outcome":"ok","commit_state":"committed","request_id":id,"note":id,"ref":reference,"root":id,"revision":{"id":id,"root":id}});
    assert_eq!(bind_write_reply(&request, reply.clone()), reply);
    for path in [
        vec!["request_id"],
        vec!["note"],
        vec!["ref"],
        vec!["root"],
        vec!["revision", "id"],
        vec!["revision", "root"],
    ] {
        let mut wrong = reply.clone();
        let mut field = &mut wrong;
        for key in path {
            field = &mut field[key];
        }
        *field = json!("01900000-0000-7000-8000-000000000999");
        let result = bind_write_reply(&request, wrong);
        assert_eq!(result["outcome"], "error");
        assert_eq!(result["request_id"], id);
        assert_eq!(result["commit_state"], "unknown");
    }
    let mut missing = reply.clone();
    missing.as_object_mut().unwrap().remove("revision");
    assert_eq!(
        bind_write_reply(&request, missing.clone())["commit_state"],
        "unknown"
    );
    missing["outcome"] = json!("degraded");
    missing["presentation"] = json!("retained_but_unpresentable");
    assert_eq!(bind_write_reply(&request, missing.clone()), missing);
}

fn read_call(world: &World, arguments: &Value) -> Value {
    let mut operation = arguments.clone();
    operation["reference"] = operation.as_object_mut().unwrap().remove("ref").unwrap();
    operation["command"] = json!("model-read");
    crate::model_read::execute(
        world.reader.as_ref().unwrap(),
        &operation,
        Instant::now() + Duration::from_secs(40),
        &mut || false,
    )
}
fn growth_request(n: u32, target: &str) -> WriteNote {
    WriteNote {
        request_id: format!("01900000-0000-7000-8000-{n:012}"),
        author: "agent:neutral-growth".into(),
        machine: "synthetic-machine".into(),
        created_at: Some(1000 - n as i64),
        kind: Some(NoteKind::Correction),
        value: format!("Interpretation {n}; retained source bytes stay unchanged."),
        targets: Some(vec![Target {
            reference: target.into(),
            seen_hash: Some(BlobHash::of(b"unchanged retained text")),
        }]),
        relations: vec![],
    }
}
#[test]
fn dense_authored_relations_degrade_context_without_suppressing_unchanged_primary_text() {
    let world = World::new();
    world.import("growth-associations", json!("unchanged retained text"));
    let target = "block:claude-code:growth-associations:answer:0";
    let before = world.read(target, 4000);
    assert_eq!(before["shown"][0]["text"], "unchanged retained text");
    for n in 1..=17 {
        assert_eq!(
            world.write(&growth_request(n, target))["commit_state"],
            "committed"
        );
    }
    let read = world.read(target, 4000);
    assert_eq!(read["outcome"], "degraded", "{read}");
    assert_eq!(
        read["shown"], before["shown"],
        "content, exact identity and citations remain unchanged"
    );
    assert_eq!(read["primary_content_complete"], true);
    assert_eq!(read["authored_associations_complete"], false);
    assert!(!read["authored_notes"]["next"].is_null());
    let cutoff = read["authored_notes"]["archive_counter"].clone();
    let mut next = read["authored_notes"]["next"].clone();
    let mut roots = vec![];
    // Intervening writes cannot join a retained association population or alter
    // only its later status pages.
    let mut withdrawal = growth_request(100, target);
    withdrawal.kind = Some(NoteKind::Withdrawal);
    withdrawal.targets = None;
    withdrawal.relations = vec![Relation {
        kind: "withdraws".into(),
        other: format!("note:{}", growth_request(1, target).request_id),
    }];
    assert_eq!(world.write(&withdrawal)["commit_state"], "committed");
    assert_eq!(
        world.write(&growth_request(18, target))["commit_state"],
        "committed"
    );
    while !next.is_null() {
        let page = read_call(&world, &next["arguments"]);
        assert_ne!(page["outcome"], "error", "{page}");
        assert_eq!(page["authored_notes"]["archive_counter"], cutoff);
        let families = page["authored_notes"]["families"].as_array().unwrap();
        assert_eq!(families.len(), 1);
        let family = &families[0];
        assert_eq!(family["shown"][0]["withdrawn"], false);
        assert_eq!(
            family["shown"][0]["revision"]["targets"][0]["seen_hash"],
            BlobHash::of(b"unchanged retained text").to_string()
        );
        roots.push(family["root"].as_str().unwrap().to_owned());
        next = page["next"].clone();
    }
    assert_eq!(roots.len(), 17);
    assert_eq!(
        roots
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        17
    );
    let fresh = read_call(
        &world,
        &json!({"ref":target,"view":"annotations","budget_tokens":4000}),
    );
    assert!(
        fresh["authored_notes"]["families"][0]["current"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    assert_eq!(world.read(target, 4000)["shown"], before["shown"]);
}
#[test]
fn authored_page_tokens_bind_reference_archive_view_and_fixed_status_basis() {
    let world = World::new();
    let target = "block:claude-code:awaited:answer:0";
    let first = growth_request(1, target);
    assert_eq!(world.write(&first)["commit_state"], "committed");
    let mut previous = first.request_id.clone();
    for n in 2..=40 {
        let mut r = growth_request(n, target);
        r.kind = None;
        r.targets = None;
        r.relations = vec![Relation {
            kind: "revises".into(),
            other: format!("note:{}@{previous}", first.request_id),
        }];
        assert_eq!(world.write(&r)["commit_state"], "committed");
        previous = r.request_id;
    }
    let root = format!("note:{}", first.request_id);
    let initial = world.read(&root, 12000);
    let next = &initial["family"]["history_next"];
    assert!(!next.is_null(), "{initial}");
    let tail = read_call(&world, &next["arguments"]);
    assert_ne!(tail["outcome"], "error", "{tail}");
    assert_eq!(tail["family"]["history"].as_array().unwrap().len(), 8);
    let mut bad = next["arguments"].clone();
    bad["ref"] = json!(format!("note:{}", growth_request(2, target).request_id));
    assert_eq!(read_call(&world, &bad)["outcome"], "error");
    let mut bad = next["arguments"].clone();
    bad["view"] = json!("withdrawals");
    assert_eq!(read_call(&world, &bad)["outcome"], "error");
    let other = World::new();
    assert_eq!(read_call(&other, &next["arguments"])["outcome"], "error");
    let mut bad = next["arguments"].clone();
    let token = bad["continuation"].as_str().unwrap().to_owned();
    let last = if token.ends_with('0') { '1' } else { '0' };
    bad["continuation"] = json!(format!("{}{last}", &token[..token.len() - 1]));
    assert_eq!(read_call(&world, &bad)["outcome"], "error");
    let mut revision = growth_request(41, target);
    revision.kind = None;
    revision.targets = None;
    revision.relations = vec![Relation {
        kind: "revises".into(),
        other: format!("{root}@{previous}"),
    }];
    assert_eq!(world.write(&revision)["commit_state"], "committed");
    let mut withdrawal = growth_request(100, target);
    withdrawal.kind = Some(NoteKind::Withdrawal);
    withdrawal.targets = None;
    withdrawal.relations = vec![Relation {
        kind: "withdraws".into(),
        other: root.clone(),
    }];
    assert_eq!(world.write(&withdrawal)["commit_state"], "committed");
    let old = read_call(&world, &next["arguments"]);
    assert_eq!(old["family"]["current"], initial["family"]["current"]);
    assert_eq!(old["outcome"], "degraded");
    assert_eq!(
        old["view"]["authored_archive_counter"],
        initial["view"]["authored_archive_counter"]
    );
    assert!(
        world.read(&root, 12000)["family"]["current"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    let mut restore = growth_request(101, target);
    restore.kind = Some(NoteKind::Withdrawal);
    restore.targets = None;
    restore.relations = vec![Relation {
        kind: "withdraws".into(),
        other: format!("note:{}", withdrawal.request_id),
    }];
    assert_eq!(world.write(&restore)["commit_state"], "committed");
    let restored = world.read(&root, 12000);
    assert_eq!(
        restored["family"]["current"],
        json!([format!("{root}@{}", revision.request_id)])
    );
    assert_eq!(
        read_call(&world, &next["arguments"])["family"]["current"],
        initial["family"]["current"]
    );
    let pinned = format!("{root}@{}", first.request_id);
    let value = read_call(
        &world,
        &json!({"ref":pinned,"view":"value","budget_tokens":4000}),
    );
    assert_eq!(value["status_evaluated"], false);
    assert_eq!(value["current"], Value::Null);
    assert_eq!(value["revision"]["value"], first.value);
}

#[test]
fn body_continuation_allows_budget_refitting_but_refuses_a_new_authored_context() {
    let world = World::new();
    let raw = "retained neutral body ".repeat(2000);
    world.import("growth-certificate", json!(raw));
    let target = "block:claude-code:growth-certificate:answer:0";
    for n in 1..=17 {
        assert_eq!(
            world.write(&growth_request(n, target))["commit_state"],
            "committed"
        );
    }
    let first = world.read(target, 2000);
    assert_ne!(first["outcome"], "error", "{first}");
    assert!(!first["next"].is_null());
    assert_eq!(first["authored_associations_complete"], false);
    world.import(
        "unrelated-growth-source",
        json!("unrelated newly retained source bytes"),
    );
    let mut next = first["next"].clone();
    next["budget_tokens"] = json!(3000);
    let second = read_call(&world, &next);
    assert_ne!(second["outcome"], "error", "{second}");
    assert_eq!(
        second["shown"][0]["display_start"],
        first["shown"][0]["display_end"]
    );
    // Native context fencing is intentionally conservative across authored
    // writes, but must not include unrelated source/index bookkeeping.
    assert_eq!(
        world.write(&growth_request(
            19,
            "block:claude-code:unrelated-growth-source:answer:0"
        ))["commit_state"],
        "committed"
    );
    assert_eq!(read_call(&world, &next)["error"]["reason"], "stale_read");
    let fresh = world.read(target, 2000);
    assert!(!fresh["next"].is_null());
    assert_eq!(
        world.write(&growth_request(18, target))["commit_state"],
        "committed"
    );
    let stale = read_call(&world, &fresh["next"]);
    assert_eq!(stale["outcome"], "error");
    assert_eq!(stale["error"]["reason"], "stale_read");
}

#[test]
fn native_identity_authored_targets_refuse_encoded_secrets_but_authors_remain_opaque() {
    let world = World::new();
    let secret = polylogue_model::NativeId::from_utf16(
        &format!("ghp_{}", "A".repeat(36))
            .encode_utf16()
            .chain([0xd800])
            .collect::<Vec<_>>(),
    );
    let session = polylogue_model::SessionId::new(
        &polylogue_model::Origin::new("claude-code").unwrap(),
        "privacy",
    )
    .unwrap();
    let reference = format!(
        "message:{}",
        polylogue_model::MessageId::from_native(session, &secret)
            .unwrap()
            .as_str()
    );
    let mut write = growth_request(1, &reference);
    let refused = world.write(&write);
    assert_eq!(refused["outcome"], "error", "{refused}");
    assert_eq!(refused["error"]["kind"], "not_yet_available");
    write.targets.as_mut().unwrap()[0].reference = "message:claude-code:privacy:ordinary".into();
    write.author = reference.strip_prefix("message:").unwrap().into();
    let accepted = world.write(&write);
    assert_eq!(accepted["commit_state"], "committed", "{accepted}");
}
