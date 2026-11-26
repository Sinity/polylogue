use super::*;
use polylogue_model::BlobHash;
use polylogue_model::authored::AuthoredEffect;
use polylogue_model::signal::{Checkpoint, Commit, Event};
use polylogue_model::user::UserWrite;
use serde_json::json;
use shared_signals::{Calendar, DeliveryOutcome, DeliveryReceipt, Evaluator};

/// A recorded line request with its retained binding and the query object it
/// ran, as the build writes them.
const BINDING: &str = include_str!("../../../../query/src/plan/tests/pre-sinex-binding.json");

/// Moves the key `now` of the object at `pointer` back to its earlier name.
fn back(value: &mut Value, pointer: &str, now: &str, earlier: &str) {
    let target = value
        .pointer_mut(pointer)
        .unwrap_or_else(|| panic!("{pointer}"));
    assert_eq!(rename(target, now, earlier), 1, "{pointer}/{now}");
}

fn back_in_text(value: &mut Value, pointer: &str, edit: impl FnOnce(&mut Value)) {
    let Value::String(text) = value.pointer_mut(pointer).unwrap() else {
        panic!("{pointer} is not text")
    };
    let mut inner: Value = serde_json::from_str(text).unwrap();
    edit(&mut inner);
    *text = inner.to_string();
}

fn archive() -> (tempfile::TempDir, Connection) {
    let dir = tempfile::tempdir().unwrap();
    let conn = Connection::open(dir.path().join("archive.db")).unwrap();
    polylogue_store::schema::ensure_archive(&conn).unwrap();
    (dir, conn)
}

fn packed(value: &Value) -> Vec<u8> {
    zstd::bulk::compress(&serde_json::to_vec(value).unwrap(), 1).unwrap()
}

fn request_row(conn: &Connection, request: &Value, effect: Option<&Value>) {
    conn.execute(
        "INSERT INTO client_request VALUES (?1, ?2, 0, ?3, ?4, ?5, 1)",
        params![
            request["id"].as_str().unwrap(),
            vec![0u8; 32],
            packed(request),
            effect
                .map(|_| packed(&json!({"id": request["id"], "repeat": false, "pending": false}))),
            effect.map(packed),
        ],
    )
    .unwrap();
}

fn stored(conn: &Connection, sql: &str, id: &str) -> Value {
    let bytes: Vec<u8> = conn.query_row(sql, [id], |r| r.get(0)).unwrap();
    unpack(&bytes).unwrap()
}

fn authorship() -> Value {
    json!({"author": "operator", "via": "cli"})
}

const AT: &str = "2026-11-20T10:00:00.001Z";

fn position() -> Value {
    json!({"index_instance": "instance", "index": 3, "archive": 4})
}

fn empty_effect() -> Value {
    json!({"annotations": [], "alternates": {"alternates": [], "alternate_parts": []},
           "saved_rows": [], "deliveries": [], "chats": [], "import_jobs": []})
}

/// A saved body and its read position text, as the build writes them.
fn saved_texts_now() -> (String, String) {
    let body = json!({"unit": "sessions", "shape": "rows", "columns": [], "rows": [],
                      "query": "sessions", "now": 1,
                      "read_position": {"index_instance": "instance", "index": 3, "archive": 4, "cursor": "instance:3:4"}});
    let read = json!({"now": 1, "position": {"index_instance": "instance", "index": 3, "archive": 4, "cursor": "instance:3:4"}});
    (body.to_string(), read.to_string())
}

fn earlier_saved_texts(value: &mut Value, pointer: &str) {
    back_in_text(value, &format!("{pointer}/rows_json"), |body| {
        back(body, "/read_position", "index_instance", "generation");
        back(body, "", "read_position", "read_at");
    });
    back_in_text(value, &format!("{pointer}/read_position"), |read| {
        back(read, "/position", "index_instance", "generation");
    });
    back(value, pointer, "read_position", "read_at");
}

fn delivery_now() -> Value {
    json!({"recipient_kind": "session", "recipient_id": "s", "sender": "hook",
           "event": "session_start", "text": "brief", "refs": [],
           "left_out": [{"item": "session:codex:other", "reason": "budget"}]})
}

fn slot_now() -> Value {
    json!({"section": "Evidence", "position": 0.5, "mode": "follow"})
}

/// One payload of each kind, as the build writes it and as format 9 wrote
/// it: the earlier one is the build's with each renamed key moved back at
/// its path, written out here rather than by the step's own code.
struct Case {
    now: Value,
    earlier: Value,
}

fn cases() -> Vec<(Case, Option<Case>)> {
    let (body, read) = saved_texts_now();
    let saved_row = json!({"id": "save", "name": null, "query": "sessions",
        "ran_at": AT, "read_position": read, "rows_json": body});
    let save = json!({"id": "save", "machine": "m", "authorship": authorship(), "at": AT,
        "write": {"write": "save", "name": null, "query": "sessions", "denominator": null,
                  "rows_json": body, "read_position": read, "watch": null,
                  "outcomes": {"request": "save", "saved": null,
                               "saved_rows": {"result": "result:save", "rows": 0, "at": AT},
                               "watch": null, "repeat": false, "pending": false}}});
    let mut save_effect = empty_effect();
    save_effect["saved_rows"] = json!([saved_row]);
    let mut earlier_save = save.clone();
    earlier_saved_texts(&mut earlier_save, "/write");
    back(&mut earlier_save, "/write/outcomes", "saved_rows", "kept");
    let mut earlier_save_effect = save_effect.clone();
    earlier_saved_texts(&mut earlier_save_effect, "/saved_rows/0");
    back(&mut earlier_save_effect, "", "saved_rows", "kept_results");

    let fixture: Value = serde_json::from_str(BINDING).unwrap();
    let mut line: Value =
        serde_json::from_str(fixture["original_request"].as_str().unwrap()).unwrap();
    let object: Value = serde_json::from_str(fixture["original_object"].as_str().unwrap()).unwrap();
    let replay = line.pointer_mut("/write/retained/binding/replay").unwrap();
    replay["definitions"] = json!([{"name": "focus", "root": "r", "revision": "r",
        "query_id": object["id"], "ast": object["query"]}]);
    replay["access_fingerprint"] = json!("access");
    replay["context"]["versions"]["access"] = json!("access");
    let mut earlier_line = line.clone();
    let replay = "/write/retained/binding/replay";
    back(
        &mut earlier_line,
        &format!("{replay}/definitions/0"),
        "query_id",
        "question",
    );
    back(
        &mut earlier_line,
        replay,
        "access_fingerprint",
        "scope_fingerprint",
    );
    back(
        &mut earlier_line,
        &format!("{replay}/context/read_positions"),
        "index_instance",
        "generation",
    );
    back(
        &mut earlier_line,
        &format!("{replay}/context/versions"),
        "access",
        "scope",
    );

    let event = json!({"id": "event", "machine": "m", "authorship": authorship(), "at": AT,
        "write": {"write": "delivery_event", "delivery": "d", "outcome": "delivered"}});
    let mut earlier_event = event.clone();
    back(&mut earlier_event, "/write", "outcome", "what");

    let mut deliver =
        json!({"id": "deliver", "machine": "m", "authorship": authorship(), "at": AT});
    deliver["write"] = delivery_now();
    deliver["write"]["write"] = json!("deliver");
    let mut deliver_effect = empty_effect();
    deliver_effect["deliveries"] = json!([{
        "delivery": {"id": "d", "prepared": delivery_now(), "at": AT,
                     "events": [{"request": "deliver", "outcome": "delivered", "at": AT}]},
        "added_events": [{"request": "deliver", "outcome": "delivered", "at": AT}]}]);
    let mut earlier_deliver = deliver.clone();
    back(&mut earlier_deliver, "/write/left_out/0", "item", "what");
    let mut earlier_deliver_effect = deliver_effect.clone();
    back(
        &mut earlier_deliver_effect,
        "/deliveries/0/delivery/events/0",
        "outcome",
        "what",
    );
    back(
        &mut earlier_deliver_effect,
        "/deliveries/0/added_events/0",
        "outcome",
        "what",
    );
    back(
        &mut earlier_deliver_effect,
        "/deliveries/0/delivery/prepared/left_out/0",
        "item",
        "what",
    );

    let annotation = json!({"kind": "slot", "key": null, "value": slot_now(), "valid_until": null,
        "links": [{"role": "slot_of", "reference": "page:p"}]});
    let slot_request = json!({"id": "slot", "machine": "m", "authorship": authorship(), "at": AT,
        "write": {"write": "assert", "annotations": [annotation]}});
    let mut slot_effect = empty_effect();
    slot_effect["annotations"] = json!([{"id": "slot", "root": "slot", "kind": "slot", "key": null,
        "value": slot_now(), "author": "operator", "via": "cli", "machine": "m", "created_at": AT,
        "valid_until": null, "links": [{"role": "slot_of", "reference": "page:p", "seen_hash": null}]}]);
    let mut earlier_slot = slot_request.clone();
    back(
        &mut earlier_slot,
        "/write/annotations/0/value",
        "position",
        "pos",
    );
    let mut earlier_slot_effect = slot_effect.clone();
    back(
        &mut earlier_slot_effect,
        "/annotations/0/value",
        "position",
        "pos",
    );

    let revise = json!({"id": "revise", "machine": "m", "authorship": authorship(), "at": AT,
        "write": {"write": "revise_annotations", "bases": ["annotation:slot@slot"],
                  "patch": {"value": {"change": "set", "value": slot_now()}}}});
    let mut revise_effect = slot_effect.clone();
    revise_effect["annotations"][0]["id"] = json!("revise");
    let mut earlier_revise = revise.clone();
    back(
        &mut earlier_revise,
        "/write/patch/value/value",
        "position",
        "pos",
    );
    let mut earlier_revise_effect = revise_effect.clone();
    back(
        &mut earlier_revise_effect,
        "/annotations/0/value",
        "position",
        "pos",
    );

    let seen = json!({"id": "seen", "machine": "m", "authorship": authorship(), "at": AT,
        "write": {"write": "mark_seen", "marks": [{"target": "session:codex:s", "item": null, "at": position()}]}});
    let mut earlier_seen = seen.clone();
    back(
        &mut earlier_seen,
        "/write/marks/0/at",
        "index_instance",
        "generation",
    );

    let pair = |now: Value, earlier: Value| Case { now, earlier };
    vec![
        (
            pair(save, earlier_save),
            Some(pair(save_effect, earlier_save_effect)),
        ),
        (pair(line, earlier_line), None),
        (pair(event, earlier_event), None),
        (
            pair(deliver, earlier_deliver),
            Some(pair(deliver_effect, earlier_deliver_effect)),
        ),
        (
            pair(slot_request, earlier_slot),
            Some(pair(slot_effect, earlier_slot_effect)),
        ),
        (
            pair(revise, earlier_revise),
            Some(pair(revise_effect, earlier_revise_effect)),
        ),
        (pair(seen, earlier_seen), None),
    ]
}

/// The query object of the fixture with one definition, its own body,
/// addressed by its hash.
fn query_object() -> (Value, Value) {
    let fixture: Value = serde_json::from_str(BINDING).unwrap();
    let mut object: Value =
        serde_json::from_str(fixture["original_object"].as_str().unwrap()).unwrap();
    object["definitions"] = json!([{"query_id": object["id"], "normal_form": object["normal_form"],
        "parameters": object["parameters"], "query": object["query"],
        "source_outputs": object["source_outputs"]}]);
    let mut earlier = object.clone();
    back(&mut earlier, "/definitions/0", "query_id", "question");
    (object, earlier)
}

fn query_hash(object: &Value) -> String {
    hex(&serde_json::from_value(object["id"].clone()).unwrap())
}

fn query_row(conn: &Connection, object: &Value) {
    conn.execute(
        "INSERT INTO query(hash, normal_form, object) VALUES (?1, ?2, ?3)",
        params![
            query_hash(object),
            object["normal_form"].to_string(),
            packed(object)
        ],
    )
    .unwrap();
}

/// A run of the fixture's line binding, as the build writes it and as format
/// 9 wrote it.
fn query_run() -> (Value, Value) {
    let (line, _) = cases().swap_remove(1);
    let run = json!({"id": "0190f6a2-7c51-7a3e-9b1d-4e2f8a6c0d13",
        "query_id": line.now["write"]["retained"]["binding"]["replay"]["context"]["question"],
        "at": AT, "read_position": position(), "machine": "m", "authorship": authorship(),
        "binding": {"kind": "original", "execution": line.now["write"]["retained"]["binding"]}});
    let mut earlier = run.clone();
    back(&mut earlier, "", "query_id", "question");
    back(
        &mut earlier,
        "/read_position",
        "index_instance",
        "generation",
    );
    back(&mut earlier, "", "read_position", "read_at");
    let replay = "/binding/execution/replay";
    back(
        &mut earlier,
        &format!("{replay}/definitions/0"),
        "query_id",
        "question",
    );
    back(
        &mut earlier,
        replay,
        "access_fingerprint",
        "scope_fingerprint",
    );
    back(
        &mut earlier,
        &format!("{replay}/context/read_positions"),
        "index_instance",
        "generation",
    );
    back(
        &mut earlier,
        &format!("{replay}/context/versions"),
        "access",
        "scope",
    );
    (run, earlier)
}

fn event_record(request: &str, request_position: u64) -> Value {
    serde_json::to_value(EventRecord {
        position: request_position + 1,
        request: request.into(),
        request_position,
        at: polylogue_model::Timestamp::from_unix_ms(1).unwrap(),
        event: Event::Outlet {
            receipt: DeliveryReceipt {
                delivery_id: "d".into(),
                at: 1,
                outlet: "value".into(),
                outcome: DeliveryOutcome::Unknown,
            },
        },
    })
    .unwrap()
}

fn signal_event_row(conn: &Connection, body: &Value) {
    conn.execute(
        "INSERT INTO signal_event VALUES (?1, ?2, 'd', ?3, 1, ?4, 1)",
        params![
            body["request"].as_str().unwrap(),
            body.get("request_position")
                .or(body.get("ordinal"))
                .unwrap()
                .as_i64(),
            body["position"].as_i64(),
            packed(body)
        ],
    )
    .unwrap();
}

/// Writes every earlier payload into an archive; returns what each should
/// read as after the step.
fn earlier_archive(conn: &Connection) -> Vec<(Case, Option<Case>)> {
    let cases = cases();
    for (request, effect) in &cases {
        request_row(conn, &request.earlier, effect.as_ref().map(|e| &e.earlier));
    }
    let (_, object) = query_object();
    query_row(conn, &object);
    let (_, run) = query_run();
    conn.execute(
        "INSERT INTO query_run(id, binding_hash, run_hash, query, run) VALUES (?1, ?2, ?2, ?3, ?4)",
        params![
            run["id"].as_str(),
            "0".repeat(64),
            query_hash(&object),
            packed(&run)
        ],
    )
    .unwrap();
    let mut saved = json!({});
    let (body, read) = saved_texts_now();
    saved["rows_json"] = json!(body);
    saved["read_position"] = json!(read);
    earlier_saved_texts(&mut saved, "");
    conn.execute(
        &format!(
            "INSERT INTO {} VALUES ('save', NULL, 'sessions', 1, ?1, ?2)",
            SAVED_ROWS.0
        ),
        params![saved["read_at"].as_str(), saved["rows_json"].as_str()],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO delivery(id, recipient_kind, sender, event, at, content_hash, bytes, body, refs_json, left_out_json, commit_seq)
         VALUES ('d', 'session', 'hook', 'session_start', 1, ?1, 0, ?2, '[]', ?3, 1)",
        params![vec![0u8; 32], packed(&json!("")), r#"[{"what":"session:codex:other","reason":"budget"}]"#],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO annotation(id, root, kind, value, author, via, machine, commit_seq)
         VALUES ('slot', 'slot', 'slot', ?1, 'operator', 'cli', 'm', 1)",
        [r#"{"section":"Evidence","pos":0.5,"mode":"follow"}"#],
    )
    .unwrap();
    let mut event = event_record("signal", 0);
    rename(&mut event, "request_position", "ordinal");
    signal_event_row(conn, &event);
    cases
}

/// Each earlier payload reads, after the step, as the value the build
/// writes; the covering hashes are those of the values as they now read;
/// every table reports what it rewrote, and a second run changes nothing.
#[test]
fn earlier_keys_read_as_the_build_writes_them() {
    let (_dir, conn) = archive();
    let cases = earlier_archive(&conn);
    assert_eq!(prove_query_ids(&conn).unwrap(), 1);
    let counts = convert(&conn).unwrap();
    let rewritten = |table: &str| {
        let (_, c) = counts.iter().find(|(name, _)| *name == table).unwrap();
        (c.visited, c.rewritten, c.before, c.after)
    };
    assert_eq!(rewritten("client_request"), (7, 7, 7, 0));
    for table in [
        "query",
        "query_run",
        "saved rows",
        "delivery",
        "annotation",
        "signal_event",
    ] {
        assert_eq!(rewritten(table), (1, 1, 1, 0), "{table}");
    }

    for (request, effect) in &cases {
        let id = request.now["id"].as_str().unwrap();
        let now: UserRequest = serde_json::from_value(request.now.clone()).unwrap();
        let read: UserRequest = serde_json::from_value(stored(
            &conn,
            "SELECT request FROM client_request WHERE id = ?1",
            id,
        ))
        .unwrap();
        assert_eq!(read, now, "{id}");
        let hash: Vec<u8> = conn
            .query_row(
                "SELECT content_hash FROM client_request WHERE id = ?1",
                [id],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            hash,
            polylogue_store::authored_records::request_hash(&now).unwrap(),
            "{id}"
        );
        if let Some(effect) = effect {
            let now: AuthoredEffect = serde_json::from_value(effect.now.clone()).unwrap();
            let read: AuthoredEffect = serde_json::from_value(stored(
                &conn,
                "SELECT effect FROM client_request WHERE id = ?1",
                id,
            ))
            .unwrap();
            assert_eq!(read, now, "{id}");
        }
    }

    let (object, _) = query_object();
    let read: QueryObject = serde_json::from_value(stored(
        &conn,
        "SELECT object FROM query WHERE hash = ?1",
        &query_hash(&object),
    ))
    .unwrap();
    assert_eq!(read, serde_json::from_value::<QueryObject>(object).unwrap());

    let (run, _) = query_run();
    let id = run["id"].as_str().unwrap();
    let now: QueryRun = serde_json::from_value(run.clone()).unwrap();
    let read: QueryRun =
        serde_json::from_value(stored(&conn, "SELECT run FROM query_run WHERE id = ?1", id))
            .unwrap();
    assert_eq!(read, now);
    let hashes: (String, String) = conn
        .query_row(
            "SELECT binding_hash, run_hash FROM query_run WHERE id = ?1",
            [id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(
        hashes,
        (
            now.binding_hash().unwrap().to_string(),
            now.record_hash().unwrap().to_string()
        )
    );

    let saved: (String, String) = conn
        .query_row(
            &format!(
                "SELECT rows_json, {} FROM {} WHERE id = 'save'",
                SAVED_ROWS.1, SAVED_ROWS.0
            ),
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(saved, saved_texts_now());
    let left_out: String = conn
        .query_row(
            "SELECT left_out_json FROM delivery WHERE id = 'd'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        left_out,
        r#"[{"item":"session:codex:other","reason":"budget"}]"#
    );
    let value: String = conn
        .query_row("SELECT value FROM annotation WHERE id = 'slot'", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(
        value,
        r#"{"section":"Evidence","position":0.5,"mode":"follow"}"#
    );
    let body: Vec<u8> = conn
        .query_row(
            "SELECT body FROM signal_event WHERE delivery = 'd'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(unpack(&body).unwrap(), event_record("signal", 0));

    for (table, counts) in convert(&conn).unwrap() {
        assert_eq!(
            (counts.rewritten, counts.before, counts.after),
            (0, 0, 0),
            "{table}"
        );
    }
}

/// A query object whose stored ID is not the hash of its normal form is
/// refused before any row is rewritten, and so is a definition whose
/// declared ID its body does not hash to.
#[test]
fn a_query_id_that_does_not_hash_from_its_normal_form_is_refused() {
    let (_dir, conn) = archive();
    let (_, mut object) = query_object();
    let hash = query_hash(&object);
    object["definitions"][0]["question"] = json!([7; 32]);
    query_row(&conn, &object);
    assert!(matches!(
        prove_query_ids(&conn).unwrap_err().downcast_ref::<Refusal>(),
        Some(Refusal::DefinitionId { query, .. }) if *query == hash
    ));

    let (_dir, conn) = archive();
    let (_, mut object) = query_object();
    object["normal_form"]["stages"] = json!([{"kind": "limit", "count": 1}]);
    conn.execute(
        "INSERT INTO query(hash, normal_form, object) VALUES (?1, ?2, ?3)",
        params![hash, object["normal_form"].to_string(), packed(&object)],
    )
    .unwrap();
    assert!(matches!(
        prove_query_ids(&conn).unwrap_err().downcast_ref::<Refusal>(),
        Some(Refusal::QueryId { query, .. }) if *query == hash
    ));
}

struct CalendarFixture;
impl Calendar for CalendarFixture {
    fn logical_day(&self, at: i64) -> String {
        (at / 86_400_000).to_string()
    }
    fn minute(&self, at: i64) -> u16 {
        ((at / 60_000) % 1440) as u16
    }
    fn next_at(&self, after: i64, _: u16) -> i64 {
        after + 86_400_000
    }
}

fn checkpoint(now: i64) -> Checkpoint {
    Checkpoint {
        owner: "host".into(),
        evaluator: Evaluator::new(now, vec![], &CalendarFixture).unwrap(),
        rules: vec![],
        host: None,
        notifications: None,
    }
}

/// Three signal commits chain each checkpoint's hash into the next commit's
/// `prior`, and their events are renamed: the chain reads the same after the
/// step, each prior the hash of the checkpoint before it.
#[test]
fn a_signal_checkpoint_chain_keeps_its_hashes() {
    let (_dir, conn) = archive();
    let mut prior = None;
    let mut identities = Vec::new();
    for n in 0..3_i64 {
        let checkpoint = checkpoint(n * 1_000);
        let commit = Commit {
            prior,
            checkpoint: checkpoint.clone(),
            pending_delivery: vec![],
            events: vec![],
        };
        let id = format!("signal-{n}");
        let request = UserRequest {
            id: id.clone(),
            machine: "host".into(),
            authorship: polylogue_model::authorship::Authorship {
                author: polylogue_model::Author::Script("signals".into()),
                via: polylogue_model::authorship::Channel::Daemon,
            },
            at: polylogue_model::Timestamp::from_unix_ms(n).unwrap(),
            write: UserWrite::SignalCommit(Box::new(commit)),
        };
        let mut effect = serde_json::to_value(AuthoredEffect {
            signal_checkpoints: vec![checkpoint.clone()],
            ..Default::default()
        })
        .unwrap();
        let mut event = event_record(&id, 0);
        event["position"] = json!(n + 1);
        effect["signal_events"] = json!([event.clone()]);
        back(
            &mut effect,
            "/signal_events/0",
            "request_position",
            "ordinal",
        );
        request_row(
            &conn,
            &serde_json::to_value(&request).unwrap(),
            Some(&effect),
        );
        rename(&mut event, "request_position", "ordinal");
        signal_event_row(&conn, &event);
        let identity = checkpoint.identity().unwrap();
        identities.push(identity);
        prior = Some(identity);
    }
    conn.execute(
        "INSERT INTO signal_state(owner, body, commit_seq) VALUES ('host', ?1, 3)",
        [packed(&serde_json::to_value(checkpoint(2_000)).unwrap())],
    )
    .unwrap();

    convert(&conn).unwrap();
    let mut before: Option<BlobHash> = None;
    for n in 0..3 {
        let request: UserRequest = serde_json::from_value(stored(
            &conn,
            "SELECT request FROM client_request WHERE id = ?1",
            &format!("signal-{n}"),
        ))
        .unwrap();
        let UserWrite::SignalCommit(commit) = request.write else {
            panic!("a signal commit")
        };
        assert_eq!(commit.prior, before);
        assert_eq!(commit.checkpoint.identity().unwrap(), identities[n]);
        before = Some(identities[n]);
    }
    let state: Checkpoint = serde_json::from_value(stored(
        &conn,
        "SELECT body FROM signal_state WHERE owner = ?1",
        "host",
    ))
    .unwrap();
    assert_eq!(state.identity().unwrap(), identities[2]);
}
