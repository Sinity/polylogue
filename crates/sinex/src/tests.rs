//! Native excision publication controls with synthetic source and authored data.
//! Runs real Sinex over owned PostgreSQL/NATS instances. `SINEXD` and
//! `SINEX_SCHEMA` name the built Sinex daemon and schema binaries.
use crate::{Mode, Sinex};
use async_nats::jetstream;
use polylogue_engine::{Config, Engine};
use polylogue_store::{Archive, ReadPool};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};

const EXAMPLES: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/sources/claude-code"
);

fn converge(root: &Path, home: &Path, blob_roots: Vec<PathBuf>) {
    let (archive, _lock) = Archive::open_or_create(root).unwrap();
    let config = Config {
        home: home.to_path_buf(),
        tool_dirs: Default::default(),
        workers: 2,
        reader_memory_bytes: 128 << 20,
        bytes_in_flight: 16 << 20,
        machine: "proof-machine".into(),
        host: "proof".into(),
        chunk: 64 << 20,
        blob_roots,
    };
    let mut engine = Engine::new(config, archive).unwrap();
    let pass = engine.converge().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
}

/// Every session and its messages, as the reader answers them.
fn answers(root: &Path) -> String {
    let pool = ReadPool::open(&Archive::at(root)).unwrap();
    let reader = pool.reader().unwrap();
    let mut out = String::new();
    let sessions = reader
        .query(
            "SELECT id, kind, coalesce(title, derived_title) AS title, started_at, last_activity_at, cwd \
             FROM session ORDER BY id",
            &[],
        )
        .unwrap();
    for s in sessions {
        out.push_str(&format!("session {s}\n"));
        let id = polylogue_model::SessionId::parse(s["id"].as_str().unwrap()).unwrap();
        for m in reader.messages(&id, false).unwrap() {
            out.push_str(&format!("  {m:?}\n"));
        }
    }
    out
}

struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[tokio::test]
#[ignore = "needs Sinex's binaries; set SINEXD and SINEX_SCHEMA"]
async fn excision_restores_current_units_after_retired_payloads_and_unknown_ack() {
    let sinexd = std::env::var("SINEXD").expect("SINEXD names a built sinexd");
    let schema = std::env::var("SINEX_SCHEMA").expect("SINEX_SCHEMA names a built sinex-schema");
    let dir = tempfile::tempdir().unwrap();

    // 1. A standalone archive.
    let home = dir.path().join("home");
    let case = Path::new(EXAMPLES).join("turn-with-tool-call");
    let rel = fs::read_to_string(case.join("path")).unwrap();
    let target = home.join(rel.trim());
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::copy(case.join("input.jsonl"), target).unwrap();
    let a = dir.path().join("a");
    converge(&a, &home, vec![]);
    let expected = answers(&a);
    assert!(
        expected.contains("session {"),
        "the examples produced sessions"
    );

    // Authored rows share the same native publication path as source revisions.
    let source_archive = Archive::at(&a);
    let conn = polylogue_store::archive::open_write(
        &source_archive.archive_db(),
        polylogue_store::archive::cache::SETUP,
    )
    .unwrap();
    let seq: i64 = conn
        .query_row(
            "SELECT CAST(value AS INTEGER) FROM meta WHERE key='write_counter'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    for (offset, id, value) in [
        (1, "note1", "before retire-me after later"),
        (2, "note2", "unrelated exact text"),
    ] {
        conn.execute("INSERT INTO note(id,root,kind,value,author,machine,created_at,commit_seq) VALUES (?1,?1,'note',?2,'operator','proof',1,?3)", (id,value,seq+offset)).unwrap();
    }
    conn.execute(
        "UPDATE meta SET value=?1 WHERE key='write_counter'",
        [(seq + 2).to_string()],
    )
    .unwrap();
    drop(conn);
    let note_unit = polylogue_store::sinex_rows::Unit {
        commit_seq: seq + 1,
        kind: "note".into(),
        key: "note1".into(),
    };
    let mut old_batch = polylogue_store::sinex_rows::Published::open(&a)
        .unwrap()
        .read(&note_unit)
        .unwrap();
    let mut old_digest = blake3::Hasher::new();
    std::io::copy(&mut old_batch.file, &mut old_digest).unwrap();
    let old_hash = hashed_blobs::Hash(*old_digest.finalize().as_bytes());

    // 2. A Sinex: PostgreSQL with its schema, NATS, sinexd.
    let pg = devkit::pg::Server::start(&devkit::pg::ClusterOptions {
        preload: vec!["timescaledb".into()],
        settings: vec![("timescaledb.enable_chunk_skipping".into(), "on".into())],
    })
    .unwrap();
    pg.sql("postgres", "CREATE DATABASE sinex").unwrap();
    let url = pg.url("sinex", None);
    let applied = Command::new(&schema)
        .args(["apply", &url])
        .status()
        .unwrap();
    assert!(applied.success(), "sinex-schema apply");
    let nats = devkit::nats::Server::start().unwrap();
    let store = dir.path().join("sinex-store");
    let tokens = dir.path().join("tokens");
    fs::create_dir_all(&tokens).unwrap();
    for t in ["read", "agent", "operator"] {
        fs::write(
            tokens.join(format!("{t}.token")),
            format!("{t}-proof-token"),
        )
        .unwrap();
    }
    let daemon = Daemon(
        Command::new(&sinexd)
            .arg("run")
            .args(["--database", &url])
            .arg("--store")
            .arg(&store)
            .args(["--nats", nats.url(), "--subjects", "proof"])
            .arg("--tokens")
            .arg(&tokens)
            .arg("--socket")
            .arg(dir.path().join("sinexd.sock"))
            .spawn()
            .unwrap(),
    );
    let client = async_nats::connect(nats.url()).await.unwrap();
    let js = jetstream::new(client);
    let started = Instant::now();
    while js.get_stream("PROOF_INGRESS").await.is_err() {
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "sinexd did not start"
        );
        tokio::time::sleep(Duration::from_millis(200)).await;
    }

    // 3. Publish; a second pass sends nothing.
    let config = crate::Config {
        mode: Mode::Primary,
        nats: nats.url().into(),
        subjects: "proof".into(),
        producer: "polylogue".into(),
        store: store.clone(),
        archive: None,
    };
    let mut sinex = Sinex::connect(&config).await.unwrap();
    let deadline = Instant::now();
    loop {
        match sinex.have(&["readiness-probe".into()]).await {
            Ok(_) => break,
            Err(crate::Error::Unreachable(_)) if deadline.elapsed() < Duration::from_secs(10) => {
                tokio::time::sleep(Duration::from_millis(50)).await
            }
            other => panic!("producer have endpoint did not become ready: {other:?}"),
        }
    }
    if std::env::var_os("POLYLOGUE_OLD8").is_some() {
        historical_machine::proof(&mut sinex, &config, dir.path()).await;
    }
    historical_machine::cached_comparison(&mut sinex, dir.path()).await;
    let first = sinex.publish(&a).await.unwrap();
    assert!(first.units > 0 && first.held == 0, "{first:?}");
    let second = sinex.publish(&a).await.unwrap();
    assert_eq!((second.units, second.held), (0, first.units), "{second:?}");

    // 4. Restore into an empty archive once every unit is confirmed.
    let id = polylogue_store::sinex_rows::Published::open(&a)
        .unwrap()
        .archive_id()
        .unwrap();
    let b = dir.path().join("b");
    let started = Instant::now();
    let restored = loop {
        let (archive, lock) = Archive::open_or_create(&b).unwrap();
        let r = sinex.restore(&archive, &id).await.unwrap();
        drop(lock);
        if r.units == first.units {
            break r;
        }
        assert!(
            started.elapsed() < Duration::from_secs(20),
            "only {r:?} of {first:?} confirmed"
        );
        tokio::time::sleep(Duration::from_millis(300)).await;
    };
    assert_eq!(restored.units, first.units);

    // 5. Converge with no provider files: every byte comes from Sinex's store.
    let empty = dir.path().join("empty-home");
    fs::create_dir_all(&empty).unwrap();
    converge(&b, &empty, vec![store.clone()]);
    assert_eq!(answers(&b), expected);

    // Confirm the replacement, then make its before-image physically unavailable.
    let writer = polylogue_store::Writer::start(&source_archive).unwrap();
    let local = std::sync::Arc::new(hashed_blobs::Store::open(a.clone(), vec![], 0o600).unwrap());
    writer
        .excise(
            polylogue_store::excise::ExciseRequest {
                id: "erase1".into(),
                scope: polylogue_store::excise::Scope::Text("retire-me".into()),
                reason: None,
                dry_run: false,
            },
            local.clone(),
        )
        .unwrap();
    let changed = sinex.publish(&a).await.unwrap();
    assert_eq!(changed.units, 1);
    assert_eq!(
        changed.physical_retirement,
        crate::PhysicalRetirement::Completed
    );
    let (restored_archive, _restored_lock) = Archive::open_or_create(&b).unwrap();
    let deadline = Instant::now();
    loop {
        let result = sinex.restore(&restored_archive, &id).await.unwrap();
        if result.units == first.units + 1 {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let physical = hashed_blobs::Store::open(store.clone(), vec![], 0o600).unwrap();
    assert!(!physical.contains(&old_hash));
    assert_eq!(
        sinex.publish(&a).await.unwrap().physical_retirement,
        crate::PhysicalRetirement::Completed
    );
    let fresh_path = dir.path().join("after-retirement");
    let (fresh, _fresh_lock) = Archive::open_or_create(&fresh_path).unwrap();
    let restored = sinex.restore(&fresh, &id).await.unwrap();
    assert_eq!(restored.units, first.units + 1);
    let values = |archive: &Archive| {
        let conn = polylogue_store::archive::open_read(&archive.archive_db()).unwrap();
        conn.prepare("SELECT id,root,value FROM note ORDER BY id")
            .unwrap()
            .query_map([], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    };
    assert_eq!(values(&fresh), values(&source_archive));
    assert_eq!(values(&fresh)[1].2, "unrelated exact text");
    assert!(!values(&fresh)[0].2.contains("retire-me"));
    assert_eq!(sinex.restore(&fresh, &id).await.unwrap().rows, 0);

    // A second excision may retire the first replacement without making fresh
    // restore replay either unavailable before-image.
    let published = polylogue_store::sinex_rows::Published::open(&a).unwrap();
    let erase1 = inventory_units(&published)
        .into_iter()
        .find(|u| u.key == "erase1")
        .unwrap();
    let replacement = published
        .publication(&erase1)
        .unwrap()
        .unwrap()
        .replacements[0]
        .batch
        .clone()
        .unwrap();
    writer
        .excise(
            polylogue_store::excise::ExciseRequest {
                id: "erase2".into(),
                scope: polylogue_store::excise::Scope::Text("before".into()),
                reason: None,
                dry_run: false,
            },
            local.clone(),
        )
        .unwrap();
    assert_eq!(sinex.publish(&a).await.unwrap().units, 1);
    let deadline = Instant::now();
    loop {
        if sinex.restore(&fresh, &id).await.unwrap().units == first.units + 2 {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    let replacement_hash = hashed_blobs::Hash(
        polylogue_store::sinex_rows::unhex(replacement.strip_prefix("blake3:").unwrap())
            .unwrap()
            .try_into()
            .unwrap(),
    );
    assert!(
        published.replacement(&replacement).is_err(),
        "later excision removes the old local pending batch too"
    );
    assert!(!physical.contains(&replacement_hash));
    let (latest, _latest_lock) =
        Archive::open_or_create(&dir.path().join("after-second-retirement")).unwrap();
    sinex.restore(&latest, &id).await.unwrap();
    assert_eq!(values(&latest), values(&source_archive));
    assert_eq!(values(&restored_archive)[1].2, "unrelated exact text");
    // Two local excisions can retire an unpublished intermediate batch. Publish
    // only the final unit, then lose its application ACK and reconnect.
    for (request, text) in [("erase3", "after"), ("erase4", "later")] {
        writer
            .excise(
                polylogue_store::excise::ExciseRequest {
                    id: request.into(),
                    scope: polylogue_store::excise::Scope::Text(text.into()),
                    reason: None,
                    dry_run: false,
                },
                local.clone(),
            )
            .unwrap();
    }
    let published = super::ArchiveRows::open(&a).unwrap();
    let units = inventory_units(&published);
    let last = units.last().unwrap();
    assert_eq!(last.key, "erase4");
    let manifest = published.publication(last).unwrap().unwrap();
    let mut counters = super::Published::default();
    for replacement in &manifest.replacements {
        if let Some(hash) = &replacement.batch {
            sinex
                .place_batch(&id, &mut published.replacement(hash).unwrap())
                .await
                .unwrap();
        }
    }
    let (frame, _) = sinex
        .unit_frame(
            &published,
            &id,
            units.len() as u64,
            last,
            Some(&manifest),
            &mut counters,
        )
        .await
        .unwrap();
    js.publish(
        sinex.subjects.ingress(&sinex.producer),
        serde_json::to_vec(&frame).unwrap().into(),
    )
    .await
    .unwrap()
    .await
    .unwrap();
    drop(sinex); // No producer acknowledgment was consumed for this frame.
    let mut sinex = Sinex::connect(&config).await.unwrap();
    let (incomplete, _incomplete_lock) =
        Archive::open_or_create(&dir.path().join("incomplete-suffix")).unwrap();
    let retained_before = values(&latest);
    let deadline = Instant::now();
    loop {
        match sinex.restore(&latest, &id).await {
            Err(crate::Error::Store(polylogue_store::StoreError::Invalid(message)))
                if message.contains("but not unit") =>
            {
                break;
            }
            Ok(_) => {}
            other => panic!("unexpected incomplete restore: {other:?}"),
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(
        values(&latest),
        retained_before,
        "gap refusal preserves previous complete rows"
    );
    assert!(matches!(
        sinex.restore(&incomplete, &id).await,
        Err(crate::Error::Store(polylogue_store::StoreError::Invalid(_)))
    ));
    let retried = sinex.publish(&a).await.unwrap();
    assert_eq!(
        retried.units, 1,
        "the highest frame survived its unknown acknowledgment"
    );
    assert_eq!(retried.held, units.len() as u64 - 1);
    let deadline = Instant::now();
    loop {
        match sinex.restore(&incomplete, &id).await {
            Ok(result) if result.units == units.len() as u64 => break,
            Err(crate::Error::Store(polylogue_store::StoreError::Invalid(message)))
                if message.contains("but not unit") => {}
            other => panic!("unexpected retry restore: {other:?}"),
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(values(&incomplete), values(&source_archive));
    assert_eq!(values(&incomplete)[1].2, "unrelated exact text");
    assert_eq!(sinex.publish(&a).await.unwrap().units, 0);

    // A later lifecycle fact has its own frame after its prepared delivery is held.
    use polylogue_store::user_writes::{UserRequest, UserWrite};
    writer
        .user(UserRequest::now(
            "prepared:[opaque]".into(),
            UserWrite::Deliver(polylogue_store::Delivery {
                recipient_kind: "operator".into(),
                recipient_id: None,
                sender: "cli".into(),
                event: "advisory".into(),
                cwd: None,
                text: "Synthetic event fixture".into(),
                refs: vec![],
                left_out: vec![],
            }),
        ))
        .unwrap();
    assert_eq!(sinex.publish(&a).await.unwrap().units, 1);
    assert_eq!(sinex.publish(&a).await.unwrap().units, 0);
    writer
        .user(UserRequest::now(
            "late-event".into(),
            UserWrite::DeliveryEvent {
                delivery: "prepared:[opaque]".into(),
                what: polylogue_store::DeliveryOutcome::Failed,
                detail: Some("synthetic event-private-token evidence".into()),
            },
        ))
        .unwrap();
    let sent = sinex.publish(&a).await.unwrap();
    assert_eq!(sent.units, 1, "late fact must append exactly one unit");
    let published = polylogue_store::sinex_rows::Published::open(&a).unwrap();
    let expected_units = published.inventory().unwrap().len();
    let event = inventory_units(&published)
        .into_iter()
        .find(|u| u.kind == "delivery_event")
        .unwrap();
    let mut event_batch = published.read(&event).unwrap();
    let mut digest = blake3::Hasher::new();
    std::io::copy(&mut event_batch.file, &mut digest).unwrap();
    let event_hash = hashed_blobs::Hash(*digest.finalize().as_bytes());
    drop(published);
    let event_values = |archive: &Archive| {
        let db = polylogue_store::archive::open_read(&archive.archive_db()).unwrap();
        db.query_row(
            "SELECT delivery,what,at,detail,commit_seq FROM delivery_event",
            [],
            |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, String>(3)?,
                    r.get::<_, i64>(4)?,
                ))
            },
        )
        .unwrap()
    };
    let deadline = Instant::now();
    loop {
        if sinex.restore(&incomplete, &id).await.unwrap().units == expected_units {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert_eq!(event_values(&incomplete), event_values(&source_archive));
    writer
        .excise(
            polylogue_store::excise::ExciseRequest {
                id: "redact-event".into(),
                scope: polylogue_store::excise::Scope::Text("event-private-token".into()),
                reason: None,
                dry_run: false,
            },
            local.clone(),
        )
        .unwrap();
    assert_eq!(sinex.publish(&a).await.unwrap().units, 1);
    let deadline = Instant::now();
    loop {
        if sinex.restore(&incomplete, &id).await.unwrap().units == expected_units + 1 {
            break;
        }
        assert!(deadline.elapsed() < Duration::from_secs(20));
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    assert!(!physical.contains(&event_hash));
    let (event_fresh, _event_lock) =
        Archive::open_or_create(&dir.path().join("after-event-retirement")).unwrap();
    sinex.restore(&event_fresh, &id).await.unwrap();
    assert_eq!(event_values(&event_fresh), event_values(&source_archive));
    assert!(!event_values(&event_fresh).3.contains("event-private-token"));
    assert_eq!(sinex.restore(&event_fresh, &id).await.unwrap().rows, 0);
    let machine_alias = machine_alias::rename(&mut sinex, dir.path()).await;
    session_policy::input_removal(&mut sinex, &store, dir.path()).await;
    legacy::parent_retirement(&mut sinex, &store, dir.path()).await;
    // Rebuild the stream from durable descriptors after all before-images go.
    drop(daemon);
    js.delete_stream("PROOF_CONFIRMED").await.unwrap();
    pg.sql("sinex", "DELETE FROM core.consumer_positions WHERE consumer='publisher'; DELETE FROM core.published").unwrap();
    let published = Command::new(&sinexd)
        .arg("publish")
        .args([
            "--database",
            &url,
            "--nats",
            nats.url(),
            "--subjects",
            "proof",
        ])
        .status()
        .unwrap();
    assert!(published.success());
    let (rebuilt, _rebuilt_lock) =
        Archive::open_or_create(&dir.path().join("after-stream-reconstruction")).unwrap();
    let restored = sinex.restore(&rebuilt, &id).await.unwrap();
    assert_eq!(restored.units, expected_units + 1);
    assert_eq!(values(&rebuilt), values(&source_archive));
    assert_eq!(event_values(&rebuilt), event_values(&source_archive));
    assert!(!physical.contains(&old_hash));
    assert!(!physical.contains(&replacement_hash));
    assert!(!physical.contains(&event_hash));
    machine_alias::after_restart(&mut sinex, dir.path(), machine_alias).await;
}

fn inventory_units(published: &super::ArchiveRows) -> Vec<polylogue_store::sinex_rows::Unit> {
    let inventory = published.inventory().unwrap();
    (1..=inventory.len())
        .map(|position| inventory.unit(position).unwrap())
        .collect()
}

mod placement;

mod legacy;
mod session_policy;

mod machine_alias;

mod historical_machine;
