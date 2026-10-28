use super::*;
use polylogue_model::{Json, RecordCounts, RecordOutcome};
use polylogue_sources::{Arrival, OriginSpec, Root, Step, StepParser};
use std::sync::mpsc::{Receiver, Sender};

struct Control {
    started: Sender<()>,
    release: Receiver<()>,
}
static CONTROL: Mutex<Option<Control>> = Mutex::new(None);

struct Held;
impl StepParser for Held {
    fn step(
        &self,
        _: Option<&Json>,
        bytes: &[u8],
        _: u64,
    ) -> Result<(Step, Json), polylogue_sources::ParseError> {
        let control = CONTROL.lock().unwrap().take();
        if let Some(Control { started, release }) = control {
            let _ = started.send(());
            let _ = release.recv();
        }
        let mut counts = RecordCounts {
            records: 1,
            ..RecordCounts::default()
        };
        counts.add("synthetic", &RecordOutcome::skipped("synthetic input"));
        Ok((
            Step {
                output: Default::default(),
                counts,
                start: 0,
                consumed: bytes.len() as u64,
                repeats_whole_conversations: false,
            },
            serde_json::json!({}),
        ))
    }
}

#[test]
fn new_files_and_database_records_run_while_another_parser_waits() {
    let home = ClaudeHome::new();
    let db = home.home.join(".codex/state_5.sqlite");
    std::fs::create_dir_all(db.parent().unwrap()).unwrap();
    let source = rusqlite::Connection::open(&db).unwrap();
    let native = "00000000-0000-4000-8000-000000000042";
    source
        .execute_batch("CREATE TABLE threads(id TEXT PRIMARY KEY,name TEXT);")
        .unwrap();
    source
        .execute("INSERT INTO threads VALUES(?1,'before')", [native])
        .unwrap();
    let (mut engine, _lock) = open_engine(&home);
    assert!(engine.run_pass().unwrap().errors.is_empty());
    engine.config.workers = 2;
    engine.config.live_every = Duration::from_millis(10);
    let (wake, _changes) = channel();
    engine.watcher = Some(watch::Roots::new(&engine.config, wake.clone()).unwrap());
    engine.wake = Some(wake);
    let reads = ReadPool::open(&engine.archive).unwrap();

    source
        .execute("UPDATE threads SET name='after'", [])
        .unwrap();
    engine.captures = Databases::default();
    engine
        .captures
        .start(
            vec![databases::DatabaseInput {
                path: db.clone(),
                input: InputKey {
                    id: engine.location_id(&db),
                    format: "codex-database".into(),
                    arrival_rank: 0,
                },
                observation: engine.observation(&db, None),
            }],
            engine.archive.clone(),
            engine.writer.sender().unwrap(),
            engine.wake.clone(),
        )
        .unwrap();

    let path = home.dir.path().join("held-input.jsonl");
    std::fs::write(&path, b"{}\n").unwrap();
    let origin = Box::leak(Box::new(OriginSpec {
        name: "synthetic",
        harness_prefixes: &[],
        modules: &[],
        support: Support::Parsed,
        arrival: Arrival::Live,
        root: Root::HOME,
        locations: &[],
        capture: Capture::Bytes,
        database: None,
        recognize: |_, _| true,
        open: |_| Some(polylogue_sources::Parse::Lines(Box::new(Held))),
        resume: None,
        bare_uuid_ids: false,
        quota: None,
        agent_env: &[],
        web: None,
        assets: None,
        collector: None,
    }));
    let (started, waiting) = channel();
    let (release, continue_reading) = channel();
    *CONTROL.lock().unwrap() = Some(Control {
        started,
        release: continue_reading,
    });
    let mut plan = Plan::stored(
        path.clone(),
        origin,
        InputKey {
            id: "synthetic-held-input".into(),
            format: "synthetic".into(),
            arrival_rank: 0,
        },
        Change::First,
        None,
        3,
        0,
        pool::Bytes::File,
        64 << 20,
        false,
    );
    plan.stored_upto = 0;
    plan.observation = Some(engine.observation(&path, None));
    let thread = std::thread::spawn(move || {
        let mut pass = Pass::default();
        let result = engine.execute_pool(vec![plan], &mut pass);
        (result, pass)
    });
    waiting.recv_timeout(Duration::from_secs(10)).unwrap();
    let fresh = home.file("fresh/fresh.jsonl");
    std::fs::create_dir_all(fresh.parent().unwrap()).unwrap();
    std::fs::write(
        &fresh,
        claude_user("fresh-message", "fresh-session", "fresh input"),
    )
    .unwrap();
    let expected = blake3::hash(
        serde_json::json!({"id":native,"name":"after"})
            .to_string()
            .as_bytes(),
    )
    .to_hex()
    .to_string()
    .to_uppercase();
    let observed = (|| -> polylogue_store::Result<(bool, bool)> {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let reader = reads.reader()?;
            let database = reader
                .query(
                    "SELECT hex(hash) AS hash FROM tool_row WHERE tbl='threads'",
                    &[],
                )?
                .iter()
                .any(|row| row["hash"] == expected);
            let message = reader.query("SELECT count(*) AS n FROM message", &[])?[0]["n"] == 1;
            if (database && message) || Instant::now() >= deadline {
                return Ok((database, message));
            }
            drop(reader);
            std::thread::sleep(Duration::from_millis(10));
        }
    })();
    let _ = release.send(());
    let (result, pass) = thread.join().unwrap();
    result.unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(
        observed.unwrap(),
        (true, true),
        "both inputs must be parsed before the held parser resumes"
    );
}
