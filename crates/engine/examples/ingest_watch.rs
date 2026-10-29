//! Paired ingest measurement. Run through AgentCTL's measure operation.
//! The producer is a child process so its I/O is separate from the loop's.
use polylogue_engine::{Config, Engine, LoopMsg, LoopStatus};
use polylogue_store::{Archive, Batch, Change, InputKey, Observation, ParseStep, ReadPool, Writer};
use serde_json::json;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex, mpsc::channel};
use std::time::{Duration, Instant};

fn io() -> serde_json::Value {
    fs::read_to_string("/proc/self/io")
        .unwrap()
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.to_owned(), json!(value.trim().parse::<u64>().unwrap())))
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}

fn rollout(n: u64) -> String {
    format!(
        "{}\n",
        json!({"type":"response_item","payload":{"type":"message","id":format!("message-{n}"),"role":"user","content":[{"type":"input_text","text":format!("{n} {}", "synthetic content ".repeat(1024))}]}})
    )
}

fn produce(dir: &Path, seconds: u64) {
    let db = rusqlite::Connection::open(dir.join("home/.codex/state_5.sqlite")).unwrap();
    let mut rollout = OpenOptions::new()
        .append(true)
        .open(dir.join("home/.codex/sessions/2026/10/27/rollout-measure.jsonl"))
        .unwrap();
    let mut written = 0;
    let started = Instant::now();
    for n in 1..=seconds {
        let line = self::rollout(n);
        rollout.write_all(line.as_bytes()).unwrap();
        rollout.sync_data().unwrap();
        written += line.len();
        // Separate the two event bursts so unrelated file wakes cannot make
        // a continuously updated database appear quiet.
        std::thread::sleep(Duration::from_millis(250));
        db.execute(
            "UPDATE kept_rows SET n=?1 WHERE id=1",
            [i64::try_from(n).expect("measurement ordinal fits SQLite INTEGER")],
        )
        .unwrap();
        fs::write(
            dir.join("producer.json"),
            json!({"rollout_bytes":written,"database_updates":n}).to_string(),
        )
        .unwrap();
        let target = Duration::from_secs(n);
        if let Some(delay) = target.checked_sub(started.elapsed()) {
            std::thread::sleep(delay);
        }
    }
    fs::write(
        dir.join("producer.json"),
        json!({"rollout_bytes":written,"database_updates":seconds,"io":io()}).to_string(),
    )
    .unwrap();
}

fn watch(dir: &Path, seconds: u64) {
    fs::create_dir_all(dir.join("home/.codex/sessions/2026/10/27")).unwrap();
    let db = rusqlite::Connection::open(dir.join("home/.codex/state_5.sqlite")).unwrap();
    db.execute_batch("PRAGMA journal_mode=WAL; CREATE TABLE kept_rows(id INTEGER PRIMARY KEY,n INTEGER,body TEXT); INSERT INTO kept_rows VALUES(1,0,printf('%0128000d',1));").unwrap();
    drop(db);
    fs::write(dir.join("home/.codex/sessions/2026/10/27/rollout-measure.jsonl"), format!("{}\n", json!({"type":"session_meta","payload":{"id":"00000000-0000-4000-8000-000000000042","source":"cli"}}))).unwrap();
    let (archive, lock) = Archive::open_or_create(&dir.join("archive")).unwrap();
    let mut config = Config::from_env();
    config.home = dir.join("home");
    config.tool_dirs.clear();
    config.workers = 2;
    let mut engine = Engine::new(config, archive).unwrap();
    let first = engine.run_pass().unwrap();
    assert!(first.errors.is_empty(), "{:?}", first.errors);
    let reads = polylogue_store::ReadPool::open(&engine.archive).unwrap();
    let before = io();
    polylogue_store::timing::reset();
    let measurement_clock = polylogue_store::timing::clock_seconds();
    let samples = Arc::new(Mutex::new(Vec::new()));
    let recorded = samples.clone();
    let (stop_sample, sample_stop) = channel();
    let sampler = std::thread::spawn(move || {
        let before = thread_io();
        let started = Instant::now();
        loop {
            recorded.lock().unwrap().push(json!({"seconds":started.elapsed().as_secs_f64(),"threads":threads(),"stages":stages(),"queues":polylogue_store::timing::queues()}));
            if sample_stop.recv_timeout(Duration::from_secs(1)).is_ok() {
                break;
            }
        }
        json!({"before_io":before,"after_io":thread_io()})
    });
    let status = Arc::new(Mutex::new(LoopStatus::default()));
    let shared = status.clone();
    let (tx, rx) = channel();
    let (ready, started) = channel();
    let worker_tx = tx.clone();
    let thread = std::thread::spawn(move || {
        let _lock = lock;
        engine
            .run(worker_tx, &rx, &shared, || ready.send(()).unwrap())
            .unwrap();
    });
    started.recv().unwrap();
    let mut producer = std::process::Command::new(std::env::current_exe().unwrap())
        .arg("produce")
        .arg(dir)
        .arg(seconds.to_string())
        .spawn()
        .unwrap();
    assert!(producer.wait().unwrap().success());
    // End the measurement after the last append and final comparison are read.
    let checking_before = thread_io();
    loop {
        let s = status.lock().unwrap().clone();
        assert!(s.last_pass.errors.is_empty(), "{:?}", s.last_pass.errors);
        let rows = reads
            .reader()
            .unwrap()
            .query("SELECT count(*) AS n FROM message", &[])
            .unwrap();
        let count = rows[0]["n"].as_u64().unwrap();
        let value = json!({"body":format!("{}1","0".repeat(127999)),"id":1,"n":seconds});
        let hash = blake3::hash(value.to_string().as_bytes())
            .to_hex()
            .to_string()
            .to_uppercase();
        let captured = reads
            .reader()
            .unwrap()
            .query(
                "SELECT hex(hash) AS hash FROM tool_row WHERE tbl='kept_rows'",
                &[],
            )
            .unwrap()
            .iter()
            .any(|r| r["hash"] == hash);
        if count >= seconds && captured {
            break;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    let checking_after = thread_io();
    tx.send(LoopMsg::Stop).unwrap();
    thread.join().unwrap();
    stop_sample.send(()).unwrap();
    let sampling_io = sampler.join().unwrap();
    let after = io();
    fs::write(dir.join("result.json"), json!({"seconds":seconds,"before_io":before,"after_io":after,"memory":fs::read_to_string("/proc/self/status").unwrap().lines().filter(|line| line.starts_with("VmHWM:") || line.starts_with("VmRSS:")).collect::<Vec<_>>(),"load":fs::read_to_string("/proc/loadavg").unwrap(),"status":*status.lock().unwrap(),"sampling_io":sampling_io,"checking_io":{"before_io":checking_before,"after_io":checking_after},"measurement_clock":measurement_clock,"workers":polylogue_store::timing::workers(),"samples":*samples.lock().unwrap(),"stages":stages(),"producer":serde_json::from_slice::<serde_json::Value>(&fs::read(dir.join("producer.json")).unwrap()).unwrap()}).to_string()).unwrap();
}

fn thread_io() -> serde_json::Value {
    fs::read_to_string("/proc/thread-self/io")
        .unwrap()
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(':')?;
            Some((name.to_owned(), json!(value.trim().parse::<u64>().unwrap())))
        })
        .collect::<serde_json::Map<_, _>>()
        .into()
}
fn stages() -> serde_json::Value {
    polylogue_store::timing::stage_seconds()
        .into_iter()
        .map(|(stage, seconds)| (stage.name().into(), json!(seconds)))
        .collect::<serde_json::Map<_, _>>()
        .into()
}
fn threads() -> serde_json::Value {
    fs::read_dir("/proc/self/task").unwrap().filter_map(|entry| {
        let path=entry.ok()?.path();
        let stat=fs::read_to_string(path.join("stat")).ok()?;
        let end=stat.rfind(')')?;
        let fields:Vec<_>=stat[end+2..].split_whitespace().collect();
        Some(json!({"id":path.file_name()?.to_str()?,"name":fs::read_to_string(path.join("comm")).ok()?.trim(),"cpu_ticks":fields[11].parse::<u64>().ok()?+fields[12].parse::<u64>().ok()?,"wait":fs::read_to_string(path.join("wchan")).ok()?.trim()}))
    }).collect::<Vec<_>>().into()
}
fn config(dir: &Path) -> Config {
    let mut config = Config::from_env();
    config.home = dir.join("home");
    config.tool_dirs.clear();
    config.workers = 2;
    config
}
fn seed_restart(dir: &Path) {
    let source = dir.join("home/.codex/state_5.sqlite");
    fs::create_dir_all(source.parent().unwrap()).unwrap();
    rusqlite::Connection::open(&source).unwrap().execute_batch("CREATE TABLE kept(id INTEGER PRIMARY KEY,body TEXT); INSERT INTO kept VALUES(1,'synthetic');").unwrap();
    let (archive, _lock) = Archive::open_or_create(&dir.join("archive")).unwrap();
    let writer = Writer::start(&archive).unwrap();
    let id = blake3::hash(format!("{}\0{}", config(dir).machine, source.display()).as_bytes())
        .to_hex()[..32]
        .to_owned();
    let input = InputKey {
        id,
        format: "codex-database".into(),
        arrival_rank: 0,
    };
    let line = format!(
        "{}\n",
        json!({"t":"kept","k":[1],"r":{"id":1,"body":"synthetic ".repeat(800)},"at_ms":1,"encoding":"sqlite"})
    );
    let bytes = line.repeat((8 << 20) / line.len()).into_bytes();
    let mut length = 0;
    while length < 2_u64 << 30 {
        let from = length;
        length += bytes.len() as u64;
        let (steps, _) = polylogue_sources::codex::database::open(&source)
            .unwrap()
            .steps(None, &bytes, from)
            .unwrap();
        let mut parse: Vec<_> = steps
            .into_iter()
            .map(|s| ParseStep::new(s.output, s.counts, from + s.start..from + s.consumed, true))
            .collect();
        for step in &mut parse {
            let start = (step.start - from) as usize;
            let end = (step.parsed_upto - from) as usize;
            step.compare_records(&bytes[start..end]).unwrap();
        }
        let (blob, file) = archive.prepare_blob(&bytes).unwrap();
        let batch = Batch {
            input: input.clone(),
            change: if from == 0 {
                Change::First
            } else {
                Change::Appended { from }
            },
            observation: Some(Observation {
                machine: config(dir).machine,
                host: "test".into(),
                path: source.display().to_string(),
                facts: None,
                seen_at_ms: 1,
                import: None,
                container: None,
            }),
            continues: false,
            new_blob: Some(blob),
            blobs: vec![],
            length,
            parse,
            release: Default::default(),
        };
        for result in writer
            .sender()
            .unwrap()
            .store_group(
                vec![polylogue_store::staging::PreparedBatch {
                    batch,
                    files: file.into_iter().collect(),
                }],
                Arc::new(archive.blobs().unwrap()),
            )
            .unwrap()
        {
            result.unwrap();
        }
    }
    writer.derive().unwrap();
    fs::write(dir.join("history-bytes"), length.to_string()).unwrap();
}
fn restart(dir: &Path) {
    let before = io();
    let started = Instant::now();
    let (archive, _lock) = Archive::open_or_create(&dir.join("archive")).unwrap();
    let mut engine = Engine::new(config(dir), archive).unwrap();
    let result = engine.run_pass().unwrap();
    assert!(result.errors.is_empty(), "{:?}", result.errors);
    let elapsed = started.elapsed().as_secs_f64();
    let reads = ReadPool::open(&engine.archive).unwrap();
    let rows = reads
        .reader()
        .unwrap()
        .query("SELECT sum(length) AS n FROM tool_database", &[])
        .unwrap();
    println!(
        "{}",
        json!({"seconds":elapsed,"before_io":before,"after_io":io(),"history_bytes":rows[0]["n"],"stages":stages()})
    );
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let dir = Path::new(&args[2]);
    match args[1].as_str() {
        "produce" => produce(dir, args[3].parse().unwrap()),
        "watch" => watch(dir, args[3].parse().unwrap()),
        "restart" => restart(dir),
        "seed-restart" => seed_restart(dir),
        _ => panic!("expected watch, produce or restart"),
    }
}
