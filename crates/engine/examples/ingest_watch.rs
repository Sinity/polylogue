//! Paired ingest measurement. Run through AgentCTL's measure operation.
//! The producer is a child process so its I/O is separate from the loop's.
use polylogue_engine::{Config, Engine, LoopMsg, LoopStatus};
use polylogue_store::{Archive, tool_databases::Exporter};
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
    let database_blobs = engine.archive.blobs().unwrap();
    let reader = reads.reader().unwrap();
    let initial = reader.query("SELECT id FROM a.input WHERE format='codex-database'", &[]).unwrap().iter().any(|input| {
        last_archived_record(&reader,&database_blobs,input["id"].as_str().unwrap()).is_some_and(|row|row["r"]["n"].as_u64()==Some(0))
    });
    assert!(initial,"baseline checker must read the stored initial database row");
    drop(reader);

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
        let reader = reads.reader().unwrap();
        let captured = reader.query("SELECT id FROM a.input WHERE format='codex-database'", &[]).unwrap().iter().any(|input| {
            last_archived_record(&reader,&database_blobs,input["id"].as_str().unwrap())
                .is_some_and(|row| row["r"]["n"].as_u64()==Some(seconds))
        });
        drop(reader);
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

fn restart(dir: &Path) {
    let before = io();
    let started = Instant::now();
    let result = Exporter::default()
        .export(&dir.join("source.sqlite"), &dir.join("export.jsonl"), None)
        .unwrap()
        .unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"seconds":elapsed,"before_io":before,"after_io":io(),"rows":result.rows,"gone":result.gone})
    );
}

fn last_archived_record(reader: &polylogue_store::Reader<'_>, blobs: &hashed_blobs::Store, input: &str) -> Option<serde_json::Value> {
    let progress = reader.input_progress(input).ok()??;
    let chain = reader.stored_chain(input,progress.chain,progress.length).ok()?;
    let mut end = progress.length;
    let mut bytes = Vec::new();
    while end > 0 {
        let start = end.saturating_sub(4096);
        let mut part = chain.read_range(blobs,start,end).ok()?;
        part.extend(bytes);
        bytes = part;
        if bytes.last()!=Some(&b'\n') { return None; }
        if let Some(record_end) = bytes.iter().rposition(|b| !matches!(*b,b'\n'|b'\r')).map(|i|i+1) {
            if let Some(previous) = bytes[..record_end].iter().rposition(|b|*b==b'\n') {
                return serde_json::from_slice(&bytes[previous+1..record_end]).ok();
            }
            if start==0 { return serde_json::from_slice(&bytes[..record_end]).ok(); }
        }
        end=start;
    }
    None
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

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let dir = Path::new(&args[2]);
    match args[1].as_str() {
        "produce" => produce(dir, args[3].parse().unwrap()),
        "watch" => watch(dir, args[3].parse().unwrap()),
        "restart" => restart(dir),
        _ => panic!("expected watch, produce or restart"),
    }
}
