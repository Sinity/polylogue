//! Compare a first build and one origin's re-parse in separate processes.
//! Run through AgentCTL's measure operation over an isolated sample home.
use polylogue_engine::{Config, Engine};
use polylogue_store::{Archive, ReadPool};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};

#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn io() -> BTreeMap<String, u64> {
    fs::read_to_string("/proc/self/io")
        .unwrap_or_default()
        .lines()
        .filter_map(|line| {
            let (name, n) = line.split_once(':')?;
            Some((name.into(), n.trim().parse().ok()?))
        })
        .collect()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let arg = |key: &str| {
        args.iter()
            .position(|a| a == key)
            .and_then(|n| args.get(n + 1))
            .cloned()
    };
    let home = PathBuf::from(arg("--home").expect("--home"));
    let root = PathBuf::from(arg("--archive").expect("--archive"));
    let mode = arg("--mode").expect("--mode fresh or reparse");
    assert!(mode == "fresh" || mode == "reparse");
    if mode == "fresh" {
        assert!(!root.exists(), "a fresh build needs an absent archive");
    }
    let (archive, _lock) = Archive::open_or_create(&root).unwrap();
    let mut selected = None;
    if mode == "reparse" {
        let reads = ReadPool::open(&archive).unwrap();
        let rows=reads.reader().unwrap().query("SELECT i.format,coalesce(sum((SELECT max(length) FROM source_revision r WHERE r.input=i.id AND r.chain=p.chain)),0) AS bytes FROM parse_state p JOIN input_ref i ON i.pk=p.input_pk GROUP BY i.format ORDER BY bytes DESC,i.format",&[]).unwrap();
        let origin = rows[0]["format"].as_str().unwrap().to_owned();
        drop(reads);
        let (chains, bytes) = polylogue_store::fixture::older_reading(&archive, &origin).unwrap();
        selected = Some(json!({"origin":origin,"chains":chains,"bytes":bytes}));
    }
    let mut config = Config::from_env();
    config.home = home;
    config.tool_dirs.clear();
    if let Some(workers) = arg("--workers") {
        config.workers = workers.parse().unwrap();
    }
    let workers = config.workers;
    let bytes_in_flight = config.bytes_in_flight;
    let clock_ticks_per_second = std::process::Command::new("getconf")
        .arg("CLK_TCK")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .and_then(|out| out.trim().parse::<u64>().ok());
    let setup = Instant::now();
    let mut engine = Engine::new(config, archive.clone()).unwrap();
    let setup_seconds = setup.elapsed().as_secs_f64();
    let before = io();
    let load_before = fs::read_to_string("/proc/loadavg").ok();
    let started = Instant::now();
    polylogue_store::timing::reset();
    let (stop, stopped) = std::sync::mpsc::channel();
    let sample = std::thread::spawn(move || {
        let began = Instant::now();
        let mut samples = Vec::new();
        loop {
            let stages: BTreeMap<_, _> = polylogue_store::timing::stage_seconds()
                .into_iter()
                .map(|(stage, seconds)| (stage.name(), seconds))
                .collect();
            let threads: Vec<_> = fs::read_dir("/proc/self/task")
                .unwrap()
                .filter_map(Result::ok)
                .filter_map(|entry| {
                    let stat = fs::read_to_string(entry.path().join("stat")).ok()?;
                    let (name, fields) = stat.rsplit_once(") ")?;
                    let fields: Vec<_> = fields.split_whitespace().collect();
                    Some(
                        json!({"thread":entry.file_name().to_str(),"name":name.split_once('(')?.1,
                    "cpu_ticks":fields[11].parse::<u64>().ok()? + fields[12].parse::<u64>().ok()?,
                    "io_wait_ticks":fields.get(39).and_then(|v|v.parse::<u64>().ok()),
                    "waiting_on":fs::read_to_string(entry.path().join("wchan")).ok()}),
                    )
                })
                .collect();
            samples.push(json!({"seconds":began.elapsed().as_secs_f64(),"queues":polylogue_store::timing::reading_pipeline(),"stages":stages,"threads":threads}));
            if stopped
                .recv_timeout(std::time::Duration::from_secs(1))
                .is_ok()
            {
                break;
            }
        }
        samples
    });
    let pass = engine.run_pass().unwrap();
    drop(engine);
    let seconds = started.elapsed().as_secs_f64();
    let _ = stop.send(());
    let pipeline = sample.join().unwrap();
    let after = io();
    assert!(
        pass.errors.is_empty(),
        "sample ingestion failed with {} errors",
        pass.errors.len()
    );
    assert!(
        pass.unreadable_roots.is_empty(),
        "sample has unreadable roots"
    );
    if mode == "fresh" {
        assert!(pass.messages_written > 0, "sample wrote no messages");
    } else {
        assert!(pass.bytes_read > 0, "sample read no chains");
    }
    let stages: serde_json::Map<String, Value> = polylogue_store::timing::stage_seconds()
        .into_iter()
        .map(|(s, n)| (s.name().into(), json!(n)))
        .collect();
    let peak_rss_mb = fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find_map(|line| {
                line.strip_prefix("VmHWM:")
                    .and_then(|v| v.split_whitespace().next())
                    .and_then(|n| n.parse::<u64>().ok())
            })
        })
        .map(|kb| kb * 1024 / 1_000_000);
    let reads = ReadPool::open(&archive).unwrap();
    let r = reads.reader().unwrap();
    let storage = r.query(
        "SELECT name,sum(pgsize) AS bytes FROM dbstat GROUP BY name ORDER BY name",
        &[],
    );
    let (storage, storage_error) = match storage {
        Ok(rows) => (Some(rows), None),
        Err(e) => (None, Some(e.to_string())),
    };
    println!(
        "{}",
        json!({"mode":mode,"workers":workers,"bytes_in_flight":bytes_in_flight,"clock_ticks_per_second":clock_ticks_per_second,"peak_rss_mb":peak_rss_mb,"load_before":load_before,"load_after":fs::read_to_string("/proc/loadavg").ok(),"setup_seconds":setup_seconds,"seconds":seconds,"selected":selected,
        "read_bytes":after.get("read_bytes").zip(before.get("read_bytes")).map(|(a,b)|a-b),
        "write_bytes":after.get("write_bytes").zip(before.get("write_bytes")).map(|(a,b)|a-b),
        "pass":pass,"stages":stages,"pipeline":pipeline,"reading_pipeline":polylogue_store::timing::reading_pipeline(),"storage":storage,"storage_error":storage_error})
    );
}
