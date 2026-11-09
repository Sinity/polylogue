use super::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Run time in nanoseconds of each parse worker thread alive now, by thread
/// ID (`/proc/self/task/<tid>/schedstat`, first field).
fn parse_worker_times() -> BTreeMap<String, u64> {
    let mut times = BTreeMap::new();
    let Ok(tasks) = fs::read_dir("/proc/self/task") else {
        return times;
    };
    for task in tasks.flatten() {
        let path = task.path();
        let named = fs::read_to_string(path.join("comm"))
            .is_ok_and(|name| name.trim_end().starts_with("polylogue-parse"));
        let ran = fs::read_to_string(path.join("schedstat"))
            .ok()
            .and_then(|stat| stat.split_whitespace().next()?.parse().ok());
        if let (true, Some(ran)) = (named, ran) {
            times.insert(task.file_name().to_string_lossy().into_owned(), ran);
        }
    }
    times
}

/// A lone large input is read one range at a time. Each range goes to the
/// worker that ran the one before, so the pass holds one worker's memory,
/// not a range's worth with every worker in turn
/// (docs/architecture.md, pipeline).
#[test]
fn the_ranges_of_a_lone_input_run_on_one_worker() {
    let w = ClaudeHome::new();
    let filler = "quokka ".repeat(100);
    let mut lines = String::new();
    for i in 0..2_000 {
        lines += &user(&format!("u{i}"), "long", &format!("{filler}{i}"));
        lines += &assistant(&format!("a{i}"), "long", &format!("{filler}{i}"));
    }
    w.append("long.jsonl", &lines);
    const CHUNK: u64 = 64 << 10;
    let ranges = lines.len() as u64 / CHUNK;
    assert!(ranges >= 20, "{ranges} ranges");

    let (archive, _lock) = Archive::open_or_create(&w.root).unwrap();
    let mut config = w.config(CHUNK);
    config.workers = 8;
    let mut engine = Engine::new(config, archive, polylogue_testkit::stopped_attempt()).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let sampler = {
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            let mut most = BTreeMap::<String, u64>::new();
            while !stop.load(Ordering::Acquire) {
                for (thread, ran) in parse_worker_times() {
                    let held = most.entry(thread).or_default();
                    *held = (*held).max(ran);
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            most
        })
    };
    let pass = engine.run_pass().unwrap();
    stop.store(true, Ordering::Release);
    let times = sampler.join().unwrap();
    assert!(pass.errors.is_empty(), "{:?}", pass.errors);
    assert_eq!(w.texts("long").len(), 4_000);

    assert!(times.len() >= 8, "every worker was seen: {times:?}");
    let busiest = times.values().copied().max().unwrap();
    let busy: Vec<_> = times.values().filter(|&&ran| ran * 4 >= busiest).collect();
    assert_eq!(busy.len(), 1, "run time by worker: {times:?}");
}
