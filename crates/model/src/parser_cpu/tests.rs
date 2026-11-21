use super::{Current, Part, Sample};
use std::time::{Duration, Instant};

#[test]
fn nested_parts_charge_each_interval_once_and_restore_the_parent() {
    let start = Instant::now();
    let at = |wall, cpu| Sample {
        wall: start + Duration::from_nanos(wall),
        cpu: Some(cpu),
    };
    let mut current = Current::default();
    assert!(current.switch(Some(Part::Other), at(0, 0)).is_none());
    assert_eq!(
        current.switch(Some(Part::Provider), at(20, 10)),
        Some((Part::Other, 20, Some(10)))
    );
    assert_eq!(
        current.switch(Some(Part::ContextGet), at(50, 30)),
        Some((Part::Provider, 30, Some(20)))
    );
    assert_eq!(
        current.switch(Some(Part::Provider), at(90, 45)),
        Some((Part::ContextGet, 40, Some(15)))
    );
    assert_eq!(
        current.switch(Some(Part::Other), at(110, 60)),
        Some((Part::Provider, 20, Some(15)))
    );
    assert_eq!(
        current.switch(None, at(130, 70)),
        Some((Part::Other, 20, Some(10)))
    );
    // Exclusive totals: writer 40/20, row 50/35, claim 40/15.
    // The parent does not include its child's 40 wall / 15 CPU nanoseconds.
    assert!(current.switch(None, at(150, 80)).is_none());
}

#[test]
fn an_unavailable_cpu_clock_remains_unknown() {
    let start = Instant::now();
    let mut current = Current::default();
    current.switch(
        Some(Part::Provider),
        Sample {
            wall: start,
            cpu: None,
        },
    );
    assert_eq!(
        current.switch(
            None,
            Sample {
                wall: start + Duration::from_nanos(17),
                cpu: Some(42),
            }
        ),
        Some((Part::Provider, 17, None))
    );
}

#[test]
fn diagnostic_guard_cost_idle_and_short_worker_control() {
    use super::{part, records, reset, set_enabled};
    fn cpu() -> u64 {
        Sample::now().cpu.expect("thread CPU clock")
    }
    fn decode(bytes: &[u8], guarded: bool) -> u64 {
        let begin = cpu();
        for _ in 0..100_000 {
            let _guard = guarded.then(|| part(Part::JsonDecode));
            let value: serde_json::Value =
                serde_json::from_slice(std::hint::black_box(bytes)).unwrap();
            std::hint::black_box(value);
        }
        cpu() - begin
    }
    let bytes = serde_json::to_vec(&serde_json::json!({"type":"assistant", "message":{"content":"synthetic text ".repeat(80)}})).unwrap();
    // Warm the same decoder before either CPU measurement.
    decode(&bytes, false);
    let baseline = decode(&bytes, false);
    reset();
    set_enabled(true);
    fn early_return() {
        let _root = part(Part::Other);
        let _nested = part(Part::ContextGet);
    }
    early_return();
    assert!(super::CURRENT.with(|current| current.borrow().part.is_none()));
    reset();
    let begin = cpu();
    let measured;
    {
        let _root = part(Part::Other);
        measured = decode(&bytes, true);
    }
    let short_cpu = cpu() - begin;
    let parts = records();
    let accounted = parts.values().map(|p| p.cpu_seconds.unwrap()).sum::<f64>();
    reset();
    let begin = cpu();
    {
        let _root = part(Part::Other);
        std::thread::sleep(Duration::from_millis(20));
    }
    let idle_cpu = cpu() - begin;
    let idle = records();
    reset();
    let guard_pairs = 100_000;
    let workers: Vec<_> = (0..4)
        .map(|_| {
            std::thread::spawn(move || {
                let begin = cpu();
                {
                    let _root = part(Part::Other);
                    for _ in 0..guard_pairs {
                        let _guard = part(Part::JsonDecode);
                        std::hint::black_box(1);
                    }
                }
                cpu() - begin
            })
        })
        .collect();
    let guard_cpu: u64 = workers
        .into_iter()
        .map(|worker| worker.join().unwrap())
        .sum();
    let guard_pair_cpu_ns = guard_cpu as f64 / (4.0 * f64::from(guard_pairs));
    set_enabled(false);
    reset();
    println!(
        "PARSER_CPU_CONTROL_JSON={}",
        serde_json::json!({
        "guard_pair_cpu_ns":guard_pair_cpu_ns,"iterations":100000,"record_bytes":bytes.len(),"baseline_decode_cpu_ns":baseline,
        "guarded_decode_cpu_ns":measured,"short_worker_cpu_ns":short_cpu,
        "accounted_cpu_seconds":accounted,"parts":parts,
        "idle_cpu_ns":idle_cpu,"idle_parts":idle,
        "guard_cpu_fraction":(measured as f64-baseline as f64)/measured as f64,
        "note":"CPU control, not a product throughput result; clock costs are included"})
    );
}
