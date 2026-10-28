use super::super::*;
use std::io::BufReader;

#[test]
#[ignore = "real input throughput measurement through AgentCTL measure"]
fn observer_throughput() {
    let path = std::env::var_os("POLYLOGUE_SCHEMA_SAMPLE").expect("sample path");
    for run in 0..3 {
        let file = std::fs::File::open(&path).unwrap();
        let bytes = file.metadata().unwrap().len();
        let start = std::time::Instant::now();
        let mut observer = Observer::default();
        observer.read(BufReader::new(file), Framing::Lines).unwrap();
        let schema = observer.finish("claude-code".into(), vec![]);
        let seconds = start.elapsed().as_secs_f64();
        assert!(schema.records >= 100_000);
        assert_eq!(schema.unreadable_records, 0);
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let peak = status
            .lines()
            .find(|line| line.starts_with("VmHWM:"))
            .unwrap();
        println!(
            "run={run} bytes={bytes} records={} fields={} seconds={seconds:.6} records_per_second={:.1} mib_per_second={:.3} {peak}",
            schema.records,
            schema.fields.len(),
            schema.records as f64 / seconds,
            bytes as f64 / 1048576.0 / seconds
        );
    }
}
