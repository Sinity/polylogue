//! Repeatable synthetic day/cost timings; kept out of correctness checks.
use super::*;
use std::io::{Read, Write};
use std::time::Instant;

fn http(server: &Server, path: &str) -> (f64, usize) {
    let clock = Instant::now();
    let mut peer = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    write!(peer,"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",server.port,server.token).unwrap();
    let mut bytes = Vec::new();
    peer.read_to_end(&mut bytes).unwrap();
    assert!(
        bytes.starts_with(b"HTTP/1.1 200"),
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    (clock.elapsed().as_secs_f64() * 1000., bytes.len())
}
fn samples(mut sample: impl FnMut() -> (f64, usize)) -> serde_json::Value {
    let first = sample();
    let raw = (0..30).map(|_| sample()).collect::<Vec<_>>();
    let mut times = raw.iter().map(|v| v.0).collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    serde_json::json!({"first_ms":first.0,"bytes":first.1,"median_ms":times[15],"p95_ms":times[28],"samples":raw})
}
#[test]
#[ignore = "synthetic day and cost latency samples; use measure operation"]
fn day_and_cost_measurements() {
    let r = reader();
    let source=(0..1000).map(|i| {
        let mut value:serde_json::Value=serde_json::from_str(&polylogue_testkit::claude_assistant(&format!("cost-{i}"),"s1","Synthetic response")).unwrap();
        value["message"]["model"]="claude-opus-4-1".into();
        value["message"]["usage"]=serde_json::json!({"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":0,"cache_creation_input_tokens":0});
        format!("{value}\n")
    }).collect::<String>();
    std::fs::OpenOptions::new()
        .append(true)
        .open(r.w.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    r.w.wait_for(|s| s["store"]["messages"] == 1001 && s["store"]["search_pending"] == 0);
    let mut server = start_on(Arc::clone(&r.web.daemon), 0).unwrap();
    let baseline = || Request::Query {
        text: "sessions | measure usage".into(),
        cursor: None,
        page_size: None,
        why: false,
        session: None,
        fields: vec![],
    };
    let cost = || Request::Cost {
        month: None,
        query: "sessions".into(),
        date: None,
        model: None,
        undated: false,
        unknown_model: false,
        cursor: None,
        page_size: None,
    };
    let day = || Request::Timeline {
        date: Some("2026-03-01".into()),
        across_years: false,
        undated: false,
        cursor: None,
        page_size: None,
    };
    let daemon = |request| {
        let start = Instant::now();
        let answer = handle(request, &r.web.daemon);
        assert!(matches!(answer, Response::Ok { .. }), "{answer:?}");
        let elapsed = start.elapsed().as_secs_f64() * 1000.;
        (elapsed, serde_json::to_vec(&answer).unwrap().len())
    };
    let report = serde_json::json!({"population":{"sessions":1,"known_response_records":1000,"messages":1001,"page_size":50},"host_load":std::fs::read_to_string("/proc/loadavg").ok(),"memory":std::fs::read_to_string("/proc/self/status").ok().map(|s|s.lines().filter(|l|l.starts_with("VmRSS:")||l.starts_with("VmHWM:")).collect::<Vec<_>>().join("; ")),"ingestion":"caught up before samples; watcher remains running","baseline_daemon_usage":samples(||daemon(baseline())),"after_daemon_cost":samples(||daemon(cost())),"after_daemon_day":samples(||daemon(day())),"baseline_http_usage":samples(||http(&server,"/api/v1/query?query=sessions%20%7C%20measure%20usage")),"after_http_cost":samples(||http(&server,"/api/v1/cost?query=sessions")),"after_http_day":samples(||http(&server,"/api/v1/timeline?date=2026-03-01")),"after_html_cost":samples(||http(&server,"/cost")),"after_html_day":samples(||http(&server,"/day/2026-03-01"))});
    let path = std::env::var_os("POLYLOGUE_DAY_COST_REPORT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/day-cost-measurements.json")
        });
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    eprintln!("DAY_COST_MEASUREMENTS={report}");
    server.shutdown().unwrap();
}
