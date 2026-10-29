//! Paired transport measurements over the same synthetic archive. Opt-in;
//! correctness belongs to the ordinary paging and access tests.
use super::*;
use std::io::{Read, Write};
use std::time::Instant;

fn sample(server: &Server, path: &str) -> (f64, usize) {
    let started = Instant::now();
    let mut peer = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    write!(peer,"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n",server.port,server.token).unwrap();
    let mut bytes = Vec::new();
    peer.read_to_end(&mut bytes).unwrap();
    assert!(
        bytes.starts_with(b"HTTP/1.1 200"),
        "{}",
        String::from_utf8_lossy(&bytes)
    );
    (started.elapsed().as_secs_f64() * 1000.0, bytes.len())
}

fn repeated(server: &Server, path: &str) -> serde_json::Value {
    let first = sample(server, path);
    let raw = (0..30).map(|_| sample(server, path)).collect::<Vec<_>>();
    let mut times = raw.iter().map(|(time, _)| *time).collect::<Vec<_>>();
    times.sort_by(f64::total_cmp);
    serde_json::json!({"first_ms":first.0,"bytes":first.1,"warm_median_ms":times[15],"warm_p95_ms":times[28],"samples":raw})
}

#[test]
#[ignore = "paired workbench HTTP measurements; run in the measure queue"]
fn paged_reader_http_measurements() {
    let r = reader();
    let text = "synthetic paragraph λ🙂 <markup>\n".repeat(32);
    let source = (0..1000)
        .map(|i| {
            polylogue_testkit::claude_user(
                &if i == 0 {
                    "u1".into()
                } else {
                    format!("bench-{i}")
                },
                "s1",
                &text,
            )
        })
        .collect::<String>();
    std::fs::write(r.w.home.join(".claude/projects/-p/s1.jsonl"), source).unwrap();
    r.w.wait_for(|data| data["store"]["messages"] == 1000 && data["store"]["search_pending"] == 0);
    let mut server = start_on(Arc::clone(&r.web.daemon), Arc::clone(&r.web.operations), 0).unwrap();
    let report = serde_json::json!({
        "population":{"messages":1000,"characters_per_message":text.chars().count(),"source_text_bytes":text.len()*1000},
        "host_load":std::fs::read_to_string("/proc/loadavg").ok().map(|line|line.split_whitespace().take(3).collect::<Vec<_>>().join(" ")),
        "process_memory":std::fs::read_to_string("/proc/self/status").ok().map(|status|status.lines().filter(|line|line.starts_with("VmRSS:")||line.starts_with("VmHWM:")).collect::<Vec<_>>().join("; ")),
        "budget":8192,"boundary":"authenticated loopback HTTP including socket ownership check; same running daemon and archive",
        "before_whole_read":repeated(&server,"/api/v1/read?session=claude-code:s1&with_parents=true"),
        "after_first_page":repeated(&server,"/api/v1/read?session=claude-code:s1&with_parents=true&budget=8192"),
        "after_middle_page":repeated(&server,"/api/v1/read?session=claude-code:s1&with_parents=true&budget=8192&message=claude-code:bench-500"),
        "after_latest_page":repeated(&server,"/api/v1/read?session=claude-code:s1&with_parents=true&budget=8192&latest=true"),
        "after_html_page":repeated(&server,"/s/claude-code:s1"),
    });
    eprintln!("WORKBENCH_HTTP_MEASUREMENTS={report}");
    server.shutdown().unwrap();
}
