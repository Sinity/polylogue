//! Opt-in HTTP and browser measurements over a synthetic real daemon.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "run in the measure queue with POLYLOGUE_WORKBENCH_DRIVER"]
fn workbench_measurements() {
    let mut r = reader();
    let now = polylogue_model::Timestamp::from_unix_ms(crate::daemon::service::now_ms())
        .unwrap()
        .to_string();
    let current = |id: &str, session: &str| {
        polylogue_testkit::claude_user(id, session, "synthetic parser prompt λ")
            .replace("2026-03-01T10:00:00Z", &now)
    };
    std::fs::write(
        r.w.home.join(".claude/projects/-p/s1.jsonl"),
        current("u1", "s1"),
    )
    .unwrap();
    for n in 0..120 {
        std::fs::write(
            r.w.home
                .join(format!(".claude/projects/-p/extra-{n}.jsonl")),
            current(&format!("u-{n}"), &format!("extra-{n}")),
        )
        .unwrap();
    }
    r.w.wait_for(|s| s["store"]["sessions"] == 121 && s["store"]["search_pending"] == 0);
    let mut server = start_on(Arc::clone(&r.web.daemon), 0).unwrap();
    let output = std::process::Command::new("python3")
        .arg(
            std::env::var("POLYLOGUE_WORKBENCH_DRIVER")
                .expect("set POLYLOGUE_WORKBENCH_DRIVER to the browser measurement driver"),
        )
        .arg(format!(
            "http://127.0.0.1:{}/auth#{}",
            server.port, server.token
        ))
        .arg(&r.w.home)
        .output()
        .unwrap();
    eprintln!(
        "WORKBENCH_BROWSER={}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut samples = Vec::new();
    for _ in 0..30 {
        let started = Instant::now();
        assert!(matches!(
            handle(
                Request::Query {
                    text: "sessions where origin:claude-code parser".into(),
                    cursor: None,
                    page_size: Some(50),
                    why: true,
                    session: None,
                    fields: vec![]
                },
                &r.web.daemon
            ),
            Response::Ok { .. }
        ));
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    eprintln!("WORKBENCH_DAEMON_SAMPLES={samples:?}");
    server.shutdown().unwrap();
    for _ in 0..5 {
        let started = Instant::now();
        r.w.restart();
        let startup = started.elapsed().as_secs_f64() * 1000.0;
        let mut server = start_on(Arc::clone(&r.w.daemon), 0).unwrap();
        let output = std::process::Command::new("python3")
            .arg(std::env::var("POLYLOGUE_WORKBENCH_DRIVER").unwrap())
            .arg(format!(
                "http://127.0.0.1:{}/auth#{}",
                server.port, server.token
            ))
            .arg(&r.w.home)
            .arg("restart")
            .output()
            .unwrap();
        eprintln!(
            "WORKBENCH_RESTART startup_ms={startup} {}",
            String::from_utf8_lossy(&output.stdout)
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        server.shutdown().unwrap();
    }
}
