//! Opt-in daemon measurements over a synthetic archive.
use super::*;
use std::time::Instant;

#[test]
#[ignore = "run through the AgentCTL measure operation"]
fn search_measurements() {
    let r = reader();
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
    let provider = r.w.local_embeddings();
    for (name, text) in [
        (
            "live",
            "sessions where -live:ended | sort by live_rank, last_event_at desc",
        ),
        ("text", "sessions where origin:claude-code parser"),
        (
            "facets",
            "sessions where origin:claude-code parser | group by origin | count",
        ),
        (
            "semantic",
            "messages where origin:claude-code near:\"parser\"",
        ),
        (
            "mixed",
            "messages where origin:claude-code parser near:\"parser\"",
        ),
    ] {
        let request = Request::Query {
            text: text.into(),
            cursor: None,
            page_size: Some(50),
            why: true,
            session: None,
            fields: vec![],
        };
        let started = Instant::now();
        assert!(matches!(
            handle(request.clone(), &r.web.daemon),
            Response::Ok { .. }
        ));
        let first_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut samples = Vec::new();
        for _ in 0..30 {
            let started = Instant::now();
            assert!(matches!(
                handle(request.clone(), &r.web.daemon),
                Response::Ok { .. }
            ));
            samples.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        eprintln!(
            "SEARCH_DAEMON={}",
            serde_json::json!({"name":name,"query":text,"page_size":50,"first_ms":first_ms,"samples_ms":samples})
        );
    }
    let provider_ms = provider.join().unwrap();
    eprintln!("SEARCH_SYNTHETIC_PROVIDER_RESPONSE_MS={provider_ms}");
}
