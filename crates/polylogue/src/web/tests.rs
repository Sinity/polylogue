//! The HTTP port over a real daemon on a temporary archive and home:
//! access rules, the reader's pages and its JSON API.

use super::*;
use crate::test_daemon::{self, TestDaemon};
use axum::body::{Body, to_bytes};
use axum::http::Request as Http;
use tower::ServiceExt;

const PORT: u16 = 8765;
const TOKEN: &str = "t0ken";

struct Reader {
    w: TestDaemon,
    web: Arc<Web>,
}

fn reader() -> Reader {
    let w = test_daemon::start("look at <script>alert(1)</script> & the parser");
    let web = Arc::new(Web {
        daemon: Arc::clone(&w.daemon),
        operations: Arc::clone(w.operations()),
        key: "abc123def456".into(),
        token: TOKEN.into(),
        port: PORT,
    });
    Reader { w, web }
}

fn get(path: &str) -> axum::http::request::Builder {
    Http::get(path).header(header::HOST, format!("127.0.0.1:{PORT}"))
}

fn post(path: &str) -> axum::http::request::Builder {
    Http::post(path)
        .header(header::HOST, format!("127.0.0.1:{PORT}"))
        .header(
            header::CONTENT_TYPE,
            if path == "/auth" {
                "application/x-www-form-urlencoded"
            } else {
                "application/json"
            },
        )
}

fn authed(b: axum::http::request::Builder) -> axum::http::request::Builder {
    b.header(header::COOKIE, format!("polylogue_abc123def456={TOKEN}"))
}

fn setting_envelope(id: &str) -> String {
    serde_json::to_string(&command_surface::Request {
        operation: "setting".into(),
        arguments: serde_json::json!({"key":"reading.variants", "value":"[]"}),
        request_id: id.into(),
        deadline: command_surface::Deadline::Default,
        budget_tokens: Default::default(),
        cursor: None,
        fresh: false,
        format: Default::default(),
    })
    .unwrap()
}

fn assert_copy_command(html: &str, command: &str) {
    assert!(html.contains("data-copy-forms="), "{html}");
    assert!(
        html.contains(&format!(
            "<code id=\"page-command\">{}</code>",
            polylogue_render::html::escape(command)
        )),
        "{html}"
    );
}

fn own_origin(b: axum::http::request::Builder) -> axum::http::request::Builder {
    b.header(header::ORIGIN, format!("http://127.0.0.1:{PORT}"))
}

async fn send(r: &Reader, req: Http<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = router(Arc::clone(&r.web)).oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    let body = String::from_utf8_lossy(&body).into_owned();
    (status, headers, application_body(body))
}

/// Domain assertions use the payload; protocol tests inspect the wire envelope.
pub(super) fn application_body(body: String) -> String {
    let Ok(answer) = serde_json::from_str::<command_surface::Answer<serde_json::Value>>(&body)
    else {
        return body;
    };
    match answer {
        command_surface::Answer { outcome: command_surface::Outcome::Error(error), .. } => serde_json::json!({
            "error": error.code, "message": error.message,
            "detail": {"position": error.position, "suggestion": error.suggestion(), "hint": error.hint, "retryable": error.retryable},
        }).to_string(),
        answer => answer.value().unwrap().to_string(),
    }
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

#[test]
fn access_rules() {
    let r = reader();
    rt().block_on(async {
        // DNS rebinding: a foreign Host is refused before anything else.
        let req = Http::get("/")
            .header(header::HOST, "evil.example:8765")
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::MISDIRECTED_REQUEST);

        // No token: a page sends the browser to sign in, the API refuses.
        let (s, h, _) = send(&r, get("/").body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::SEE_OTHER);
        assert_eq!(h[header::LOCATION], "/auth");
        let req = get("/api/v1/status").body(Body::empty()).unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::UNAUTHORIZED);
        let req = get("/api/v1/status")
            .header(header::AUTHORIZATION, "Bearer nope")
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::UNAUTHORIZED);
        // The extension's routes are not served here.
        let req = authed(get("/v1/status")).body(Body::empty()).unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::NOT_FOUND);

        // The cookie and the bearer header both work.
        let (s, h, _) = send(&r, authed(get("/")).body(Body::empty()).unwrap()).await;
        assert_eq!(s, StatusCode::OK);
        let csp = h[header::CONTENT_SECURITY_POLICY].to_str().unwrap();
        assert!(csp.contains("script-src 'self'"), "{csp}");
        assert!(h.get(header::ACCESS_CONTROL_ALLOW_ORIGIN).is_none());
        let req = get("/api/v1/status")
            .header(header::AUTHORIZATION, format!("Bearer {TOKEN}"))
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::OK);

        // A write from another origin is refused even with the cookie.
        let req = authed(post("/api/v1/setting"))
            .header(header::ORIGIN, "https://evil.example")
            .body(Body::from(setting_envelope("access-setting")))
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::FORBIDDEN);
        let req = own_origin(authed(post("/api/v1/setting")))
            .body(Body::from(setting_envelope("access-setting")))
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::OK);

        // Excision is not offered over HTTP.
        let req = own_origin(authed(post("/api/v1/excise")))
            .body(Body::from(
                r#"{"target":"session:claude-code:s1","yes":true}"#,
            ))
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::NOT_FOUND);

        // Sign-in: the token buys the archive's cookie.
        let req = own_origin(post("/auth"))
            .body(Body::from(format!("token={TOKEN}")))
            .unwrap();
        let (s, h, _) = send(&r, req).await;
        assert_eq!(s, StatusCode::NO_CONTENT);
        let cookie = h[header::SET_COOKIE].to_str().unwrap();
        assert!(cookie.starts_with("polylogue_abc123def456="), "{cookie}");
        assert!(cookie.contains("SameSite=Strict") && cookie.contains("HttpOnly"));
        let req = own_origin(post("/auth"))
            .body(Body::from("token=wrong"))
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::UNAUTHORIZED);
    });
}

#[test]
fn pages_and_api_never_carry_the_archive_path() {
    let r = reader();
    let root = r.web.daemon.archive.root.to_string_lossy().into_owned();
    rt().block_on(async {
        let req = authed(get("/api/v1/status")).body(Body::empty()).unwrap();
        let (s, _, body) = send(&r, req).await;
        assert_eq!(s, StatusCode::OK);
        let v: serde_json::Value = serde_json::from_str(&body).unwrap();
        assert!(v.get("archive").is_none(), "{body}");
        assert!(!body.contains(&root), "{body}");
        assert_eq!(v["store"]["messages"], 1);
        let (_, _, page) = send(&r, authed(get("/")).body(Body::empty()).unwrap()).await;
        assert!(!page.contains(&root));
        assert!(page.contains("/s/claude-code%3As1"), "{page}");
    });
}

#[test]
fn the_session_page_escapes_archive_text_and_matches_the_socket() {
    use polylogue_model::read::ReadPage;
    let mut r = reader();
    // Signal commits advance the archive independently of this read comparison.
    r.w.pause_signals();
    let wait_for_runs = |expected| {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let completed =
                r.w.daemon
                    .reads
                    .reader()
                    .unwrap()
                    .query(
                        "SELECT count(*) AS n FROM source_query_run WHERE completion IS NOT NULL",
                        &[],
                    )
                    .unwrap()[0]["n"]
                    .as_u64()
                    .unwrap();
            if completed == expected {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "expected {expected} completed read runs, found {completed}"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
    };
    rt().block_on(async {
        let (status, _, page) = send_html(&r, "/s/claude-code%3As1").await;
        assert_eq!(status, StatusCode::OK, "{page}");
        assert!(
            page.contains("&lt;script&gt;alert(1)&lt;/script&gt; &amp; the parser"),
            "{page}"
        );
        assert!(!page.contains("<script>alert"));
        assert!(page.contains("polylogue read --request"));
        assert!(page.contains("/assets/shell.js"));
        // A first page records its run after answering. Finish the previous
        // reading before comparing the next reading's exact snapshot position.
        wait_for_runs(1);
        let api_read_at = {
            r.w.daemon
                .reads
                .reader()
                .unwrap()
                .change_position()
                .unwrap()
        };
        let api = read_page(&r, "session:claude-code:s1", &[]).await;
        assert_eq!(api.snapshot.read_at, api_read_at);
        wait_for_runs(2);
        let socket_read_at = {
            r.w.daemon
                .reads
                .reader()
                .unwrap()
                .change_position()
                .unwrap()
        };
        let Ok(data) = crate::protocol::into_value(
            crate::test_daemon::Call::new(
                "read",
                serde_json::json!({"target": "session:claude-code:s1", "format": "json"}),
            )
            .answer(&r.web.daemon),
        ) else {
            panic!("socket read failed")
        };
        let socket: ReadPage = serde_json::from_value(data).unwrap();
        assert_eq!(api.document.items, socket.document.items);
        assert_eq!(api.document.header, socket.document.header);
        assert_eq!(socket.snapshot.read_at, socket_read_at);
        let (status, _, page) = send_html(&r, "/s/claude-code%3Anope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(page.contains("not_found"));
    });
}

#[test]
fn search_renders_the_cli_answer_with_session_links() {
    let r = reader();
    rt().block_on(async {
        let req = authed(get("/search?q=parser")).body(Body::empty()).unwrap();
        let (s, _, page) = send(&r, req).await;
        assert_eq!(s, StatusCode::OK, "{page}");
        assert!(page.contains("<mark>"), "{page}");
        assert_copy_command(&page, "polylogue list parser");
        // A query error shows its message instead of failing the page.
        let req = authed(get("/search?q=repo%3A%28"))
            .body(Body::empty())
            .unwrap();
        let (s, _, page) = send(&r, req).await;
        assert_eq!(s, StatusCode::OK);
        assert!(page.contains("role=\"alert\""), "{page}");
    });
}

fn chat_send(body: &serde_json::Value) -> axum::http::Request<Body> {
    own_origin(authed(post("/api/v1/chat/send")))
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[test]
fn the_operator_queues_chat_actions_only_while_switched_on() {
    let r = reader();
    let body = serde_json::json!({
        "request_id": "draft-1",
        "provider": "chatgpt",
        "text": "summarize this",
        "model": "GPT X",
    });
    rt().block_on(async {
        let (s, _, refused) = send(&r, chat_send(&body)).await;
        assert_eq!(s, StatusCode::CONFLICT, "{refused}");
        assert!(refused.contains("browser_actions_off"));
        r.w.set("browser.actions", "on");
        let (s, _, created) = send(&r, chat_send(&body)).await;
        assert_eq!(s, StatusCode::OK, "{created}");
        let v: serde_json::Value = serde_json::from_str(&created).unwrap();
        assert_eq!(v["action"]["status"], "queued");
        assert_eq!(v["action"]["submit"], false);
        // The same request again, as after a lost reply, is the same action.
        let (s, _, again) = send(&r, chat_send(&body)).await;
        assert_eq!(s, StatusCode::OK);
        let again: serde_json::Value = serde_json::from_str(&again).unwrap();
        assert_eq!(again["action"]["id"], v["action"]["id"]);
        // The same request ID with other content is refused.
        let mut changed = body.clone();
        changed["text"] = "Different draft".into();
        let (s, _, raw) = send(&r, chat_send(&changed)).await;
        assert_eq!(s, StatusCode::CONFLICT);
        assert!(raw.contains("idempotency_key_reused"));
    });
    assert_eq!(
        r.w.daemon
            .reads
            .reader()
            .unwrap()
            .chat_actions()
            .unwrap()
            .len(),
        1
    );
}

#[test]
fn chat_send_needs_the_origin_and_token() {
    let r = reader();
    rt().block_on(async {
        let path = "/api/v1/chat/send";
        assert_eq!(
            send(&r, own_origin(post(path)).body(Body::from("{}")).unwrap())
                .await
                .0,
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            send(&r, authed(post(path)).body(Body::from("{}")).unwrap())
                .await
                .0,
            StatusCode::FORBIDDEN
        );
    });
}

#[test]
fn an_unavailable_browser_permission_is_not_reported_as_disabled() {
    let r = reader();
    let archive = rusqlite::Connection::open(r.w.daemon.archive.archive_db()).unwrap();
    archive
        .execute_batch("ALTER TABLE assertion RENAME TO unavailable_assertion")
        .unwrap();
    let body = serde_json::json!({
        "provider": "chatgpt",
        "text": "Neutral synthetic draft",
        "model": "X",
    });
    let response = rt().block_on(async { send(&r, chat_send(&body)).await });
    archive
        .execute_batch("ALTER TABLE unavailable_assertion RENAME TO assertion")
        .unwrap();
    assert_eq!(
        response.0,
        StatusCode::INTERNAL_SERVER_ERROR,
        "{}",
        response.2
    );
    let body: serde_json::Value = serde_json::from_str(&response.2).unwrap();
    assert_eq!(body["error"], "store");
}

#[test]
fn a_chat_send_answer_preserves_exact_draft_credentials_as_data() {
    let r = reader();
    r.w.set("browser.actions", "on");
    let text = format!("Keep this draft {}", "AKIAIOSFODNN7EXAMPLE");
    let body = serde_json::json!({"provider":"chatgpt", "text":text, "model":"X"});
    rt().block_on(async {
        let (status, _, shown) = send(&r, chat_send(&body)).await;
        assert_eq!(status, StatusCode::OK);
        let shown: serde_json::Value = serde_json::from_str(&shown).unwrap();
        assert_eq!(shown["action"]["text"], text);
    });
    let queued = r.w.daemon.reads.reader().unwrap().chat_actions().unwrap();
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].request["text"], text.as_str());
}

#[test]
fn backfill_jobs_start_and_list_over_http() {
    let r = reader();
    rt().block_on(async {
        let start = own_origin(authed(post("/api/v1/backfill/start")))
            .body(Body::from(r#"{"provider":"claude-ai"}"#))
            .unwrap();
        let (s, _, started) = send(&r, start).await;
        assert_eq!(s, StatusCode::OK, "{started}");
        let started: serde_json::Value = serde_json::from_str(&started).unwrap();
        assert_eq!(started["job"]["status"], "waiting");
        let (s, _, listed) = send(
            &r,
            authed(get("/api/v1/backfill/list"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(s, StatusCode::OK, "{listed}");
        let listed: serde_json::Value = serde_json::from_str(&listed).unwrap();
        assert_eq!(listed["jobs"][0]["id"], started["job"]["id"]);
    });
}

#[test]
fn every_http_command_is_served_and_nothing_else() {
    use crate::commands::{COMMANDS, Kind};
    let r = reader();
    rt().block_on(async {
        for c in COMMANDS.iter() {
            let path = format!("/api/v1{}", c.http_path());
            let mut object = serde_json::Map::new();
            for a in c.args.iter().filter(|a| a.required && a.default.is_none()) {
                let value = match a.kind {
                    Kind::List => serde_json::json!(["session:claude-code:s1"]),
                    _ => serde_json::json!("claude-code:s1"),
                };
                object.insert(a.name.into(), value);
            }
            let req = if c.writes {
                own_origin(authed(post(&path)))
                    .body(Body::from(serde_json::Value::Object(object).to_string()))
                    .unwrap()
            } else {
                let query: Vec<String> = object
                    .iter()
                    .map(|(k, v)| {
                        let v = v
                            .as_str()
                            .map_or_else(|| v[0].as_str().unwrap().to_owned(), str::to_owned);
                        format!("{k}={}", v.replace(':', "%3A"))
                    })
                    .collect();
                authed(get(&format!("{path}?{}", query.join("&"))))
                    .body(Body::empty())
                    .unwrap()
            };
            let (status, _, body) = send(&r, req).await;
            if c.http {
                // A 404 is an answer about an unknown ID, not a missing route.
                assert!(
                    status != StatusCode::NOT_FOUND
                        || body.contains("not_found")
                        || body.contains("\"error\":\"unknown_"),
                    "{path}: {status} {body}"
                );
                assert_ne!(status, StatusCode::METHOD_NOT_ALLOWED, "{path}");
                assert!(
                    !body.contains("has no argument") && !body.contains("needs `"),
                    "{path}: {body}"
                );
            } else {
                assert_eq!(status, StatusCode::NOT_FOUND, "{path} is served");
            }
        }
        // The routes before the table are gone.
        let req = authed(get("/api/v1/cmd/status"))
            .body(Body::empty())
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::NOT_FOUND);
    });
}

#[test]
fn owned_http_shutdown_closes_partial_peers_and_releases_daemon() {
    use std::io::Write;
    let daemon = test_daemon::start("HTTP lifecycle fixture");
    let weak = Arc::downgrade(&daemon.daemon);
    let mut server = start_on(
        Arc::clone(&daemon.daemon),
        Arc::clone(daemon.operations()),
        0,
    )
    .unwrap();
    let _idle = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    let mut partial = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    partial
        .write_all(b"GET /api/v1/status HTTP/1.1\r\n")
        .unwrap();
    let (done, stopped) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::spawn(move || {
        done.send(server.shutdown()).unwrap();
    });
    stopped
        .recv()
        .expect("HTTP worker kept idle clients open")
        .unwrap();
    thread.join().unwrap();
    drop(daemon);
    assert!(weak.upgrade().is_none());
}

#[test]
fn application_shutdown_joins_http_handlers_while_all_reader_slots_remain_held() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(std::path::PathBuf::from("/synthetic/home"));
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let mut runtime = crate::test_daemon::start_runtime(
        &dir.path().join("archive"),
        config,
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    let mut server = start_on(Arc::clone(&daemon), Arc::clone(&runtime.operations), 0).unwrap();
    let mut peer = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    write!(
        peer,
        "GET /api/v1/status HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\n\r\n",
        server.port, server.token
    )
    .unwrap();
    runtime.stop_accepting();
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
}

#[test]
fn shutdown_cancels_a_started_http_read_before_joining_its_transport() {
    use std::io::Write;
    let dir = tempfile::tempdir().unwrap();
    let mut config = crate::test_daemon::engine_config(dir.path().to_owned());
    config.tool_dirs.clear();
    config.workers = 1;
    let mut runtime = crate::test_daemon::start_runtime(
        &dir.path().join("archive"),
        config,
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    let mut server = start_on(
        Arc::clone(&runtime.daemon),
        Arc::clone(&runtime.operations),
        0,
    )
    .unwrap();
    let (observed, reads) = std::sync::mpsc::channel();
    *runtime.daemon.read_limits.lock().unwrap() = Some(observed);
    let mut peer = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    let sql = "WITH%20RECURSIVE%20n(x)%20AS%20(SELECT%201%20UNION%20ALL%20SELECT%20x%2B1%20FROM%20n%20WHERE%20x%3C2000000000)%20SELECT%20count(*)%20FROM%20n";
    write!(peer, "GET /api/v1/sql?statement={sql} HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\n\r\n", server.port, server.token).unwrap();
    let read = reads
        .recv_timeout(std::time::Duration::from_secs(5))
        .unwrap();
    crate::test_daemon::TestDaemon::wait_for_sql(&read);
    runtime.stop_accepting();
    assert!(read.cancellation.is_cancelled());
    server.shutdown().unwrap();
    runtime.shutdown().unwrap();
    assert_eq!(runtime.daemon.reads.checked_out(), 0);
}

#[test]
fn session_html_and_json_read_routes_preserve_stored_titles_and_message_content() {
    use std::io::Write;
    let r = reader();
    let literal = "AKIAIOSFODNN7EXAMPLE";
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(r.w.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap();
    writeln!(file,"{}",serde_json::json!({"type":"custom-title","sessionId":"s1","customTitle":format!("Synthetic title {literal}")})).unwrap();
    write!(
        file,
        "{}",
        polylogue_testkit::claude_user("u2", "s1", literal)
    )
    .unwrap();
    drop(file);
    r.w.wait_for(|status| {
        status["store"]["messages"] == 2 && status["store"]["search_pending"] == 0
    });
    rt().block_on(async {
        let (status, _, html) = send_html(&r, "/s/claude-code:s1").await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains(literal), "{html}");
        assert!(html.contains("Synthetic title "), "{html}");
        let page = read_page(&r, "session:claude-code:s1", &[]).await;
        let payload = serde_json::to_string(&page.document).unwrap();
        assert!(payload.contains(literal));
        assert!(payload.contains("Synthetic title "));
        let invalid = read_url(
            "session:claude-code:s1",
            &[("view", "unsupported-synthetic-view")],
        );
        let (status, _, body) = send(&r, authed(get(&invalid)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
    });
}

#[test]
fn cost_uses_public_accounting_and_session_scope() {
    let r = reader();
    rt().block_on(async {
        let (status, _, _) = send(&r, get("/cost").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        for (path, scope) in [
            ("/cost", "sessions"),
            ("/cost?q=id%3Aclaude-code%3As1", "id:claude-code:s1"),
        ] {
            let (status, headers, html) =
                send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{html}");
            assert_eq!(headers[header::CACHE_CONTROL], "no-store");
            let Ok(data) = crate::protocol::into_value(crate::test_daemon::Call::new("query", serde_json::json!({"query": format!("{scope} | measure usage")})).answer(&r.web.daemon)) else {
                panic!("usage failed")
            };
            let report =
                polylogue_render::html::escape(&polylogue_render::measures::usage(&data["data"]));
            assert!(html.contains(&report), "{html}");
            assert!(html.contains("API-equivalent unknown"), "{html}");
            assert!(html.contains("Browse selected sessions"));
            assert_copy_command(
                &html,
                &format!(
                    "polylogue measure usage {}",
                    polylogue_render::shell_quote(scope)
                ),
            );
        }
        let (status, _, html) = send(
            &r,
            authed(get("/cost?q=messages")).body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{html}");
        // Accounting is checked against the exact scoped cost report above;
        // the typed reading declares lineage rather than a retired cost panel.
        use polylogue_model::read::{Card, ItemKind, ValueState};
        let items = read_items(&r, "session:claude-code:s1", &[("show", "inherited")]).await;
        assert!(items.iter().any(|item| matches!(&item.kind,
            ItemKind::Card { card } if matches!(card.as_ref(), Card::Session { session, lineage_complete, .. }
                if session.id.to_string()=="claude-code:s1" && lineage_complete==&ValueState::known(true)))));
        assert!(!items.iter().any(|item| matches!(item.kind, ItemKind::LinkIssue { .. })));
    });
}

#[test]
fn session_parent_links_are_scoped_and_missing_parent_stays_incomplete() {
    let r = reader();
    for (child, parent) in [
        ("child", "s1"),
        ("orphan", "missing"),
        ("other", "unrelated"),
    ] {
        let mut record: serde_json::Value = serde_json::from_str(&polylogue_testkit::claude_user(
            &format!("u-{child}"),
            child,
            "synthetic child",
        ))
        .unwrap();
        record["forkedFrom"] = serde_json::json!({"sessionId": parent, "messageUuid": "u1"});
        std::fs::write(
            r.w.home.join(format!(".claude/projects/-p/{child}.jsonl")),
            format!("{record}\n"),
        )
        .unwrap();
    }
    r.w.wait_for(|status| status["store"]["messages"] == 4);
    rt().block_on(async {
        use polylogue_model::read::{Card, ItemKind, ParentIssue, ReadRecord, ValueState};
        let (status, _, html) = send_html(&r, "/s/claude-code:child?view=links").await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("href=\"/s/claude-code%3As1\""), "{html}");
        assert!(!html.contains("claude-code:other"), "{html}");
        let scoped=read_items(&r,"session:claude-code:child",&[("view","links")]).await;
        let edges:Vec<_>=scoped.iter().filter_map(|item| match &item.kind {
            ItemKind::Card {card} => match card.as_ref() { Card::LinkEdge {from,to,relation,how,..} => Some((from.as_str(),to.as_str(),relation.as_str(),how.as_deref())), _=>None }, _=>None,
        }).collect();
        assert_eq!(edges,[("session:claude-code:s1","session:claude-code:child","fork",None)]);
        // Native marker and anchor are canonical Link facts, not invented HTML text.
        let path=read_url("session:claude-code:child",&[("view","links"),("delivery","stream")]);
        let (status,_,body)=send(&r,authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status,StatusCode::OK,"{body}");
        let frames=command_surface::SseDecoder::new().push(body.as_bytes()).unwrap();
        assert_eq!(frames.first().unwrap().event,"begin");
        assert_eq!(frames.last().unwrap().event,"complete");
        let links:Vec<_>=frames.iter().filter(|frame|frame.event=="record").filter_map(|frame| {
            match serde_json::from_str::<ReadRecord>(&frame.data).unwrap() {
                ReadRecord::Canonical {record,..} => match *record {polylogue_model::bundle::Record::Link(row)=>Some(row.link),_=>None}, _=>None,
            }
        }).collect();
        assert_eq!(links.len(),1);
        assert_eq!(links[0].parent.to_string(),"claude-code:s1");
        assert_eq!(links[0].child.to_string(),"claude-code:child");
        assert_eq!(links[0].kind,polylogue_model::LinkKind::Fork);
        assert_eq!(links[0].at.as_ref().unwrap().to_string(),"claude-code:u1");
        assert_eq!(links[0].evidence,"forkedFrom");
        let (status, _, html) = send_html(&r, "/s/claude-code:orphan?view=links").await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("href=\"/s/claude-code%3Amissing\""), "{html}");
        let items=read_items(&r,"session:claude-code:orphan",&[("show","inherited")]).await;
        let issues:Vec<_>=items.iter().filter_map(|item|match &item.kind {ItemKind::LinkIssue {issue}=>Some(issue.clone()),_=>None}).collect();
        assert_eq!(issues,[ParentIssue::MissingParent {child:polylogue_model::SessionId::parse("claude-code:orphan").unwrap(),parent:polylogue_model::SessionId::parse("claude-code:missing").unwrap()}]);
        assert_eq!(serde_json::to_value(&issues[0]).unwrap()["kind"],"missing_parent");
        assert!(items.iter().any(|item|matches!(&item.kind,ItemKind::Card {card} if matches!(card.as_ref(),Card::Session {lineage_complete,..} if lineage_complete==&ValueState::known(false)))));
        let (status,_,html)=send_html(&r,"/s/claude-code:orphan?show=inherited").await;
        assert_eq!(status,StatusCode::OK,"{html}");
        assert!(html.contains("Parents incomplete"),"{html}");

    });
}

#[test]
fn cost_is_served_through_the_owned_loopback_http_listener() {
    use std::io::{Read, Write};
    let r = reader();
    let mut server = start_on(Arc::clone(&r.web.daemon), Arc::clone(&r.web.operations), 0).unwrap();
    let mut peer = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    write!(peer, "GET /cost?q=id%3Aclaude-code%3As1 HTTP/1.1\r\nHost: 127.0.0.1:{}\r\nAuthorization: Bearer {}\r\nConnection: close\r\n\r\n", server.port, server.token).unwrap();
    let mut response = String::new();
    peer.read_to_string(&mut response).unwrap();
    assert!(response.starts_with("HTTP/1.1 200"), "{response}");
    assert!(response.contains("0 known response records"), "{response}");
    assert!(response.contains("API-equivalent unknown"), "{response}");
    assert!(response.contains("Browse selected sessions"), "{response}");
    server.shutdown().unwrap();
}

#[test]
fn priced_native_usage_and_drilldown_keep_the_selected_session_population() {
    use std::io::Write;
    let r = reader();
    let assistant = |session: &str, input: u64| {
        let mut record: serde_json::Value =
            serde_json::from_str(&polylogue_testkit::claude_assistant(
                &format!("a-{session}"),
                session,
                "Synthetic priced response",
            ))
            .unwrap();
        record["message"]["model"] = "claude-opus-4-1".into();
        record["message"]["usage"] = serde_json::json!({
            "input_tokens": input, "output_tokens": 0,
            "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
        });
        format!("{record}\n")
    };
    std::fs::OpenOptions::new()
        .append(true)
        .open(r.w.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap()
        .write_all(assistant("s1", 1_000_000).as_bytes())
        .unwrap();
    std::fs::write(
        r.w.home.join(".claude/projects/-p/other.jsonl"),
        assistant("other", 2_000_000),
    )
    .unwrap();
    r.w.wait_for(|status| status["store"]["messages"] == 3);
    rt().block_on(async {
        let (status, _, page) = send(
            &r,
            authed(get("/cost?q=id%3Aclaude-code%3As1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{page}");
        let (status, _, api) = send(
            &r,
            authed(get(
                "/api/v1/query?query=id%3Aclaude-code%3As1%20%7C%20measure%20usage",
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{api}");
        let api: serde_json::Value = serde_json::from_str(&api).unwrap();
        let usage = &api["data"];
        assert_eq!(usage["sessions"], 1);
        assert_eq!(usage["messages"], 1);
        assert_eq!(usage["tokens"]["input"]["tokens"], 1_000_000);
        assert_eq!(usage["api_cost"]["known"]["nusd"], 15_000_000_000_i64);
        assert_eq!(usage["api_cost"]["priced_messages"], 1);
        assert_eq!(usage["api_cost"]["unpriced_messages"], 0);
        assert_eq!(usage["by_model"][0]["model"], "claude-opus-4-1");
        assert_eq!(usage["by_model"][0]["messages"], 1);
        assert_eq!(
            usage["by_model"][0]["api_cost"]["known"]["nusd"],
            15_000_000_000_i64
        );
        assert!(page.contains("API-equivalent $15.00"), "{page}");
        assert!(page.contains("1 known response records"), "{page}");
        assert!(
            page.contains(&polylogue_render::html::escape(
                &polylogue_render::measures::usage(usage)
            )),
            "{page}"
        );
        assert!(
            page.contains("href=\"/search?q=id%3Aclaude-code%3As1\">Browse selected sessions"),
            "{page}"
        );
        let (status, _, selected) = send(
            &r,
            authed(get("/search?q=id%3Aclaude-code%3As1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{selected}");
        assert!(
            selected.contains("href=\"/s/claude-code%3As1\""),
            "{selected}"
        );
        assert!(!selected.contains("claude-code:other"), "{selected}");
    });
}

mod documents;

mod attachments;

mod pages;

mod notes;
mod search_api;
mod timeline;

mod cancellation;

fn read_url(target: &str, extra: &[(&str, &str)]) -> String {
    let encode = polylogue_render::html::encode_path_segment;
    let mut path = format!("/api/v1/read?target={}", encode(target));
    for (key, value) in extra {
        path.push('&');
        path.push_str(&encode(key));
        path.push('=');
        path.push_str(&encode(value));
    }
    path
}
async fn send_html(r: &Reader, path: &str) -> (StatusCode, HeaderMap, String) {
    let mut path = path.to_owned();
    for _ in 0..4 {
        let reply = send(
            r,
            authed(get(&path))
                .header(header::ACCEPT, "text/html")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        if reply.0 == StatusCode::TEMPORARY_REDIRECT || reply.0 == StatusCode::SEE_OTHER {
            path = reply.1[header::LOCATION].to_str().unwrap().to_owned();
            assert!(path.starts_with('/') && !path.starts_with("//"));
        } else {
            return reply;
        }
    }
    panic!("object URL redirect loop")
}
async fn read_page(
    r: &Reader,
    target: &str,
    extra: &[(&str, &str)],
) -> polylogue_model::read::ReadPage {
    let path = read_url(target, extra);
    let (status, _, body) = send(r, authed(get(&path)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).unwrap()
}
async fn follow_read(
    r: &Reader,
    call: &polylogue_model::read::ReadyCall,
) -> polylogue_model::read::ReadPage {
    let path =
        polylogue_render::web::http_url(crate::commands::declarations(), &call.request).unwrap();
    let (status, _, body) = send(r, authed(get(&path)).body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    serde_json::from_str(&body).unwrap()
}

async fn read_items(
    r: &Reader,
    target: &str,
    extra: &[(&str, &str)],
) -> Vec<polylogue_model::read::ReadItem> {
    use polylogue_model::{
        BlobHash,
        read::{ItemKind, ReadItem, SelectionLocator},
    };
    let mut partial =
        std::collections::BTreeMap::<(SelectionLocator, u64, BlobHash), (String, u64)>::new();
    let mut values = Vec::new();
    let mut calls = std::collections::BTreeSet::new();
    let mut page = read_page(r, target, extra).await;
    loop {
        for item in &page.document.items {
            if let ItemKind::Fragment {
                text,
                position,
                until,
                total,
            } = &item.kind
            {
                let key = (position.selection, position.item, position.item_hash);
                let (encoded, characters) = partial.entry(key).or_default();
                assert_eq!(
                    position.character, *characters,
                    "fragment gap or duplicate prefix"
                );
                assert!(*until > *characters && *until <= *total);
                assert_eq!(*until - *characters, text.chars().count() as u64);
                encoded.push_str(text);
                *characters = *until;
                if until == total {
                    let (encoded, _) = partial.remove(&key).unwrap();
                    assert_eq!(BlobHash::of(encoded.as_bytes()), position.item_hash);
                    let item: ReadItem = serde_json::from_str(&encoded).unwrap();
                    assert!(!matches!(item.kind, ItemKind::Fragment { .. }));
                    assert!(!values.contains(&item), "duplicate reconstructed item");
                    values.push(item);
                }
            } else {
                assert!(!values.contains(item), "duplicate item");
                values.push(item.clone());
            }
        }
        let Some(next) = page.document.footer.next.as_ref() else {
            break;
        };
        assert!(
            calls.insert(serde_json::to_string(&next.request).unwrap()),
            "repeated continuation"
        );
        assert!(
            !page.document.items.is_empty(),
            "continuation without item progress"
        );
        eprintln!(
            "typed web continuation: items={}, shown={}, next_page={}",
            values.len(),
            page.document.footer.shown,
            calls.len()
        );
        page = follow_read(r, next).await;
    }
    assert!(partial.is_empty(), "delivery ended within an encoded item");
    values
}

#[test]
fn setting_http_keeps_typed_write_repeat_and_current_results() {
    let reader = reader();
    rt().block_on(async {
        let invoke = || {
            authed(own_origin(post("/api/v1/setting")))
                .body(Body::from(setting_envelope("http-setting-receipt")))
                .unwrap()
        };
        let (status, _, first) = send(&reader, invoke()).await;
        assert_eq!(status, StatusCode::OK, "{first}");
        let crate::protocol::setting::Answer::Receipt(first) =
            serde_json::from_str(&first).unwrap()
        else {
            panic!("write must return a receipt")
        };
        assert!(!first.pending);
        let (status, _, repeat) = send(&reader, invoke()).await;
        assert_eq!(status, StatusCode::OK, "{repeat}");
        let crate::protocol::setting::Answer::Receipt(repeat) =
            serde_json::from_str(&repeat).unwrap()
        else {
            panic!("repeat must return a receipt")
        };
        assert_eq!(repeat.id, first.id);
        assert!(repeat.repeat);
        assert!(!repeat.pending);
        let request = command_surface::Request {
            operation: "setting".into(),
            arguments: serde_json::json!({"key":"reading.variants"}),
            request_id: "http-setting-current".into(),
            deadline: command_surface::Deadline::None,
            budget_tokens: Default::default(),
            cursor: None,
            fresh: false,
            format: Default::default(),
        };
        let call = authed(own_origin(post("/api/v1/setting")))
            .body(Body::from(serde_json::to_vec(&request).unwrap()))
            .unwrap();
        let (status, _, shown) = send(&reader, call).await;
        assert_eq!(status, StatusCode::OK, "{shown}");
        let crate::protocol::setting::Answer::Current(current) =
            serde_json::from_str(&shown).unwrap()
        else {
            panic!("read must return a current setting")
        };
        assert_eq!(current.key, "reading.variants");
    });
}
