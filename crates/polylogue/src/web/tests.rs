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

fn own_origin(b: axum::http::request::Builder) -> axum::http::request::Builder {
    b.header(header::ORIGIN, format!("http://127.0.0.1:{PORT}"))
}

async fn send(r: &Reader, req: Http<Body>) -> (StatusCode, HeaderMap, String) {
    let resp = router(Arc::clone(&r.web)).oneshot(req).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
    (status, headers, String::from_utf8_lossy(&body).into_owned())
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
            .body(Body::from(r#"{"key":"a","value":"b"}"#))
            .unwrap();
        assert_eq!(send(&r, req).await.0, StatusCode::FORBIDDEN);
        let req = own_origin(authed(post("/api/v1/setting")))
            .body(Body::from(r#"{"key":"a","value":"b"}"#))
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
    let r = reader();
    rt().block_on(async {
        let req = authed(get("/s/claude-code%3As1"))
            .body(Body::empty())
            .unwrap();
        let (s, _, page) = send(&r, req).await;
        assert_eq!(s, StatusCode::OK, "{page}");
        assert!(
            page.contains("&lt;script&gt;alert(1)&lt;/script&gt; &amp; the parser"),
            "{page}"
        );
        assert!(!page.contains("<script>alert"));
        assert!(
            page.contains("data-cli=\"polylogue read claude-code:s1 --with-parents --view full --budget 8192 --cursor "),
            "{page}"
        );
        // The same read through the API is the socket's answer.
        let req = authed(get("/api/v1/read?session=claude-code%3As1"))
            .body(Body::empty())
            .unwrap();
        let (s, _, body) = send(&r, req).await;
        assert_eq!(s, StatusCode::OK);
        let api: serde_json::Value = serde_json::from_str(&body).unwrap();
        let Response::Ok { data, .. } = handle(
            Request::Read {
            slices: true, selection: None,
                target: crate::daemon::ReadTarget::Session(
                    polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
                ),
                view: "full".into(),
                format: "json".into(),
                with_parents: false,
                budget: None,
                cursor: None,
            },
            &r.web.daemon,
        ) else {
            panic!("socket read failed")
        };
        assert_eq!(api, scrubbed(&r.web, data));
        // An unknown session is a typed not-found, without a local path.
        let req = authed(get("/s/claude-code%3Anope"))
            .body(Body::empty())
            .unwrap();
        let (s, _, page) = send(&r, req).await;
        assert_eq!(s, StatusCode::NOT_FOUND);
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
        assert!(page.contains("data-cli=\"polylogue list parser\""));
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
        .execute_batch("ALTER TABLE setting RENAME TO unavailable_setting")
        .unwrap();
    let body = serde_json::json!({
        "provider": "chatgpt",
        "text": "Neutral synthetic draft",
        "model": "X",
    });
    let response = rt().block_on(async { send(&r, chat_send(&body)).await });
    archive
        .execute_batch("ALTER TABLE unavailable_setting RENAME TO setting")
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
fn a_chat_send_answer_masks_credentials_without_changing_the_queued_draft() {
    let r = reader();
    r.w.set("browser.actions", "on");
    let text = format!("Keep this draft {}", "AKIAIOSFODNN7EXAMPLE");
    let body = serde_json::json!({"provider":"chatgpt", "text":text, "model":"X"});
    rt().block_on(async {
        let (status, _, shown) = send(&r, chat_send(&body)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(!shown.contains("AKIAIOSFODNN7EXAMPLE"));
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
    let mut server = start_on(Arc::clone(&daemon.daemon), 0).unwrap();
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
    let mut config = polylogue_engine::Config::from_env();
    config.home = dir.path().to_owned();
    config.tool_dirs.clear();
    config.workers = 1;
    let mut runtime = crate::daemon::service::start(
        &dir.path().join("archive"),
        config,
        crate::daemon::service::ArchiveMode::Standalone,
    )
    .unwrap();
    let daemon = Arc::clone(&runtime.daemon);
    let reader = daemon.reads.reader().unwrap();
    assert_eq!(reader.query("SELECT 42 AS n", &[]).unwrap()[0]["n"], 42);
    let mut server = start_on(Arc::clone(&daemon), 0).unwrap();
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
fn session_html_and_json_read_routes_mask_titles_and_message_content() {
    use std::io::Write;
    let r = reader();
    let secret = format!("sk-ant-api03-{}-AAAAAAAA", "A".repeat(94));
    let mut file = std::fs::OpenOptions::new()
        .append(true)
        .open(r.w.home.join(".claude/projects/-p/s1.jsonl"))
        .unwrap();
    writeln!(file, "{}", serde_json::json!({"type": "custom-title", "sessionId": "s1", "customTitle": format!("Synthetic title {secret}")})).unwrap();
    write!(
        file,
        "{}",
        polylogue_testkit::claude_user("u2", "s1", &secret)
    )
    .unwrap();
    drop(file);
    r.w.wait_for(|status| {
        if status["store"]["messages"] != 2 { return false; }
        let response = handle(Request::Read {
            slices: true, selection: None, target: crate::daemon::ReadTarget::Session(polylogue_model::SessionId::parse("claude-code:s1").unwrap()), view: "full".into(), format: "json".into(), with_parents: false, budget: None, cursor: None }, &r.w.daemon);
        matches!(response, Response::Ok { data, .. } if data["session"]["title"]["text"].as_str().is_some_and(|t| t.starts_with("Synthetic title ")))
    });
    rt().block_on(async {
        for path in [
            "/s/claude-code:s1",
            "/api/v1/read?session=claude-code%3As1&format=json",
        ] {
            let (status, _, body) = send(&r, authed(get(path)).body(Body::empty()).unwrap()).await;
            assert_eq!(status, StatusCode::OK, "{path}: {body}");
            assert!(body.contains("Synthetic title "));
            assert!(
                !body.contains(&secret),
                "{path} exposed a synthetic credential"
            );
            assert!(body.contains("masked"));
        }
        let path = format!("/api/v1/read?session=claude-code%3As1&view={secret}");
        let (status, _, body) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(
            !body.contains(&secret),
            "argument validation exposed a synthetic credential"
        );
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&body).unwrap()["error"],
            "bad_request"
        );
        let path = format!("/api/v1/read?session=claude-code%3A{secret}");
        let (status, _, body) = send(&r, authed(get(&path)).body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(!body.contains(&secret));
        assert!(body.contains("masked"));
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
            let Response::Ok { data, .. } = handle(
                Request::Query {
                    text: format!("{scope} | measure usage"),
                    cursor: None,
                    page_size: None,
                    why: false,
                    session: None,
                    fields: vec![],
                },
                &r.web.daemon,
            ) else {
                panic!("usage failed")
            };
            let report =
                polylogue_render::html::escape(&polylogue_render::measures::usage(&data["data"]));
            assert!(html.contains(&report), "{html}");
            assert!(html.contains("API-equivalent unknown"), "{html}");
            assert!(html.contains("Browse selected sessions"));
            assert!(
                html.contains("data-cli=\"polylogue measure usage "),
                "{html}"
            );
        }
        let (status, _, html) = send(
            &r,
            authed(get("/cost?q=messages")).body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{html}");
        let (status, _, html) = send(
            &r,
            authed(get("/s/claude-code:s1"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(
            html.contains("Every selected parent session is here."),
            "{html}"
        );
        assert!(!html.contains("Links unavailable"), "{html}");
        assert!(html.contains("Session cost and usage"));
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
        let (status, _, html) = send(
            &r,
            authed(get("/s/claude-code:child"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("href=\"/s/claude-code%3As1\""), "{html}");
        assert!(html.contains("forkedFrom"), "{html}");
        assert!(html.contains("claude-code:u1"), "{html}");
        assert!(!html.contains("claude-code:other"), "{html}");
        let (status, _, html) = send(
            &r,
            authed(get("/s/claude-code:orphan"))
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{html}");
        assert!(html.contains("Some parent sessions are missing"), "{html}");
        assert!(html.contains("missing_parent"), "{html}");
        assert!(html.contains("href=\"/s/claude-code%3Amissing\""), "{html}");
    });
}

#[test]
fn cost_is_served_through_the_owned_loopback_http_listener() {
    use std::io::{Read, Write};
    let r = reader();
    let mut server = start_on(Arc::clone(&r.web.daemon), 0).unwrap();
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
        assert_eq!(usage["api_cost"]["known_usd"], 15.0);
        assert_eq!(usage["api_cost"]["priced_messages"], 1);
        assert_eq!(usage["api_cost"]["unpriced_messages"], 0);
        assert_eq!(usage["by_model"][0]["model"], "claude-opus-4-1");
        assert_eq!(usage["by_model"][0]["messages"], 1);
        assert_eq!(usage["by_model"][0]["api_cost"]["known_usd"], 15.0);
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

mod benchmark;

mod notes;
mod search_api;
mod search_measurements;
mod timeline;

mod day_cost_measurements;
mod measurement;
