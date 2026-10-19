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
    let browser = Arc::new(crate::browser::Ctx::at(&w.daemon.config, &w.daemon.archive).unwrap());
    let web = Arc::new(Web {
        daemon: Arc::clone(&w.daemon),
        key: "abc123def456".into(),
        token: TOKEN.into(),
        port: PORT,
        browser,
    });
    Reader { w, web }
}

fn get(path: &str) -> axum::http::request::Builder {
    Http::get(path).header(header::HOST, format!("127.0.0.1:{PORT}"))
}

fn post(path: &str) -> axum::http::request::Builder {
    Http::post(path).header(header::HOST, format!("127.0.0.1:{PORT}"))
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
            page.contains("data-cli=\"polylogue read claude-code:s1 --view full\""),
            "{page}"
        );
        // The same read through the API is the socket's answer.
        let req = authed(get("/api/v1/read?session=claude-code%3As1"))
            .body(Body::empty())
            .unwrap();
        let (s, _, body) = send(&r, req).await;
        assert_eq!(s, StatusCode::OK);
        let api: serde_json::Value = serde_json::from_str(&body).unwrap();
        let Response::Ok { data } = handle(
            Request::Read {
                session: polylogue_model::SessionId::parse("claude-code:s1").unwrap(),
                view: "full".into(),
                format: "json".into(),
                logical: false,
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

#[test]
fn the_operator_queues_chat_actions_only_while_switched_on() {
    let r = reader();
    let body = serde_json::json!({
        "provider": "chatgpt",
        "operation": "conversation.create",
        "text": "summarize this",
        "presentation": {"model_slug": "gpt-x", "model_label": "GPT X", "effort_label": "standard"}
    })
    .to_string();
    rt().block_on(async {
        let enqueue = || {
            own_origin(authed(post("/api/v1/browser-actions")))
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body.clone()))
                .unwrap()
        };
        assert_eq!(send(&r, enqueue()).await.0, StatusCode::CONFLICT);
        r.w.set("browser.actions", "on");
        let (s, _, created) = send(&r, enqueue()).await;
        assert_eq!(s, StatusCode::CREATED, "{created}");
        let v: serde_json::Value = serde_json::from_str(&created).unwrap();
        assert_eq!(v["action"]["status"], "queued");
        assert_eq!(v["action"]["submit_policy"], "stage_only");
        // The same request again is the same action.
        let (s, _, again) = send(&r, enqueue()).await;
        assert_eq!(s, StatusCode::OK);
        let again: serde_json::Value = serde_json::from_str(&again).unwrap();
        assert_eq!(again["action"]["action_id"], v["action"]["action_id"]);
    });
}

#[test]
fn every_http_command_is_served_and_nothing_else() {
    use crate::commands::{COMMANDS, Kind};
    let r = reader();
    rt().block_on(async {
        for c in COMMANDS {
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
                assert!(
                    status != StatusCode::NOT_FOUND || body.contains("not_found"),
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
