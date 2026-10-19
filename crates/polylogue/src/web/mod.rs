//! The daemon's loopback HTTP port (spec `surfaces/web.md`): the web reader
//! and its JSON API, for the browser only. Every route answers through the
//! same `Request` as the Unix socket, so a page and a command cannot drift
//! apart. Other programs use the socket; the extension uses its native
//! messaging host (`crate::browser`).

pub mod access;
mod api;
pub mod pages;

use crate::daemon::server::{Daemon, handle};
use crate::daemon::{Request, Response};
use axum::Router;
use axum::extract::{Path as UrlPath, Query, Request as HttpRequest, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Redirect, Response as HttpResponse};
use axum::routing::get;
use polylogue_render::View;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

/// What every handler shares.
pub struct Web {
    pub daemon: Arc<Daemon>,
    pub key: String,
    pub token: String,
    pub port: u16,
    /// The extension's files, for the operator's action requests.
    pub browser: Arc<crate::browser::Ctx>,
}

/// Binds the port, writes the token and port files and serves on a thread
/// of its own. A port already taken is reported and the daemon goes on
/// without HTTP: the socket is the path every client but the browser uses.
pub fn start(daemon: Arc<Daemon>) -> std::io::Result<u16> {
    let key = access::archive_key(&daemon.archive.root);
    let listener = std::net::TcpListener::bind(("127.0.0.1", access::wanted_port()))?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();
    let token = access::new_token()?;
    let dir = access::runtime_dir(&key);
    access::publish(&dir, &token, port)?;
    let browser = Arc::new(crate::browser::Ctx::at(&daemon.state, &daemon.archive)?);
    let web = Arc::new(Web {
        daemon,
        key,
        token,
        port,
        browser,
    });
    let uid = access::my_uid().ok_or_else(|| std::io::Error::other("cannot read this uid"))?;
    std::thread::Builder::new()
        .name("polylogue-http".into())
        .spawn(move || {
            let rt = match tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
            {
                Ok(rt) => rt,
                Err(e) => {
                    eprintln!("polylogue: HTTP runtime: {e}");
                    return;
                }
            };
            rt.block_on(async move {
                let listener = match tokio::net::TcpListener::from_std(listener) {
                    Ok(l) => l,
                    Err(e) => {
                        eprintln!("polylogue: HTTP listener: {e}");
                        return;
                    }
                };
                let app = router(web);
                if let Err(e) = axum::serve(
                    OwnerListener {
                        inner: listener,
                        uid,
                    },
                    app,
                )
                .await
                {
                    eprintln!("polylogue: HTTP stopped: {e}");
                }
            });
        })?;
    Ok(port)
}

/// Accepts only connections whose peer socket belongs to the operator's
/// uid; others are closed unanswered. A peer whose owner cannot be read is
/// refused too.
struct OwnerListener {
    inner: tokio::net::TcpListener,
    uid: u32,
}

impl axum::serve::Listener for OwnerListener {
    type Io = tokio::net::TcpStream;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok((stream, peer)) = self.inner.accept().await else {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            };
            if access::socket_owner(peer) == Some(self.uid) {
                return (stream, peer);
            }
        }
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        self.inner.local_addr()
    }
}

pub fn router(web: Arc<Web>) -> Router {
    Router::new()
        .route("/", get(home))
        .route("/search", get(search))
        .route("/s/{id}", get(session))
        .route("/auth", get(auth_page).post(auth_exchange))
        .route("/assets/reader.css", get(css))
        .route("/assets/reader.js", get(js))
        .nest("/api/v1", api::routes())
        .layer(middleware::from_fn_with_state(Arc::clone(&web), guard))
        .with_state(web)
}

/// The sign-in page and the static assets are open; everything else needs
/// the token.
fn open_route(path: &str) -> bool {
    path == "/auth" || path.starts_with("/assets/")
}

pub(crate) fn has_token(web: &Web, headers: &HeaderMap) -> bool {
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    let cookie = access::cookie_token(
        headers.get(header::COOKIE).and_then(|v| v.to_str().ok()),
        &access::cookie_name(&web.key),
    );
    bearer
        .or(cookie)
        .is_some_and(|t| access::same_secret(t, &web.token))
}

/// Host check, credentials, POST origin check, and the headers every
/// response carries. No CORS headers are ever sent.
async fn guard(State(web): State<Arc<Web>>, req: HttpRequest, next: Next) -> HttpResponse {
    let headers = req.headers();
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok());
    if !access::host_allowed(host, web.port) {
        return (StatusCode::MISDIRECTED_REQUEST, "unknown host").into_response();
    }
    let origin = headers.get(header::ORIGIN).and_then(|v| v.to_str().ok());
    let path = req.uri().path().to_owned();
    if !open_route(&path) && !has_token(&web, headers) {
        return if req.method() == Method::GET && !path.starts_with("/api/") {
            Redirect::to("/auth").into_response()
        } else {
            (
                StatusCode::UNAUTHORIZED,
                "the access token is missing or stale",
            )
                .into_response()
        };
    }
    // A write from a page must come from the reader's own origin.
    if req.method() == Method::POST {
        let ok = origin.is_some_and(|o| access::host_allowed(o.strip_prefix("http://"), web.port));
        let bearer_only = origin.is_none()
            && headers
                .get(header::AUTHORIZATION)
                .is_some_and(|v| v.as_bytes().starts_with(b"Bearer "));
        if !ok && !bearer_only {
            return (
                StatusCode::FORBIDDEN,
                "writes come from the reader's own pages",
            )
                .into_response();
        }
    }
    let mut response = next.run(req).await;
    let h = response.headers_mut();
    h.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; form-action 'self'; base-uri 'none'; frame-ancestors 'none'",
        ),
    );
    h.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    h.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

/// Runs a request on a blocking thread, as the socket would.
pub async fn ask(web: &Arc<Web>, request: Request) -> Response {
    let d = Arc::clone(&web.daemon);
    tokio::task::spawn_blocking(move || handle(request, &d))
        .await
        .unwrap_or_else(|e| Response::Error {
            kind: "internal".into(),
            message: e.to_string(),
            detail: None,
        })
}

/// A response's data with the archive directory's path taken out.
pub fn scrubbed(web: &Web, mut value: serde_json::Value) -> serde_json::Value {
    if let Some(o) = value.as_object_mut() {
        o.remove("archive");
    }
    let text = value.to_string();
    let clean = access::scrub(&text, &web.daemon.archive.root);
    if clean == text {
        value
    } else {
        serde_json::from_str(&clean).unwrap_or(value)
    }
}

fn page(html: String) -> HttpResponse {
    Html(html).into_response()
}

fn failed(status: StatusCode, kind: &str, message: &str) -> HttpResponse {
    (status, Html(pages::error_page(kind, message))).into_response()
}

async fn home(State(web): State<Arc<Web>>) -> HttpResponse {
    let status = match ask(&web, Request::Status).await {
        Response::Ok { data } => scrubbed(&web, data),
        Response::Error { kind, message, .. } => {
            return failed(StatusCode::INTERNAL_SERVER_ERROR, &kind, &message);
        }
        Response::Accepted { .. } => serde_json::Value::Null,
    };
    let sessions = match ask(
        &web,
        Request::List {
            limit: 30,
            before: None,
        },
    )
    .await
    {
        Response::Ok { data } => scrubbed(&web, data),
        _ => serde_json::Value::Null,
    };
    page(pages::home(&status, &sessions))
}

#[derive(Deserialize)]
struct SearchQuery {
    #[serde(default)]
    q: String,
    cursor: Option<String>,
}

async fn search(State(web): State<Arc<Web>>, Query(p): Query<SearchQuery>) -> HttpResponse {
    if p.q.trim().is_empty() {
        return Redirect::to("/").into_response();
    }
    let request = Request::Query {
        text: p.q.clone(),
        cursor: p.cursor,
        page_size: None,
        why: true,
        session: None,
        fields: vec![],
    };
    match ask(&web, request).await {
        Response::Ok { data } => page(pages::search(&p.q, Some(&scrubbed(&web, data)), None)),
        Response::Error {
            kind,
            message,
            detail,
        } => {
            let e = serde_json::json!({ "kind": kind, "message": access::scrub(&message, &web.daemon.archive.root), "detail": detail });
            page(pages::search(&p.q, None, Some(&e)))
        }
        Response::Accepted { .. } => page(pages::search(&p.q, None, None)),
    }
}

#[derive(Deserialize)]
struct ReadQuery {
    view: Option<String>,
}

async fn session(
    State(web): State<Arc<Web>>,
    UrlPath(id): UrlPath<String>,
    Query(p): Query<ReadQuery>,
) -> HttpResponse {
    let view_name = p.view.unwrap_or_else(|| "full".into());
    let Some(view) = View::parse(&view_name) else {
        return failed(StatusCode::BAD_REQUEST, "bad_request", "unknown view");
    };
    let Ok(session_id) = polylogue_model::SessionId::parse(&id) else {
        return failed(StatusCode::BAD_REQUEST, "bad_request", "not a session ID");
    };
    let request = Request::Read {
        session: session_id,
        view: view_name.clone(),
        format: "json".into(),
        logical: true,
        budget: None,
        cursor: None,
    };
    match ask(&web, request).await {
        Response::Ok { data } => {
            let data = scrubbed(&web, data);
            let session = serde_json::from_value(data["session"].clone());
            let messages = serde_json::from_value(data["messages"].clone());
            match (session, messages) {
                (Ok(s), Ok(m)) => {
                    let m: Vec<polylogue_model::Message> = m;
                    page(pages::session(&s, &m, view, &view_name))
                }
                _ => failed(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "internal",
                    "the session could not be read",
                ),
            }
        }
        Response::Error { kind, message, .. } => {
            let status = if kind == "not_found" {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            failed(
                status,
                &kind,
                &access::scrub(&message, &web.daemon.archive.root),
            )
        }
        Response::Accepted { .. } => {
            failed(StatusCode::INTERNAL_SERVER_ERROR, "internal", "no data")
        }
    }
}

async fn auth_page() -> HttpResponse {
    page(pages::auth())
}

/// Exchanges the token for the archive's cookie. The page posts the token
/// from its URL fragment; the cookie is `SameSite=Strict` and `HttpOnly`.
async fn auth_exchange(State(web): State<Arc<Web>>, body: String) -> HttpResponse {
    let form: HashMap<String, String> = serde_urlencoded_pairs(&body);
    let token = form.get("token").map_or("", String::as_str);
    if !access::same_secret(token, &web.token) {
        return (
            StatusCode::UNAUTHORIZED,
            "the access token is missing or stale",
        )
            .into_response();
    }
    let cookie = format!(
        "{}={}; Path=/; HttpOnly; SameSite=Strict",
        access::cookie_name(&web.key),
        web.token
    );
    let mut r = StatusCode::NO_CONTENT.into_response();
    if let Ok(v) = HeaderValue::from_str(&cookie) {
        r.headers_mut().insert(header::SET_COOKIE, v);
    }
    r
}

fn serde_urlencoded_pairs(body: &str) -> HashMap<String, String> {
    body.split('&')
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            Some((percent_decode(k), percent_decode(v)))
        })
        .collect()
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
                match hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                    Some(b) => {
                        out.push(b);
                        i += 2;
                    }
                    None => out.push(b'%'),
                }
            }
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

async fn css() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/css; charset=utf-8")],
        include_str!("assets/reader.css"),
    )
        .into_response()
}

async fn js() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("assets/reader.js"),
    )
        .into_response()
}

/// The body of a JSON answer: the data as the socket gives it, the error
/// as a typed kind with a status code.
pub fn json_answer(web: &Web, response: Response) -> HttpResponse {
    match response {
        Response::Ok { data } => axum::Json(scrubbed(web, data)).into_response(),
        Response::Accepted { request_id } => (
            StatusCode::ACCEPTED,
            axum::Json(serde_json::json!({ "request_id": request_id })),
        )
            .into_response(),
        Response::Error {
            kind,
            message,
            detail,
        } => {
            let status = match kind.as_str() {
                "not_found" => StatusCode::NOT_FOUND,
                "bad_request" | "query" | "sql" => StatusCode::BAD_REQUEST,
                "conflict" => StatusCode::CONFLICT,
                "not_built" => StatusCode::NOT_IMPLEMENTED,
                _ => StatusCode::INTERNAL_SERVER_ERROR,
            };
            let message = access::scrub(&message, &web.daemon.archive.root);
            (
                status,
                axum::Json(
                    serde_json::json!({ "error": kind, "message": message, "detail": detail }),
                ),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
mod tests;
