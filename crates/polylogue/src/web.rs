//! The daemon's loopback HTTP port: the web reader
//! and its JSON API, for the browser only. Every route answers through the
//! same `Request` as the Unix socket, so a page and a command cannot drift
//! apart. Other programs use the socket; the extension uses its native
//! messaging host (`crate::browser`).

pub mod access;
mod api;
mod connection;
pub mod pages;
mod reader;

use crate::daemon::service::Daemon;
use crate::daemon::{Request, Response};
use axum::Router;
use axum::extract::{Path as UrlPath, Query, Request as HttpRequest, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Redirect, Response as HttpResponse};
use axum::routing::get;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

/// What every handler shares.
pub struct Web {
    pub daemon: Arc<Daemon>,
    pub operations: Arc<command_surface::OperationTable>,
    pub key: String,
    pub token: String,
    pub port: u16,
}

/// Owns the HTTP runtime. Dropping it aborts network input in progress and
/// joins blocking application handlers before the application itself stops.
pub struct Server {
    pub port: u16,
    stop: Option<tokio::sync::oneshot::Sender<()>>,
    thread: Option<std::thread::JoinHandle<std::io::Result<()>>>,
    directory: std::path::PathBuf,
    token: String,
}

impl Server {
    pub fn shutdown(&mut self) -> std::io::Result<()> {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        let result = self.thread.take().map_or(Ok(()), |thread| {
            thread
                .join()
                .map_err(|_| std::io::Error::other("HTTP worker panicked"))?
        });
        if std::fs::read_to_string(self.directory.join("token"))
            .ok()
            .as_deref()
            == Some(&self.token)
        {
            let _ = std::fs::remove_file(self.directory.join("token"));
            let _ = std::fs::remove_file(self.directory.join("port"));
            let _ = std::fs::remove_dir(&self.directory);
        }
        result
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Err(error) = self.shutdown() {
            eprintln!("polylogue: HTTP shutdown: {error}");
        }
    }
}

/// Binds the port and returns the owner of the HTTP worker. Socket serving
/// can continue independently if HTTP startup fails.
pub fn start(
    daemon: Arc<Daemon>,
    operations: Arc<command_surface::OperationTable>,
) -> std::io::Result<Server> {
    start_on(daemon, operations, access::wanted_port())
}

fn start_on(
    daemon: Arc<Daemon>,
    operations: Arc<command_surface::OperationTable>,
    port: u16,
) -> std::io::Result<Server> {
    use std::future::IntoFuture;
    let key = crate::runtime_dir::archive_key(&daemon.archive.root);
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;
    let listener = {
        let _entered = runtime.enter();
        tokio::net::TcpListener::from_std(listener)?
    };
    let token = access::new_token()?;
    let directory = crate::runtime_dir::runtime_dir(&key);
    let uid = access::my_uid().ok_or_else(|| std::io::Error::other("cannot read this uid"))?;
    let web = Arc::new(Web {
        daemon,
        operations,
        key,
        token: token.clone(),
        port,
    });
    let (stop, stopped) = tokio::sync::oneshot::channel();
    let mut server = Server {
        port,
        stop: Some(stop),
        thread: None,
        directory,
        token,
    };
    access::write_access_files(&server.directory, &server.token, port)?;
    server.thread = Some(std::thread::Builder::new().name("polylogue-http".into()).spawn(move || {
        runtime.block_on(async move {
            tokio::select! {
                result = axum::serve(OwnerListener { inner: listener, uid }, router(web).into_make_service_with_connect_info::<connection::Peer>()).into_future() => result,
                _ = stopped => Ok(()),
            }
        })
        // Runtime drop aborts async network tasks and waits for every
        // spawn_blocking application request before this owner can join.
    })?);
    Ok(server)
}

/// Accepts only connections whose peer socket belongs to the operator's
/// uid; others are closed unanswered. A peer whose owner cannot be read is
/// refused too.
struct OwnerListener {
    inner: tokio::net::TcpListener,
    uid: u32,
}

impl axum::serve::Listener for OwnerListener {
    type Io = connection::Connection;
    type Addr = std::net::SocketAddr;

    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok((stream, peer)) = self.inner.accept().await else {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            };
            let Ok(server) = stream.local_addr() else {
                continue;
            };
            if access::socket_owner(peer, server) == Some(self.uid)
                && let Ok(connection) = connection::Connection::new(stream)
            {
                return (connection, peer);
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
        .route("/cost", get(cost))
        .route("/s/{id}", get(reader::session))
        .route("/d/{id}", get(reader::document))
        .route("/m/{id}", get(reader::message))
        .route("/n/{id}", get(reader::note))
        .route("/auth", get(auth_page).post(auth_exchange))
        .route("/assets/reader.css", get(css))
        .route("/assets/reader.js", get(js))
        .route("/assets/live.js", get(live_js))
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
async fn guarded(web: Arc<Web>, req: HttpRequest, next: Next) -> HttpResponse {
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
    next.run(req).await
}

tokio::task_local! {
    static READ_LIMIT: (Option<i64>, polylogue_store::read::limit::ReadCancellation);
}

async fn guard(State(web): State<Arc<Web>>, req: HttpRequest, next: Next) -> HttpResponse {
    let cancellation = req
        .extensions()
        .get::<axum::extract::ConnectInfo<connection::Peer>>()
        .map(|peer| peer.0.0.child())
        .unwrap_or_default();
    let deadline_ms = match req.headers().get("x-polylogue-deadline-ms") {
        Some(value) => match value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<i64>().ok())
        {
            Some(value) => Some(value),
            None => {
                return json_answer(
                    &web,
                    crate::daemon::service::error(
                        "bad_request",
                        "x-polylogue-deadline-ms must be Unix milliseconds",
                    ),
                );
            }
        },
        None => None,
    };
    let mut response = READ_LIMIT
        .scope((deadline_ms, cancellation), guarded(web, req, next))
        .await;
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
    let scope = if request.read_only() {
        match web.daemon.reads.scope_token() {
            Ok(scope) => Some(scope),
            Err(error) => return crate::daemon::service::store_error(&error),
        }
    } else {
        None
    };
    let d = Arc::clone(&web.daemon);
    let (deadline_ms, cancellation) = READ_LIMIT.try_with(Clone::clone).unwrap_or_default();
    let limit = crate::daemon::execution::limit_for(&request, deadline_ms, cancellation);
    let mut cancel = CancelRead(Some(limit.cancellation.clone()));
    let authorship = polylogue_model::authorship::compute_authorship(
        polylogue_model::authorship::Channel::Web,
        &Default::default(),
        None,
    );
    let response = tokio::task::spawn_blocking(move || {
        crate::daemon::service::handle_with_caller(request, &d, &limit, &authorship)
    })
    .await
    .unwrap_or_else(|error| crate::daemon::service::error("internal", error));
    cancel.0.take();
    let answer = crate::protocol::scope_answer(response.answer(), scope.as_ref());
    browser_response(answer.into(), &web.daemon.archive.root)
}

pub(super) fn request_limit() -> polylogue_store::read::limit::ReadLimit {
    READ_LIMIT
        .try_with(|(deadline_ms, cancellation)| {
            crate::daemon::execution::limit(*deadline_ms, cancellation.clone())
        })
        .unwrap_or_else(|_| crate::daemon::execution::limit(None, Default::default()))
}

pub(super) struct CancelRead(Option<polylogue_store::read::limit::ReadCancellation>);
impl Drop for CancelRead {
    fn drop(&mut self) {
        if let Some(cancellation) = &self.0 {
            cancellation.cancel();
        }
    }
}

fn browser_response(response: Response, root: &std::path::Path) -> Response {
    let mut response = response;
    match &mut response {
        Response::Ok { data } => access::scrub_value(data, root),
        Response::Error {
            message, detail, ..
        } => {
            *message = access::scrub(message, root);
            if let Some(detail) = detail {
                access::scrub_value(detail, root);
            }
        }
    }
    if let Response::Ok { data, .. } = &mut response
        && let Some(data) = data.as_object_mut()
    {
        data.remove("archive");
    }
    response
}

/// A response's data with the archive directory's path taken out.
pub fn scrubbed(web: &Web, mut value: serde_json::Value) -> serde_json::Value {
    if let Some(o) = value.as_object_mut() {
        o.remove("archive");
    }
    access::scrub_value(&mut value, &web.daemon.archive.root);
    value
}

fn page(html: String) -> HttpResponse {
    Html(html).into_response()
}

fn failed(status: StatusCode, kind: &str, message: &str) -> HttpResponse {
    (status, Html(pages::error_page(kind, message))).into_response()
}

async fn home(State(web): State<Arc<Web>>) -> HttpResponse {
    let status = match ask(&web, Request::Status).await {
        Response::Ok { data, .. } => data,
        Response::Error { kind, message, .. } => {
            return failed(error_status(&kind), &kind, &message);
        }
    };
    let sessions = match ask(
        &web,
        Request::Query {
            text: "sessions".into(),
            cursor: None,
            page_size: Some(30),
            why: false,
            session: None,
            fields: vec![],
        },
    )
    .await
    {
        Response::Ok { data, .. } => data,
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
        cursor: p.cursor.clone(),
        page_size: None,
        why: true,
        session: None,
        fields: vec![],
    };
    match ask(&web, request).await {
        Response::Ok { data, .. } => {
            page(pages::search(&p.q, Some(&data), None, p.cursor.as_deref()))
        }
        Response::Error {
            kind,
            message,
            detail,
        } => {
            let e = serde_json::json!({ "kind": kind, "message": access::scrub(&message, &web.daemon.archive.root), "detail": detail });
            page(pages::search(&p.q, None, Some(&e), p.cursor.as_deref()))
        }
    }
}

/// Cost uses the cost measure and its session selection, as the CLI does.
async fn cost(State(web): State<Arc<Web>>, Query(p): Query<SearchQuery>) -> HttpResponse {
    let scope = if p.q.trim().is_empty() {
        "sessions"
    } else {
        p.q.trim()
    };
    let text = format!("{scope} | measure usage");
    match ask(
        &web,
        Request::Query {
            text,
            cursor: None,
            page_size: None,
            why: false,
            session: None,
            fields: vec![],
        },
    )
    .await
    {
        Response::Ok { data, .. } => page(pages::cost(scope, &data["data"])),
        Response::Error { kind, message, .. } => failed(
            error_status(&kind),
            &kind,
            &access::scrub(&message, &web.daemon.archive.root),
        ),
    }
}

async fn auth_page() -> HttpResponse {
    page(pages::auth())
}

/// Exchanges the token for the archive's cookie. The page posts the token
/// from its URL fragment; the cookie is `SameSite=Strict` and `HttpOnly`.
async fn auth_exchange(
    State(web): State<Arc<Web>>,
    headers: HeaderMap,
    body: String,
) -> HttpResponse {
    if !access::content_type(&headers, "application/x-www-form-urlencoded") {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "sign-in requires application/x-www-form-urlencoded",
        )
            .into_response();
    }
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
        include_str!("web/assets/reader.css"),
    )
        .into_response()
}

async fn js() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("web/assets/reader.js"),
    )
        .into_response()
}

async fn live_js() -> HttpResponse {
    (
        [(header::CONTENT_TYPE, "text/javascript; charset=utf-8")],
        include_str!("web/assets/live.js"),
    )
        .into_response()
}

fn error_status(kind: &str) -> StatusCode {
    crate::protocol::kind(kind)
        .http_status()
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
}

/// The data as the socket gives it, or a typed error with its HTTP status.
pub fn json_answer(
    web: &Web,
    response: impl Into<command_surface::Answer<serde_json::Value>>,
) -> HttpResponse {
    let answer = response.into();
    let status = answer
        .http_status()
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let mut value = serde_json::to_value(answer).expect("answer serializes");
    if let Some(data) = value
        .get_mut("value")
        .and_then(serde_json::Value::as_object_mut)
    {
        data.remove("archive");
    }
    access::scrub_value(&mut value, &web.daemon.archive.root);
    (status, axum::Json(value)).into_response()
}

#[cfg(test)]
mod tests;
