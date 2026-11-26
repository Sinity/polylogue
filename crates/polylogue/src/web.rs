//! The daemon's loopback HTTP port: the web reader
//! and its JSON API, for the browser only. Every route answers through the
//! same `answer` as the Unix socket, in a blocking task, so a page and a
//! command cannot drift apart. Other programs use the socket; the extension
//! uses its native messaging host (`crate::browser`).

pub mod access;
mod api;
mod connection;
pub mod pages;
mod reader;

use crate::daemon::service::Daemon;
use axum::Router;
use axum::extract::{Extension, Form, Path as UrlPath, Query, Request as HttpRequest, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Redirect, Response as HttpResponse};
use axum::routing::get;
use command_surface::Answer;
use serde::Deserialize;
use serde_json::Value;
use std::sync::Arc;

/// What every handler shares.
pub struct Web {
    pub daemon: Arc<Daemon>,
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
            polylogue_store::log!(error, "HTTP shutdown: {error}");
        }
    }
}

/// Binds the port and returns the holder of the HTTP worker. Socket serving
/// can continue independently if HTTP startup fails.
pub fn start(daemon: Arc<Daemon>) -> std::io::Result<Server> {
    let port = daemon.config.http.listen.port();
    start_on(daemon, port)
}

fn start_on(daemon: Arc<Daemon>, port: u16) -> std::io::Result<Server> {
    use std::future::IntoFuture;
    let key = crate::runtime_dir::archive_key(&daemon.archive.root);
    let listener = std::net::TcpListener::bind(std::net::SocketAddr::new(
        daemon.config.http.listen.ip(),
        port,
    ))?;
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
    let token = match &daemon.config.http.token {
        Some(token) => token.expose().to_owned(),
        None => access::new_token()?,
    };
    let directory = daemon.config.paths.runtime_dir(&daemon.archive.root);
    let uid = access::my_uid().ok_or_else(|| std::io::Error::other("cannot read this uid"))?;
    let stop_request = daemon.archive.stop().clone();
    let web = Arc::new(Web {
        daemon,
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
    server.thread = Some(polylogue_model::spawn_named("polylogue-http", move || {
        let result = runtime.block_on(async move {
            tokio::select! {
                result = axum::serve(OwnerListener { inner: listener, uid, stop: stop_request }, router(web).into_make_service_with_connect_info::<connection::Peer>()).into_future() => result,
                _ = stopped => Ok(()),
            }
        });
        // Async network tasks are aborted; requests answering in blocking
        // tasks have been cancelled by the daemon's stop and get the
        // daemon's bound to end.
        runtime.shutdown_timeout(std::time::Duration::from_secs(5));
        result
    })?);
    Ok(server)
}

/// Accepts only connections whose peer socket belongs to the operator's
/// uid; others are closed unanswered. A peer whose owner cannot be read is
/// refused too.
struct OwnerListener {
    inner: tokio::net::TcpListener,
    uid: u32,
    /// The archive's stop request; each connection's requests derive from it.
    stop: polylogue_store::cancellation::Cancellation,
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
                && let Ok(connection) = connection::Connection::new(stream, self.stop.child())
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
        .fallback(unknown_page)
        .route("/search", get(search))
        .route("/cost", get(cost))
        .route("/day/{date}", get(day))
        .route("/rows", get(reader::rows))
        .route("/s/{id}", get(reader::session))
        .route("/d/{id}", get(reader::document))
        .route("/m/{id}", get(reader::message))
        .route("/a/{reference}", get(reader::annotation))
        .route("/auth", get(auth_page).post(auth_exchange))
        .route("/assets/auth.js", get(auth_js))
        .route("/assets/{*path}", get(asset))
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

/// The credential of an `Authorization: Bearer` header. The scheme name is
/// case-insensitive and one or more spaces may follow it (RFC 9110).
fn bearer_token(headers: &HeaderMap) -> Option<&str> {
    let value = headers.get(header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, credential) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("Bearer")
        .then(|| credential.trim_start_matches(' '))
}

fn has_token(web: &Web, headers: &HeaderMap) -> bool {
    let bearer = bearer_token(headers);
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
            let next = polylogue_render::html::encode_path_segment(
                req.uri().path_and_query().map_or("/", |uri| uri.as_str()),
            );
            let mut response = Redirect::to("/auth").into_response();
            let cookie = format!(
                "{}_page={next}; Path=/auth; HttpOnly; SameSite=Strict",
                access::cookie_name(&web.key)
            );
            if let Ok(cookie) = HeaderValue::from_str(&cookie) {
                response.headers_mut().insert(header::SET_COOKIE, cookie);
            }
            response
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
        let bearer_only = origin.is_none() && bearer_token(headers).is_some();
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

/// What bounds one HTTP request's life: its connection's cancellation,
/// itself a child of the daemon's stop, and the `x-polylogue-deadline-ms`
/// bound when the caller sent one.
#[derive(Clone)]
pub(crate) struct Lifetime {
    stop: polylogue_store::cancellation::Cancellation,
    outer_deadline: Option<std::time::Instant>,
}

impl Lifetime {
    /// The caller of one request carried by this connection.
    pub(crate) fn caller(
        &self,
        evidence: &command_surface::CallerEvidence,
    ) -> crate::operation::Caller {
        let mut caller =
            crate::operation::Caller::local(command_surface::Via::Web, self.stop.child());
        caller.authorship =
            command_surface::author_of(evidence, command_surface::Via::Web, None).into();
        caller.calling_session = evidence.session.as_ref().map(ToString::to_string);
        caller.outer_deadline = self.outer_deadline;
        caller
    }
}

async fn guard(State(web): State<Arc<Web>>, mut req: HttpRequest, next: Next) -> HttpResponse {
    let stop = req
        .extensions()
        .get::<axum::extract::ConnectInfo<connection::Peer>>()
        .map_or_else(
            || web.daemon.archive.stop().child(),
            |peer| peer.0.0.child(),
        );
    let deadline_ms = match req.headers().get("x-polylogue-deadline-ms") {
        Some(value) => match value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<i64>().ok())
        {
            Some(value) => Some(value),
            None => {
                return web_shell::assets::secure(json_answer(
                    &web,
                    crate::daemon::service::error(
                        polylogue_model::Code::BadRequest,
                        "x-polylogue-deadline-ms must be Unix milliseconds",
                    ),
                ));
            }
        },
        None => None,
    };
    // The outer wall-clock header becomes one instant for the whole request.
    let outer_deadline = deadline_ms.and_then(|deadline| {
        std::time::Instant::now().checked_add(crate::daemon::execution::remaining(Some(deadline)))
    });
    req.extensions_mut().insert(Lifetime {
        stop,
        outer_deadline,
    });
    let is_asset = req.uri().path().starts_with("/assets/");
    let mut response = guarded(web, req, next).await;
    if !is_asset {
        response
            .headers_mut()
            .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    web_shell::assets::secure(response)
}

/// Answers one operation for a view, in a blocking task, as the typed reply
/// in the JSON form the page renders. Leaving the page cancels it.
pub(crate) async fn view<O>(
    web: &Arc<Web>,
    lifetime: &Lifetime,
    arguments: O::Arguments,
    cursor: Option<String>,
) -> Answer<Value>
where
    O: crate::daemon::handler::Operation,
    O::Arguments: Send,
    O::Reply: Send,
{
    let daemon = Arc::clone(&web.daemon);
    let caller = lifetime.caller(&Default::default());
    let mut cancel = polylogue_store::cancellation::CancelOnDrop::new(caller.stop.clone());
    let answer = tokio::task::spawn_blocking(move || {
        crate::daemon::handler::ask::<O>(&daemon, arguments, cursor, &caller)
    })
    .await
    .unwrap_or_else(|_| {
        crate::daemon::Failure::internal("the view's task ended without an answer").answer()
    });
    cancel.disarm();
    let answer = answer.map(|reply| serde_json::to_value(reply).unwrap_or_default());
    browser_response(answer, &web.daemon.archive.root)
}

/// Archive paths never reach a browser, in values, failures or notes.
fn browser_response<T: serde::Serialize + serde::de::DeserializeOwned>(
    answer: Answer<T>,
    root: &std::path::Path,
) -> Answer<T> {
    let mut value = serde_json::to_value(answer).expect("answer serializes");
    access::scrub_value(&mut value, root);
    if let Some(object) = value.get_mut("value").and_then(Value::as_object_mut) {
        object.remove("archive");
    }
    serde_json::from_value(value).expect("scrubbed answer deserializes")
}

/// All answer details appear together before the view's content.
fn answer_page<T>(answers: &[(&Answer<T>, &str)], render: impl FnOnce() -> String) -> HttpResponse {
    let failure = answers
        .iter()
        .find_map(|(answer, cli)| match &answer.outcome {
            command_surface::Outcome::Error(failure) => Some((failure, *cli)),
            _ => None,
        });
    let status = failure
        .and_then(|(failure, _)| failure.kind.http_status())
        .and_then(|status| StatusCode::from_u16(status).ok())
        .unwrap_or(if failure.is_some() {
            StatusCode::INTERNAL_SERVER_ERROR
        } else {
            StatusCode::OK
        });
    let content = pages::answer_content(answers);
    let html = if let Some((_, cli)) = failure {
        pages::layout("Read failed", "", cli, &content)
    } else {
        // Both product templates and typed read pages use the shared frame.
        render().replacen(
            "<main id=\"main\" tabindex=\"-1\">",
            &format!("<main id=\"main\" tabindex=\"-1\">{content}"),
            1,
        )
    };
    (status, Html(html)).into_response()
}

async fn unknown_page(request: HttpRequest) -> HttpResponse {
    failed(
        StatusCode::NOT_FOUND,
        "Page not found",
        &format!(
            "{} is not a page. Use Home, Search or Cost and usage in the navigation.",
            request.uri().path()
        ),
    )
}

fn page(html: String) -> HttpResponse {
    Html(html).into_response()
}

fn failed(status: StatusCode, kind: &str, message: &str) -> HttpResponse {
    (status, Html(pages::error_page(kind, message))).into_response()
}

/// The line of a view: a selection, with the reasons for its rows when asked.
fn line(selection: String, page_size: Option<u64>, why: bool) -> crate::protocol::line::Rows {
    crate::protocol::line::Rows {
        selection: Some(selection),
        page_size: page_size.and_then(std::num::NonZeroU64::new),
        why,
        ..Default::default()
    }
}

type Line = crate::daemon::service::line::Line;

async fn home(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
) -> HttpResponse {
    let status =
        view::<crate::daemon::service::handlers::Status>(&web, &lifetime, Default::default(), None)
            .await;
    let sessions = view::<Line>(
        &web,
        &lifetime,
        line("sessions".into(), Some(30), false),
        None,
    )
    .await;
    answer_page(
        &[
            (&status, "polylogue status"),
            (&sessions, "polylogue --page-size 30 -- sessions"),
        ],
        || pages::home(status.value().unwrap(), &sessions.value().unwrap()["reply"]),
    )
}

#[derive(Deserialize)]
struct SearchQuery {
    #[serde(default)]
    q: String,
    cursor: Option<String>,
}

/// The cost measure is not paged, so the form sends only the selection.
#[derive(Deserialize)]
struct CostQuery {
    #[serde(default)]
    q: String,
}

async fn search(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    Query(p): Query<SearchQuery>,
) -> HttpResponse {
    if p.q.trim().is_empty() {
        return Redirect::to("/").into_response();
    }
    let answer = view::<Line>(
        &web,
        &lifetime,
        line(p.q.clone(), None, true),
        p.cursor.clone(),
    )
    .await;
    let mut cli = String::from("polylogue --why");
    if let Some(cursor) = &p.cursor {
        cli.push_str(&format!(
            " --cursor {}",
            polylogue_render::shell_quote(cursor)
        ));
    }
    cli.push_str(&format!(" -- {}", polylogue_render::shell_quote(&p.q)));
    answer_page(&[(&answer, &cli)], || {
        pages::search(&p.q, &answer.value().unwrap()["reply"], p.cursor.as_deref())
    })
}

/// Cost uses the cost measure and its session selection, as the CLI does.
async fn cost(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    Query(p): Query<CostQuery>,
) -> HttpResponse {
    let scope = if p.q.trim().is_empty() {
        "sessions"
    } else {
        p.q.trim()
    };
    let text = format!("{scope} | measure usage");
    let cli = format!("polylogue -- {}", polylogue_render::shell_quote(&text));
    let answer = view::<Line>(&web, &lifetime, line(text, None, false), None).await;
    answer_page(&[(&answer, &cli)], || {
        pages::cost(scope, &answer.value().unwrap()["reply"]["data"])
    })
}

/// One logical day as queries: its sessions, their usage during the day,
/// its month by day and the same day in other years.
async fn day(
    State(web): State<Arc<Web>>,
    Extension(lifetime): Extension<Lifetime>,
    UrlPath(date): UrlPath<String>,
) -> HttpResponse {
    let Ok(date) = date.parse::<jiff::civil::Date>() else {
        return failed(
            StatusCode::BAD_REQUEST,
            "bad_request",
            "a day is written YYYY-MM-DD",
        );
    };
    let mut answers = Vec::new();
    for query in pages::day_queries(date) {
        let cli = format!("polylogue -- {}", polylogue_render::shell_quote(&query));
        let answer = view::<Line>(&web, &lifetime, line(query, None, false), None).await;
        answers.push((answer, cli));
    }
    let details = answers
        .iter()
        .map(|(answer, cli)| (answer, cli.as_str()))
        .collect::<Vec<_>>();
    answer_page(&details, || {
        let replies = std::array::from_fn(|i| answers[i].0.value().unwrap()["reply"].clone());
        pages::day(date, &replies)
    })
}

async fn auth_page() -> HttpResponse {
    page(pages::auth())
}

#[derive(Deserialize)]
struct SignIn {
    #[serde(default)]
    token: String,
}

/// Exchanges the token for the archive's cookie. The page posts the token
/// from its URL fragment; the cookie is `SameSite=Strict` and `HttpOnly`.
async fn auth_exchange(
    State(web): State<Arc<Web>>,
    headers: HeaderMap,
    form: Result<Form<SignIn>, axum::extract::rejection::FormRejection>,
) -> HttpResponse {
    if !access::content_type(&headers, "application/x-www-form-urlencoded") {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "sign-in requires application/x-www-form-urlencoded",
        )
            .into_response();
    }
    let token = form.map(|Form(form)| form.token).unwrap_or_default();
    if !access::same_secret(&token, &web.token) {
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
    let page_cookie = format!("{}_page", access::cookie_name(&web.key));
    if let Some(page) = access::cookie_token(
        headers
            .get(header::COOKIE)
            .and_then(|value| value.to_str().ok()),
        &page_cookie,
    ) && let Ok(page) = HeaderValue::from_str(page)
    {
        r.headers_mut().insert("x-polylogue-return-to", page);
    }
    if let Ok(cookie) = HeaderValue::from_str(&format!(
        "{page_cookie}=; Path=/auth; HttpOnly; SameSite=Strict; Max-Age=0"
    )) {
        r.headers_mut().append(header::SET_COOKIE, cookie);
    }
    r
}

async fn asset(request: HttpRequest) -> HttpResponse {
    let (parts, _) = request.into_parts();
    let request = axum::http::Request::from_parts(parts, ());
    let theme = web_shell::theme::Theme {
        stylesheet: Some(include_str!("web/assets/theme.css").into()),
        fallback_reason: None,
    };
    web_shell::assets::serve_with_theme(&request, Some(&theme)).map(axum::body::Body::from)
}
async fn auth_js() -> HttpResponse {
    (
        [
            (header::CONTENT_TYPE, "text/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "no-store"),
        ],
        include_str!("web/assets/auth.js"),
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

/// An HTML failure page with the failure's shared HTTP status. The message
/// was already scrubbed of archive paths by `view`.
fn failed_with(failure: &command_surface::Failure) -> HttpResponse {
    let answer = Answer::<Value>::error(failure.clone());
    answer_page(&[(&answer, "")], String::new)
}

/// The data as the socket gives it, or a typed error with its HTTP status.
pub(crate) fn prepare_json_answer(
    web: &Web,
    response: impl Into<command_surface::Answer<serde_json::Value>>,
) -> (StatusCode, Vec<u8>) {
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
    (
        status,
        serde_json::to_vec(&value).expect("answer serializes"),
    )
}

pub fn json_answer(
    web: &Web,
    response: impl Into<command_surface::Answer<serde_json::Value>>,
) -> HttpResponse {
    let (status, bytes) = prepare_json_answer(web, response);
    (
        status,
        [(axum::http::header::CONTENT_TYPE, "application/json")],
        bytes,
    )
        .into_response()
}

#[cfg(test)]
mod tests;
